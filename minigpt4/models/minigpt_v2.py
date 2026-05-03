import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from minigpt4.common.registry import registry
from minigpt4.models.base_model import disabled_train
from minigpt4.models.minigpt_base import MiniGPTBase
from minigpt4.models.Qformer import BertConfig, BertLMHeadModel


@registry.register_model("minigpt_v2")
class MiniGPTv2(MiniGPTBase):
    """
    MiniGPT-v2 model for ECG
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/minigpt_v2.yaml",
    }

    def __init__(
            self,
            ecg_model="ecg_vit",
            q_former_model="",
            seq_len=1000,
            patch_size=(1, 200),
            freeze_ecg=True,
            llama_model="",
            prompt_template='###Human: {} ###Assistant: ',
            max_txt_len=300,
            end_sym='\n',
            lora_r=0,
            lora_target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            lora_alpha=16,
            lora_dropout=0.05,
            chat_template=False,
            use_grad_checkpoint_llm=False,
            max_context_len=3800,
            low_resource=False,  # use 8 bit and put vit in cpu
            device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
            freeze_phi=True,
    ):
        super().__init__(
            ecg_model=ecg_model,
            seq_len=seq_len,
            patch_size=patch_size,
            llama_model=llama_model,
            max_txt_len=max_txt_len,
            max_context_len=max_context_len,
            end_sym=end_sym,
            prompt_template=prompt_template,
            low_resource=low_resource,
            device_8bit=device_8bit,
            lora_r=lora_r,
            lora_target_modules=lora_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            freeze_ecg=freeze_ecg,
            freeze_phi=freeze_phi,
        )

        logging.info("Initializing ECG Q-Former")
        state_dict = None
        qformer_width = self.visual_encoder.dim

        if q_former_model:
            logging.info("Loading Q-Former weights from %s", q_former_model)
            if q_former_model.startswith("http"):
                from minigpt4.common.dist_utils import download_cached_file
                cached_file = download_cached_file(q_former_model, check_hash=False, progress=True)
                checkpoint = torch.load(cached_file, map_location="cpu")
            else:
                checkpoint = torch.load(q_former_model, map_location="cpu")

            state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint

        # Detect encoder_width from checkpoint if provided.
        qformer_key = "Qformer.bert.encoder.layer.0.crossattention.self.key.weight"
        fallback_key = "bert.encoder.layer.0.crossattention.self.key.weight"
        if state_dict is not None and qformer_key in state_dict:
            qformer_width = state_dict[qformer_key].shape[1]
        elif state_dict is not None and fallback_key in state_dict:
            qformer_width = state_dict[fallback_key].shape[1]

        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token = 32, vision_width = qformer_width, freeze = True
        )
        
        if self.visual_encoder.dim != qformer_width:
            self.vision_proj = nn.Linear(self.visual_encoder.dim, qformer_width)
        else:
            self.vision_proj = nn.Identity()

        if state_dict is not None:
            qformer_weights = {
                k[len("Qformer."):]: v
                for k, v in state_dict.items()
                if k.startswith("Qformer.")
            }

            if not qformer_weights:
                qformer_weights = {
                    k: v
                    for k, v in state_dict.items()
                    if k.startswith("bert.")
                }

            if qformer_weights:
                msg = self.Qformer.load_state_dict(qformer_weights, strict=False)
                if "query_tokens" in state_dict:
                    self.query_tokens.data.copy_(state_dict["query_tokens"])
                logging.info(
                    "Loaded only Q-Former weights/query tokens from %s. "
                    "No BLIP vision encoder weights were connected to the ECG model.",
                    q_former_model,
                )
                logging.info("Q-Former load msg: %s", msg)
            else:
                logging.warning(
                    "No Q-Former-compatible weights were found in %s. "
                    "Continuing with a randomly initialized Q-Former.",
                    q_former_model,
                )
        else:
            logging.info(
                "No Q-Former checkpoint provided. "
                "The ECG model is not connected to any BLIP checkpoint."
            )
        logging.info("ECG Q-Former initialization done")

        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, 4096
        )
        self.llama_proj2 = nn.Linear(
            4096, self.llama_model.config.hidden_size
        )
        self.chat_template = chat_template

        if use_grad_checkpoint_llm:
            self.llama_model.gradient_checkpointing_enable()
    
    @classmethod
    def init_Qformer(cls, num_query_token, vision_width, freeze):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 2
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)

        Qformer.cls = None
        Qformer.bert.embeddings.word_embeddings = None
        Qformer.bert.embeddings.position_embeddings = None
        for layer in Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        if freeze:
            for name, param in Qformer.named_parameters():
                param.requires_grad = False
            Qformer = Qformer.eval()
            Qformer.train = disabled_train
            query_tokens.requires_grad = False
            logging.info("freeze Qformer")

        return Qformer, query_tokens

    def encode_ecg(self, ecg_signal):
        device = ecg_signal.device

        with self.maybe_autocast():
            ecg_embeds = self.ln_vision(self.visual_encoder.encode(ecg_signal)).to(device)
            if hasattr(self, 'vision_proj'):
                ecg_embeds = self.vision_proj(ecg_embeds)
            ecg_atts = torch.ones(ecg_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(ecg_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                    query_embeds=query_tokens,
                    encoder_hidden_states=ecg_embeds,
                    encoder_attention_mask=ecg_atts,
                    return_dict=True,
                )

            inputs_llama = self.llama_proj(query_output.last_hidden_state)
            inputs_llama = self.llama_proj2(inputs_llama)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(ecg_signal.device)
        return inputs_llama, atts_llama

    def encode_img(self, image):
        return self.encode_ecg(image)

    @classmethod
    def from_config(cls, cfg):
        ecg_model = cfg.get("ecg_model", "ecg_vit")
        q_former_model = cfg.get("q_former_model", "")
        seq_len = cfg.get("seq_len", 1000)
        patch_size = cfg.get("patch_size", (1, 200))
        llama_model = cfg.get("llama_model")

        freeze_ecg = cfg.get("freeze_ecg", True)
        freeze_phi = cfg.get("freeze_phi", cfg.get("freeze_llm", True))
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_template = cfg.get("prompt_template", '###Human: {} ###Assistant: ')
        max_txt_len = cfg.get("max_txt_len", 300)
        end_sym = cfg.get("end_sym", '\n')

        lora_r = cfg.get("lora_r", 0 if freeze_phi else 64)
        lora_target_modules = cfg.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
        lora_alpha = cfg.get("lora_alpha", 16)
        lora_dropout = cfg.get("lora_dropout", 0.05)
        chat_template = cfg.get("chat_template", False)

        use_grad_checkpoint_llm = cfg.get("use_grad_checkpoint_llm", False)
        max_context_len = cfg.get("max_context_len", 3800)

        model = cls(
            ecg_model=ecg_model,
            q_former_model=q_former_model,
            seq_len=seq_len,
            patch_size=patch_size,
            freeze_ecg=freeze_ecg,
            freeze_phi=freeze_phi,
            llama_model=llama_model,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            low_resource=low_resource,
            device_8bit=device_8bit,
            end_sym=end_sym,
            lora_r=lora_r,
            lora_target_modules=lora_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            chat_template=chat_template,
            use_grad_checkpoint_llm=use_grad_checkpoint_llm,
            max_context_len=max_context_len,
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load ECG-GPT Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        return model
