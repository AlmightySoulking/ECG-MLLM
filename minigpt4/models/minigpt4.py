import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from minigpt4.common.registry import registry
from minigpt4.models.base_model import disabled_train
from minigpt4.models.minigpt_base import MiniGPTBase
from minigpt4.models.Qformer import BertConfig, BertLMHeadModel


@registry.register_model("minigpt4")
class MiniGPT4(MiniGPTBase):
    """
    MiniGPT-4 model for ECG
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna0": "configs/models/minigpt4_vicuna0.yaml",
        "pretrain_llama2": "configs/models/minigpt4_llama2.yaml",
    }

    def __init__(
            self,
            ecg_model="ecg_vit",
            q_former_model="",
            seq_len=1000,
            patch_size=(1, 200),
            freeze_ecg=True,
            freeze_qformer=True,
            num_query_token=32,
            llama_model="",
            prompt_path="",
            prompt_template="",
            max_txt_len=32,
            end_sym='\n',
            low_resource=False,  # use 8 bit and put vit in cpu
            device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
            lora_r=0,
            lora_target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            lora_alpha=16,
            lora_dropout=0.05,
            freeze_phi=True,
            pretrained_ecg="/home/cmpdil/iit_profbehra2/pretrained/vit_base_sigmoid/model_epoch9 copy.bin",
    ):
        super().__init__(
            ecg_model=ecg_model,
            seq_len=seq_len,
            patch_size=patch_size,
            llama_model=llama_model,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            lora_r=lora_r,
            lora_target_modules=lora_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            freeze_ecg=freeze_ecg,
            freeze_phi=freeze_phi,
        )

        self.has_qformer = True
        if self.has_qformer:
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
                num_query_token, qformer_width, freeze_qformer
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
        
        if pretrained_ecg:
            self.load_ecg_encoder(pretrained_ecg)
        
        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, 4096
        )
        self.llama_proj2 = nn.Linear(
            4096, self.llama_model.config.hidden_size
        )

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []

    def load_ecg_encoder(self, ckpt_path):
        print(f"Loading ECG encoder from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        
        # Robust key matching for different prefixes
        new_state_dict = {}
        prefixes = ["visual_encoder.", "ecg_model.model.", "model."]
        for k, v in state_dict.items():
            matched = False
            for p in prefixes:
                if k.startswith(p):
                    new_state_dict[k[len(p):]] = v
                    matched = True
                    break
            if not matched:
                new_state_dict[k] = v
        
        msg = self.visual_encoder.load_state_dict(new_state_dict, strict=False)
        print(f"ECG encoder load msg: {msg}")

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
            if self.has_qformer:
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

            else:
                inputs_llama = self.llama_proj(ecg_embeds)
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
        num_query_token = cfg.get("num_query_token", 32)
        llama_model = cfg.get("llama_model")

        freeze_ecg = cfg.get("freeze_ecg", True)
        freeze_phi = cfg.get("freeze_phi", cfg.get("freeze_llm", True))
        has_qformer = cfg.get("has_qformer", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')

        lora_r = cfg.get("lora_r", 0 if freeze_phi else 64)
        lora_target_modules = cfg.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
        lora_alpha = cfg.get("lora_alpha", 16)
        lora_dropout = cfg.get("lora_dropout", 0.05)

        pretrained_ecg = cfg.get("pretrained_ecg", "")

        model = cls(
            ecg_model=ecg_model,
            q_former_model=q_former_model,
            seq_len=seq_len,
            patch_size=patch_size,
            freeze_ecg=freeze_ecg,
            freeze_phi=freeze_phi,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            lora_r=lora_r,
            lora_target_modules=lora_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            pretrained_ecg=pretrained_ecg,
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load ECG-GPT Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt['model'] if 'model' in ckpt else ckpt
            msg = model.load_state_dict(state_dict, strict=False)
            print(f"Full model load msg: {msg}")

        return model
