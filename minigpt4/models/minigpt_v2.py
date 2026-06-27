import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from minigpt4.common.registry import registry
from minigpt4.models.minigpt_base import MiniGPTBase


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
            seq_len=1000,
            patch_size=(1, 200),
            freeze_ecg=True,
            llama_model="Qwen/Qwen3-8B",
            prompt_template='###Human: {} ###Assistant: ',
            max_txt_len=300,
            end_sym='\n',
            lora_r=8,
            lora_target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            lora_alpha=16,
            lora_dropout=0.05,
            chat_template=False,
            use_grad_checkpoint_llm=False,
            max_context_len=3800,
            low_resource=False,  # use 8 bit and put vit in cpu
            device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
            freeze_phi=True,
            connector_hidden_size=None,
            pretrained_ecg="",
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

        self.connector_hidden_size = connector_hidden_size or self.llama_model.config.hidden_size
        logging.info(
            "Initializing ECG 2-layer MLP connector: %s -> %s -> %s",
            self.visual_encoder.dim,
            self.connector_hidden_size,
            self.llama_model.config.hidden_size,
        )
        self.llama_proj = nn.Sequential(
            nn.Linear(self.visual_encoder.dim, self.connector_hidden_size),
            nn.GELU(),
            nn.Linear(self.connector_hidden_size, self.llama_model.config.hidden_size),
        )

        if pretrained_ecg:
            self.load_ecg_encoder(pretrained_ecg)
        else:
            logging.info("No pretrained ECG encoder checkpoint configured.")
        self.chat_template = chat_template

        if use_grad_checkpoint_llm:
            self.llama_model.gradient_checkpointing_enable()
    
    def encode_ecg(self, ecg_signal):
        device = ecg_signal.device

        with self.maybe_autocast():
            ecg_embeds = self.ln_vision(self.visual_encoder.encode(ecg_signal)).to(device)
            inputs_llama = self.llama_proj(ecg_embeds)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(ecg_signal.device)
        return inputs_llama, atts_llama

    def encode_img(self, image):
        return self.encode_ecg(image)

    @classmethod
    def from_config(cls, cfg):
        ecg_model = cfg.get("ecg_model", "ecg_vit")
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

        lora_r = cfg.get("lora_r", 8)
        lora_target_modules = cfg.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
        lora_alpha = cfg.get("lora_alpha", 16)
        lora_dropout = cfg.get("lora_dropout", 0.05)
        chat_template = cfg.get("chat_template", False)
        pretrained_ecg = cfg.get("pretrained_ecg", "")
        connector_hidden_size = cfg.get("connector_hidden_size", None)

        use_grad_checkpoint_llm = cfg.get("use_grad_checkpoint_llm", False)
        max_context_len = cfg.get("max_context_len", 3800)

        model = cls(
            ecg_model=ecg_model,
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
            connector_hidden_size=connector_hidden_size,
            pretrained_ecg=pretrained_ecg,
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load ECG-GPT Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)

        return model
