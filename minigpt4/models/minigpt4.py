import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from minigpt4.common.registry import registry
from minigpt4.models.minigpt_base import MiniGPTBase


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
            seq_len=1000,
            patch_size=(1, 200),
            freeze_ecg=True,
            llama_model="",
            prompt_path="",
            prompt_template="",
            max_txt_len=32,
            end_sym='\n',
            low_resource=False,  # use 8 bit and put vit in cpu
            device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
            lora_r=8,
            lora_target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            lora_alpha=16,
            lora_dropout=0.05,
            freeze_phi=True,
            pretrained_ecg="",
            connector_hidden_size=None,
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
        logging.info("Loading ECG encoder from %s", ckpt_path)
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
        logging.info("ECG encoder load msg: %s", msg)
        print(f"ECG encoder load msg: {msg}")

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

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')

        lora_r = cfg.get("lora_r", 8)
        lora_target_modules = cfg.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
        lora_alpha = cfg.get("lora_alpha", 16)
        lora_dropout = cfg.get("lora_dropout", 0.05)

        pretrained_ecg = cfg.get("pretrained_ecg", "")
        connector_hidden_size = cfg.get("connector_hidden_size", None)

        model = cls(
            ecg_model=ecg_model,
            seq_len=seq_len,
            patch_size=patch_size,
            freeze_ecg=freeze_ecg,
            freeze_phi=freeze_phi,
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
            connector_hidden_size=connector_hidden_size,
        )

        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load ECG-GPT Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt['model'] if 'model' in ckpt else ckpt
            msg = model.load_state_dict(state_dict, strict=False)
            print(f"Full model load msg: {msg}")

        return model
