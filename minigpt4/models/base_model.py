"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import logging
import contextlib

from omegaconf import OmegaConf
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)

from minigpt4.common.dist_utils import download_cached_file
from minigpt4.common.utils import get_abs_path, is_url


class BaseModel(nn.Module):
    """Base class for models."""

    def __init__(self):
        super().__init__()

    @property
    def device(self):
        return list(self.parameters())[-1].device

    def load_checkpoint(self, url_or_filename):
        """
        Load from a finetuned checkpoint.

        This should expect no mismatch in the model keys and the checkpoint keys.
        """

        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        if "model" in checkpoint.keys():
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        msg = self.load_state_dict(state_dict, strict=False)

        logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Build a pretrained model from default configuration file, specified by model_type.

        Args:
            - model_type (str): model type, specifying architecture and checkpoints.

        Returns:
            - model (nn.Module): pretrained or finetuned model, depending on the configuration.
        """
        model_cfg = OmegaConf.load(cls.default_config_path(model_type)).model
        model = cls.from_config(model_cfg)

        return model

    @classmethod
    def default_config_path(cls, model_type):
        assert (
            model_type in cls.PRETRAINED_MODEL_CONFIG_DICT
        ), "Unknown model type {}".format(model_type)
        return get_abs_path(cls.PRETRAINED_MODEL_CONFIG_DICT[model_type])

    def load_checkpoint_from_config(self, cfg, **kwargs):
        """
        Load checkpoint as specified in the config file.

        If load_finetuned is True, load the finetuned model; otherwise, load the pretrained model.
        When loading the pretrained model, each task-specific architecture may define their
        own load_from_pretrained() method.
        """
        load_finetuned = cfg.get("load_finetuned", True)
        if load_finetuned:
            finetune_path = cfg.get("finetuned", None)
            assert (
                finetune_path is not None
            ), "Found load_finetuned is True, but finetune_path is None."
            self.load_checkpoint(url_or_filename=finetune_path)
        else:
            # load pre-trained weights
            pretrain_path = cfg.get("pretrained", None)
            assert "Found load_finetuned is False, but pretrain_path is None."
            self.load_from_pretrained(url_or_filename=pretrain_path, **kwargs)

    def before_evaluation(self, **kwargs):
        pass

    def show_n_params(self, return_str=True):
        tot = 0
        for p in self.parameters():
            w = 1
            for x in p.shape:
                w *= x
            tot += w
        if return_str:
            if tot >= 1e6:
                return "{:.1f}M".format(tot / 1e6)
            else:
                return "{:.1f}K".format(tot / 1e3)
        else:
            return tot

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    @classmethod
    def init_ecg_encoder(
        cls, model_name, patch_size=(1, 200), seq_len=1000, freeze=True
    ):
        logging.info('Loading ECG VIT')
        from .ecg_vit import ecg_vit, ecg_vit_base, ecg_vit_large, ecg_vit_giant                                                                 
                                                                                                                                                
        if model_name == "ecg_vit":
            visual_encoder = ecg_vit(patch_size=patch_size, seq_len=seq_len)
        elif model_name == "ecg_vit_base":
            visual_encoder = ecg_vit_base(patch_size=patch_size, seq_len=seq_len)
        elif model_name == "ecg_vit_large":
            visual_encoder = ecg_vit_large(patch_size=patch_size, seq_len=seq_len)
        elif model_name == "ecg_vit_giant":
            visual_encoder = ecg_vit_giant(patch_size=patch_size, seq_len=seq_len)
        else:
            raise ValueError(f"Unknown ECG encoder: {model_name}")                                                                               

        ln_vision = LayerNorm(visual_encoder.dim)

        if freeze:
            for name, param in visual_encoder.named_parameters():
                param.requires_grad = False
            visual_encoder = visual_encoder.eval()
            visual_encoder.train = disabled_train
            for name, param in ln_vision.named_parameters():
                param.requires_grad = False
            ln_vision = ln_vision.eval()
            ln_vision.train = disabled_train
            logging.info("freeze ecg encoder")

        logging.info('Loading ECG VIT Done')
        return visual_encoder, ln_vision

    def init_llm(cls, llama_model_path, low_resource=False, low_res_device=0, lora_r=0,
                 lora_target_modules=None, freeze_phi=True, **lora_kargs):
        logging.info("Loading LLM")

        if freeze_phi and lora_r > 0:
            logging.warning(
                "freeze_phi=True with lora_r>0 leaves LoRA adapters trainable. "
                "Set lora_r=0 to keep the Qwen LLM fully frozen."
            )

        llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_path)
        if llama_tokenizer.pad_token is None:
            if llama_tokenizer.eos_token is not None:
                llama_tokenizer.pad_token = llama_tokenizer.eos_token
            elif llama_tokenizer.unk_token is not None:
                llama_tokenizer.pad_token = llama_tokenizer.unk_token
            else:
                raise ValueError(
                    f"Tokenizer for {llama_model_path} is missing pad/eos/unk tokens, "
                    "so a pad token could not be inferred."
                )

        if low_resource:
            llama_model = AutoModelForCausalLM.from_pretrained(
                llama_model_path,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': low_res_device},
            )
        else:
            llama_model = AutoModelForCausalLM.from_pretrained(
                llama_model_path,
                torch_dtype=torch.float16,
            )

        llama_model.config.pad_token_id = llama_tokenizer.pad_token_id
        if getattr(llama_model, "generation_config", None) is not None:
            if llama_model.generation_config.pad_token_id is None:
                llama_model.generation_config.pad_token_id = llama_tokenizer.pad_token_id
            if (
                llama_model.generation_config.eos_token_id is None
                and llama_tokenizer.eos_token_id is not None
            ):
                llama_model.generation_config.eos_token_id = llama_tokenizer.eos_token_id
            if (
                llama_model.generation_config.bos_token_id is None
                and llama_tokenizer.bos_token_id is not None
            ):
                llama_model.generation_config.bos_token_id = llama_tokenizer.bos_token_id

        lora_target_modules = list(
            lora_target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
        )

        if lora_r > 0:
            if low_resource:
                llama_model = prepare_model_for_int8_training(llama_model)
            loraconfig = LoraConfig(
                r=lora_r,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=lora_target_modules,
                **lora_kargs
            )
            llama_model = get_peft_model(llama_model, loraconfig)

            llama_model.print_trainable_parameters()
            if not freeze_phi:
                for name, param in llama_model.named_parameters():
                    if "layernorm" in name.lower() or "norm" in name.lower():
                        param.requires_grad = True
                        param.data = param.data.float()

        else:
            for name, param in llama_model.named_parameters():
                param.requires_grad = False
                
            if not freeze_phi:
                for name, param in llama_model.named_parameters():
                    if "layernorm" in name.lower() or "norm" in name.lower():
                        param.requires_grad = True
                        param.data = param.data.float()

        logging.info("Loading LLM Done")
        return llama_model, llama_tokenizer


    def load_from_pretrained(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location="cpu")
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location="cpu")
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]

        msg = self.load_state_dict(state_dict, strict=False)

        # logging.info("Missing keys {}".format(msg.missing_keys))
        logging.info("load checkpoint from %s" % url_or_filename)

        return msg


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)



