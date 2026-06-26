import yaml
from omegaconf import OmegaConf

class MockRegistry:
    def get_model_class(self, arch):
        return MockModel

class MockModel:
    @classmethod
    def from_config(cls, cfg):
        pretrained_ecg = cfg.get("pretrained_ecg", "")
        if pretrained_ecg:
            print("FOUND:", pretrained_ecg)
        else:
            print("NOT FOUND, default to empty string")
        return cls()

import minigpt4.common.config as config_module
config_module.registry = MockRegistry()

class MockArgs:
    def __init__(self):
        self.cfg_path = "train_configs/tinygptv_stage1.yaml"
        self.options = None

config = config_module.Config(MockArgs())
print("cfg.model_cfg.pretrained_ecg =", config.model_cfg.get("pretrained_ecg", ""))
model = MockRegistry().get_model_class(config.model_cfg.arch).from_config(config.model_cfg)
