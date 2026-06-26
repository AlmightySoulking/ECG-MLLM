import sys
import yaml

with open("train_configs/tinygptv_stage1.yaml", "r") as f:
    cfg = yaml.safe_load(f)

print(cfg['model'].get('pretrained_ecg', 'NOT_FOUND'))
