"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import json
import os
from typing import Iterable

from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data.dataloader import default_collate




class BaseDataset(Dataset):
    def __init__(
        self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=[]
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.annotation = []
        normalized_ann_paths = self._normalize_ann_paths(ann_paths)
        for ann_path in normalized_ann_paths:
            if os.path.isdir(ann_path):
                raise IsADirectoryError(
                    f"Annotation path must be a JSON file, but got directory: {ann_path}"
                )
            with open(ann_path, "r") as f:
                ann = json.load(f)
            if isinstance(ann, dict):
                self.annotation.extend(ann.get('annotations', []))
            else:
                self.annotation.extend(ann)
    
        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        return default_collate(samples)

    def set_processors(self, vis_processor, text_processor):
        self.vis_processor = vis_processor
        self.text_processor = text_processor

    @staticmethod
    def _normalize_ann_paths(ann_paths):
        if ann_paths is None:
            return []

        if isinstance(ann_paths, str):
            ann_paths = [ann_paths]

        normalized_ann_paths = []
        for ann_path in ann_paths:
            if ann_path is None:
                continue

            normalized_ann_path = str(ann_path).strip()
            if not normalized_ann_path:
                continue

            normalized_ann_paths.append(os.path.expanduser(normalized_ann_path))

        return normalized_ann_paths

    def _add_instance_ids(self, key="instance_id"):
        for idx, ann in enumerate(self.annotation):
            ann[key] = str(idx)



class ConcatDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)

    def collater(self, samples):
        # TODO For now only supports datasets with same underlying collater implementations

        all_keys = set()
        for s in samples:
            all_keys.update(s)

        shared_keys = all_keys
        for s in samples:
            shared_keys = shared_keys & set(s.keys())

        samples_shared_keys = []
        for s in samples:
            samples_shared_keys.append({k: s[k] for k in s.keys() if k in shared_keys})

        return self.datasets[0].collater(samples_shared_keys)
