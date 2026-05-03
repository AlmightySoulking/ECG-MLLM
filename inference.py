import argparse
import importlib
import json
import os
import warnings
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark inference for the ECG-encoder + Qwen2.5-3B model in this repo."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mimic-multi",
        choices=[
            "ptbxl",
            "cpsc",
            "csn",
            "european_st_t",
            "mit_bih_st",
            "mit_bih_arrhythmia",
            "european_st_t_long",
            "mit_bih_st_long",
            "mit_bih_arrhythmia_long",
            "ecgqa",
            "mimic-multi",
        ],
        help="Dataset to evaluate.",
    )
    parser.add_argument(
        "--dataset_subtype",
        type=str,
        choices=["all", "diag", "form", "rhythm", "sub-diag", "super-diag"],
        default="all",
    )
    parser.add_argument("--cfg-path", type=str, required=True, help="Model config path.")
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Checkpoint path to evaluate. When the config already defines a base checkpoint, this checkpoint is loaded on top of it.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for inference.",
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="Optional config overrides in key=value form.",
    )
    parser.add_argument("--sampling_freq", type=int, default=100, help="Sampling frequency.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Generation temperature.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p.")
    parser.add_argument("--num_beams", type=int, default=1, help="Beam count.")
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max generation length.")
    parser.add_argument("--model_seq_len", type=int, default=1000, help="Encoder sequence length.")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument(
        "--dataset-root",
        action="append",
        default=[],
        metavar="ALIAS=PATH",
        help="Map dataset alias to ECG root, for example mimic=/path/to/mimic.",
    )
    parser.add_argument(
        "--vis-root",
        type=str,
        default=None,
        help="Fallback ECG root for relative paths.",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default=None,
        help="Optional explicit annotation JSON path.",
    )
    parser.add_argument(
        "--classification_csv_dir",
        type=str,
        default=None,
        help="Directory containing ptbxl.csv / cpsc.csv / csn.csv.",
    )
    parser.add_argument(
        "--classification_json",
        type=str,
        default=None,
        help="Optional classification annotation JSON file for ptbxl/cpsc/csn instead of CSV metadata.",
    )
    parser.add_argument("--mask_first_non_zero_lead", action="store_true")
    parser.add_argument("--mask_second_non_zero_lead", action="store_true")
    parser.add_argument("--mask_random_non_zero_lead", action="store_true")
    parser.add_argument(
        "--text_embedding_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Text embedding model for metrics. Can be a HuggingFace model ID or local path.",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="outputs/inference_results.json",
        help="Where to save results.",
    )
    return parser


def ensure_parent_dir(path_value):
    Path(path_value).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def parse_dataset_roots(items):
    dataset_roots = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --dataset-root value '{item}'. Expected ALIAS=PATH.")
        alias, path = item.split("=", 1)
        alias = alias.strip().lower()
        path = os.path.expanduser(path.strip())
        if alias and path:
            dataset_roots[alias] = path
    return dataset_roots


def resolve_default_data_file(args):
    if args.data_file:
        return args.data_file

    candidates = []
    if args.dataset == "mimic-multi":
        candidates.append("data/mimic_llama3.3-70b_test.json")
    elif args.dataset == "ecgqa":
        candidates.extend(["data/ecgqa_test.json", "data/ecgqa_train.json"])
    else:
        candidates.append(f"data/{args.dataset}_test.json")

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        f"Could not find a data file for dataset '{args.dataset}'. Tried: {', '.join(candidates)}"
    )


def get_local_utils():
    return importlib.import_module("utils")


def update_local_utils_roots(local_utils, dataset_roots, vis_root=None):
    for alias, root in dataset_roots.items():
        local_utils.ecg_dir[alias] = root

    if vis_root:
        for alias in [
            "ptbxl",
            "cpsc",
            "csn",
            "mimic",
            "european_st_t",
            "mit_bih_st",
            "mit_bih_arrhythmia",
        ]:
            local_utils.ecg_dir.setdefault(alias, vis_root)


def resolve_ecg_candidates(raw_path, dataset_name, dataset_roots, vis_root):
    raw_path = os.path.expanduser(str(raw_path).strip())
    if not raw_path:
        return []

    if os.path.isabs(raw_path):
        return [raw_path]

    candidates = []
    dataset_name = (dataset_name or "").strip().lower()
    if dataset_name and dataset_name in dataset_roots:
        candidates.append(os.path.join(dataset_roots[dataset_name], raw_path))
    if vis_root:
        candidates.append(os.path.join(vis_root, raw_path))
    candidates.append(raw_path)

    deduped = []
    seen = set()
    for candidate in candidates:
        normalized = os.path.normpath(candidate)
        if normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped


def load_single_ecg_tensor(
    raw_path,
    dataset_name,
    sampling_freq,
    local_utils,
    dataset_roots,
    vis_root,
    ecg_start=None,
    ecg_end=None,
):
    last_error = None
    for candidate in resolve_ecg_candidates(raw_path, dataset_name, dataset_roots, vis_root):
        try:
            ecg = local_utils.get_ecg_from_path(candidate, sampling_freq, ecg_start, ecg_end)
            ecg = local_utils.ecg_transform(ecg)
            return torch.tensor(ecg, dtype=torch.float32).permute(1, 0)
        except Exception as exc:
            last_error = exc

        try:
            from minigpt4.datasets.datasets.ecg_dataset import ECGDataset

            ecg = ECGDataset._load_ecg_signal(candidate).float()
            return ecg
        except Exception as exc:
            last_error = exc

    raise FileNotFoundError(
        f"Could not load ECG '{raw_path}' for dataset '{dataset_name}'. Last error: {last_error}"
    )


class CompatFinetuningDataset(Dataset):
    def __init__(
        self,
        dataset_name,
        dataset_subtype,
        sampling_freq,
        local_utils,
        dataset_roots,
        csv_dir,
        data_file=None,
        split_fold="test",
    ):
        import ast
        import pandas as pd

        self.dataset = dataset_name
        self.dataset_subtype = dataset_subtype
        self.sampling_freq = sampling_freq
        self.local_utils = local_utils
        self._ast = ast
        self._pd = pd

        if csv_dir is None and data_file is None:
            raise ValueError(
                "Classification evaluation requires --classification_csv_dir or --classification_json because this repo does not ship the "
                "classification metadata."
            )

        if csv_dir is not None:
            if os.path.isfile(csv_dir):
                filename = csv_dir
            else:
                filename = os.path.join(csv_dir, f"{dataset_name}.csv")

            if not os.path.exists(filename):
                raise FileNotFoundError(f"Classification CSV not found: {filename}")

            df = pd.read_csv(filename)
        else:
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"Classification JSON not found: {data_file}")

            with open(data_file, "r", encoding="utf-8") as handle:
                raw = json.load(handle)

            if isinstance(raw, dict) and "samples" in raw:
                raw = raw["samples"]

            df = pd.DataFrame(raw)

        df = self._normalize_classification_metadata(df)

        df = df[df["split_fold"] == split_fold]
        self._label_test_, self._text_test_, self._ecg_root_ = self._get_label_text(local_utils, dataset_roots)

        if dataset_name == "ptbxl" and dataset_subtype in ["sub-diag", "super-diag"]:
            label_column = "sub_diag_labels" if dataset_subtype == "sub-diag" else "super_diag_labels"
            if label_column in df.columns:
                df[label_column] = df[label_column].apply(self._parse_label_list)
                for label in self._label_test_:
                    if label not in df.columns:
                        df[label] = df[label_column].apply(lambda x: 1 if label in x else 0)

        if dataset_name == "ptbxl":
            missing_label_columns = [label for label in self._label_test_ if label not in df.columns]
            if missing_label_columns:
                preview = ", ".join(missing_label_columns[:10])
                raise KeyError(
                    "Classification metadata is missing one-hot label columns required for "
                    f"'{dataset_name}/{dataset_subtype}'. Missing examples: {preview}"
                )
            df["label_len"] = df[self._label_test_].sum(axis=1)
            df = df[df["label_len"] > 0]
        else:
            missing_label_columns = [label for label in self._label_test_ if label not in df.columns]
            if missing_label_columns:
                preview = ", ".join(missing_label_columns[:10])
                raise KeyError(
                    "Classification metadata is missing one-hot label columns required for "
                    f"'{dataset_name}'. Missing examples: {preview}"
                )

        self.df = df.reset_index(drop=True)

    def _normalize_classification_metadata(self, df):
        df = df.copy()

        if "split_fold" not in df.columns:
            if "split" in df.columns:
                df = df.rename(columns={"split": "split_fold"})
            elif "strat_fold" in df.columns:
                df["split_fold"] = df["strat_fold"].apply(self._map_ptbxl_strat_fold)
            else:
                raise KeyError(
                    "Classification metadata is missing a split column. Expected 'split_fold' "
                    "or a compatible alternative such as 'split' or 'strat_fold'."
                )

        path_candidates = self._path_column_candidates()
        self._path_column = next((column for column in path_candidates if column in df.columns), None)
        if self._path_column is None:
            available = ", ".join(df.columns.astype(str).tolist())
            expected = ", ".join(path_candidates)
            raise KeyError(
                "Classification metadata is missing an ECG path column. "
                f"Expected one of: {expected}. Available columns: {available}"
            )

        if self._path_column != "path":
            df["path"] = df[self._path_column]

        return df

    def _path_column_candidates(self):
        if self.dataset == "ptbxl":
            preferred = ["filename_lr", "filename_hr"] if self.sampling_freq <= 100 else ["filename_hr", "filename_lr"]
            return ["path", "ecg_path", *preferred, "record_path", "signal_path", "file_name", "filename", "record", "image_path"]
        return ["path", "ecg_path", "record_path", "signal_path", "file_name", "filename", "record", "image_path"]

    @staticmethod
    def _map_ptbxl_strat_fold(value):
        try:
            fold = int(value)
        except (TypeError, ValueError):
            return value

        if fold == 10:
            return "test"
        if fold == 9:
            return "val"
        return "train"

    def _parse_label_list(self, value):
        if isinstance(value, (list, tuple, set)):
            return [str(item) for item in value]

        if value is None or self._pd.isna(value):
            return []

        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            try:
                parsed = self._ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                return [stripped]
            if isinstance(parsed, (list, tuple, set)):
                return [str(item) for item in parsed]
            return [str(parsed)]

        return [str(value)]

    def _resolve_sample_path(self, sample):
        raw_path = sample["path"]
        if isinstance(raw_path, (list, tuple)):
            if not raw_path:
                raise ValueError("Encountered an empty ECG path list in classification metadata.")
            raw_path = raw_path[0]

        if raw_path is None or self._pd.isna(raw_path):
            raise ValueError("Encountered a missing ECG path in classification metadata.")

        if isinstance(raw_path, str):
            stripped = raw_path.strip()
            if not stripped:
                raise ValueError("Encountered an empty ECG path in classification metadata.")
            if stripped[0] in "[(":
                try:
                    parsed = self._ast.literal_eval(stripped)
                except (ValueError, SyntaxError):
                    parsed = None
                if isinstance(parsed, (list, tuple)):
                    if not parsed:
                        raise ValueError("Encountered an empty ECG path list in classification metadata.")
                    raw_path = parsed[0]
                else:
                    raw_path = stripped
            else:
                raw_path = stripped

        return str(raw_path)

    def _get_label_text(self, local_utils, dataset_roots):
        if self.dataset == "ptbxl":
            label_test = {
                "all": local_utils._labels_,
                "diag": local_utils._diag_labels_,
                "form": local_utils._form_labels_,
                "rhythm": local_utils._rhythm_labels_,
                "sub-diag": local_utils._sub_diag_labels_,
                "super-diag": local_utils._super_diag_labels_,
            }[self.dataset_subtype]
        elif self.dataset == "csn":
            label_test = local_utils._csn_labels_
        elif self.dataset == "cpsc":
            label_test = local_utils._cpsc_labels_
        else:
            raise NotImplementedError(f"Unsupported classification dataset: {self.dataset}")

        if self.dataset == "ptbxl" and self.dataset_subtype in ["sub-diag", "super-diag"]:
            text_test = {
                "sub-diag": local_utils._sub_diag_text_,
                "super-diag": local_utils._super_diag_text_,
            }[self.dataset_subtype]
        else:
            text_test = label_test

        ecg_root = dataset_roots.get(self.dataset)
        if ecg_root is None:
            ecg_root = {
                "ptbxl": getattr(local_utils, "_ptbxl_dir_", None),
                "csn": getattr(local_utils, "_csn_dir_", None),
                "cpsc": getattr(local_utils, "_cpsc_dir_", None),
            }.get(self.dataset)

        if not ecg_root:
            raise ValueError(
                f"No ECG root configured for '{self.dataset}'. Pass --dataset-root {self.dataset}=PATH."
            )

        return label_test, text_test, ecg_root

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        sample = self.df.iloc[index]
        raw_path = self._resolve_sample_path(sample)
        last_error = None
        for ecg_path in resolve_ecg_candidates(
            raw_path=raw_path,
            dataset_name=self.dataset,
            dataset_roots={self.dataset: self._ecg_root_},
            vis_root=self._ecg_root_,
        ):
            try:
                ecg = self.local_utils.get_ecg_from_path(ecg_path, self.sampling_freq)
                break
            except Exception as exc:
                last_error = exc
        else:
            raise FileNotFoundError(
                "Could not load ECG from classification metadata row "
                f"{index} using column '{self._path_column}' and value '{raw_path}'. "
                f"Last error: {last_error}"
            )
        ecg = self.local_utils.ecg_transform(ecg)
        label = sample[self._label_test_].values.astype(np.float32)
        return ecg, label


class CompatFinetuningCollator:
    def __call__(self, batch):
        ecgs = np.array([item[0] for item in batch], dtype=np.float32)
        ecgs = torch.as_tensor(ecgs).permute(0, 2, 1)
        labels = np.array([item[1] for item in batch], dtype=np.float32)
        labels = torch.as_tensor(labels, dtype=torch.float32)
        return ecgs, labels


class InferenceCollator:
    def __init__(
        self,
        local_utils,
        dataset_roots,
        vis_root,
        sampling_freq,
        return_abnormal_type=False,
        return_question_type=False,
    ):
        self.local_utils = local_utils
        self.dataset_roots = dataset_roots
        self.vis_root = vis_root
        self.sampling_freq = sampling_freq
        self.return_abnormal_type = return_abnormal_type
        self.return_question_type = return_question_type

    def __call__(self, batch):
        ecgs, messages = [], []
        abnormal_types = []
        question_types = []

        for example in batch:
            dataset_name = example["dataset"]
            task = example["task"]
            ecg_paths = example["ecg_path"]

            ecg_start = example.get("ecg_start") if task in ["location", "location_long"] else None
            ecg_end = example.get("ecg_end") if task in ["location", "location_long"] else None

            if isinstance(ecg_paths, list):
                ecg_sample = [
                    load_single_ecg_tensor(
                        raw_path=path,
                        dataset_name=dataset_name,
                        sampling_freq=self.sampling_freq,
                        local_utils=self.local_utils,
                        dataset_roots=self.dataset_roots,
                        vis_root=self.vis_root,
                        ecg_start=ecg_start,
                        ecg_end=ecg_end,
                    )
                    for path in ecg_paths
                ]
            else:
                ecg_sample = load_single_ecg_tensor(
                    raw_path=ecg_paths,
                    dataset_name=dataset_name,
                    sampling_freq=self.sampling_freq,
                    local_utils=self.local_utils,
                    dataset_roots=self.dataset_roots,
                    vis_root=self.vis_root,
                    ecg_start=ecg_start,
                    ecg_end=ecg_end,
                )

            ecgs.append(ecg_sample)
            messages.append(
                [
                    {"role": "user", "content": example["question"]},
                    {"role": "assistant", "content": example["answer"]},
                ]
            )

            if self.return_abnormal_type:
                abnormal_types.append(example["abnormal_type"])
            if self.return_question_type:
                question_types.append(example["question_type"])

        if self.return_abnormal_type:
            return ecgs, messages, abnormal_types
        if self.return_question_type:
            return ecgs, messages, question_types
        return ecgs, messages


class RepoECGChatBackend:
    def __init__(self, model, args):
        self.model = model
        self.args = args

    @property
    def device(self):
        return next(self.model.parameters()).device

    def _extract_user_content(self, messages):
        for message in reversed(messages):
            if message.get("role") == "user":
                return message.get("content", "")
        return messages[0].get("content", "") if messages else ""

    def _build_prompt(self, question, num_ecgs):
        question = (
            question.replace("<|reserved_special_token_1|>", "")
            .replace("<|reserved_special_token_2|>", "")
            .replace("<|reserved_special_token_3|>", "")
            .strip()
        )

        # The model internals still expect the <ImageHere> placeholder token.
        # There is no image pipeline left in this script; this is only the
        # multimodal placeholder used by minigpt_v2 to splice ECG embeddings in.
        if num_ecgs == 1:
            instruction = f"<Img><ImageHere></Img> {question}".strip()
        else:
            slots = " ".join(f"ECG{i}: <Img><ImageHere></Img>" for i in range(1, num_ecgs + 1))
            instruction = f"{slots} Question: {question}".strip()

        if getattr(self.model, "chat_template", False):
            return self.model.prompt_template.format(instruction)
        return instruction

    def _split_signal(self, ecg):
        target_len = self.args.model_seq_len
        if ecg.shape[-1] <= target_len:
            if ecg.shape[-1] == target_len:
                return [ecg]
            return [torch.nn.functional.pad(ecg, (0, target_len - ecg.shape[-1]))]

        segments = []
        for start in range(0, ecg.shape[-1], target_len):
            segment = ecg[:, start : start + target_len]
            if segment.shape[-1] < target_len:
                segment = torch.nn.functional.pad(segment, (0, target_len - segment.shape[-1]))
            segments.append(segment)
        return segments

    def _encode_single_ecg(self, ecg):
        ecg = ecg.float().to(self.device)
        segments = self._split_signal(ecg)
        batch = torch.stack(segments, dim=0)
        embeddings, _ = self.model.encode_img(batch)

        if embeddings.shape[0] == 1:
            return embeddings[0]

        cls_token = embeddings[:, 0, :].mean(dim=0, keepdim=True)
        remaining_tokens = embeddings[:, 1:, :].reshape(-1, embeddings.shape[-1])
        return torch.cat([cls_token, remaining_tokens], dim=0)

    def _decode(self, output_tokens):
        bos_token_id = self.model.get_bos_token_id()
        if bos_token_id is not None and len(output_tokens) > 0 and output_tokens[0].item() == bos_token_id:
            output_tokens = output_tokens[1:]

        output_text = self.model.llama_tokenizer.decode(output_tokens, skip_special_tokens=True)
        output_text = output_text.split("###")[0]
        output_text = output_text.replace("<s>", "").replace("</s>", "")
        output_text = output_text.split(r"[/INST]")[-1].strip()
        output_text = output_text.split("Assistant:")[-1].strip()
        return output_text.strip()

    def generate_one(self, ecg_sample, messages):
        if isinstance(ecg_sample, torch.Tensor):
            ecg_list = [ecg_sample]
        else:
            ecg_list = list(ecg_sample)

        question = self._extract_user_content(messages)
        prompt = self._build_prompt(question, len(ecg_list))
        ecg_embedding_list = [self._encode_single_ecg(ecg).unsqueeze(0) for ecg in ecg_list]

        context_embeds = self.model.get_context_emb(prompt, ecg_embedding_list)
        attention_mask = torch.ones(
            context_embeds.shape[:2],
            dtype=torch.long,
            device=context_embeds.device,
        )

        with self.model.maybe_autocast():
            outputs = self.model.llama_model.generate(
                inputs_embeds=context_embeds,
                attention_mask=attention_mask,
                max_new_tokens=self.args.max_new_tokens,
                num_beams=self.args.num_beams,
                do_sample=self.args.do_sample,
                temperature=float(self.args.temperature),
                top_p=self.args.top_p,
                repetition_penalty=1.0,
                pad_token_id=self.model.llama_tokenizer.pad_token_id,
                bos_token_id=self.model.get_bos_token_id(),
                eos_token_id=self.model.get_eos_token_id(),
            )

        return self._decode(outputs[0])

    def generate_batch(self, ecgs, messages):
        return [self.generate_one(ecg_sample, message) for ecg_sample, message in zip(ecgs, messages)]


def load_repo_model(args):
    import minigpt4.tasks  # noqa: F401
    import minigpt4.datasets.builders  # noqa: F401
    import minigpt4.models  # noqa: F401
    import minigpt4.processors  # noqa: F401
    import minigpt4.runners  # noqa: F401
    from minigpt4.common.config import Config
    from minigpt4.common.registry import registry

    cfg = Config(args)
    base_ckpt = cfg.model_cfg.get("ckpt", "")
    target_ckpt = args.ckpt

    def _normalize_path(path_value):
        return os.path.abspath(os.path.expanduser(path_value)) if path_value else ""

    def _load_checkpoint_state_dict(path_value):
        checkpoint = torch.load(path_value, map_location="cpu")
        return checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    # Keep the config checkpoint as the base initialization when present.
    # Later-stage checkpoints in this repo often save only trainable parameters.
    if base_ckpt:
        cfg.model_cfg.ckpt = base_ckpt
    else:
        cfg.model_cfg.ckpt = target_ckpt

    model_cls = registry.get_model_class(cfg.model_cfg.arch)
    model = model_cls.from_config(cfg.model_cfg)

    if _normalize_path(target_ckpt) and _normalize_path(target_ckpt) != _normalize_path(base_ckpt):
        print("Load inference overlay checkpoint: {}".format(target_ckpt))
        overlay_state_dict = _load_checkpoint_state_dict(target_ckpt)
        msg = model.load_state_dict(overlay_state_dict, strict=False)
        print(f"Inference overlay load msg: {msg}")

    model = model.to(args.device)
    model.eval()
    return model


def get_json_loader(args, local_utils, dataset_roots, return_abnormal_type=False, return_question_type=False):
    from datasets import load_dataset

    data_file = resolve_default_data_file(args)
    dataset = load_dataset("json", data_files=data_file)["train"]
    collate_fn = InferenceCollator(
        local_utils=local_utils,
        dataset_roots=dataset_roots,
        vis_root=args.vis_root,
        sampling_freq=args.sampling_freq,
        return_abnormal_type=return_abnormal_type,
        return_question_type=return_question_type,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    return dataset, loader


def test_classification(args, get_response, local_utils, dataset_roots):
    from nltk.translate.bleu_score import corpus_bleu
    from rouge import Rouge
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics.pairwise import cosine_similarity

    test_data = CompatFinetuningDataset(
        dataset_name=args.dataset,
        dataset_subtype=args.dataset_subtype,
        sampling_freq=args.sampling_freq,
        local_utils=local_utils,
        dataset_roots=dataset_roots,
        csv_dir=args.classification_csv_dir,
        data_file=args.classification_json or args.data_file,
        split_fold="test",
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        collate_fn=CompatFinetuningCollator(),
        shuffle=False,
        pin_memory=True,
        num_workers=args.num_workers,
    )

    text_model = SentenceTransformer(args.text_embedding_model)
    label_embeddings = text_model.encode(test_data._text_test_)
    labels_all, prediction_all = [], []
    reports_all, reports_prediction_all = [], []

    for ecgs, labels in tqdm(test_loader, desc="Classification"):
        message = [[{"role": "user", "content": "Please provide the report for the following ECG."}] for _ in range(ecgs.shape[0])]
        response = get_response(ecgs, message)

        reports = [
            "Report: " + ", ".join([test_data._text_test_[i] for i, val in enumerate(label) if val == 1])
            for label in labels.cpu().numpy()
        ]
        reports_all.extend(reports)
        reports_prediction_all.extend(response)

        prediction_labels = [item.replace("Report: ", "").split(", ") for item in response]
        prediction_labels_embeddings = [text_model.encode(item) for item in prediction_labels]
        similarity = [cosine_similarity(label_embeddings, item) for item in prediction_labels_embeddings]
        prediction = [item.max(axis=1) for item in similarity]
        prediction_all.extend(prediction)
        labels_all.extend(labels.cpu().numpy())

    labels_all = np.array(labels_all)
    prediction_all = np.array(prediction_all)

    if len(labels_all) == 0:
        return {
            "error": "No test samples found in classification dataset. Check that your CSV/JSON has rows with split_fold='test' and valid labels."
        }

    auc_all = roc_auc_score(labels_all, prediction_all, average=None)

    bleu_1 = corpus_bleu([[r.split()] for r in reports_all], [r.split() for r in reports_prediction_all], weights=(1, 0, 0, 0))
    bleu_2 = corpus_bleu([[r.split()] for r in reports_all], [r.split() for r in reports_prediction_all], weights=(0, 1, 0, 0))
    bleu_3 = corpus_bleu([[r.split()] for r in reports_all], [r.split() for r in reports_prediction_all], weights=(0, 0, 1, 0))
    bleu_4 = corpus_bleu([[r.split()] for r in reports_all], [r.split() for r in reports_prediction_all], weights=(0, 0, 0, 1))
    rouge_scores = Rouge().get_scores(reports_prediction_all, reports_all, avg=True)

    return {
        "auc": auc_all.mean(),
        "detail": {test_data._text_test_[i]: auc_all[i] for i in range(len(auc_all))},
        "bleu_1": bleu_1,
        "bleu_2": bleu_2,
        "bleu_3": bleu_3,
        "bleu_4": bleu_4,
        "rouge_1": rouge_scores["rouge-1"]["f"],
        "rouge_2": rouge_scores["rouge-2"]["f"],
        "rouge_l": rouge_scores["rouge-l"]["f"],
    }


def test_localization(args, get_response, local_utils, dataset_roots):
    _, test_loader = get_json_loader(args, local_utils, dataset_roots, return_abnormal_type=True)
    mean_iou = {}

    for ecgs, messages, abnormal_types in tqdm(test_loader, desc="Localization"):
        question_message = [[item[0]] for item in messages]
        answer_message = [[item[1]] for item in messages]

        for ecg in ecgs:
            non_zero_leads = [i for i in range(ecg.shape[0]) if torch.sum(ecg[i]) != 0]
            if len(non_zero_leads) <= 1:
                continue
            if args.mask_first_non_zero_lead:
                ecg[non_zero_leads[0]] = 0
            if args.mask_second_non_zero_lead:
                ecg[non_zero_leads[1]] = 0
            if args.mask_random_non_zero_lead:
                ecg[np.random.choice(non_zero_leads)] = 0

        response = get_response(ecgs, question_message)

        for i in range(len(response)):
            truth = answer_message[i][0]["content"]
            prediction = response[i]
            abnormal_type = abnormal_types[i]
            if truth == prediction:
                iou = 1.0
            else:
                try:
                    iou = local_utils.compute_iou(truth, prediction)
                except Exception:
                    iou = 0.0

            mean_iou.setdefault(abnormal_type, {"mean_iou": 0.0, "count": 0})
            mean_iou[abnormal_type]["mean_iou"] += iou
            mean_iou[abnormal_type]["count"] += 1

    for key in mean_iou:
        mean_iou[key]["mean_iou"] /= mean_iou[key]["count"]

    macro_iou = np.mean([mean_iou[key]["mean_iou"] for key in mean_iou])
    micro_iou = (
        np.sum([mean_iou[key]["mean_iou"] * mean_iou[key]["count"] for key in mean_iou])
        / np.sum([mean_iou[key]["count"] for key in mean_iou])
    )

    return {
        "macro_iou": macro_iou,
        "micro_iou": micro_iou,
        "detail": mean_iou,
    }


def test_ecgqa(args, get_response, local_utils, dataset_roots):
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    _, test_loader = get_json_loader(args, local_utils, dataset_roots, return_question_type=True)
    text_model = SentenceTransformer(args.text_embedding_model)

    mean_acc = {}
    for ecgs, messages, question_types in tqdm(test_loader, desc="ECGQA"):
        question_message = [[item[0]] for item in messages]
        answer_message = [[item[1]] for item in messages]
        response = get_response(ecgs, question_message)

        for i in range(len(response)):
            truth = answer_message[i][0]["content"]
            prediction = response[i]
            question_type = question_types[i]

            if "verify" in question_type:
                label_embeddings = text_model.encode(["yes", "no", "not sure"])
                prediction_embeddings = text_model.encode([prediction])
                similarity = cosine_similarity(label_embeddings, prediction_embeddings)
                prediction = ["yes", "no", "not sure"][similarity.argmax()]

            acc = 1 if truth.lower() == prediction.lower() else 0
            mean_acc.setdefault(question_type, {"mean_acc": 0.0, "count": 0})
            mean_acc[question_type]["mean_acc"] += acc
            mean_acc[question_type]["count"] += 1

    for key in mean_acc:
        mean_acc[key]["mean_acc"] /= mean_acc[key]["count"]

    macro_acc = np.mean([mean_acc[key]["mean_acc"] for key in mean_acc])
    micro_acc = (
        np.sum([mean_acc[key]["mean_acc"] * mean_acc[key]["count"] for key in mean_acc])
        / np.sum([mean_acc[key]["count"] for key in mean_acc])
    )

    return {
        "macro_acc": macro_acc,
        "micro_acc": micro_acc,
        "detail": mean_acc,
    }


def test_mimic_multi(args, get_response, local_utils, dataset_roots):
    _, test_loader = get_json_loader(args, local_utils, dataset_roots)
    count = 0
    samples_all = []

    for ecgs, messages in tqdm(test_loader, desc="MIMIC multi"):
        question_message = [[item[0]] for item in messages]
        answer_message = [[item[1]] for item in messages]
        response = get_response(ecgs, question_message)
        for i in range(len(response)):
            samples_all.append(
                {
                    "id": count,
                    "question": question_message[i][0]["content"]
                    .replace("<|reserved_special_token_1|>", "")
                    .replace("<|reserved_special_token_2|>", "")
                    .replace("<|reserved_special_token_3|>", ""),
                    "answer": answer_message[i][0]["content"],
                    "prediction": response[i],
                }
            )
            count += 1

    return {"samples": samples_all}


def main():
    args = build_parser().parse_args()
    ensure_parent_dir(args.result_path)

    local_utils = get_local_utils()
    dataset_roots = parse_dataset_roots(args.dataset_root)
    update_local_utils_roots(local_utils, dataset_roots, args.vis_root)

    model = load_repo_model(args)
    get_response = RepoECGChatBackend(model, args).generate_batch

    if args.dataset in ["ptbxl", "cpsc", "csn"]:
        output_json = test_classification(args, get_response, local_utils, dataset_roots)
    elif args.dataset == "ecgqa":
        output_json = test_ecgqa(args, get_response, local_utils, dataset_roots)
    elif args.dataset == "mimic-multi":
        output_json = test_mimic_multi(args, get_response, local_utils, dataset_roots)
    else:
        output_json = test_localization(args, get_response, local_utils, dataset_roots)

    print(output_json)
    with open(args.result_path, "w") as handle:
        json.dump(output_json, handle, indent=4)


if __name__ == "__main__":
    main()
