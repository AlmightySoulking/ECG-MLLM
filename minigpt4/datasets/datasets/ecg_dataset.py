import logging
import os
import re

import numpy as np
import torch
from scipy.signal import resample

from minigpt4.datasets.datasets.base_dataset import BaseDataset


STANDARD_12_LEADS = [
    "i",
    "ii",
    "iii",
    "avr",
    "avl",
    "avf",
    "v1",
    "v2",
    "v3",
    "v4",
    "v5",
    "v6",
]
TENSOR_EXTENSIONS = (".npy", ".pt", ".pth")
WFDB_EXTENSIONS = (".hea", ".dat")
WFDB_MAX_SAMPLES = 5000
TARGET_SAMPLING_FREQ = 100
TARGET_SIGNAL_LENGTH = 1000
LIKELY_ABSOLUTE_UNIX_ROOTS = (
    "etc",
    "home",
    "media",
    "mnt",
    "opt",
    "root",
    "srv",
    "tmp",
    "usr",
    "var",
)
TRAILING_ALPHA_SUFFIX_AFTER_VERSION_RE = re.compile(r"^(?P<prefix>.+\d(?:\.\d+)*)(?P<suffix>[A-Za-z]+)$")

class ECGDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of ECG signals
        ann_paths (list): list of paths to the annotation files
        """
        vis_root = self._normalize_optional_path(vis_root)
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        normalized_ann_paths = self._normalize_ann_paths(ann_paths)
        self.annotation_roots = tuple(
            os.path.dirname(os.path.abspath(ann_path))
            for ann_path in normalized_ann_paths
            if ann_path
        )
        self._warned_missing_vis_root = False
        self._warned_missing_dataset_root = False

    @staticmethod
    def _normalize_optional_path(path_value):
        if path_value is None:
            return None

        normalized = str(path_value).strip()
        if not normalized or normalized.lower() == "none":
            return None

        return os.path.expanduser(normalized)

    @staticmethod
    def _extract_signal_id(annotation):
        signal_id = annotation.get("ecg_path", annotation.get("ecg_id", annotation.get("image_path", "")))

        if isinstance(signal_id, list):
            signal_id = signal_id[0] if signal_id else ""

        if signal_id is None:
            return ""

        signal_id = str(signal_id).strip()
        if signal_id.lower() == "none":
            return ""

        return signal_id

    @staticmethod
    def _extract_dataset_root(annotation):
        dataset_root = annotation.get("dataset")

        if isinstance(dataset_root, list):
            dataset_root = dataset_root[0] if dataset_root else ""

        if dataset_root is None:
            return None

        dataset_root = str(dataset_root).strip()
        if not dataset_root or dataset_root.lower() == "none":
            return None

        return os.path.expanduser(dataset_root)

    @staticmethod
    def _strip_accidental_dataset_suffix(dataset_root):
        root_dir = os.path.dirname(dataset_root)
        root_name = os.path.basename(dataset_root)
        match = TRAILING_ALPHA_SUFFIX_AFTER_VERSION_RE.match(root_name)
        if not match:
            return dataset_root

        return os.path.normpath(os.path.join(root_dir, match.group("prefix")))

    @staticmethod
    def _resolve_dataset_root_candidates(dataset_root):
        if dataset_root is None:
            return []

        normalized_root = os.path.normpath(dataset_root)
        candidate_roots = []
        seen_roots = set()

        def add_candidate(candidate_root):
            normalized_candidate = os.path.normpath(candidate_root)
            if normalized_candidate not in seen_roots:
                seen_roots.add(normalized_candidate)
                candidate_roots.append(normalized_candidate)

        # Some annotations store Unix absolute paths without the leading slash
        # (for example "home/..."). Prefer the corrected absolute root.
        if not os.path.isabs(normalized_root):
            root_head = normalized_root.split(os.path.sep, 1)[0]
            if root_head in LIKELY_ABSOLUTE_UNIX_ROOTS:
                add_candidate(os.path.join(os.path.sep, normalized_root))
            else:
                add_candidate(normalized_root)
        else:
            add_candidate(normalized_root)

        # Some generated annotations accidentally append a dataset alias directly
        # to a versioned directory name, e.g. "...subset-1.0mimic". Strip that
        # obvious corruption, but still keep the JSON-provided root as primary.
        for candidate_root in list(candidate_roots):
            repaired_root = ECGDataset._strip_accidental_dataset_suffix(candidate_root)
            add_candidate(repaired_root)

        return candidate_roots

    def _resolve_signal_base_paths(self, annotation):
        signal_id = self._extract_signal_id(annotation)
        if not signal_id:
            raise FileNotFoundError("Missing ECG path in annotation entry.")

        signal_id = os.path.expanduser(signal_id)
        if os.path.isabs(signal_id):
            return [signal_id]

        dataset_root = self._extract_dataset_root(annotation)
        dataset_root_candidates = self._resolve_dataset_root_candidates(dataset_root)
        if dataset_root_candidates:
            return [
                os.path.normpath(os.path.join(dataset_root_candidate, signal_id))
                for dataset_root_candidate in dataset_root_candidates
            ]

        if not self._warned_missing_dataset_root:
            logging.warning(
                "ECG annotation entries without a `dataset` field are falling back to `vis_root` and annotation "
                "relative resolution."
            )
            self._warned_missing_dataset_root = True

        if self.vis_root is None and not self._warned_missing_vis_root:
            logging.warning(
                "ECG dataset `vis_root` is not set; resolving relative ECG paths from the current working "
                "directory and annotation file locations."
            )
            self._warned_missing_vis_root = True

        candidate_paths = []
        if self.vis_root is not None:
            candidate_paths.append(os.path.join(self.vis_root, signal_id))

        candidate_paths.append(signal_id)
        candidate_paths.extend(os.path.join(ann_root, signal_id) for ann_root in self.annotation_roots)

        deduped_paths = []
        seen_paths = set()
        for candidate in candidate_paths:
            normalized_candidate = os.path.normpath(candidate)
            if normalized_candidate not in seen_paths:
                seen_paths.add(normalized_candidate)
                deduped_paths.append(normalized_candidate)

        return deduped_paths

    @staticmethod
    def _resolve_tensor_path(base_path):
        if os.path.isfile(base_path) and base_path.endswith(TENSOR_EXTENSIONS):
            return base_path

        for extension in TENSOR_EXTENSIONS:
            candidate = base_path if base_path.endswith(extension) else base_path + extension
            if os.path.isfile(candidate):
                return candidate

        return None

    @staticmethod
    def _parse_gain_and_baseline(signal_spec):
        spec = signal_spec.split("/", 1)[0]
        baseline = None

        if "(" in spec and ")" in spec:
            gain_spec, baseline_spec = spec.split("(", 1)
            baseline = float(baseline_spec.split(")", 1)[0])
        else:
            gain_spec = spec

        gain = float(gain_spec) if gain_spec not in {"", "0"} else 1.0

        return gain, baseline

    @classmethod
    def _parse_wfdb_header(cls, header_path):
        with open(header_path, "r") as handle:
            lines = [line.strip() for line in handle if line.strip() and not line.startswith("#")]

        if not lines:
            raise ValueError(f"WFDB header is empty: {header_path}")

        record_fields = lines[0].split()
        if len(record_fields) < 3:
            raise ValueError(f"Malformed WFDB header: {header_path}")

        num_signals = int(record_fields[1])
        sampling_freq = float(record_fields[2].split("/", 1)[0])
        signal_lines = lines[1 : 1 + num_signals]

        if len(signal_lines) != num_signals:
            raise ValueError(f"WFDB header lists {num_signals} signals but only {len(signal_lines)} were found.")

        dat_file = None
        gains = []
        baselines = []
        lead_names = []

        for line in signal_lines:
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"Malformed WFDB signal line: {line}")

            if dat_file is None:
                dat_file = fields[0]

            wfdb_format = fields[1]
            # Accept both '16' and '212' formats, let wfdb handle reading
            if wfdb_format not in ("16", "212"):
                raise ValueError(f"Unsupported WFDB format '{wfdb_format}' in {header_path}")

            gain, baseline = cls._parse_gain_and_baseline(fields[2] if len(fields) > 2 else "1")
            if baseline is None:
                baseline = float(fields[4]) if len(fields) > 4 else 0.0

            gains.append(gain if abs(gain) > 1e-6 else 1.0)
            baselines.append(baseline)
            lead_names.append(fields[-1].lower())

        dat_path = os.path.join(os.path.dirname(header_path), dat_file)

        return dat_path, num_signals, sampling_freq, np.asarray(gains), np.asarray(baselines), lead_names

    @staticmethod
    def _reorder_leads(signal, lead_names):
        reordered = np.zeros((signal.shape[0], len(STANDARD_12_LEADS)), dtype=np.float32)

        for lead_index, lead_name in enumerate(STANDARD_12_LEADS):
            if lead_name in lead_names:
                reordered[:, lead_index] = signal[:, lead_names.index(lead_name)]

        return reordered

    @staticmethod
    def _normalize_signal(signal):
        min_vals = np.min(signal, axis=0, keepdims=True)
        max_vals = np.max(signal, axis=0, keepdims=True)
        ranges = max_vals - min_vals
        ranges[ranges < 1e-6] = 1.0

        normalized = (signal - min_vals) / ranges * 2 - 1

        return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    @staticmethod
    def _pad_or_trim_time_major(signal, target_length=TARGET_SIGNAL_LENGTH):
        if signal.shape[0] < target_length:
            pad_width = target_length - signal.shape[0]
            signal = np.pad(signal, ((0, pad_width), (0, 0)))
        elif signal.shape[0] > target_length:
            signal = signal[:target_length]

        return signal.astype(np.float32)

    @classmethod
    def _load_wfdb_signal(cls, base_path):
        import wfdb
        if base_path.endswith(WFDB_EXTENSIONS):
            base_path = os.path.splitext(base_path)[0]

        header_path = base_path + ".hea"
        if not os.path.isfile(header_path):
            raise FileNotFoundError(f"WFDB header not found at {header_path}")

        dat_path, num_signals, sampling_freq, gains, baselines, lead_names = cls._parse_wfdb_header(header_path)
        if not os.path.isfile(dat_path):
            raise FileNotFoundError(f"WFDB data file not found at {dat_path}")

        # Use wfdb to read the signal (supports both format 16 and 212)
        record = wfdb.rdrecord(os.path.splitext(dat_path)[0], physical=True)
        signal = record.p_signal
        if signal is None:
            # Fallback: try reading digital signal and convert manually
            record = wfdb.rdrecord(os.path.splitext(dat_path)[0], physical=False)
            if record.d_signal is not None:
                signal = record.d_signal.astype(np.float32)
                signal = (signal - baselines.reshape(1, -1)) / gains.reshape(1, -1)
            else:
                raise ValueError(f"Could not read signal from {dat_path}")
        else:
            signal = signal.astype(np.float32)
            signal = (signal - baselines.reshape(1, -1)) / gains.reshape(1, -1)

        signal = signal[:WFDB_MAX_SAMPLES]

        if sampling_freq != TARGET_SAMPLING_FREQ:
            target_length = max(1, int(round(signal.shape[0] * TARGET_SAMPLING_FREQ / sampling_freq)))
            signal = resample(signal, target_length, axis=0)

        signal = cls._reorder_leads(signal, lead_names)
        signal = cls._normalize_signal(signal)
        signal = cls._pad_or_trim_time_major(signal)

        return torch.from_numpy(np.ascontiguousarray(signal.T))

    @classmethod
    def _load_tensor_signal(cls, signal_path):
        if signal_path.endswith(".npy"):
            signal = np.load(signal_path)
        else:
            signal = torch.load(signal_path, map_location="cpu")

        if isinstance(signal, torch.Tensor):
            signal = signal.detach().cpu().numpy()

        signal = np.asarray(signal)
        signal = np.squeeze(signal)

        if signal.ndim != 2:
            raise ValueError(f"Expected a 2D ECG tensor, but got shape {signal.shape} from {signal_path}")

        if signal.shape[0] == len(STANDARD_12_LEADS):
            lead_first = signal
        elif signal.shape[1] == len(STANDARD_12_LEADS):
            lead_first = signal.T
        else:
            raise ValueError(f"Could not interpret ECG shape {signal.shape} from {signal_path}")

        if lead_first.shape[1] < TARGET_SIGNAL_LENGTH:
            pad_width = TARGET_SIGNAL_LENGTH - lead_first.shape[1]
            lead_first = np.pad(lead_first, ((0, 0), (0, pad_width)))
        elif lead_first.shape[1] > TARGET_SIGNAL_LENGTH:
            lead_first = lead_first[:, :TARGET_SIGNAL_LENGTH]

        return torch.from_numpy(np.ascontiguousarray(lead_first.astype(np.float32)))

    @classmethod
    def _load_ecg_signal(cls, base_path):
        tensor_path = cls._resolve_tensor_path(base_path)
        if tensor_path is not None:
            return cls._load_tensor_signal(tensor_path)

        wfdb_base_path = os.path.splitext(base_path)[0] if base_path.endswith(WFDB_EXTENSIONS) else base_path
        if os.path.isfile(wfdb_base_path + ".hea"):
            return cls._load_wfdb_signal(wfdb_base_path)

        tried_paths = [base_path]
        tried_paths.extend(base_path + extension for extension in TENSOR_EXTENSIONS)
        tried_paths.extend(wfdb_base_path + extension for extension in WFDB_EXTENSIONS)

        raise FileNotFoundError("ECG signal not found. Tried: " + ", ".join(tried_paths))

    def __getitem__(self, index):
        dataset_size = len(self.annotation)

        for offset in range(dataset_size):
            ann = self.annotation[(index + offset) % dataset_size]

            # Use various keys for signal path, instruction/question, and answer/output
            signal_id = self._extract_signal_id(ann)
            instruction = ann.get("question", ann.get("instruction", "describe this ECG signal."))
            answer = ann.get("answer", ann.get("output", ""))
            try:
                signal_paths = self._resolve_signal_base_paths(ann)
            except FileNotFoundError:
                logging.warning("Skipping ECG annotation with no signal path: %s", ann.get("instance_id", index))
                continue

            ecg_signal = None
            for signal_path in signal_paths:
                try:
                    ecg_signal = self._load_ecg_signal(signal_path)
                    break
                except FileNotFoundError:
                    continue

            if ecg_signal is None:
                dataset_root = self._extract_dataset_root(ann)
                logging.warning(
                    "Skipping missing ECG signal '%s' from dataset root '%s'. Tried: %s",
                    signal_id,
                    dataset_root,
                    ", ".join(signal_paths),
                )
                continue

            # Ensure signal is float32 and pass through processor
            ecg_signal = ecg_signal.float()
            ecg_signal = self.vis_processor(ecg_signal)

            # Wrap instruction with image placeholder
            instruction = "<Img><ImageHere></Img> {} ".format(instruction)

            return {
                "image": ecg_signal,
                "answer": answer,
                "instruction_input": instruction,
            }

        raise RuntimeError("No valid ECG samples were found in the dataset.")
