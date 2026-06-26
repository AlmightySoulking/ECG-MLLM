import argparse
import importlib
import json
import os
import re
import warnings
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)


CLASSIFICATION_TABLE_SPECS = [
    ("PTBXL Super", "ptbxl", "super-diag"),
    ("PTBXL Sub", "ptbxl", "sub-diag"),
    ("PTBXL Form", "ptbxl", "form"),
    ("PTBXL Rhythm", "ptbxl", "rhythm"),
    ("CPSC", "cpsc", "all"),
    ("CSN", "csn", "all"),
]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark inference for the ECG-encoder + Qwen3.5-9B model in this repo."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mimic-multi",
        choices=[
            "classification",
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
        default="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
        help="Text embedding model for metrics. Can be a HuggingFace model ID or local path.",
    )
    parser.add_argument(
        "--classification_scoring",
        type=str,
        default="report",
        choices=["report", "label_chunks"],
        help=(
            "How to turn generated report text into per-label classification scores. "
            "'report' matches the paper-style flow by embedding the full generated report "
            "and each label name, then using cosine similarity. "
            "'label_chunks' keeps the legacy comma-split label matching behavior."
        ),
    )
    parser.add_argument(
        "--method_name",
        type=str,
        default=None,
        help="Display name used in classification table outputs. Defaults to the checkpoint filename stem.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Model name used by ECG-QA's official evaluation logic, for example anyECG-chat.",
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


def infer_method_name(args):
    if args.method_name:
        return args.method_name
    ckpt_name = Path(args.ckpt).expanduser().name
    stem, _ = os.path.splitext(ckpt_name)
    return stem or "model"


def format_percentage(value):
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}"


def normalize_report_text(text):
    text = "" if text is None else str(text)
    text = text.replace("<|reserved_special_token_1|>", "")
    text = text.replace("<|reserved_special_token_2|>", "")
    text = text.replace("<|reserved_special_token_3|>", "")
    text = text.strip()
    if text.lower().startswith("report:"):
        text = text.split(":", 1)[1].strip()
    return " ".join(text.split())


def split_report_label_chunks(text):
    normalized = normalize_report_text(text)
    if not normalized:
        return []
    return [chunk.strip() for chunk in normalized.split(",") if chunk.strip()]


def normalize_ecgqa_text(text):
    text = normalize_report_text(text).lower()
    text = text.replace("\n", " ").replace(";", ",").replace("|", ",")
    text = text.replace("’", "'").replace("`", "'")
    substitutions = [
        (r"\bdigitalis-effect\b", "digitalis effect"),
        (r"\blong qt interval\b", "long qt-interval"),
        (r"\bqtc interval\b", "qt corrected"),
        (r"\bqtc\b", "qt corrected"),
        (r"\bqtc corrected\b", "qt corrected"),
        (r"\bqt corrected interval\b", "qt corrected"),
        (r"\bp-r\b", "pr"),
        (r"\br-r\b", "rr"),
        (r"\bnon specific\b", "non-specific"),
        (r"\bt waves\b", "t-waves"),
        (r"\bt wave\b", "t-wave"),
    ]
    for pattern, replacement in substitutions:
        text = re.sub(pattern, replacement, text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,.")


def normalize_ecgqa_key(text):
    normalized = normalize_ecgqa_text(text)
    normalized = normalized.replace("/", " ")
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"[^a-z0-9]+", "", normalized)
    return normalized


def strip_ecgqa_preamble(text):
    normalized = normalize_ecgqa_text(text)
    prefixes = [
        "answer:",
        "final answer:",
        "the answer is",
        "answer is",
        "it is",
        "they are",
        "this ecg shows",
        "the ecg shows",
        "the finding is",
        "the findings are",
        "the symptom is",
        "the symptoms are",
        "the diagnosis is",
        "the diagnoses are",
        "the noise is",
        "the noises are",
        "present symptoms are",
        "present findings are",
        "the first ecg shows",
        "the second ecg shows",
        "ecg shows",
    ]
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :].strip(" ,.:")
                changed = True
    return normalized


def split_ecgqa_prediction_chunks(text):
    normalized = strip_ecgqa_preamble(text)
    if not normalized:
        return []
    normalized = re.sub(r"\b(?:and|plus|as well as)\b", ",", normalized)
    normalized = re.sub(r"\b(?:while|whereas|but)\b", ",", normalized)
    pieces = []
    for raw_piece in normalized.split(","):
        piece = strip_ecgqa_preamble(raw_piece).strip(" .,:")
        if piece:
            pieces.append(piece)
    return pieces


def build_ecgqa_alignment_resources(data_file, text_model):
    from collections import Counter, defaultdict

    annotation_files = []
    train_file = os.path.join(os.path.dirname(os.path.abspath(data_file)), "ecgqa_train.json")
    for candidate in [train_file, data_file]:
        candidate_path = os.path.abspath(candidate)
        if os.path.exists(candidate_path) and candidate_path not in annotation_files:
            annotation_files.append(candidate_path)

    canonical_chunks = []
    seen_chunks = set()
    chunk_variants = {}
    full_answer_lookup = {}
    full_answer_counts = Counter()
    full_answers_by_type = defaultdict(list)
    full_answer_seen_by_type = defaultdict(set)
    answer_counts_by_chunk_set = defaultdict(Counter)

    for annotation_file in annotation_files:
        with open(annotation_file, "r", encoding="utf-8") as handle:
            samples = json.load(handle)

        for sample in samples:
            answer = normalize_ecgqa_text(sample.get("answer", ""))
            if not answer:
                continue

            chunks = [chunk.strip() for chunk in answer.split(",") if chunk.strip()]
            if not chunks:
                continue

            canonical_answer = ", ".join(chunks)
            answer_key = normalize_ecgqa_text(canonical_answer)
            full_answer_lookup[answer_key] = canonical_answer
            full_answer_counts[canonical_answer] += 1

            question_type = sample.get("question_type")
            if question_type and canonical_answer not in full_answer_seen_by_type[question_type]:
                full_answer_seen_by_type[question_type].add(canonical_answer)
                full_answers_by_type[question_type].append(canonical_answer)

            answer_counts_by_chunk_set[frozenset(chunks)][canonical_answer] += 1

            for chunk in chunks:
                if chunk not in seen_chunks:
                    seen_chunks.add(chunk)
                    canonical_chunks.append(chunk)

    for canonical_chunk in canonical_chunks:
        variants = {
            canonical_chunk,
            canonical_chunk.replace("-", " "),
            canonical_chunk.replace("/", " "),
            canonical_chunk.replace("-", " ").replace("/", " "),
        }
        chunk_variants[canonical_chunk] = sorted(
            {normalize_ecgqa_text(variant) for variant in variants if variant},
            key=len,
            reverse=True,
        )

    answer_by_chunk_set = {
        chunk_set: counts.most_common(1)[0][0]
        for chunk_set, counts in answer_counts_by_chunk_set.items()
    }

    full_answers = list(full_answer_counts.keys())
    chunk_embeddings = text_model.encode(canonical_chunks, convert_to_numpy=True) if canonical_chunks else None
    full_answer_embeddings_by_type = {
        question_type: text_model.encode(answers, convert_to_numpy=True)
        for question_type, answers in full_answers_by_type.items()
        if answers
    }
    verify_labels = ["yes", "no", "not sure"]
    verify_embeddings = text_model.encode(verify_labels, convert_to_numpy=True)

    return {
        "canonical_chunks": canonical_chunks,
        "chunk_index": {chunk: index for index, chunk in enumerate(canonical_chunks)},
        "chunk_variants": chunk_variants,
        "chunk_lookup": {normalize_ecgqa_key(chunk): chunk for chunk in canonical_chunks},
        "chunk_embeddings": chunk_embeddings,
        "full_answer_lookup": full_answer_lookup,
        "answer_by_chunk_set": answer_by_chunk_set,
        "full_answers_by_type": dict(full_answers_by_type),
        "full_answer_embeddings_by_type": full_answer_embeddings_by_type,
        "verify_labels": verify_labels,
        "verify_embeddings": verify_embeddings,
        "full_answers": full_answers,
    }


def extract_ecgqa_chunks_from_text(text, alignment, allowed_chunks=None):
    normalized = strip_ecgqa_preamble(text)
    if not normalized:
        return []

    allowed = set(allowed_chunks) if allowed_chunks is not None else None
    matches = {}
    for canonical_chunk in alignment["canonical_chunks"]:
        if allowed is not None and canonical_chunk not in allowed:
            continue
        for variant in alignment["chunk_variants"].get(canonical_chunk, [canonical_chunk]):
            pattern = rf"(?<![a-z0-9]){re.escape(variant)}(?![a-z0-9])"
            for match in re.finditer(pattern, normalized):
                prefix = normalized[max(0, match.start() - 16) : match.start()]
                if re.search(r"(?:\bno\b|\bnot\b|\bwithout\b|\babsent\b)\s*$", prefix):
                    continue
                if canonical_chunk not in matches or match.start() < matches[canonical_chunk]:
                    matches[canonical_chunk] = match.start()

    return [chunk for chunk, _ in sorted(matches.items(), key=lambda item: item[1])]


def resolve_ecgqa_question_candidates(question, alignment):
    matches = extract_ecgqa_chunks_from_text(question, alignment)
    if "none" in alignment["chunk_index"] and "none" not in matches:
        matches.append("none")
    return matches


def project_ecgqa_verify_answer(prediction, alignment, text_model):
    from sklearn.metrics.pairwise import cosine_similarity

    normalized = strip_ecgqa_preamble(prediction)
    if not normalized:
        return ""

    if re.search(r"\bnot sure\b", normalized):
        return "not sure"
    if re.search(r"\byes\b", normalized):
        return "yes"
    if re.search(r"\bno\b", normalized):
        return "no"

    prediction_embedding = text_model.encode([normalized], convert_to_numpy=True)
    similarity = cosine_similarity(alignment["verify_embeddings"], prediction_embedding)
    return alignment["verify_labels"][int(similarity.argmax())]


def render_ecgqa_chunks(chunks, alignment):
    deduped = []
    seen = set()
    for chunk in chunks:
        if chunk and chunk not in seen:
            seen.add(chunk)
            deduped.append(chunk)

    if not deduped:
        return ""

    canonical_answer = alignment["answer_by_chunk_set"].get(frozenset(deduped))
    if canonical_answer is not None:
        return canonical_answer

    return ", ".join(deduped)


def match_ecgqa_chunks_by_similarity(raw_chunks, candidate_chunks, alignment, text_model, threshold):
    from sklearn.metrics.pairwise import cosine_similarity

    if not raw_chunks or not candidate_chunks:
        return []

    candidate_indices = [alignment["chunk_index"][chunk] for chunk in candidate_chunks]
    candidate_embeddings = alignment["chunk_embeddings"][candidate_indices]

    mapped_chunks = []
    seen = set()
    for raw_chunk in raw_chunks:
        lookup_chunk = alignment["chunk_lookup"].get(normalize_ecgqa_key(raw_chunk))
        if lookup_chunk in candidate_chunks and lookup_chunk not in seen:
            seen.add(lookup_chunk)
            mapped_chunks.append(lookup_chunk)
            continue

        prediction_embedding = text_model.encode([raw_chunk], convert_to_numpy=True)
        similarity = cosine_similarity(prediction_embedding, candidate_embeddings)[0]
        best_index = int(similarity.argmax())
        if float(similarity[best_index]) < threshold:
            continue

        canonical_chunk = candidate_chunks[best_index]
        if canonical_chunk not in seen:
            seen.add(canonical_chunk)
            mapped_chunks.append(canonical_chunk)

    return mapped_chunks


def match_ecgqa_full_answer_by_similarity(prediction, question_type, alignment, text_model, threshold):
    from sklearn.metrics.pairwise import cosine_similarity

    candidates = alignment["full_answers_by_type"].get(question_type, [])
    candidate_embeddings = alignment["full_answer_embeddings_by_type"].get(question_type)
    if not candidates or candidate_embeddings is None:
        return None

    prediction_embedding = text_model.encode([strip_ecgqa_preamble(prediction)], convert_to_numpy=True)
    similarity = cosine_similarity(prediction_embedding, candidate_embeddings)[0]
    best_index = int(similarity.argmax())
    if float(similarity[best_index]) < threshold:
        return None

    return candidates[best_index]


def canonicalize_ecgqa_prediction(prediction, question, question_type, alignment, text_model):
    normalized_prediction = strip_ecgqa_preamble(prediction)
    if not normalized_prediction:
        return ""

    if "verify" in question_type:
        return project_ecgqa_verify_answer(normalized_prediction, alignment, text_model)

    direct_full_match = alignment["full_answer_lookup"].get(normalized_prediction)
    if direct_full_match is not None:
        return direct_full_match

    if "choose" in question_type:
        question_candidates = resolve_ecgqa_question_candidates(question, alignment)
        direct_chunks = extract_ecgqa_chunks_from_text(normalized_prediction, alignment, question_candidates)
        if direct_chunks:
            return render_ecgqa_chunks(direct_chunks[:1], alignment)

        whole_chunk_match = match_ecgqa_chunks_by_similarity(
            [normalized_prediction],
            question_candidates or alignment["canonical_chunks"],
            alignment,
            text_model,
            threshold=0.35,
        )
        if whole_chunk_match:
            return render_ecgqa_chunks(whole_chunk_match[:1], alignment)

        full_answer_match = match_ecgqa_full_answer_by_similarity(
            normalized_prediction,
            question_type,
            alignment,
            text_model,
            threshold=0.45,
        )
        if full_answer_match is not None:
            return full_answer_match

        return normalized_prediction

    direct_chunks = extract_ecgqa_chunks_from_text(normalized_prediction, alignment)
    if direct_chunks:
        return render_ecgqa_chunks(direct_chunks, alignment)

    full_answer_match = match_ecgqa_full_answer_by_similarity(
        normalized_prediction,
        question_type,
        alignment,
        text_model,
        threshold=0.58,
    )
    if full_answer_match is not None:
        return full_answer_match

    chunk_candidates = split_ecgqa_prediction_chunks(normalized_prediction)
    chunk_matches = match_ecgqa_chunks_by_similarity(
        chunk_candidates,
        alignment["canonical_chunks"],
        alignment,
        text_model,
        threshold=0.42,
    )
    if chunk_matches:
        return render_ecgqa_chunks(chunk_matches, alignment)

    return normalized_prediction


def compute_label_scores_from_reports(predicted_reports, label_embeddings, text_model, scoring_mode):
    from sklearn.metrics.pairwise import cosine_similarity

    if scoring_mode == "report":
        report_embeddings = text_model.encode(predicted_reports, convert_to_numpy=True)
        return cosine_similarity(report_embeddings, label_embeddings)

    all_scores = []
    for report in predicted_reports:
        label_chunks = split_report_label_chunks(report)
        if not label_chunks:
            all_scores.append(np.zeros(label_embeddings.shape[0], dtype=np.float32))
            continue
        chunk_embeddings = text_model.encode(label_chunks, convert_to_numpy=True)
        similarity = cosine_similarity(label_embeddings, chunk_embeddings)
        all_scores.append(similarity.max(axis=1))
    return np.array(all_scores, dtype=np.float32)


def compute_multilabel_auc(labels_all, prediction_all, label_names):
    from sklearn.metrics import roc_auc_score

    auc_all = []
    detail = {}
    for index, label_name in enumerate(label_names):
        try:
            auc_value = float(roc_auc_score(labels_all[:, index], prediction_all[:, index]))
        except ValueError:
            auc_value = None
        auc_all.append(auc_value)
        detail[label_name] = auc_value

    valid_auc = [value for value in auc_all if value is not None]
    macro_auc = float(np.mean(valid_auc)) if valid_auc else None
    return auc_all, macro_auc, detail


def build_classification_table(method_name, ordered_results):
    headers = ["macro-AUC", *[item["column"] for item in ordered_results]]
    values = [format_percentage(item["macro_auc"]) for item in ordered_results]
    markdown_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---", *(["---:"] * len(ordered_results))]) + " |",
        "| " + " | ".join([method_name, *values]) + " |",
    ]
    latex_header = " & ".join(headers) + r" \\"
    latex_row = " & ".join([method_name, *values]) + r" \\"
    return {
        "headers": headers,
        "row": [method_name, *values],
        "markdown": "\n".join(markdown_lines),
        "latex_header": latex_header,
        "latex_row": latex_row,
    }


def format_metric_cell(value, already_percent=False):
    if value is None:
        return "N/A"
    numeric = float(value)
    if already_percent:
        return f"{numeric:.2f}"
    return f"{numeric * 100:.2f}"


def build_markdown_table(headers, rows):
    align = ["---", *(["---:"] * (len(headers) - 1))]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(align) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_plain_text_table(headers, rows):
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def format_row(row):
        cells = []
        for index, cell in enumerate(row):
            if index == 0:
                cells.append(cell.ljust(widths[index]))
            else:
                cells.append(cell.rjust(widths[index]))
        return " | ".join(cells)

    separator = "-+-".join("-" * width for width in widths)
    lines = [format_row(headers), separator]
    lines.extend(format_row(row) for row in rows)
    return "\n".join(lines)


def build_classification_paper_table(method_name, ordered_results):
    headers = [
        "Model",
        "PTBXL Super",
        "PTBXL Sub",
        "PTBXL Form",
        "PTBXL Rhythm",
        "CPSC",
        "CSN",
    ]
    metric_by_column = {item["column"]: item.get("macro_auc_pct") for item in ordered_results}
    row = [
        method_name,
        format_metric_cell(metric_by_column.get("PTBXL Super"), already_percent=True),
        format_metric_cell(metric_by_column.get("PTBXL Sub"), already_percent=True),
        format_metric_cell(metric_by_column.get("PTBXL Form"), already_percent=True),
        format_metric_cell(metric_by_column.get("PTBXL Rhythm"), already_percent=True),
        format_metric_cell(metric_by_column.get("CPSC"), already_percent=True),
        format_metric_cell(metric_by_column.get("CSN"), already_percent=True),
    ]
    rows = [row]
    return {
        "title": "Table 4: Results of Classification.",
        "headers": headers,
        "rows": rows,
        "markdown": build_markdown_table(headers, rows),
        "plain_text": build_plain_text_table(headers, rows),
    }


def normalize_report_metric_lookup(payload):
    if "report_generation_metrics" in payload and isinstance(payload["report_generation_metrics"], dict):
        payload = payload["report_generation_metrics"]
    aliases = {
        "bleu_1": ["bleu_1", "bleu1", "BLEU-1", "BLEU1"],
        "bleu_2": ["bleu_2", "bleu2", "BLEU-2", "BLEU2"],
        "bleu_3": ["bleu_3", "bleu3", "BLEU-3", "BLEU3"],
        "bleu_4": ["bleu_4", "bleu4", "BLEU-4", "BLEU4"],
        "rouge_1": ["rouge_1", "rouge1", "ROUGE-1", "ROUGE1"],
        "rouge_2": ["rouge_2", "rouge2", "ROUGE-2", "ROUGE2"],
        "rouge_l": ["rouge_l", "rouge_l_f1", "rougel", "ROUGE-L", "ROUGEL"],
    }
    normalized = {}
    for canonical_key, candidates in aliases.items():
        for candidate in candidates:
            if candidate in payload:
                normalized[canonical_key] = payload[candidate]
                break
    return normalized


def build_report_generation_paper_table(method_name, metrics):
    normalized = normalize_report_metric_lookup(metrics)
    expected = ["bleu_1", "bleu_2", "bleu_3", "bleu_4", "rouge_1", "rouge_2", "rouge_l"]
    if not all(key in normalized for key in expected):
        return None

    headers = ["Model", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-1", "ROUGE-2", "ROUGE-L"]
    row = [
        method_name,
        format_metric_cell(normalized["bleu_1"]),
        format_metric_cell(normalized["bleu_2"]),
        format_metric_cell(normalized["bleu_3"]),
        format_metric_cell(normalized["bleu_4"]),
        format_metric_cell(normalized["rouge_1"]),
        format_metric_cell(normalized["rouge_2"]),
        format_metric_cell(normalized["rouge_l"]),
    ]
    rows = [row]
    return {
        "title": "Table 5: Results of Report Generation",
        "headers": headers,
        "rows": rows,
        "markdown": build_markdown_table(headers, rows),
        "plain_text": build_plain_text_table(headers, rows),
    }


def print_named_table(table_payload):
    if not table_payload:
        return
    print(table_payload["title"])
    print(table_payload["plain_text"])


def parse_dataset_roots(items):
    dataset_roots = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --dataset-root value '{item}'. Expected ALIAS=PATH.")
        alias, path = item.split("=", 1)
        alias = alias.strip().lower()
        path = os.path.abspath(os.path.expanduser(path.strip()))
        if alias and path:
            dataset_roots[alias] = path
    return dataset_roots


def resolve_default_data_file(args):
    if args.data_file:
        return args.data_file

    candidates = []
    if args.dataset == "mimic-multi":
        candidates.append("data/mimic_multiECG_test.json")
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
    legacy_root_attrs = {
        "ptbxl": "_ptbxl_dir_",
        "cpsc": "_cpsc_dir_",
        "csn": "_csn_dir_",
        "mimic": "_mimic_dir_",
        "european_st_t": "_european_st_t_dir_",
        "mit_bih_st": "_mit_bih_st_dir_",
        "mit_bih_arrhythmia": "_mit_bih_arrhythmia_dir_",
    }
    for alias, root in dataset_roots.items():
        local_utils.ecg_dir[alias] = root
        legacy_attr = legacy_root_attrs.get(alias)
        if legacy_attr:
            setattr(local_utils, legacy_attr, root)

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

            df = pd.read_csv(filename, low_memory=False)
        else:
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"Classification JSON not found: {data_file}")

            with open(data_file, "r", encoding="utf-8") as handle:
                raw = json.load(handle)

            if isinstance(raw, dict) and "samples" in raw:
                raw = raw["samples"]

            df = pd.DataFrame(raw)

        self._label_test_, self._text_test_, self._ecg_root_ = self._get_label_text(local_utils, dataset_roots)
        self._label_alias_lookup = self._build_label_alias_lookup()
        self._cpsc_label_id_lookup = self._build_cpsc_label_id_lookup()
        self._ptbxl_statement_table = self._load_ptbxl_statement_table()
        self._ptbxl_super_diag_fallback = self._build_ptbxl_super_diag_fallback()

        df = self._normalize_classification_metadata(df)
        df = self._ensure_label_columns(df)
        df = df[df["split_fold"] == split_fold]

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
            split_fold = self._build_split_fold_column(df)
            if split_fold is None:
                warnings.warn(
                    "Classification metadata has no recognized split column. Treating all rows "
                    "as split_fold='test'. If this file contains train/val rows too, provide a "
                    "CSV/JSON with explicit split metadata."
                )
                df["split_fold"] = "test"
            else:
                df["split_fold"] = split_fold

        path_candidates = self._path_column_candidates()
        self._path_columns = [column for column in path_candidates if column in df.columns]
        self._path_column = self._path_columns[0] if self._path_columns else None
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

    def _build_split_fold_column(self, df):
        if "split" in df.columns:
            normalized = self._normalize_split_series(df["split"])
            if normalized is not None:
                return normalized

        if "strat_fold" in df.columns:
            return df["strat_fold"].apply(self._map_ptbxl_strat_fold)

        alias_candidates = [
            "split_name",
            "split_type",
            "subset",
            "phase",
            "partition",
            "fold_type",
            "set",
            "group",
            "eval_split",
            "data_split",
        ]
        for column in alias_candidates:
            if column in df.columns:
                normalized = self._normalize_split_series(df[column])
                if normalized is not None:
                    return normalized

        fold_candidates = [
            "fold",
            "split_id",
            "fold_id",
            "cv_fold",
            "stratified_fold",
        ]
        for column in fold_candidates:
            if column in df.columns:
                normalized = self._normalize_fold_series(df[column], column)
                if normalized is not None:
                    return normalized

        indicator_specs = [
            ("is_test", "test"),
            ("test", "test"),
            ("in_test", "test"),
            ("is_eval", "test"),
            ("is_validation", "val"),
            ("is_valid", "val"),
            ("is_val", "val"),
            ("val", "val"),
            ("valid", "val"),
            ("is_train", "train"),
            ("train", "train"),
        ]
        for column, split_name in indicator_specs:
            if column in df.columns:
                normalized = self._normalize_indicator_split(df[column], split_name)
                if normalized is not None:
                    return normalized

        return None

    def _normalize_split_series(self, series):
        normalized = series.apply(self._normalize_split_value)
        valid = normalized.notna().sum()
        return normalized if valid > 0 else None

    def _normalize_fold_series(self, series, column_name):
        if "strat" in column_name.lower() or self.dataset == "ptbxl":
            normalized = series.apply(self._map_ptbxl_strat_fold)
            valid = normalized.notna().sum()
            return normalized if valid > 0 else None

        normalized = series.apply(self._normalize_split_value)
        valid = normalized.notna().sum()
        if valid > 0:
            return normalized
        return None

    def _normalize_indicator_split(self, series, positive_split):
        positive_mask = series.apply(self._coerce_bool_or_none)
        if positive_mask.notna().sum() == 0:
            return None

        if positive_split == "test":
            negative_split = "train"
        elif positive_split == "val":
            negative_split = "train"
        else:
            negative_split = "test"

        normalized = positive_mask.apply(
            lambda value: positive_split if value is True else (negative_split if value is False else None)
        )
        return normalized if normalized.notna().sum() > 0 else None

    @staticmethod
    def _coerce_bool_or_none(value):
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        text = str(value).strip().lower()
        if not text:
            return None
        if text in {"1", "true", "t", "yes", "y"}:
            return True
        if text in {"0", "false", "f", "no", "n"}:
            return False
        return None

    @staticmethod
    def _normalize_split_value(value):
        if value is None:
            return None
        text = str(value).strip().lower()
        if not text or text == "nan":
            return None
        compact = "".join(ch for ch in text if ch.isalnum())

        if compact in {"test", "testing", "eval", "evaluation", "holdout", "heldout"}:
            return "test"
        if compact in {"val", "valid", "validation", "dev"}:
            return "val"
        if compact in {"train", "training", "tr"}:
            return "train"

        if compact in {"10"}:
            return "test"
        if compact in {"9"}:
            return "val"
        if compact.isdigit():
            return "train"
        return None

    def _ensure_label_columns(self, df):
        df = df.copy()
        found_explicit_label_signal = False

        for column in list(df.columns):
            canonical_label = self._map_label_token(column)
            if canonical_label and canonical_label not in df.columns:
                df[canonical_label] = df[column]
            if canonical_label:
                found_explicit_label_signal = True

        missing_label_columns = [label for label in self._label_test_ if label not in df.columns]
        if missing_label_columns:
            derived_labels = self._extract_label_sets(df)
            if derived_labels is not None:
                found_explicit_label_signal = True
                for label in missing_label_columns:
                    df[label] = [1.0 if label in sample_labels else 0.0 for sample_labels in derived_labels]
                missing_label_columns = [label for label in self._label_test_ if label not in df.columns]

        if missing_label_columns and found_explicit_label_signal:
            for label in missing_label_columns:
                df[label] = 0.0

        for label in self._label_test_:
            if label in df.columns:
                df[label] = df[label].apply(lambda value, expected=label: self._coerce_indicator(value, expected))

        return df

    def _path_column_candidates(self):
        if self.dataset == "ptbxl":
            preferred = ["filename_lr", "filename_hr"] if self.sampling_freq <= 100 else ["filename_hr", "filename_lr"]
            return [*preferred, "path", "ecg_path", "record_path", "signal_path", "file_name", "filename", "record", "image_path"]
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

    @staticmethod
    def _normalize_label_key(value):
        return "".join(ch.lower() for ch in str(value) if ch.isalnum())

    def _build_label_alias_lookup(self):
        alias_lookup = {}
        for label in self._label_test_:
            alias_lookup[self._normalize_label_key(label)] = label

        if self.dataset == "cpsc":
            cpsc_aliases = {
                "Normal": ["N", "normal", "norm"],
                "Atrial fibrillation": ["AF", "AFIB", "atrial fibrillation"],
                "First-degree atrioventricular block": [
                    "I-AVB",
                    "IAVB",
                    "1st degree AV block",
                    "1 degree atrioventricular block",
                    "first degree atrioventricular block",
                ],
                "Left bundle branch block": ["LBBB", "left bundle branch block"],
                "Right bundle branch block": ["RBBB", "right bundle branch block"],
                "Premature atrial contraction": [
                    "PAC",
                    "APC",
                    "premature atrial contraction",
                    "atrial premature contraction",
                    "atrial premature beat",
                ],
                "Premature ventricular contraction": [
                    "PVC",
                    "VPC",
                    "premature ventricular contraction",
                    "ventricular premature contraction",
                    "ventricular premature beat",
                ],
                "ST-segment depression": ["STD", "ST depression", "ST-segment depression"],
                "ST-segment elevated": [
                    "STE",
                    "ST elevation",
                    "ST elevated",
                    "ST-segment elevation",
                ],
            }
            for label, aliases in cpsc_aliases.items():
                for alias in aliases:
                    alias_lookup[self._normalize_label_key(alias)] = label

        return alias_lookup

    def _build_cpsc_label_id_lookup(self):
        if self.dataset != "cpsc":
            return {}
        return {
            str(index): label
            for index, label in enumerate(self._label_test_, start=1)
        }

    def _load_ptbxl_statement_table(self):
        if self.dataset != "ptbxl":
            return None

        scp_path = Path(self._ecg_root_) / "scp_statements.csv"
        if not scp_path.exists():
            warnings.warn(
                f"PTB-XL statement metadata not found at {scp_path}. Falling back to limited built-in mappings."
            )
            return None

        table = self._pd.read_csv(scp_path, index_col=0)
        table.index = table.index.map(lambda value: str(value).strip())
        return table

    @staticmethod
    def _build_ptbxl_super_diag_fallback():
        return {
            "ISCI": "STTC",
            "ISCA": "STTC",
            "ISC_": "STTC",
            "NST_": "STTC",
            "STTC": "STTC",
            "NORM": "NORM",
            "LVH": "HYP",
            "RVH": "HYP",
            "SEHYP": "HYP",
            "RAO/RAE": "HYP",
            "LAO/LAE": "HYP",
            "CRBBB": "CD",
            "CLBBB": "CD",
            "IRBBB": "CD",
            "ILBBB": "CD",
            "LAFB/LPFB": "CD",
            "LPFB": "CD",
            "LAFB": "CD",
            "WPW": "CD",
            "_AVB": "CD",
            "IVCD": "CD",
            "IMI": "MI",
            "AMI": "MI",
            "PMI": "MI",
            "LMI": "MI",
        }

    def _label_source_columns(self, df):
        explicit_names = {
            "labels",
            "label",
            "label_names",
            "label_name",
            "label_text",
            "diagnosis",
            "diagnoses",
            "dx",
            "annotation",
            "annotations",
            "class",
            "classes",
            "target",
            "targets",
            "scp_codes",
        }
        normalized_explicit = {self._normalize_label_key(name) for name in explicit_names}

        source_columns = []
        for column in df.columns:
            normalized = self._normalize_label_key(column)
            if normalized in normalized_explicit:
                source_columns.append(column)
                continue

            if self.dataset == "cpsc":
                if normalized in {
                    "firstlabel",
                    "secondlabel",
                    "thirdlabel",
                    "fourthlabel",
                    "fifthlabel",
                    "firstdiagnosis",
                    "seconddiagnosis",
                    "thirddiagnosis",
                }:
                    source_columns.append(column)
                    continue
                if normalized.startswith("label") and normalized[5:].isdigit():
                    source_columns.append(column)

        deduped = []
        seen = set()
        for column in source_columns:
            if column not in seen:
                seen.add(column)
                deduped.append(column)
        return deduped

    def _extract_label_sets(self, df):
        if self.dataset == "ptbxl" and "scp_codes" in df.columns:
            derived = self._extract_ptbxl_label_sets(df["scp_codes"])
            if derived is not None:
                return derived

        source_columns = self._label_source_columns(df)
        if not source_columns:
            return None

        label_sets = []
        for _, row in df.iterrows():
            sample_labels = set()
            for column in source_columns:
                sample_labels.update(self._labels_from_value(row[column]))
            label_sets.append(sample_labels)
        return label_sets

    def _extract_ptbxl_label_sets(self, scp_series):
        label_sets = []
        any_mapped = False
        for value in scp_series:
            sample_labels = set()
            for token in self._iter_label_tokens(value):
                sample_labels.update(self._map_ptbxl_token(token))
            if sample_labels:
                any_mapped = True
            label_sets.append(sample_labels)
        return label_sets if any_mapped else None

    def _map_ptbxl_token(self, token):
        if token is None:
            return set()

        raw_token = str(token).strip()
        if not raw_token:
            return set()

        normalized_code = raw_token.upper()
        labels = set()

        # Direct match against the expected label space.
        direct = self._map_label_token(raw_token)
        if direct is not None:
            labels.add(direct)

        if self.dataset_subtype == "sub-diag":
            if normalized_code in self._label_test_:
                labels.add(normalized_code)
                return labels

            row = self._ptbxl_lookup_statement_row(normalized_code)
            if row is not None:
                diagnostic_subclass = row.get("diagnostic_subclass")
                mapped = self._map_label_token(diagnostic_subclass)
                if mapped is not None:
                    labels.add(mapped)
            return labels

        if self.dataset_subtype == "super-diag":
            mapped = self._ptbxl_map_to_super_diag(normalized_code)
            if mapped is not None:
                labels.add(mapped)
            return labels

        row = self._ptbxl_lookup_statement_row(normalized_code)
        if row is None:
            return labels

        if self.dataset_subtype == "diag":
            if self._ptbxl_truthy(row.get("diagnostic")):
                labels.update(self._ptbxl_labels_from_row_description(row))
        elif self.dataset_subtype == "form":
            if self._ptbxl_truthy(row.get("form")):
                labels.update(self._ptbxl_labels_from_row_description(row))
        elif self.dataset_subtype == "rhythm":
            if self._ptbxl_truthy(row.get("rhythm")):
                labels.update(self._ptbxl_labels_from_row_description(row))
        else:
            labels.update(self._ptbxl_labels_from_row_description(row))

        return labels

    def _ptbxl_lookup_statement_row(self, code):
        table = self._ptbxl_statement_table
        if table is None:
            return None
        if code not in table.index:
            return None
        return table.loc[code]

    def _ptbxl_map_to_super_diag(self, code):
        row = self._ptbxl_lookup_statement_row(code)
        if row is not None:
            diagnostic_class = row.get("diagnostic_class")
            mapped = self._map_label_token(diagnostic_class)
            if mapped is not None:
                return mapped

        return self._ptbxl_super_diag_fallback.get(code)

    def _ptbxl_labels_from_row_description(self, row):
        description_columns = ["description", "statement", "label", "diagnosis"]
        labels = set()
        for column in description_columns:
            if column not in row.index:
                continue
            mapped = self._map_label_token(row.get(column))
            if mapped is not None:
                labels.add(mapped)
        return labels

    @staticmethod
    def _ptbxl_truthy(value):
        if value is None:
            return False
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        try:
            return float(value) > 0
        except (TypeError, ValueError):
            text = str(value).strip().lower()
            return text in {"true", "t", "yes", "y"}

    def _labels_from_value(self, value):
        labels = set()
        for token in self._iter_label_tokens(value):
            mapped = self._map_label_token(token)
            if mapped is not None:
                labels.add(mapped)
        return labels

    def _iter_label_tokens(self, value):
        if isinstance(value, dict):
            tokens = []
            for key, dict_value in value.items():
                if dict_value:
                    tokens.extend(self._iter_label_tokens(key))
            return tokens

        if isinstance(value, (list, tuple, set)):
            tokens = []
            for item in value:
                tokens.extend(self._iter_label_tokens(item))
            return tokens

        if value is None or self._pd.isna(value):
            return []

        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []

            if stripped[0] in "[{(":
                try:
                    parsed = self._ast.literal_eval(stripped)
                except (ValueError, SyntaxError):
                    parsed = None
                if parsed is not None and parsed is not value:
                    return self._iter_label_tokens(parsed)

            for separator in [",", ";", "|", "/"]:
                if separator in stripped:
                    tokens = []
                    for item in stripped.split(separator):
                        tokens.extend(self._iter_label_tokens(item))
                    return tokens

            return [stripped]

        return [value]

    def _map_label_token(self, token):
        if token is None:
            return None

        normalized = self._normalize_label_key(token)
        if not normalized:
            return None

        mapped = self._label_alias_lookup.get(normalized)
        if mapped is not None:
            return mapped

        if self.dataset == "cpsc":
            challenge_id = None
            if isinstance(token, (int, np.integer)):
                challenge_id = str(int(token))
            elif isinstance(token, float) and token.is_integer():
                challenge_id = str(int(token))
            elif normalized.isdigit():
                challenge_id = str(int(normalized))

            if challenge_id is not None:
                return self._cpsc_label_id_lookup.get(challenge_id)

        return None

    def _coerce_indicator(self, value, expected_label):
        if isinstance(value, dict):
            return 1.0 if expected_label in self._labels_from_value(value) else 0.0

        if isinstance(value, (list, tuple, set)):
            return 1.0 if expected_label in self._labels_from_value(value) else 0.0

        if value is None or self._pd.isna(value):
            return 0.0

        if isinstance(value, (bool, np.bool_)):
            return float(value)

        if isinstance(value, (int, float, np.integer, np.floating)):
            return 1.0 if float(value) > 0 else 0.0

        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return 0.0

            normalized = stripped.lower()
            if normalized in {"1", "true", "yes", "y", "present", "positive"}:
                return 1.0
            if normalized in {"0", "false", "no", "n", "none", "nan"}:
                return 0.0

            try:
                return 1.0 if float(stripped) > 0 else 0.0
            except ValueError:
                mapped = self._map_label_token(stripped)
                return 1.0 if mapped == expected_label else 0.0

        return 0.0

    def _normalize_raw_path(self, raw_path):
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

    def _resolve_sample_paths(self, sample):
        raw_paths = []
        seen = set()
        for column in self._path_columns:
            value = sample[column]
            try:
                raw_path = self._normalize_raw_path(value)
            except ValueError:
                continue
            if raw_path not in seen:
                seen.add(raw_path)
                raw_paths.append(raw_path)

        if not raw_paths:
            raise ValueError(
                "Encountered a missing ECG path in classification metadata. "
                f"Tried columns: {', '.join(self._path_columns)}"
            )

        return raw_paths

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
        raw_paths = self._resolve_sample_paths(sample)
        last_error = None
        attempted_paths = []
        ecg = None
        for raw_path in raw_paths:
            for ecg_path in resolve_ecg_candidates(
                raw_path=raw_path,
                dataset_name=self.dataset,
                dataset_roots={self.dataset: self._ecg_root_},
                vis_root=self._ecg_root_,
            ):
                attempted_paths.append(ecg_path)
                try:
                    ecg = self.local_utils.get_ecg_from_path(ecg_path, self.sampling_freq)
                    break
                except Exception as exc:
                    last_error = exc
            if ecg is not None:
                break

        if ecg is None:
            raise FileNotFoundError(
                "Could not load ECG from classification metadata row "
                f"{index} using columns {self._path_columns} and values {raw_paths}. "
                f"Tried paths: {attempted_paths}. Last error: {last_error}"
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

        if self.args.dataset == "ecgqa":
            lower_question = question.lower()
            if re.match(r"^(is|does|do|did|has|have|had|can|could|would|will|are|was|were)\b", lower_question):
                instruction = (
                    f"{instruction} Answer with exactly one of: yes, no, not sure. "
                    "Use lowercase only and no explanation."
                )
            elif " or " in lower_question:
                instruction = (
                    f"{instruction} Answer with exactly one option from the question, or none. "
                    "Use lowercase only and no explanation."
                )
            else:
                instruction = (
                    f"{instruction} Answer with only the final finding text as a lowercase comma-separated list. "
                    "Use canonical ECG labels, no sentences, and no explanation."
                )

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


def test_classification(
    args,
    get_response,
    local_utils,
    dataset_roots,
    dataset_name=None,
    dataset_subtype=None,
    text_model=None,
):
    from sentence_transformers import SentenceTransformer

    dataset_name = dataset_name or args.dataset
    dataset_subtype = dataset_subtype or args.dataset_subtype

    test_data = CompatFinetuningDataset(
        dataset_name=dataset_name,
        dataset_subtype=dataset_subtype,
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

    if text_model is None:
        text_model = SentenceTransformer(args.text_embedding_model)
    label_embeddings = text_model.encode(test_data._text_test_, convert_to_numpy=True)
    labels_all, prediction_all = [], []

    for ecgs, labels in tqdm(test_loader, desc="Classification"):
        message = [[{"role": "user", "content": "Please provide the report for the following ECG."}] for _ in range(ecgs.shape[0])]
        response = get_response(ecgs, message)
        normalized_response = [normalize_report_text(item) for item in response]

        prediction = compute_label_scores_from_reports(
            predicted_reports=normalized_response,
            label_embeddings=label_embeddings,
            text_model=text_model,
            scoring_mode=args.classification_scoring,
        )
        prediction_all.extend(prediction)
        labels_all.extend(labels.cpu().numpy())

    labels_all = np.array(labels_all)
    prediction_all = np.array(prediction_all)

    if len(labels_all) == 0:
        return {
            "dataset": dataset_name,
            "dataset_subtype": dataset_subtype,
            "error": (
                "No evaluation samples found in classification dataset after split normalization. "
                "Check that your CSV/JSON has test rows, usable ECG paths, and valid labels."
            ),
        }

    auc_all, macro_auc, auc_detail = compute_multilabel_auc(labels_all, prediction_all, test_data._text_test_)

    return {
        "dataset": dataset_name,
        "dataset_subtype": dataset_subtype,
        "auc": macro_auc,
        "macro_auc": macro_auc,
        "macro_auc_pct": macro_auc * 100 if macro_auc is not None else None,
        "detail": auc_detail,
        "classification_scoring": args.classification_scoring,
    }


def test_classification_suite(args, get_response, local_utils, dataset_roots):
    from sentence_transformers import SentenceTransformer

    if args.classification_json:
        raise ValueError(
            "--dataset classification expects --classification_csv_dir to point to a directory containing "
            "ptbxl.csv, cpsc.csv, and csn.csv. A single --classification_json file is only supported for "
            "per-dataset runs such as --dataset ptbxl."
        )
    if args.classification_csv_dir and os.path.isfile(args.classification_csv_dir):
        raise ValueError(
            "--dataset classification expects --classification_csv_dir to point to a directory containing "
            "ptbxl.csv, cpsc.csv, and csn.csv, but a file was provided: "
            f"{args.classification_csv_dir}"
        )

    text_model = SentenceTransformer(args.text_embedding_model)
    method_name = infer_method_name(args)
    ordered_results = []
    task_details = {}

    for column_name, dataset_name, dataset_subtype in CLASSIFICATION_TABLE_SPECS:
        metrics = test_classification(
            args=args,
            get_response=get_response,
            local_utils=local_utils,
            dataset_roots=dataset_roots,
            dataset_name=dataset_name,
            dataset_subtype=dataset_subtype,
            text_model=text_model,
        )
        ordered_results.append(
            {
                "column": column_name,
                "dataset": dataset_name,
                "dataset_subtype": dataset_subtype,
                "macro_auc": metrics.get("macro_auc"),
                "macro_auc_pct": metrics.get("macro_auc_pct"),
                "error": metrics.get("error"),
            }
        )
        task_details[column_name] = metrics

    table = build_classification_table(method_name, ordered_results)
    paper_table = build_classification_paper_table(method_name, ordered_results)
    macro_auc_pct = {
        item["column"]: item["macro_auc_pct"]
        for item in ordered_results
    }

    return {
        "method_name": method_name,
        "metric": "macro_auc_pct",
        "columns": [item["column"] for item in ordered_results],
        "macro_auc_pct": macro_auc_pct,
        "table": table,
        "paper_table": paper_table,
        "task_details": task_details,
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
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    data_file = args.data_file or f"./data/{args.dataset}_test.json"
    if not os.path.exists(data_file):
        raise FileNotFoundError(
            "Official ECG-QA evaluation expects an explicit test split file. "
            f"Could not find: {data_file}"
        )

    dataset = load_dataset("json", data_files=data_file)["train"]
    model_name = args.model_name or infer_method_name(args)
    text_model = SentenceTransformer(args.text_embedding_model)
    if model_name != "anyECG-chat":
        dataset = dataset.shuffle(seed=42).select(range(int(len(dataset) * 0.1)))

    alignment = build_ecgqa_alignment_resources(data_file, text_model)

    test_loader = DataLoader(
        dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=InferenceCollator(
            local_utils=local_utils,
            dataset_roots=dataset_roots,
            vis_root=args.vis_root,
            sampling_freq=args.sampling_freq,
            return_question_type=True,
        ),
        num_workers=64,
    )

    mean_acc = {}
    official_mean_acc = {}
    alignment_stats = {
        "prediction_changed_count": 0,
        "prediction_changed_fraction": 0.0,
        "sample_count": 0,
    }
    for ecgs, messages, question_types in tqdm(test_loader, desc="ECGQA"):
        question_message = [[item[0]] for item in messages]
        answer_message = [[item[1]] for item in messages]
        response = get_response(ecgs, question_message)

        for i in range(len(response)):
            truth = answer_message[i][0]["content"]
            question = question_message[i][0]["content"]
            raw_prediction = response[i]
            question_type = question_types[i]
            official_prediction = raw_prediction

            if "verify" in question_type and model_name != "anyECG-chat":
                label_embeddings = text_model.encode(["yes", "no", "not sure"])
                prediction_embeddings = text_model.encode([raw_prediction])
                similarity = cosine_similarity(label_embeddings, prediction_embeddings)
                official_prediction = ["yes", "no", "not sure"][similarity.argmax()]

            aligned_prediction = canonicalize_ecgqa_prediction(
                raw_prediction,
                question,
                question_type,
                alignment,
                text_model,
            )

            official_acc = 1 if truth.lower() == official_prediction.lower() else 0
            acc = 1 if normalize_ecgqa_text(truth) == normalize_ecgqa_text(aligned_prediction) else 0

            if normalize_ecgqa_text(official_prediction) != normalize_ecgqa_text(aligned_prediction):
                alignment_stats["prediction_changed_count"] += 1
            alignment_stats["sample_count"] += 1

            mean_acc.setdefault(question_type, {"mean_acc": 0.0, "count": 0})
            mean_acc[question_type]["mean_acc"] += acc
            mean_acc[question_type]["count"] += 1
            official_mean_acc.setdefault(question_type, {"mean_acc": 0.0, "count": 0})
            official_mean_acc[question_type]["mean_acc"] += official_acc
            official_mean_acc[question_type]["count"] += 1

    for key in mean_acc:
        mean_acc[key]["mean_acc"] /= mean_acc[key]["count"]
    for key in official_mean_acc:
        official_mean_acc[key]["mean_acc"] /= official_mean_acc[key]["count"]

    macro_acc = np.mean([mean_acc[key]["mean_acc"] for key in mean_acc])
    micro_acc = (
        np.sum([mean_acc[key]["mean_acc"] * mean_acc[key]["count"] for key in mean_acc])
        / np.sum([mean_acc[key]["count"] for key in mean_acc])
    )
    official_macro_acc = np.mean([official_mean_acc[key]["mean_acc"] for key in official_mean_acc])
    official_micro_acc = (
        np.sum(
            [
                official_mean_acc[key]["mean_acc"] * official_mean_acc[key]["count"]
                for key in official_mean_acc
            ]
        )
        / np.sum([official_mean_acc[key]["count"] for key in official_mean_acc])
    )
    if alignment_stats["sample_count"] > 0:
        alignment_stats["prediction_changed_fraction"] = (
            alignment_stats["prediction_changed_count"] / alignment_stats["sample_count"]
        )

    return {
        "macro_acc": macro_acc,
        "micro_acc": micro_acc,
        "detail": mean_acc,
        "official_macro_acc": official_macro_acc,
        "official_micro_acc": official_micro_acc,
        "official_detail": official_mean_acc,
        "alignment_stats": alignment_stats,
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

    if args.dataset == "classification":
        output_json = test_classification_suite(args, get_response, local_utils, dataset_roots)
    elif args.dataset in ["ptbxl", "cpsc", "csn"]:
        output_json = test_classification(args, get_response, local_utils, dataset_roots)
    elif args.dataset == "ecgqa":
        output_json = test_ecgqa(args, get_response, local_utils, dataset_roots)
    elif args.dataset == "mimic-multi":
        output_json = test_mimic_multi(args, get_response, local_utils, dataset_roots)
    else:
        output_json = test_localization(args, get_response, local_utils, dataset_roots)

    if args.dataset == "classification":
        print_named_table(output_json.get("paper_table"))
        print("Classification macro-AUC")
        print(output_json["table"]["markdown"])
        print(output_json["table"]["latex_header"])
        print(output_json["table"]["latex_row"])
    else:
        print(output_json)
        report_table = build_report_generation_paper_table(infer_method_name(args), output_json)
        print_named_table(report_table)
        if report_table is not None:
            output_json["paper_table"] = report_table
    with open(args.result_path, "w") as handle:
        json.dump(output_json, handle, indent=4)


if __name__ == "__main__":
    main()
