import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path

import requests
import torch
from tqdm import tqdm


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MiniGPT ECG checkpoint on the mimic multi-ECG benchmark."
    )
    parser.add_argument(
        "--cfg-path",
        required=True,
        help="Path to the training/eval config used to build the model.",
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Checkpoint path for the trained model.",
    )
    parser.add_argument(
        "--dataset-json",
        default="data/mimic_llama3.3-70b_test.json",
        help="Benchmark JSON file with question/answer/ecg_path entries.",
    )
    parser.add_argument(
        "--predictions-path",
        default="outputs/mimic_multi_predictions.json",
        help="Where to save or load model predictions.",
    )
    parser.add_argument(
        "--judge-output-path",
        default=None,
        help="Where to save judge outputs. Defaults to predictions path with '_judged'.",
    )
    parser.add_argument(
        "--mode",
        choices=["generate", "judge", "all"],
        default="generate",
        help="Run generation only, judge only, or both.",
    )
    parser.add_argument(
        "--dataset-root",
        action="append",
        default=[],
        metavar="ALIAS=PATH",
        help=(
            "Map dataset aliases from the JSON to actual ECG roots, for example "
            "'mimic=/data/mimic-iv-ecg' or 'ptbxl=/data/ptb-xl'. Can be repeated."
        ),
    )
    parser.add_argument(
        "--vis-root",
        default=None,
        help="Fallback root used when ECG paths in the JSON are relative.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for the benchmark model.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of tokens to generate per answer.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature for the benchmark model.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling value for the benchmark model.",
    )
    parser.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="Beam count used during generation.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable stochastic decoding for the benchmark model.",
    )
    parser.add_argument(
        "--report-csv",
        default=None,
        help=(
            "Optional CSV with ECG report text. If supplied, judge prompts will include "
            "per-ECG reports, matching the original baseline more closely."
        ),
    )
    parser.add_argument(
        "--report-path-column",
        default="path",
        help="CSV column containing ECG record paths.",
    )
    parser.add_argument(
        "--report-text-column",
        default="report",
        help="CSV column containing ECG report text.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help=(
            "OpenAI-compatible model name for scoring answers. Required for '--mode judge' "
            "and '--mode all'."
        ),
    )
    parser.add_argument(
        "--judge-base-url",
        default=None,
        help="OpenAI-compatible base URL, for example a local vLLM server.",
    )
    parser.add_argument(
        "--judge-api-key",
        default="EMPTY",
        help="API key for the judge endpoint.",
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0.0,
        help="Temperature for the judge model.",
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate for each judge response.",
    )
    parser.add_argument(
        "--judge-fallback-model",
        default=None,
        help=(
            "Optional fallback OpenAI-compatible model name for judge requests. "
            "If the primary judge model fails with a server error, the fallback will be tried."
        ),
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="Optional config overrides in key=value form.",
    )
    return parser


def parse_dataset_roots(items):
    dataset_roots = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --dataset-root value '{item}'. Expected ALIAS=PATH.")
        alias, path = item.split("=", 1)
        alias = alias.strip().lower()
        path = os.path.expanduser(path.strip())
        if not alias or not path:
            raise ValueError(f"Invalid --dataset-root value '{item}'.")
        dataset_roots[alias] = path
    return dataset_roots


def ensure_parent_dir(path_value):
    Path(path_value).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def load_json(path_value):
    with open(path_value, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(payload, path_value):
    ensure_parent_dir(path_value)
    with open(path_value, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def normalize_text(text):
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def exact_match_score(prediction, answer):
    return int(normalize_text(prediction) == normalize_text(answer))


def token_f1_score(prediction, answer):
    pred_tokens = normalize_text(prediction).split()
    answer_tokens = normalize_text(answer).split()
    if not pred_tokens and not answer_tokens:
        return 1.0
    if not pred_tokens or not answer_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    answer_counts = Counter(answer_tokens)
    overlap = sum((pred_counts & answer_counts).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(answer_tokens)
    return 2 * precision * recall / (precision + recall)


def get_processor_from_cfg(cfg):
    from minigpt4.common.registry import registry
    from minigpt4.processors.base_processor import BaseProcessor

    processor_cfg = None
    for dataset_name in cfg.datasets_cfg:
        dataset_cfg = cfg.datasets_cfg.get(dataset_name)
        processor_group = dataset_cfg.get("ecg_processor", None)
        if processor_group is None:
            processor_group = dataset_cfg.get("vis_processor", None)
        if processor_group is None:
            continue
        processor_cfg = processor_group.get("eval", None) or processor_group.get("train", None)
        if processor_cfg is not None:
            break

    if processor_cfg is None:
        return BaseProcessor()

    processor_cls = registry.get_processor_class(processor_cfg.name)
    return processor_cls.from_config(processor_cfg)


def load_model(args):
    import minigpt4.tasks  # noqa: F401
    import minigpt4.datasets.builders  # noqa: F401
    import minigpt4.models  # noqa: F401
    import minigpt4.processors  # noqa: F401
    import minigpt4.runners  # noqa: F401
    from minigpt4.common.config import Config
    from minigpt4.common.registry import registry

    cfg = Config(args)
    cfg.model_cfg.ckpt = args.ckpt

    model_cls = registry.get_model_class(cfg.model_cfg.arch)
    model = model_cls.from_config(cfg.model_cfg)
    model = model.to(args.device)
    model.eval()

    vis_processor = get_processor_from_cfg(cfg)
    return model, vis_processor


def raw_paths_from_sample(sample):
    ecg_paths = sample.get("ecg_path", [])
    if isinstance(ecg_paths, str):
        return [ecg_paths]
    if isinstance(ecg_paths, list):
        return ecg_paths
    raise TypeError(f"Unsupported ecg_path type: {type(ecg_paths)!r}")


def build_signal_candidates(raw_path, sample, dataset_roots, vis_root, ann_root):
    raw_path = os.path.expanduser(str(raw_path).strip())
    if not raw_path:
        return []

    if os.path.isabs(raw_path):
        return [raw_path]

    candidates = []
    dataset_name = str(sample.get("dataset", "")).strip()
    if dataset_name:
        mapped_root = dataset_roots.get(dataset_name.lower())
        if mapped_root is not None:
            candidates.append(os.path.join(mapped_root, raw_path))

        expanded_dataset_root = os.path.expanduser(dataset_name)
        if os.path.isabs(expanded_dataset_root) or expanded_dataset_root.startswith("."):
            candidates.append(os.path.join(expanded_dataset_root, raw_path))

    if vis_root:
        candidates.append(os.path.join(vis_root, raw_path))

    if ann_root:
        candidates.append(os.path.join(ann_root, raw_path))

    candidates.append(raw_path)

    deduped = []
    seen = set()
    for candidate in candidates:
        normalized = os.path.normpath(candidate)
        if normalized not in seen:
            seen.add(normalized)
            deduped.append(normalized)
    return deduped


def load_signal_tensor(raw_path, sample, dataset_roots, vis_root, ann_root, vis_processor):
    from minigpt4.datasets.datasets.ecg_dataset import ECGDataset

    tried_paths = []
    for candidate in build_signal_candidates(raw_path, sample, dataset_roots, vis_root, ann_root):
        tried_paths.append(candidate)
        try:
            signal = ECGDataset._load_ecg_signal(candidate).float()
            signal = vis_processor(signal)
            if not isinstance(signal, torch.Tensor):
                signal = torch.as_tensor(signal)
            return signal.float()
        except FileNotFoundError:
            continue

    raise FileNotFoundError(
        f"Could not resolve ECG path '{raw_path}'. Tried: {', '.join(tried_paths)}"
    )


def load_multi_ecg_sample(sample, dataset_roots, vis_root, ann_root, vis_processor):
    ecgs = [
        load_signal_tensor(path, sample, dataset_roots, vis_root, ann_root, vis_processor)
        for path in raw_paths_from_sample(sample)
    ]
    return torch.stack(ecgs, dim=0)


def build_prompt(question, num_ecgs, model):
    numbered_slots = " ".join(
        f"ECG{i}: <Img><ImageHere></Img>" for i in range(1, num_ecgs + 1)
    )
    instruction = f"{numbered_slots} Question: {question}".strip()

    if getattr(model, "chat_template", False):
        return model.prompt_template.format(instruction)
    return instruction


def decode_generation(model, output_tokens):
    bos_token_id = model.get_bos_token_id() if hasattr(model, "get_bos_token_id") else None
    if bos_token_id is not None and len(output_tokens) > 0 and output_tokens[0].item() == bos_token_id:
        output_tokens = output_tokens[1:]

    output_text = model.llama_tokenizer.decode(output_tokens, skip_special_tokens=True)
    output_text = output_text.split("###")[0]
    output_text = output_text.replace("<s>", "").replace("</s>", "")
    output_text = output_text.split(r"[/INST]")[-1].strip()
    output_text = output_text.split("Assistant:")[-1].strip()
    return output_text.strip()


@torch.no_grad()
def generate_prediction(model, ecgs, question, args):
    ecgs = ecgs.to(model.device)
    prompt = build_prompt(question, ecgs.shape[0], model)

    image_embeds, _ = model.encode_img(ecgs)
    image_list = [image_embeds[i : i + 1] for i in range(image_embeds.shape[0])]
    context_embeds = model.get_context_emb(prompt, image_list)
    attention_mask = torch.ones(
        context_embeds.shape[:2],
        dtype=torch.long,
        device=context_embeds.device,
    )

    with model.maybe_autocast():
        outputs = model.llama_model.generate(
            inputs_embeds=context_embeds,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            do_sample=args.do_sample,
            top_p=args.top_p,
            temperature=float(args.temperature),
            repetition_penalty=1.0,
            pad_token_id=model.llama_tokenizer.pad_token_id,
            bos_token_id=model.get_bos_token_id(),
            eos_token_id=model.get_eos_token_id(),
        )

    return decode_generation(model, outputs[0])


def summarize_prediction_metrics(samples):
    if not samples:
        return {"num_samples": 0, "exact_match": 0.0, "token_f1": 0.0}

    exact_match = sum(sample["exact_match"] for sample in samples) / len(samples)
    token_f1 = sum(sample["token_f1"] for sample in samples) / len(samples)
    return {
        "num_samples": len(samples),
        "exact_match": exact_match,
        "token_f1": token_f1,
    }


def run_generation(args):
    dataset_roots = parse_dataset_roots(args.dataset_root)
    samples = load_json(args.dataset_json)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    model, vis_processor = load_model(args)
    ann_root = os.path.dirname(os.path.abspath(args.dataset_json))

    output_samples = []
    for idx, sample in enumerate(tqdm(samples, desc="Generating benchmark answers")):
        ecgs = load_multi_ecg_sample(
            sample=sample,
            dataset_roots=dataset_roots,
            vis_root=args.vis_root,
            ann_root=ann_root,
            vis_processor=vis_processor,
        )
        prediction = generate_prediction(model, ecgs, sample["question"], args)
        output_samples.append(
            {
                "id": idx,
                "question": sample["question"],
                "answer": sample.get("answer", ""),
                "prediction": prediction,
                "num_ecgs": ecgs.shape[0],
                "ecg_paths": raw_paths_from_sample(sample),
                "exact_match": exact_match_score(prediction, sample.get("answer", "")),
                "token_f1": token_f1_score(prediction, sample.get("answer", "")),
            }
        )

    payload = {
        "summary": summarize_prediction_metrics(output_samples),
        "samples": output_samples,
    }
    save_json(payload, args.predictions_path)
    return payload


def normalize_report_key(path_value):
    normalized = str(path_value).replace("\\", "/").strip()
    return os.path.splitext(normalized)[0]


def load_report_lookup(csv_path, path_column, report_column):
    lookup = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if path_column not in row or report_column not in row:
                raise KeyError(
                    f"CSV must contain '{path_column}' and '{report_column}' columns."
                )
            lookup[normalize_report_key(row[path_column])] = row[report_column].strip()
    return lookup

class OpenAIJudge:
    def __init__(self, model_name, base_url, api_key, temperature, max_tokens, fallback_model=None):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.fallback_model = fallback_model
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")
        if self.base_url.endswith("/chat/completions"):
            self.chat_url = self.base_url
        else:
            self.chat_url = f"{self.base_url}/chat/completions"

        self.headers = {"Content-Type": "application/json"}
        if api_key and api_key != "EMPTY":
            self.headers["Authorization"] = f"Bearer {api_key}"

    def chat(self, messages):
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": float(self.temperature),
            "max_tokens": int(self.max_tokens),
        }
        response = None
        try:
            response = requests.post(self.chat_url, headers=self.headers, json=payload, timeout=120)
            response.raise_for_status()
        except requests.HTTPError as exc:
            if (
                self.fallback_model
                and self.fallback_model != self.model_name
                and response is not None
                and response.status_code == 500
            ):
                fallback_payload = payload.copy()
                fallback_payload["model"] = self.fallback_model
                fallback_response = requests.post(
                    self.chat_url,
                    headers=self.headers,
                    json=fallback_payload,
                    timeout=120,
                )
                try:
                    fallback_response.raise_for_status()
                    data = fallback_response.json()
                    if "choices" not in data or not data["choices"]:
                        raise ValueError(
                            f"Judge returned an unexpected response format from fallback model: {json.dumps(data, indent=2)}"
                        )
                    return data["choices"][0]["message"]["content"]
                except requests.RequestException:
                    pass

            server_body = response.text if response is not None else ""
            error_message = (
                f"HTTP {response.status_code if response is not None else 'N/A'} from judge endpoint {self.chat_url}\n"
                f"Primary model: {self.model_name}\n"
                f"Fallback model: {self.fallback_model or 'none'}\n"
                f"Request payload: {json.dumps(payload, indent=2)[:1000]}\n"
                f"Response body: {server_body}"
            )
            raise requests.HTTPError(error_message, response=response) from exc
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Judge request failed for {self.chat_url}: {exc}"
            ) from exc

        data = response.json()
        if "choices" not in data or not data["choices"]:
            raise ValueError(
                f"Judge returned an unexpected response format: {json.dumps(data, indent=2)}"
            )
        return data["choices"][0]["message"]["content"]


def build_judge_prompt(sample, reports):
    report_lines = []
    if reports:
        for index, report in enumerate(reports, start=1):
            if report:
                report_lines.append(f"ECG{index} report: {report}")

    prompt_parts = [
        "You are grading an answer to a multi-ECG benchmark question.",
        f"Question: {sample['question']}",
        f"Reference answer: {sample['answer']}",
    ]
    if report_lines:
        prompt_parts.append("Reference ECG reports:")
        prompt_parts.extend(report_lines)
    prompt_parts.extend(
        [
            f"Model answer: {sample['prediction']}",
            (
                "Score the model answer from 0 to 5, where 0 means completely incorrect "
                "and 5 means fully correct. Consider correctness, completeness, and whether "
                "the answer is consistent with the ECG evidence."
            ),
            'Return JSON with keys "score" and "reasoning".',
        ]
    )
    return "\n".join(prompt_parts)


def extract_score(raw_response):
    if raw_response is None:
        return None

    try:
        parsed = json.loads(raw_response)
        score = parsed.get("score")
        if score is not None:
            return float(score)
    except json.JSONDecodeError:
        pass

    match = re.search(r'"score"\s*:\s*([0-5](?:\.\d+)?)', raw_response)
    if match:
        return float(match.group(1))

    match = re.search(r"\b([0-5](?:\.\d+)?)\b", raw_response)
    if match:
        return float(match.group(1))

    return None


def default_judge_output_path(predictions_path):
    path = Path(predictions_path)
    return str(path.with_name(path.stem + "_judged" + path.suffix))


def summarize_judge_results(samples):
    scored = [sample["judge_score"] for sample in samples if sample["judge_score"] is not None]
    if not scored:
        return {"num_samples": len(samples), "mean_score": None}

    rounded_distribution = Counter(str(int(score)) if float(score).is_integer() else str(score) for score in scored)
    return {
        "num_samples": len(samples),
        "num_scored": len(scored),
        "mean_score": sum(scored) / len(scored),
        "score_distribution": dict(sorted(rounded_distribution.items())),
    }


def load_predictions_for_judging(path_value):
    payload = load_json(path_value)
    if isinstance(payload, dict) and "samples" in payload:
        return payload
    raise ValueError(f"Predictions file '{path_value}' must contain a top-level 'samples' list.")


def run_judging(args, prediction_payload=None):
    if args.judge_model is None:
        raise ValueError("--judge-model is required for '--mode judge' and '--mode all'.")

    if prediction_payload is None:
        prediction_payload = load_predictions_for_judging(args.predictions_path)

    report_lookup = None
    if args.report_csv:
        report_lookup = load_report_lookup(
            args.report_csv,
            path_column=args.report_path_column,
            report_column=args.report_text_column,
        )

    judge = OpenAIJudge(
        model_name=args.judge_model,
        base_url=args.judge_base_url,
        api_key=args.judge_api_key,
        temperature=args.judge_temperature,
        max_tokens=args.judge_max_tokens,
        fallback_model=args.judge_fallback_model,
    )

    judged_samples = []
    for sample in tqdm(prediction_payload["samples"], desc="Judging answers"):
        reports = None
        if report_lookup is not None:
            reports = [
                report_lookup.get(normalize_report_key(path), "")
                for path in sample.get("ecg_paths", [])
            ]

        prompt = build_judge_prompt(sample, reports)
        messages = [{"role": "user", "content": prompt}]
        raw_response = judge.chat(messages)
        score = extract_score(raw_response)
        judged_samples.append(
            {
                **sample,
                "judge_score": score,
                "judge_response": raw_response,
                "reference_reports": reports,
            }
        )

    output_path = args.judge_output_path or default_judge_output_path(args.predictions_path)
    payload = {
        "summary": summarize_judge_results(judged_samples),
        "samples": judged_samples,
    }
    save_json(payload, output_path)
    return payload


def main():
    parser = build_parser()
    args = parser.parse_args()

    prediction_payload = None
    if args.mode in {"generate", "all"}:
        prediction_payload = run_generation(args)
        print(f"Saved predictions to {args.predictions_path}")
        print(json.dumps(prediction_payload["summary"], indent=2))

    if args.mode in {"judge", "all"}:
        judge_payload = run_judging(args, prediction_payload=prediction_payload)
        judge_output_path = args.judge_output_path or default_judge_output_path(args.predictions_path)
        print(f"Saved judge results to {judge_output_path}")
        print(json.dumps(judge_payload["summary"], indent=2))


if __name__ == "__main__":
    main()
