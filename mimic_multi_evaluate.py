import argparse
import json
import os
from pathlib import Path

HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
OPENAI_BASE_URL = "https://api.openai.com/v1"
OLLAMA_BASE_URL = "http://localhost:11434/v1"


def resolve_default_dataset_json():
    candidates = [
        Path("data/mimic_multiECG_test.json"),
        Path("data/mimic_llama3.3-70b_test.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate a generated Multi-MIMIC QA prediction JSON with a Hugging Face, OpenAI, or custom judge API."
    )
    parser.add_argument(
        "--predictions-path",
        required=True,
        help=(
            "Path to your generated prediction JSON. Supports either a top-level "
            "'samples' object or a raw list of sample objects."
        ),
    )
    parser.add_argument(
        "--dataset-json",
        default=resolve_default_dataset_json(),
        help=(
            "Benchmark dataset JSON used to recover missing fields like answer and ecg_path. "
            "Defaults to the first available Multi-MIMIC dataset JSON in ./data."
        ),
    )
    parser.add_argument(
        "--judge-output-path",
        default=None,
        help="Where to save judged results. Defaults to <predictions>_judged.json.",
    )
    parser.add_argument(
        "--report-csv",
        default=None,
        help=(
            "Optional MIMIC report CSV. If provided, the judge prompt will include the "
            "report text corresponding to each ECG."
        ),
    )
    parser.add_argument(
        "--record-list-csv",
        default=None,
        help=(
            "Optional MIMIC-IV-ECG record_list.csv. When paired with "
            "--machine-measurements-csv, the evaluator will reconstruct a waveform-path "
            "to report lookup from the raw public tables."
        ),
    )
    parser.add_argument(
        "--machine-measurements-csv",
        default=None,
        help=(
            "Optional MIMIC-IV-ECG machine_measurements.csv. When paired with "
            "--record-list-csv, report_0..report_17 will be concatenated into the "
            "report text used by the judge."
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
        "--allow-missing-reports",
        action="store_true",
        help=(
            "Allow paper-style judging to proceed even when --report-csv is missing or some ECG "
            "paths cannot be found in the CSV. This is off by default because it does not match "
            "the original evaluation script."
        ),
    )
    parser.add_argument(
        "--judge-model",
        required=True,
        help=(
            "Judge model name. For Hugging Face, this can be a Hub model id like "
            "'Qwen/Qwen2.5-72B-Instruct:fireworks' or 'deepseek-ai/DeepSeek-R1:fastest'."
        ),
    )
    parser.add_argument(
        "--judge-provider",
        choices=["huggingface", "openai", "ollama", "custom"],
        default="ollama",
        help=(
            "Which API family to use for judging. 'huggingface' uses the Hugging Face "
            "router by default, 'openai' uses the OpenAI API, 'ollama' uses a local "
            "Ollama OpenAI-compatible endpoint, and 'custom' lets you supply your own "
            "OpenAI-compatible --judge-base-url."
        ),
    )
    parser.add_argument(
        "--judge-base-url",
        default=None,
        help=(
            "Base URL for the judge API. Defaults to the Hugging Face router for "
            "--judge-provider huggingface, the OpenAI API for --judge-provider openai, "
            "the local Ollama API for --judge-provider ollama, or must be supplied "
            "explicitly for --judge-provider custom."
        ),
    )
    parser.add_argument(
        "--judge-api-key",
        default="EMPTY",
        help="API key for the judge endpoint. For Hugging Face, this can be your HF token.",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help=(
            "Optional Hugging Face token shortcut. If omitted, the script will use "
            "HF_TOKEN from the environment when --judge-provider huggingface."
        ),
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=0.0,
        help="Temperature for the judge model.",
    )
    parser.add_argument(
        "--judge-style",
        choices=["paper", "json"],
        default="paper",
        help=(
            "Judging protocol. 'paper' matches the original two-turn mimic_multi_evaluate.py "
            "behavior; 'json' uses a single structured grading prompt."
        ),
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
        help="Optional fallback judge model if the primary one fails with a server error.",
    )
    return parser


def configure_judge_api(args):
    provider = args.judge_provider
    api_key = args.judge_api_key
    if api_key in (None, "", "EMPTY"):
        api_key = None

    if provider == "huggingface":
        if not args.judge_base_url:
            args.judge_base_url = HF_ROUTER_BASE_URL
        args.judge_api_key = api_key or args.hf_token or os.environ.get("HF_TOKEN", "EMPTY")
        if args.judge_api_key in (None, "", "EMPTY"):
            raise ValueError(
                "Hugging Face judging requires an API token. Set HF_TOKEN in your environment "
                "or pass --hf-token/--judge-api-key."
            )
        return args

    if provider == "openai":
        if not args.judge_base_url:
            args.judge_base_url = OPENAI_BASE_URL
        args.judge_api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")
        return args

    if provider == "ollama":
        if not args.judge_base_url:
            args.judge_base_url = OLLAMA_BASE_URL
        args.judge_api_key = api_key or "EMPTY"
        return args

    if not args.judge_base_url:
        raise ValueError(
            "--judge-provider custom requires --judge-base-url pointing to an OpenAI-compatible API."
        )
    args.judge_api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("HF_TOKEN", "EMPTY")
    return args


def load_prediction_payload(path_value):
    with open(path_value, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and "samples" in payload:
        samples = payload["samples"]
    elif isinstance(payload, list):
        samples = payload
        payload = {"samples": samples}
    else:
        raise ValueError(
            f"Predictions file '{path_value}' must contain either a top-level 'samples' list "
            "or be a raw JSON list."
        )

    if not isinstance(samples, list):
        raise ValueError(f"'samples' in '{path_value}' must be a list.")

    normalized_samples = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"Sample {index} in '{path_value}' is not a JSON object.")
        normalized = dict(sample)
        normalized.setdefault("id", index)
        normalized_samples.append(normalized)

    return {**payload, "samples": normalized_samples}


def format_score_cell(value):
    if value is None:
        return "-"
    return f"{float(value):.2f}"


def print_paper_row(summary):
    row = summary.get("paper_score_row")
    if not row:
        return
    header = ["2", "3", "4", "5", "6", "All"]
    print("Paper-style score row:")
    print("\t".join(header))
    print("\t".join(format_score_cell(row.get(key)) for key in header))


def main():
    args = build_parser().parse_args()
    from evaluate_mimic_multi import (
        default_judge_output_path,
        enrich_prediction_payload,
        run_judging,
    )
    args = configure_judge_api(args)

    prediction_payload = load_prediction_payload(args.predictions_path)
    prediction_payload = enrich_prediction_payload(prediction_payload, args.dataset_json)

    if args.judge_output_path is None:
        args.judge_output_path = default_judge_output_path(args.predictions_path)

    result = run_judging(args, prediction_payload=prediction_payload)

    print(f"Saved judge results to {args.judge_output_path}")
    print(json.dumps(result["summary"], indent=2))
    print_paper_row(result["summary"])


if __name__ == "__main__":
    main()
