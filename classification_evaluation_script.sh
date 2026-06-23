#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CFG_PATH="${CFG_PATH:-$ROOT_DIR/TinyGPT-V-main/eval_configs/benchmark_evaluation.yaml}"
CKPT="${CKPT:?Set CKPT to the checkpoint you want to evaluate.}"
CLASSIFICATION_CSV_DIR="${CLASSIFICATION_CSV_DIR:?Set CLASSIFICATION_CSV_DIR to the directory containing ptbxl.csv, cpsc.csv, and csn.csv.}"
PTBXL_ROOT="${PTBXL_ROOT:?Set PTBXL_ROOT to the PTB-XL ECG root directory.}"
CPSC_ROOT="${CPSC_ROOT:?Set CPSC_ROOT to the CPSC ECG root directory.}"
CSN_ROOT="${CSN_ROOT:?Set CSN_ROOT to the CSN ECG root directory.}"
TEXT_EMBEDDING_MODEL="${TEXT_EMBEDDING_MODEL:-pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb}"
CLASSIFICATION_SCORING="${CLASSIFICATION_SCORING:-report}"
RESULT_PATH="${RESULT_PATH:-$ROOT_DIR/outputs/classification_benchmark.json}"
METHOD_NAME="${METHOD_NAME:-$(basename "${CKPT%.*}")}"
LOG_PATH="${LOG_PATH:-$ROOT_DIR/outputs/classification_benchmark_nohup.log}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "$(dirname "$RESULT_PATH")"
mkdir -p "$(dirname "$LOG_PATH")"

cd "$ROOT_DIR"

nohup "$PYTHON_BIN" /home/cmpdil/Iit_profbehra/TinyGPT-ECG/TinyGPT-V-main/inference.py \
  --dataset classification \
  --cfg-path "$CFG_PATH" \
  --ckpt "$CKPT" \
  --classification_csv_dir "$CLASSIFICATION_CSV_DIR" \
  --dataset-root "ptbxl=$PTBXL_ROOT" \
  --dataset-root "cpsc=$CPSC_ROOT" \
  --dataset-root "csn=$CSN_ROOT" \
  --text_embedding_model "$TEXT_EMBEDDING_MODEL" \
  --classification_scoring "$CLASSIFICATION_SCORING" \
  --method_name "$METHOD_NAME" \
  --result_path "$RESULT_PATH" \
  "$@" \
  > "$LOG_PATH" 2>&1 &

PID=$!
echo "Started evaluation with PID $PID"
echo "Log: $LOG_PATH"
echo "Result JSON: $RESULT_PATH"
