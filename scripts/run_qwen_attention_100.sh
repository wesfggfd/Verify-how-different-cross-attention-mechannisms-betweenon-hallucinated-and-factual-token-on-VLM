#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${1:-/root/AI Trusty/.hf_home/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5}"
INPUT_JSONL="${2:-/root/AI Trusty/imagenette_100_questions.jsonl}"
OUTPUT_ROOT="${3:-/root/AI Trusty/outputs/qwen_attention_100}"
LIMIT="${4:-100}"

mkdir -p "$OUTPUT_ROOT"

python3 run_qwen_attention.py \
  --model-dir "$MODEL_DIR" \
  --input-jsonl "$INPUT_JSONL" \
  --output-jsonl "$OUTPUT_ROOT/attention_records.jsonl" \
  --limit "$LIMIT" \
  --max-new-tokens 4

python3 visualize_qwen_attention.py \
  --input-jsonl "$OUTPUT_ROOT/attention_records.jsonl" \
  --output-dir "$OUTPUT_ROOT/visualizations" \
  --per-sample-limit "$LIMIT"
