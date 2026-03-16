#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${1:-/root/fsas/AI Trusty}"
MODEL_DIR="${2:-$PROJECT_ROOT/models/Qwen2.5-VL-7B-Instruct}"
SUBSET_JSONL="${3:-/root/AI Trusty/outputs/imagenette_subset_100.jsonl}"
OUTPUT_ROOT="${4:-/root/AI Trusty/outputs/qwen_attention_imagenette_100}"
LIMIT="${5:-100}"

mkdir -p "$OUTPUT_ROOT"

python3 "$PROJECT_ROOT/prepare_imagenette_subset.py"   --imagenette-root "$PROJECT_ROOT/imagenette2-320"   --output "$SUBSET_JSONL"   --split val   --total "$LIMIT"   --seed 42

python3 "$PROJECT_ROOT/run_qwen_attention.py"   --model-dir "$MODEL_DIR"   --input-jsonl "$SUBSET_JSONL"   --output-jsonl "$OUTPUT_ROOT/all_records.jsonl"   --valid-output-jsonl "$OUTPUT_ROOT/valid_records.jsonl"   --failure-csv "$OUTPUT_ROOT/failures.csv"   --limit "$LIMIT"   --max-new-tokens 4

python3 "$PROJECT_ROOT/visualize_qwen_attention.py"   --input-jsonl "$OUTPUT_ROOT/valid_records.jsonl"   --output-dir "$OUTPUT_ROOT/viz"   --per-sample-limit 32   --top-layer-count 5   --band-width 3

python3 "$PROJECT_ROOT/analyze_attention_separability.py"   --input-jsonl "$OUTPUT_ROOT/valid_records.jsonl"   --output-dir "$OUTPUT_ROOT/analysis"   --cv-splits 5   --test-fraction 0.2
