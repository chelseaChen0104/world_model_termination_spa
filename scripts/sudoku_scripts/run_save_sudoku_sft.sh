#!/usr/bin/env bash
# Launch SAVE Sudoku f_phi SFT on autodl2.
# Run from project root: bash scripts/sudoku_scripts/run_save_sudoku_sft.sh

set -euo pipefail

PROJ=/root/autodl-tmp/world_model_termination_spa
PY=/root/miniconda3/bin/python
DATE=$(date +%Y%m%d_%H%M%S)
OUT=$PROJ/outputs/save_sudoku4_f_phi
LOG=$PROJ/logs/save_sft_sudoku_${DATE}.log

mkdir -p "$(dirname "$LOG")"

cd "$PROJ"

echo "[launch] log -> $LOG"
echo "[launch] output dir -> $OUT"

$PY scripts/sudoku_scripts/save_sft_train.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --train data/sudoku4/sft/train_balanced.sft.jsonl \
    --val   data/sudoku4/sft/val_natural_calibration.sft.jsonl \
    --output_dir "$OUT" \
    --num_train_epochs 3 \
    --learning_rate 1e-5 \
    --per_device_batch_sets 8 \
    --gradient_accumulation_steps 1 \
    --max_length 512 \
    --warmup_steps 100 \
    --logging_steps 10 \
    --eval_steps 100 \
    --save_steps 500 \
    --bf16 \
    "$@" 2>&1 | tee "$LOG"
