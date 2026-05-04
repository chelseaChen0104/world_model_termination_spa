#!/bin/bash
# Re-train Hidato f_phi with token-weighted CE (v2 trainer).
# v1 result (eval_loss 0.0068 but viability AUC 0.60-0.64): training was
# dominated by easy next-state copy tokens; viability boolean got ~1% of
# the gradient. v2 multiplies CE at true/false label positions by 100x.
#
# Run on autodl via:
#   tmux new -d -s save_fphi_hidato_v2 'bash scripts/run_save_fphi_hidato_v2.sh > logs/save_fphi_hidato_v2.log 2>&1'

set -euo pipefail

REPO=/root/autodl-tmp/world_model_termination_spa
PY=/root/miniconda3/bin/python

export HF_HOME=/root/autodl-tmp/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$REPO"

mkdir -p logs

# .sft.jsonl files were materialized by the v1 run; reuse them.
SFT_TRAIN=$REPO/data/hidato5x4/sft/train.sft.jsonl
SFT_VAL=$REPO/data/hidato5x4/sft/val.sft.jsonl

OUT=/tmp/save_fphi_hidato5x4_v2

echo "=== SAVE f_phi v2 (token-weighted CE) — Hidato 5x4 ==="
echo "  $(date)"
echo "  train: $SFT_TRAIN"
echo "  val:   $SFT_VAL"
echo "  out:   $OUT"
echo ""

$PY scripts/train_save_fphi_v2.py \
    --train  "$SFT_TRAIN" \
    --val    "$SFT_VAL" \
    --output_dir "$OUT" \
    --model  "Qwen/Qwen2.5-1.5B-Instruct" \
    --epochs 3 \
    --batch  4 \
    --grad_accum 8 \
    --lr     1e-5 \
    --weight_decay 0.01 \
    --warmup_steps 30 \
    --max_length 1024 \
    --logging_steps 5 \
    --eval_steps  20 \
    --bool-weight 100.0 \
    --gradient_checkpointing

echo ""
echo "=== TRAINING DONE — $(date) ==="
ls -lh "$OUT/final/" 2>/dev/null
