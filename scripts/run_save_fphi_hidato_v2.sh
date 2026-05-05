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
mkdir -p logs data/hidato5x4/sft_pilot

# 2026-05-05 update: retrain on PILOT data (3000/1000/1000) instead of toy
# (1500/346/345). Materialize per-candidate .sft.jsonl files first via
# save_sft_prepare.py, then train.
RAW_TRAIN=$REPO/data/hidato5x4/pilot_train_balanced.jsonl
RAW_VAL=$REPO/data/hidato5x4/pilot_val_natural_calibration.jsonl
RAW_TEST=$REPO/data/hidato5x4/pilot_test_natural_policy.jsonl

SFT_TRAIN=$REPO/data/hidato5x4/sft_pilot/train.sft.jsonl
SFT_VAL=$REPO/data/hidato5x4/sft_pilot/val.sft.jsonl
SFT_TEST=$REPO/data/hidato5x4/sft_pilot/test.sft.jsonl

OUT=/tmp/save_fphi_hidato5x4_v2

echo "=== SAVE f_phi v2 (token-weighted CE, PILOT data) — Hidato 5x4 ==="
echo "  $(date)"
echo ""

# Step 1: materialize per-candidate SFT samples from pilot raw JSONL
echo "--- Step 1: materializing pilot .sft.jsonl files ---"
$PY scripts/save_sft_prepare.py --input "$RAW_TRAIN" --output "$SFT_TRAIN"
$PY scripts/save_sft_prepare.py --input "$RAW_VAL"   --output "$SFT_VAL"
$PY scripts/save_sft_prepare.py --input "$RAW_TEST"  --output "$SFT_TEST"
echo ""

# Step 2: train v2 (token-weighted CE)
echo "--- Step 2: training v2 ---"
echo "  train: $SFT_TRAIN"
echo "  val:   $SFT_VAL"
echo "  out:   $OUT"

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
