#!/bin/bash
# Launch SAVE f_phi SFT training for Hidato 5x4 on autodl.
# Run via: tmux new -d -s save_fphi_hidato 'bash scripts/run_save_fphi_hidato.sh > logs/save_fphi_hidato.log 2>&1'

set -euo pipefail

REPO=/root/autodl-tmp/world_model_termination_spa
PY=/root/miniconda3/bin/python

# AutoDL can't reach huggingface.co — use the locally cached snapshot.
export HF_HOME=/root/autodl-tmp/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$REPO"

mkdir -p logs data/hidato5x4/sft

DATA_TRAIN=$REPO/data/hidato5x4/train_balanced.jsonl
DATA_VAL=$REPO/data/hidato5x4/val_natural_calibration.jsonl
DATA_TEST=$REPO/data/hidato5x4/test_natural_policy.jsonl

SFT_TRAIN=$REPO/data/hidato5x4/sft/train.sft.jsonl
SFT_VAL=$REPO/data/hidato5x4/sft/val.sft.jsonl
SFT_TEST=$REPO/data/hidato5x4/sft/test.sft.jsonl

# Save model to /tmp (overlay fs ~29GB free) to avoid disk pressure on
# /root/autodl-tmp; rsync to local archive immediately after training completes.
OUT=/tmp/save_fphi_hidato5x4

echo "=== SAVE f_phi SFT — Hidato 5x4 ==="
echo "  $(date)"
echo "  python: $PY"
echo ""

# --- Step 1: materialize per-candidate SFT JSONL ---
echo "--- Materializing SFT JSONL via save_sft_prepare.py ---"
$PY scripts/save_sft_prepare.py --input "$DATA_TRAIN" --output "$SFT_TRAIN"
$PY scripts/save_sft_prepare.py --input "$DATA_VAL"   --output "$SFT_VAL"
$PY scripts/save_sft_prepare.py --input "$DATA_TEST"  --output "$SFT_TEST"
echo ""

# --- Step 2: train f_phi ---
echo "--- Training f_phi ---"
$PY scripts/train_save_fphi.py \
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
    --gradient_checkpointing

echo ""
echo "=== TRAINING DONE — $(date) ==="
ls -lh "$OUT/final/" 2>/dev/null
