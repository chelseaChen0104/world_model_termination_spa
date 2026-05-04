#!/bin/bash
# Train Pentomino 5×6 base policy π_θ via SFT.
# Run on autodl3 via:
#   screen -dmS pent_pi_theta bash -c 'bash scripts/run_pi_theta_pentomino5x6.sh > logs/pi_theta_pent5x6.log 2>&1'

set -euo pipefail

REPO=/root/autodl-tmp/world_model_termination_spa
PY=/root/miniconda3/bin/python

# autodl3 reaches hf-mirror.com but not huggingface.co.
export HF_HOME=/root/autodl-tmp/cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com

cd "$REPO"

mkdir -p logs

DATA_TRAIN=$REPO/data/pentomino5x6/pi_theta_sft/train.jsonl
DATA_VAL=$REPO/data/pentomino5x6/pi_theta_sft/val.jsonl
# Save to /tmp (overlay fs ~19GB free) to avoid disk pressure on /root/autodl-tmp.
OUT=/tmp/sft_pentomino5x6_pi_theta

echo "=== π_θ SFT — Pentomino 5×6 ==="
echo "  $(date)"
echo "  train: $DATA_TRAIN"
echo "  val:   $DATA_VAL"
echo "  out:   $OUT"
echo ""

$PY scripts/train_pi_theta_sft.py \
    --train  "$DATA_TRAIN" \
    --val    "$DATA_VAL" \
    --output_dir "$OUT" \
    --model  "Qwen/Qwen2.5-1.5B-Instruct" \
    --epochs 3 \
    --batch  4 \
    --grad_accum 8 \
    --lr     1e-5 \
    --weight_decay 0.01 \
    --warmup_steps 50 \
    --max_length 512 \
    --logging_steps 10 \
    --eval_steps  50 \
    --gradient_checkpointing

echo ""
echo "=== TRAINING DONE — $(date) ==="
ls -lh "$OUT/final/" 2>/dev/null
