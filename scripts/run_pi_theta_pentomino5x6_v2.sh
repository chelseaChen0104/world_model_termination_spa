#!/bin/bash
# v2: Pentomino 5×6 base policy π_θ SFT with (1) more epochs + (2) symmetry
# augmentation. Comparison reference is the v1 run (3 epochs, no aug):
#   - v1 result: eval_loss 0.169, Pass@1 = 0/172 (per-step local_valid 65%)
# v2 hypothesis: 4× data via dihedral aug + more passes lifts both metrics.
#
# Run on autodl3:
#   screen -dmS pent_pi_theta_v2 bash -c 'bash scripts/run_pi_theta_pentomino5x6_v2.sh > logs/pi_theta_pent5x6_v2.log 2>&1'

set -euo pipefail

REPO=/root/autodl-tmp/world_model_termination_spa
PY=/root/miniconda3/bin/python

# autodl3 reaches hf-mirror.com but not huggingface.co.
export HF_HOME=/root/autodl-tmp/cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com

cd "$REPO"

mkdir -p logs

# Step 1: regenerate SFT data with --augment.
DATA_DIR=$REPO/data/pentomino5x6/pi_theta_sft_aug

echo "=== v2 step 1: regenerate augmented SFT data ==="
echo "  $(date)"
$PY scripts/generate_pi_theta_sft_pentomino.py \
    --board-h 5 --board-w 6 --k-pieces 6 \
    --max-tilings-per-subset 10 \
    --output-dir "$DATA_DIR" \
    --val-fraction 0.1 \
    --seed 42 \
    --augment

echo ""
echo "=== sample counts ==="
wc -l "$DATA_DIR/train.jsonl" "$DATA_DIR/val.jsonl"
echo ""

# Step 2: train with 6 epochs on augmented data.
DATA_TRAIN=$DATA_DIR/train.jsonl
DATA_VAL=$DATA_DIR/val.jsonl
OUT=/tmp/sft_pentomino5x6_pi_theta_v2

echo "=== v2 step 2: train π_θ — 6 epochs on augmented data ==="
echo "  $(date)"
$PY scripts/train_pi_theta_sft.py \
    --train  "$DATA_TRAIN" \
    --val    "$DATA_VAL" \
    --output_dir "$OUT" \
    --model  "Qwen/Qwen2.5-1.5B-Instruct" \
    --epochs 6 \
    --batch  4 \
    --grad_accum 8 \
    --lr     1e-5 \
    --weight_decay 0.01 \
    --warmup_steps 100 \
    --max_length 512 \
    --logging_steps 20 \
    --eval_steps  100 \
    --gradient_checkpointing

echo ""
echo "=== TRAINING DONE — $(date) ==="
ls -lh "$OUT/final/" 2>/dev/null
