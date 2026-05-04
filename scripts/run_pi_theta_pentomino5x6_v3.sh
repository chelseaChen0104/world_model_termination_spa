#!/bin/bash
# v3: Pentomino 5×6 base policy π_θ SFT.
#   v1 (3 epochs, 7K samples, no aug): per-step local_valid 65%, Pass@1 0/172
#   v2 (6 epochs, 28K samples, dihedral aug): per-step local_valid 84.5%,
#       Pass@1 STILL 0/172 — distribution shift to off-canonical-path states
#       at rollout time (model deviates → enters states no training set covers).
#
# v3 fix (per user directive 2026-05-04): add DAgger-lite off-path SFT
# samples WITHIN simple s,a format (no XML, no tags). At each step, we
# deviate to 2 different locally-valid placements, ask the solver for the
# recovery action from each off-canonical state, and add (off_state,
# recovery_action) as additional training samples. Combined with the 4-fold
# dihedral augmentation: each base tiling yields ~6 canonical + ~12
# DAgger samples, × 4 symmetries ≈ 60 samples per tiling × 1228 tilings
# ≈ 70K samples total.
#
# Run on autodl3:
#   screen -dmS pent_pi_theta_v3 bash -c 'bash scripts/run_pi_theta_pentomino5x6_v3.sh > logs/pi_theta_pent5x6_v3.log 2>&1'

set -euo pipefail

REPO=/root/autodl-tmp/world_model_termination_spa
PY=/root/miniconda3/bin/python

export HF_HOME=/root/autodl-tmp/cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com

cd "$REPO"
mkdir -p logs

# Step 1: regenerate SFT data with --augment + --dagger-deviations 2
DATA_DIR=$REPO/data/pentomino5x6/pi_theta_sft_v3

echo "=== v3 step 1: regenerate SFT data with augment + dagger ==="
echo "  $(date)"
$PY scripts/generate_pi_theta_sft_pentomino.py \
    --board-h 5 --board-w 6 --k-pieces 6 \
    --max-tilings-per-subset 10 \
    --output-dir "$DATA_DIR" \
    --val-fraction 0.1 \
    --seed 42 \
    --augment \
    --dagger-deviations 2

echo ""
echo "=== sample counts ==="
wc -l "$DATA_DIR/train.jsonl" "$DATA_DIR/val.jsonl"
echo ""

# Step 2: train with 6 epochs on the bigger data
DATA_TRAIN=$DATA_DIR/train.jsonl
DATA_VAL=$DATA_DIR/val.jsonl
OUT=/tmp/sft_pentomino5x6_pi_theta_v3

echo "=== v3 step 2: train π_θ — 6 epochs on augment+dagger data ==="
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
