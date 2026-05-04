#!/bin/bash
# Generate Hidato 5x4 SAVE data at PILOT scale (3000/1000/1000) using the
# 600-puzzle v3 bank. Toy stage was 1500/346/345 with the 200-puzzle v2 bank;
# v3 lifts the val/test ceiling by 3x and the train ceiling by ~3x.
#
# Run on autodl via:
#   tmux new -d -s hidato_pilot 'bash scripts/run_hidato_pilot_data_gen.sh > logs/hidato_pilot.log 2>&1'

set -euo pipefail

REPO=/root/autodl-tmp/world_model_termination_spa
PY=/root/miniconda3/bin/python

# Force the v3 (600-puzzle) bank — toy used v2 (200 puzzles).
export HIDATO_BANK=v3
# AutoDL HF offline (Qwen2.5 cached locally).
export HF_HOME=/root/autodl-tmp/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$REPO"

mkdir -p logs data/hidato5x4

POLICY=outputs/rl_b_h1_v8_anchor/final
POLICY_ID=rl_b_h1_v8_anchor_final

OUT_TRAIN=data/hidato5x4/pilot_train_balanced.jsonl
OUT_VAL=data/hidato5x4/pilot_val_natural_calibration.jsonl
OUT_TEST=data/hidato5x4/pilot_test_natural_policy.jsonl

echo "=== Hidato pilot SAVE data gen ==="
echo "  $(date)"
echo "  bank: v3 (600 puzzles)"
echo "  policy: $POLICY (id=$POLICY_ID)"
echo "  outputs: $OUT_TRAIN, $OUT_VAL, $OUT_TEST"
echo ""

# train_balanced — 3000 sibling sets, all 5 sources (lt:ht:rand:sol:prt)
echo "--- generating train_balanced (3000) ---"
$PY scripts/generate_save_data.py \
    --env hidato \
    --role train_balanced \
    --policy-model "$POLICY" \
    --policy-checkpoint-id "$POLICY_ID" \
    --n-target 3000 \
    --output "$OUT_TRAIN" \
    --seed 100

echo ""
echo "--- generating val_natural_calibration (1000) ---"
$PY scripts/generate_save_data.py \
    --env hidato \
    --role val_natural_calibration \
    --policy-model "$POLICY" \
    --policy-checkpoint-id "$POLICY_ID" \
    --n-target 1000 \
    --output "$OUT_VAL" \
    --seed 200

echo ""
echo "--- generating test_natural_policy (1000) ---"
$PY scripts/generate_save_data.py \
    --env hidato \
    --role test_natural_policy \
    --policy-model "$POLICY" \
    --policy-checkpoint-id "$POLICY_ID" \
    --n-target 1000 \
    --output "$OUT_TEST" \
    --seed 300

echo ""
echo "=== Pilot data gen DONE — $(date) ==="
wc -l "$OUT_TRAIN" "$OUT_VAL" "$OUT_TEST"

echo ""
echo "--- validation ---"
$PY scripts/validate_dataset.py "$OUT_TRAIN"  | tail -25
$PY scripts/validate_dataset.py "$OUT_VAL"    | tail -20
$PY scripts/validate_dataset.py "$OUT_TEST"   | tail -20
