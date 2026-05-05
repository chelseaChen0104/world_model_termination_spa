#!/bin/bash
# Pentomino 5×6 SAVE data gen on autodl3: pilot (3000/1000/1000) + paper-final
# (8000/1500/1500), sequential, using v3 SFT as π_θ for lt/ht sampling.
# Per doc/eval_2026-05-05_pentomino_pi_theta_sft_rl.md, RL on top of v3 SFT
# did not improve Pass@1; we adopt v3 SFT as the production π_θ.
#
# Background sanity checks fire after the first 50 records of each stage's
# train file, log to logs/pent_<stage>_sanity.log.
#
# Run on autodl3:
#   screen -dmS pent_save_full bash -c \
#     'bash scripts/run_pentomino_save_pilot_then_paper.sh > logs/pent_save_full.log 2>&1'

set -euo pipefail

REPO=/root/autodl-tmp/world_model_termination_spa
PY=/root/miniconda3/bin/python

export HF_HOME=/root/autodl-tmp/cache/huggingface
export HF_ENDPOINT=https://hf-mirror.com
# Pre-placed-piece variant disabled (PENT_PRE_PLACE=0) per 2026-05-05
# diagnosis: with pre-place=1, trajectories are only 5 steps and we
# emit records only from t=0,1,2 (the script skips last 2 steps). The
# 5×6 search tree is wide enough at those early steps that mixed_rate
# was ~3% (target 60%). Disabling pre-place gives full 6-step
# trajectories so t=3 (mid-game with 2 pieces remaining) gets sampled
# — that's where the constraint actually bites and doomed candidates
# become more available.
export PENT_PRE_PLACE=0

cd "$REPO"
mkdir -p logs data/pentomino5x6

POLICY=/tmp/sft_pentomino5x6_pi_theta_v3/final
POLICY_ID=sft_pentomino5x6_pi_theta_v3_final

PILOT_TRAIN=data/pentomino5x6/pilot_train_balanced.jsonl
PILOT_VAL=data/pentomino5x6/pilot_val_natural_calibration.jsonl
PILOT_TEST=data/pentomino5x6/pilot_test_natural_policy.jsonl
PAPER_TRAIN=data/pentomino5x6/paper_train_balanced.jsonl
PAPER_VAL=data/pentomino5x6/paper_val_natural_calibration.jsonl
PAPER_TEST=data/pentomino5x6/paper_test_natural_policy.jsonl

# --- background sanity check at first 50 records ---
sanity_check_at_50() {
    local file="$1" stage="$2"
    local logfile="logs/pent_${stage}_sanity.log"
    : > "$logfile"
    while [ ! -f "$file" ] || [ "$(wc -l < "$file" 2>/dev/null || echo 0)" -lt 50 ]; do
        sleep 30
    done
    echo "[$(date)] [SANITY $stage] $file has 50+ records — running validate" \
        | tee -a "$logfile"
    $PY scripts/validate_dataset.py "$file" 2>&1 | tee -a "$logfile"
    if grep -q "0 violations across all records" "$logfile"; then
        echo "[SANITY $stage] PASS — 0 schema violations" | tee -a "$logfile"
    else
        echo "[SANITY $stage] FAIL — see $logfile" | tee -a "$logfile"
    fi
}

sanity_check_at_50 "$PILOT_TRAIN" pilot &
PILOT_SANITY_PID=$!
sanity_check_at_50 "$PAPER_TRAIN" paper &
PAPER_SANITY_PID=$!

# --- helper: emit one role ---
# n_root_puzzles tuning for Pentomino 5×6 with PENT_PRE_PLACE=1:
# Each root yields up to sibling_sets_per_root=3 records, but boundary
# oversampling skips ~50% of non-boundary states, so effective ~2 records/root.
# Pilot 3000 train target → need ≥1500 train roots → ≥2143 total roots
# (70/15/15 split). Use 2500 for headroom, 6000 for paper (≥8000 train).
gen_role() {
    local role="$1" target="$2" output="$3" seed="$4" stage="$5" n_root="$6"
    echo ""
    echo "=== [$stage] $role  target=$target  seed=$seed  n_root=$n_root  ==> $output ==="
    echo "    $(date)"
    $PY scripts/generate_save_data.py \
        --env polyomino \
        --role "$role" \
        --policy-model "$POLICY" \
        --policy-checkpoint-id "$POLICY_ID" \
        --n-target "$target" \
        --n-root-puzzles "$n_root" \
        --output "$output" \
        --seed "$seed"
    echo "    [$stage] $role done $(date)"
}

echo "============================================================"
echo "  Pentomino 5×6 SAVE data gen — pilot then paper-final"
echo "  $(date)"
echo "  policy: $POLICY (id=$POLICY_ID)"
echo "  PENT_PRE_PLACE=$PENT_PRE_PLACE"
echo "============================================================"

# --- Pilot: 3000/1000/1000 ---
gen_role train_balanced 3000 "$PILOT_TRAIN" 100 PILOT 2500
gen_role val_natural_calibration 1000 "$PILOT_VAL" 200 PILOT 2500
gen_role test_natural_policy 1000 "$PILOT_TEST" 300 PILOT 2500

echo ""
echo "============================================================"
echo "  PILOT DONE — $(date)"
wc -l "$PILOT_TRAIN" "$PILOT_VAL" "$PILOT_TEST"
echo "============================================================"

wait $PILOT_SANITY_PID 2>/dev/null || true
echo ""
echo "[SANITY pilot] final state:"
tail -5 logs/pent_pilot_sanity.log 2>/dev/null || echo "no sanity log"

# --- Paper-final: 8000/1500/1500 ---
echo ""
echo "============================================================"
echo "  PAPER-FINAL stage starting — $(date)"
echo "============================================================"
gen_role train_balanced 8000 "$PAPER_TRAIN" 1000 PAPER 6000
gen_role val_natural_calibration 1500 "$PAPER_VAL" 2000 PAPER 6000
gen_role test_natural_policy 1500 "$PAPER_TEST" 3000 PAPER 6000

wait $PAPER_SANITY_PID 2>/dev/null || true
echo ""
echo "[SANITY paper] final state:"
tail -5 logs/pent_paper_sanity.log 2>/dev/null || echo "no sanity log"

echo ""
echo "============================================================"
echo "  ALL DONE — $(date)"
wc -l "$PILOT_TRAIN" "$PILOT_VAL" "$PILOT_TEST" "$PAPER_TRAIN" "$PAPER_VAL" "$PAPER_TEST"
echo "============================================================"
