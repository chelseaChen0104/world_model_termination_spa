#!/bin/bash
# Sudoku 4×4 SAVE data gen on autodl4: pilot (3000/1000/1000) + paper-final
# (8000/1500/1500), sequential. Background sanity check runs validate after
# the first 50 train records appear, logged to logs/sudoku_pilot_sanity.log.
#
# Run via:
#   tmux new -d -s sudoku_full \
#     'bash scripts/run_sudoku_pilot_then_paper.sh > logs/sudoku_full.log 2>&1'

set -euo pipefail

REPO=/root/autodl-tmp/world_model_termination_spa
PY=/root/miniconda3/bin/python

export HF_HOME=/root/autodl-tmp/cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd "$REPO"
mkdir -p logs data/sudoku4

POLICY=outputs/rl_b5_phase3_v8_anchor/final
POLICY_ID=rl_b5_phase3_v8_anchor_final

PILOT_TRAIN=data/sudoku4/pilot_train_balanced.jsonl
PILOT_VAL=data/sudoku4/pilot_val_natural_calibration.jsonl
PILOT_TEST=data/sudoku4/pilot_test_natural_policy.jsonl
PAPER_TRAIN=data/sudoku4/paper_train_balanced.jsonl
PAPER_VAL=data/sudoku4/paper_val_natural_calibration.jsonl
PAPER_TEST=data/sudoku4/paper_test_natural_policy.jsonl

# --- background sanity check at first 50 records ---
sanity_check_at_50() {
    local file="$1"
    while [ ! -f "$file" ] || [ "$(wc -l < "$file" 2>/dev/null || echo 0)" -lt 50 ]; do
        sleep 20
    done
    echo "[$(date)] [SANITY] $file has 50+ records — running validate" \
        | tee -a logs/sudoku_pilot_sanity.log
    $PY scripts/validate_dataset.py "$file" 2>&1 \
        | tee -a logs/sudoku_pilot_sanity.log
    if grep -q "0 violations across all records" logs/sudoku_pilot_sanity.log; then
        echo "[SANITY] PASS — 0 schema violations" | tee -a logs/sudoku_pilot_sanity.log
    else
        echo "[SANITY] FAIL — see logs/sudoku_pilot_sanity.log" | tee -a logs/sudoku_pilot_sanity.log
        echo "[SANITY] Continuing anyway; user should inspect."
    fi
}

# Run sanity in background; do NOT block on it (let it finish whenever).
sanity_check_at_50 "$PILOT_TRAIN" &
SANITY_PID=$!

# --- helper: emit one role ---
# n_root_puzzles tuning: Sudoku averages ~2 records/puzzle (sibling_sets_per_root=3
# cap interacts with boundary-skip). For role split 70/15/15:
#   train slice gets 70% of the pool → need ~target/(2*0.7) puzzles
#   val/test slice gets 15% → need ~target/(2*0.15)
# We pick the larger requirement and round up. v1 launcher missed this, so the
# default 500 capped pilot at ~712/143/143 vs target 3000/1000/1000.
gen_role() {
    local role="$1" target="$2" output="$3" seed="$4" stage="$5" n_root="$6"
    echo ""
    echo "=== [$stage] $role  target=$target  seed=$seed  n_root=$n_root  ==> $output ==="
    echo "    $(date)"
    $PY scripts/generate_save_data.py \
        --env sudoku \
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
echo "  Sudoku 4×4 SAVE data gen — pilot then paper-final"
echo "  $(date)"
echo "  policy: $POLICY (id=$POLICY_ID)"
echo "============================================================"

# --- Pilot: 3000/1000/1000 ---
# n_root=4000 → train slice 2800 puzzles × ~2 = 5600 ceiling (≥3000 target);
# val/test slice 600 puzzles × ~2 = 1200 ceiling (≥1000 target).
gen_role train_balanced 3000 "$PILOT_TRAIN" 100 PILOT 4000
gen_role val_natural_calibration 1000 "$PILOT_VAL" 200 PILOT 4000
gen_role test_natural_policy 1000 "$PILOT_TEST" 300 PILOT 4000

echo ""
echo "============================================================"
echo "  PILOT DONE — $(date)"
wc -l "$PILOT_TRAIN" "$PILOT_VAL" "$PILOT_TEST"
echo "============================================================"

# Wait for sanity to finish if still running.
wait $SANITY_PID 2>/dev/null || true
echo ""
echo "[SANITY] final state:"
tail -10 logs/sudoku_pilot_sanity.log 2>/dev/null || echo "no sanity log"

# --- Paper-final: 8000/1500/1500 ---
echo ""
echo "============================================================"
echo "  PAPER-FINAL stage starting — $(date)"
echo "============================================================"
# n_root=10000 → train slice 7000 × ~2 = 14000 ceiling (≥8000); val/test slice
# 1500 × ~2 = 3000 ceiling (≥1500).
gen_role train_balanced 8000 "$PAPER_TRAIN" 1000 PAPER 10000
gen_role val_natural_calibration 1500 "$PAPER_VAL" 2000 PAPER 10000
gen_role test_natural_policy 1500 "$PAPER_TEST" 3000 PAPER 10000

echo ""
echo "============================================================"
echo "  ALL DONE — $(date)"
wc -l "$PILOT_TRAIN" "$PILOT_VAL" "$PILOT_TEST" "$PAPER_TRAIN" "$PAPER_VAL" "$PAPER_TEST"
echo "============================================================"
