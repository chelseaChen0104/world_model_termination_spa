#!/usr/bin/env bash
# Phase 3: ablations for SAVE Sudoku 4×4 toy stage.
# Runs sequentially:
#   - K-ablation: K ∈ {1, 2, 4, 8} for save + oracle
#   - no-calibration (T = 1.0)
#   - single-threshold (τ_fb = τ_keep)
# All on N=50 test puzzles.
set -e
PROJ=/root/autodl-tmp/world_model_termination_spa
PY=/root/miniconda3/bin/python
LOG=$PROJ/logs/q4_phase3.log
echo "=== Phase 3 ablations: $(date) ===" > "$LOG"
cd "$PROJ"

COMMON="--policy outputs/rl_b5_phase3_v8_anchor/final \
  --save_phi outputs/save_sudoku4_f_phi/final \
  --calibration outputs/save_sudoku4_f_phi/calibration.json \
  --eval data/sudoku4/test_natural_policy.jsonl \
  --n_puzzles 50 --max_steps 20"

# K-ablation: K ∈ {1, 2, 4, 8}, save + oracle
for K in 1 2 4 8; do
  echo "" >> "$LOG"
  echo "=== K=$K ablation ===" >> "$LOG"
  date >> "$LOG"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY -u \
    scripts/sudoku_scripts/q4_rollout.py $COMMON \
    --K $K \
    --methods save,oracle \
    --output_json outputs/q4_ablation_k${K}.json \
    >> "$LOG" 2>&1 || echo "K=$K FAIL" >> "$LOG"
done

# no-calibration (T=1.0)
echo "" >> "$LOG"
echo "=== no-calibration (T=1.0) ===" >> "$LOG"
date >> "$LOG"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY -u \
  scripts/sudoku_scripts/q4_rollout.py $COMMON \
  --K 8 --no_calib --methods save \
  --output_json outputs/q4_ablation_nocalib.json \
  >> "$LOG" 2>&1 || echo "no-calib FAIL" >> "$LOG"

# single-threshold (τ_fb = τ_keep = 0.796)
echo "" >> "$LOG"
echo "=== single-threshold (τ_fb = τ_keep = 0.796) ===" >> "$LOG"
date >> "$LOG"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $PY -u \
  scripts/sudoku_scripts/q4_rollout.py $COMMON \
  --K 8 --tau_fb_override 0.796 --methods save \
  --output_json outputs/q4_ablation_singlethresh.json \
  >> "$LOG" 2>&1 || echo "single-thresh FAIL" >> "$LOG"

echo "" >> "$LOG"
echo "=== Phase 3 DONE: $(date) ===" >> "$LOG"
df -h /root/autodl-tmp >> "$LOG"
