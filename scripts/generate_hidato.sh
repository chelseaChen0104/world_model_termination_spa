#!/bin/bash
# Generate Hidato (Numbrix-variant) LLM-policy SFT data.
#
# Hidato is a constraint-propagation puzzle (place sequential numbers 1..N
# in cells such that consecutive numbers are orthogonally adjacent) chosen
# as a third game (after Sudoku + Pentomino) to test whether the recipe
# transfers to a greedy-friendly env where forced moves dominate.
#
# Predictive-gap test (2026-05-01): 67% of random rollouts hit doom mid-
# trajectory, 29% succeed. Healthy mix for SFT.
#
# Output:
#   data/hidato_llm_policy/                           (raw multi-turn parquets)
#   data/hidato_llm_policy_minimal/                   (single-step minimal format
#                                                      with <observation>+<prediction>+
#                                                      <solvable>+<answer>)
#   data/hidato_llm_policy_minimal/wm_train_no_post_bp.parquet  ← B-H1 trains on this

set -e
REPO=/root/autodl-tmp/world_model_termination_spa
PYTHON=${PYTHON:-/root/miniconda3/bin/python}
N_TRAJ=${N_TRAJ:-3000}
SUFFIX=${SUFFIX:-}
SEED=${SEED:-42}

SUF_PATH=""
if [ -n "$SUFFIX" ]; then SUF_PATH="_$SUFFIX"; fi

cd "$REPO"

LOG=logs/datagen_hidato${SUF_PATH}.log
> "$LOG"

OUT_RAW="data/hidato_llm_policy${SUF_PATH}"
OUT_MIN="data/hidato_llm_policy_minimal${SUF_PATH}"

echo "============================================================"
echo "  Hidato LLM-policy data generation"
echo "============================================================"
echo "  Trajectories: $N_TRAJ (part='${SUFFIX:-full}', seed=$SEED)"
echo "  Env: Hidato (Numbrix variant, 4-connected adjacency)"
echo "  Puzzle bank: 8 hand-curated puzzles (3x3 to 5x4)"
echo "  Output (raw):     $OUT_RAW"
echo "  Output (minimal): $OUT_MIN"
echo "============================================================"

echo
echo "=== STAGE 1: LLM-policy trajectory generation ==="
bash scripts/_run_with_env.sh $PYTHON -u -m src.data.llm_trajectory_generator \
    --env hidato \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --num-trajectories $N_TRAJ \
    --max-steps 25 \
    --output-dir "$OUT_RAW" \
    --seed $SEED \
    --temperature 0.7 \
    --val-split 0.2 \
    --device auto \
    --multi-turn --max-context-turns 6 \
    --variant hidato_minimal \
    2>&1 | tee -a "$LOG"

echo
echo "=== STAGE 2: reformat to single-step minimal ==="
$PYTHON scripts/reformat_to_minimal.py \
    --input-dir "$OUT_RAW" \
    --output-dir "$OUT_MIN" \
    --variant hidato_minimal \
    2>&1 | tee -a "$LOG"

echo
echo "=== STAGE 3: filter post-BP ==="
$PYTHON scripts/filter_post_bp.py \
    --input-dir "$OUT_MIN" \
    2>&1 | tee -a "$LOG"

echo
echo "=== Sample counts ==="
$PYTHON -c "
import pandas as pd, os
d = '$OUT_MIN'
for fn in sorted(os.listdir(d)):
    if fn.endswith('.parquet'):
        n = len(pd.read_parquet(os.path.join(d, fn)))
        print(f'  {fn:48s} {n:6d} samples')
" 2>&1 | tee -a "$LOG"

echo
echo "============================================================"
echo "  Hidato data ready at:"
echo "    $OUT_MIN/wm_train_no_post_bp.parquet"
echo "  (use this for B-H1 SFT training)"
echo "============================================================"
