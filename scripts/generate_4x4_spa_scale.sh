#!/bin/bash
# Generate 4x4 LLM-policy data at SPA scale (~6,060 samples after no_post_bp filter).
#
# Math: existing 1000-traj run yields 1,336 no_post_bp samples (1.34 samples/traj).
#       Need 6,060 → 4,500 trajectories. Default 5,000 for safety margin (single-cloud).
#
# Parameterized for split-across-clouds:
#   N_TRAJ    Trajectory count (default 5000; for split, use 2500 each)
#   SUFFIX    Output dir suffix (default '': single-cloud full run; 'A'/'B' for split)
#   SEED      Random seed (default 42; use different per part)
#
# Output: data/sudoku_4x4_llm_policy_spa_scale[_SUFFIX]/   (raw multi-turn)
# After all parts collected, run scripts/combine_4x4_spa_scale_parts.py to
# merge + reformat + filter into data/sudoku_4x4_llm_policy_minimal_spa_scale/.
#
# Wall time on H800: ~7h for 5000 trajs; ~3.7h for 2500 trajs.

set -e
REPO=/root/autodl-tmp/world_model_termination_spa
PYTHON=${PYTHON:-/root/miniconda3/bin/python}
N_TRAJ=${N_TRAJ:-5000}
SUFFIX=${SUFFIX:-}
SEED=${SEED:-42}

SUF_PATH=""
if [ -n "$SUFFIX" ]; then SUF_PATH="_$SUFFIX"; fi

cd "$REPO"

LOG=logs/datagen_4x4_spa_scale${SUF_PATH}.log
> "$LOG"

echo "============================================================"
echo "  4x4 SPA-scale data generation (part='${SUFFIX:-full}', seed=$SEED)"
echo "============================================================"
echo "  Trajectories: $N_TRAJ"
echo "  Output: data/sudoku_4x4_llm_policy_spa_scale${SUF_PATH}/"
echo "============================================================"

bash scripts/_run_with_env.sh $PYTHON -u -m src.data.llm_trajectory_generator \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --num-trajectories $N_TRAJ \
    --max-steps 12 \
    --grid-size 4 \
    --difficulty easy \
    --output-dir "data/sudoku_4x4_llm_policy_spa_scale${SUF_PATH}" \
    --seed $SEED \
    --temperature 0.7 \
    --val-split 0.2 \
    --device auto \
    --multi-turn --max-context-turns 10 \
    2>&1 | tee "$LOG"

echo
echo "============================================================"
echo "  Part '${SUFFIX:-full}' raw data done."
echo "  Run scripts/combine_4x4_spa_scale_parts.py after both parts complete."
echo "============================================================"
