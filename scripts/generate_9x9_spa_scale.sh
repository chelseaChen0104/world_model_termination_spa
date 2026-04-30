#!/bin/bash
# Generate 9x9 LLM-policy data at SPA scale (~6,060 samples after no_post_bp filter).
#
# Math: existing 4500-traj diverse run yields 2,482 no_post_bp samples (0.55 samples/traj).
#       Need 6,060 → 11,100 trajectories. Default split: 3,700 per difficulty × 3.
#
# This script is parameterized to support split-across-clouds:
#   SUFFIX    Output dir suffix (default '': single-cloud full run; 'A'/'B' for split)
#   SEED_BASE Seeds for easy/medium/hard = SEED_BASE+{0,100,200} (default 42)
#   N_EASY/N_MEDIUM/N_HARD  Per-difficulty trajectory counts (default 3700)
#
# Output: data/sudoku_llm_policy_{easy,medium,hard}_spa_scale[_SUFFIX]/  (raw multi-turn)
# Each part can be combined later with scripts/combine_9x9_spa_scale_parts.py.
#
# Wall time on H800: ~12h for full run; ~9.5h for SUFFIX=A (8,700 trajs); ~2.5h for SUFFIX=B (2,400 trajs)

set -e
REPO=/root/autodl-tmp/world_model_termination_spa
PYTHON=${PYTHON:-/root/miniconda3/bin/python}
N_EASY=${N_EASY:-3700}
N_MEDIUM=${N_MEDIUM:-3700}
N_HARD=${N_HARD:-3700}
SUFFIX=${SUFFIX:-}
SEED_BASE=${SEED_BASE:-42}

# If SUFFIX is non-empty, append underscore for path concatenation
SUF_PATH=""
if [ -n "$SUFFIX" ]; then SUF_PATH="_$SUFFIX"; fi

cd "$REPO"

LOG=logs/datagen_9x9_spa_scale${SUF_PATH}.log
> "$LOG"

echo "============================================================"
echo "  9x9 SPA-scale data generation (part='${SUFFIX:-full}', seed_base=$SEED_BASE)"
echo "============================================================"
echo "  Easy:   $N_EASY  Medium: $N_MEDIUM  Hard: $N_HARD"
echo "  Total:  $((N_EASY+N_MEDIUM+N_HARD)) trajectories"
echo "  Output: data/sudoku_llm_policy_<diff>_spa_scale${SUF_PATH}/"
echo "============================================================"

run_one() {
    local difficulty=$1
    local n=$2
    local out_dir=$3
    local seed=$4
    echo "============================================================"
    echo "  STARTING: difficulty=$difficulty n=$n out=$out_dir seed=$seed"
    echo "============================================================"
    bash scripts/_run_with_env.sh $PYTHON -u -m src.data.llm_trajectory_generator \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --num-trajectories "$n" \
        --max-steps 30 \
        --grid-size 9 \
        --difficulty "$difficulty" \
        --output-dir "$out_dir" \
        --seed "$seed" \
        --temperature 0.7 \
        --val-split 0.2 \
        --device auto \
        --multi-turn --max-context-turns 10 \
        2>&1 | tee -a "$LOG"
}

run_one easy   "$N_EASY"   "data/sudoku_llm_policy_easy_spa_scale${SUF_PATH}"   "$((SEED_BASE+0))"
run_one medium "$N_MEDIUM" "data/sudoku_llm_policy_medium_spa_scale${SUF_PATH}" "$((SEED_BASE+100))"
run_one hard   "$N_HARD"   "data/sudoku_llm_policy_hard_spa_scale${SUF_PATH}"   "$((SEED_BASE+200))"

echo
echo "============================================================"
echo "  Part '${SUFFIX:-full}' raw data done."
echo "  Run scripts/combine_9x9_spa_scale_parts.py after both A and B complete."
echo "============================================================"
