#!/bin/bash
# Generate pentomino-easy LLM-policy data (per doc/spec_pentomino.md §7).
#
# Easy variant: 5×4 board with pieces {L, P, W, Y} (locked by P-0, 20 distinct tilings).
# Target: ~3,000 trajectories → ~6,000 single-step samples after no_post_bp filter.
# Wall time on H800: ~3-4 hours (rollouts ≤ 4 steps × ~1.5s each ≈ 12s/traj × 3000 ÷ ~24% accept).
#
# Output:
#   data/pentomino_easy_llm_policy/                    (raw multi-turn parquets)
#   data/pentomino_easy_llm_policy_minimal/            (single-step minimal format,
#                                                       with <observation> + <next_state>
#                                                       + <viability> + <answer> tags)
#   data/pentomino_easy_llm_policy_minimal/wm_train_no_post_bp.parquet  ← B-7 trains on this

set -e
REPO=/root/autodl-tmp/world_model_termination_spa
PYTHON=${PYTHON:-/root/miniconda3/bin/python}
N_TRAJ=${N_TRAJ:-3000}
SUFFIX=${SUFFIX:-}
SEED=${SEED:-42}

SUF_PATH=""
if [ -n "$SUFFIX" ]; then SUF_PATH="_$SUFFIX"; fi

cd "$REPO"

LOG=logs/datagen_pentomino_easy${SUF_PATH}.log
> "$LOG"

OUT_RAW="data/pentomino_easy_llm_policy${SUF_PATH}"
OUT_MIN="data/pentomino_easy_llm_policy_minimal${SUF_PATH}"

echo "============================================================"
echo "  Pentomino-easy LLM-policy data generation"
echo "============================================================"
echo "  Trajectories: $N_TRAJ (part='${SUFFIX:-full}', seed=$SEED)"
echo "  Board: 5x4, pieces: {L, P, W, Y}"
echo "  Output (raw):     $OUT_RAW"
echo "  Output (minimal): $OUT_MIN"
echo "============================================================"

echo
echo "=== STAGE 1: LLM-policy trajectory generation ==="
bash scripts/_run_with_env.sh $PYTHON -u -m src.data.llm_trajectory_generator \
    --env polyomino \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --num-trajectories $N_TRAJ \
    --max-steps 8 \
    --board-h 5 --board-w 4 \
    --piece-set "L,P,W,Y" \
    --output-dir "$OUT_RAW" \
    --seed $SEED \
    --temperature 0.7 \
    --val-split 0.2 \
    --device auto \
    --multi-turn --max-context-turns 4 \
    --variant polyomino_minimal \
    2>&1 | tee -a "$LOG"

echo
echo "=== STAGE 2: reformat to single-step minimal ==="
$PYTHON scripts/reformat_to_minimal.py \
    --input-dir "$OUT_RAW" \
    --output-dir "$OUT_MIN" \
    --variant polyomino_minimal \
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
echo "  Pentomino-easy data ready at:"
echo "    $OUT_MIN/wm_train_no_post_bp.parquet"
echo "  (use this for B-7 SFT training)"
echo "============================================================"
