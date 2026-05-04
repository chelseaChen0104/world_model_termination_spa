#!/bin/bash
# Generate B-9 pentomino LLM-policy data (5×10 / 10-piece variant).
#
# Piece set locked by P-0 sweep (2026-05-01): {F, I, L, N, P, T, U, V, Y, Z}
# (excluded W and X). 4664 distinct tilings — 233× more variety than 5×4's 20.
# Targets the trajectory-length distribution that drove the B-7 RL collapse.
#
# Target: ~3,000 trajectories. With 5×10 board × 10 pieces, rollouts can go up
# to 10 steps; longer trajectories give richer per-step samples.
# Wall time on H800: ~6-8 hours (rollouts ≤ 10 steps × ~1.5s each ≈ 15s/traj
# × 3000 ÷ acceptance rate ≈ 6-8h depending on doom rate).
#
# Output:
#   data/pentomino_b9_llm_policy/                       (raw multi-turn parquets)
#   data/pentomino_b9_llm_policy_minimal/               (single-step minimal format)
#   data/pentomino_b9_llm_policy_minimal/wm_train_no_post_bp.parquet ← B-9 trains on this

set -e
REPO=/root/autodl-tmp/world_model_termination_spa
PYTHON=${PYTHON:-/root/miniconda3/bin/python}
N_TRAJ=${N_TRAJ:-3000}
SUFFIX=${SUFFIX:-}
SEED=${SEED:-42}
PIECE_SET=${PIECE_SET:-F,I,L,N,P,T,U,V,Y,Z}

SUF_PATH=""
if [ -n "$SUFFIX" ]; then SUF_PATH="_$SUFFIX"; fi

cd "$REPO"

LOG=logs/datagen_pentomino_b9${SUF_PATH}.log
> "$LOG"

OUT_RAW="data/pentomino_b9_llm_policy${SUF_PATH}"
OUT_MIN="data/pentomino_b9_llm_policy_minimal${SUF_PATH}"

echo "============================================================"
echo "  B-9 Pentomino LLM-policy data generation (5×10 / 10-piece)"
echo "============================================================"
echo "  Trajectories: $N_TRAJ (part='${SUFFIX:-full}', seed=$SEED)"
echo "  Board: 5x10, pieces: {$PIECE_SET}"
echo "  Output (raw):     $OUT_RAW"
echo "  Output (minimal): $OUT_MIN"
echo "============================================================"

echo
echo "=== STAGE 1: LLM-policy trajectory generation ==="
bash scripts/_run_with_env.sh $PYTHON -u -m src.data.llm_trajectory_generator \
    --env polyomino \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --num-trajectories $N_TRAJ \
    --max-steps 20 \
    --board-h 5 --board-w 10 \
    --piece-set "$PIECE_SET" \
    --output-dir "$OUT_RAW" \
    --seed $SEED \
    --temperature 0.7 \
    --val-split 0.2 \
    --device auto \
    --multi-turn --max-context-turns 6 \
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
echo "  B-9 data ready at:"
echo "    $OUT_MIN/wm_train_no_post_bp.parquet"
echo "  (use this for B-9 SFT training)"
echo "============================================================"
