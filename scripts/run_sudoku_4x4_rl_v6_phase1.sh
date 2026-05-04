#!/bin/bash
# Phase 1 RL on B-5 SFT checkpoint (per doc/plan_2026-04-29_rl_approach.md).
#
# v6 reward (multi-step rollouts, per-step <solvable> + format + end-of-trajectory success bonus).
# truncation_mode='off' for Phase 1; gate to Phase 2 = Prec(F) ≥ 0.90 at some τ.
#
# Smoke-test run: 50 steps, n_puzzles_per_batch=4, group_size=8 → 32 rollouts/step
# Wall time on H800: ~30-60 min (rough — first run will calibrate)
#
# Full Phase 1 run (after smoke test): N_TOTAL_STEPS=200 → ~6-12h
#
# Usage:
#   bash scripts/run_sudoku_4x4_rl_v6_phase1.sh                # smoke test (50 steps)
#   N_TOTAL_STEPS=200 bash scripts/run_sudoku_4x4_rl_v6_phase1.sh   # full Phase 1

set -e
cd /root/autodl-tmp/world_model_termination_spa

N_TOTAL_STEPS=${N_TOTAL_STEPS:-50}
SFT_PATH=${SFT_PATH:-outputs/sft_sudoku_4x4_minimal_b5_spa_hparams/final}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/rl_b5_phase1}
N_PUZZLES_PER_BATCH=${N_PUZZLES_PER_BATCH:-4}
GROUP_SIZE=${GROUP_SIZE:-8}
LR=${LR:-1e-6}
KL_COEF=${KL_COEF:-0.05}
EVAL_EVERY=${EVAL_EVERY:-25}
SEED=${SEED:-42}

mkdir -p "$OUTPUT_DIR"
LOG=logs/rl_b5_phase1.log
> "$LOG"

echo "============================================================"
echo "  RL Phase 1 (v6 reward) on B-5 4×4 SFT"
echo "============================================================"
echo "  SFT checkpoint:    $SFT_PATH"
echo "  Output dir:        $OUTPUT_DIR"
echo "  Total RL steps:    $N_TOTAL_STEPS"
echo "  Puzzles/batch:     $N_PUZZLES_PER_BATCH"
echo "  Group size:        $GROUP_SIZE  (rollouts/puzzle)"
echo "  Effective batch:   $((N_PUZZLES_PER_BATCH * GROUP_SIZE)) rollouts/step"
echo "  Learning rate:     $LR"
echo "  KL coefficient:    $KL_COEF"
echo "  Eval every:        $EVAL_EVERY steps (Pass@1 greedy on 30 puzzles)"
echo "============================================================"

bash scripts/_run_with_env.sh python -u src/training/rl_trainer_v6.py \
    --sft-checkpoint "$SFT_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --n-total-steps "$N_TOTAL_STEPS" \
    --grid-size 4 \
    --difficulty easy \
    --n-puzzles-per-batch "$N_PUZZLES_PER_BATCH" \
    --group-size "$GROUP_SIZE" \
    --learning-rate "$LR" \
    --kl-coef "$KL_COEF" \
    --eval-every "$EVAL_EVERY" \
    --seed "$SEED" \
    2>&1 | tee "$LOG"

echo
echo "=== Phase 1 done ==="
echo "Logs:        $LOG"
echo "Checkpoints: $OUTPUT_DIR/"
echo "Per-step JSONL: $OUTPUT_DIR/rl_log.jsonl"
