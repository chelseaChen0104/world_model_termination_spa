#!/bin/bash
# Phase 1 RL on B-7 Pentomino-easy SFT checkpoint.
#
# Distinct from run_rl_b5_phase1.sh (which is the Sudoku 4×4 RL track).
# Different env, different SFT checkpoint, different output dirs.
#
# B-7 is already RL-ready in the sense of the truncation gate
# (Prec(F) at τ=0.10 = 94.3%, > 0.90 gate from doc/plan_2026-04-29_rl_approach.md §2 Phase 2),
# but Phase 1 still runs without truncation first to verify reward signal direction.
#
# Default: 200 steps × 4 puzzles × 8 rollouts = 32 trajectories/step at lr=1e-5
# Wall time on H800: ~4-5 hours
#
# Usage:
#   bash scripts/run_rl_b7_phase1.sh                 # default
#   N_TOTAL_STEPS=500 LR=1e-5 bash scripts/run_rl_b7_phase1.sh   # longer

set -e
cd /root/autodl-tmp/world_model_termination_spa

N_TOTAL_STEPS=${N_TOTAL_STEPS:-200}
SFT_PATH=${SFT_PATH:-outputs/sft_pentomino_easy_b7_spa_hparams/final}
# v7 RL gets a fresh output dir to avoid overwriting the v6 (deprecated) checkpoint.
OUTPUT_DIR=${OUTPUT_DIR:-outputs/rl_b7_phase1_v7}
N_PUZZLES_PER_BATCH=${N_PUZZLES_PER_BATCH:-4}
GROUP_SIZE=${GROUP_SIZE:-8}
LR=${LR:-1e-5}
KL_COEF=${KL_COEF:-0.05}
EVAL_EVERY=${EVAL_EVERY:-25}
SEED=${SEED:-42}
# v7 reward is the default for Pentomino — fixes B-7 v6 collapse from short rollouts.
# Override with REWARD_VERSION=v6 to reproduce the prior (failed) baseline.
REWARD_VERSION=${REWARD_VERSION:-v7}
PROGRESS_BONUS=${PROGRESS_BONUS:-0.1}
CLASS_BALANCE_CAP=${CLASS_BALANCE_CAP:-5.0}

mkdir -p "$OUTPUT_DIR"
LOG=logs/rl_b7_phase1.log
> "$LOG"

echo "============================================================"
echo "  RL Phase 1 ($REWARD_VERSION reward) on B-7 Pentomino-easy SFT"
echo "============================================================"
echo "  SFT checkpoint:    $SFT_PATH"
echo "  Output dir:        $OUTPUT_DIR"
echo "  Env:               polyomino (5x4 board, {L, P, W, Y} pieces)"
echo "  Total RL steps:    $N_TOTAL_STEPS"
echo "  Puzzles/batch:     $N_PUZZLES_PER_BATCH"
echo "  Group size:        $GROUP_SIZE  (rollouts/puzzle)"
echo "  Effective batch:   $((N_PUZZLES_PER_BATCH * GROUP_SIZE)) rollouts/step"
echo "  Learning rate:     $LR"
echo "  KL coefficient:    $KL_COEF"
echo "  Reward version:    $REWARD_VERSION (v7 = symmetric + class-balanced + progress bonus)"
echo "  Progress bonus:    $PROGRESS_BONUS per valid step (v7 only)"
echo "  Class-balance cap: $CLASS_BALANCE_CAP (v7 only)"
echo "  Eval every:        $EVAL_EVERY steps (Pass@1 greedy on 30 puzzles)"
echo "============================================================"

bash scripts/_run_with_env.sh python -u src/training/rl_trainer_v6.py \
    --env polyomino \
    --sft-checkpoint "$SFT_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --n-total-steps "$N_TOTAL_STEPS" \
    --board-h 5 --board-w 4 \
    --piece-set "L,P,W,Y" \
    --n-puzzles-per-batch "$N_PUZZLES_PER_BATCH" \
    --group-size "$GROUP_SIZE" \
    --learning-rate "$LR" \
    --kl-coef "$KL_COEF" \
    --eval-every "$EVAL_EVERY" \
    --seed "$SEED" \
    --reward-version "$REWARD_VERSION" \
    --progress-bonus "$PROGRESS_BONUS" \
    --class-balance-cap "$CLASS_BALANCE_CAP" \
    2>&1 | tee "$LOG"

echo
echo "=== B-7 RL Phase 1 done ==="
echo "Logs:        $LOG"
echo "Checkpoints: $OUTPUT_DIR/"
echo "Per-step JSONL: $OUTPUT_DIR/rl_log.jsonl"
