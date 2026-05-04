#!/bin/bash
# B-8 RL with v8 anchor: Pentomino-easy RL on top of the augmented SFT checkpoint.
#
# B-8 SFT (data = B-7 + 30× solution-path augmented samples) lifted Pass@1
# stochastic from 0% (B-7) to 22.25% — the late-stage SFT data scarcity was
# fixed. Now we test: does the same v8 anchor mechanism that worked on Sudoku
# (Pass@1 33% → 50% with calibration held) also work on Pentomino, given the
# better SFT starting point?
#
# Three possible outcomes:
#   1. Pass@1 lifts further (e.g., 22% → 35%) with calibration held →
#      full recipe works on both envs ✓
#   2. Pass@1 plateaus near SFT (~22%) →
#      SFT augmentation was sufficient; RL adds little
#   3. Calibration regresses despite anchor →
#      anchor is Sudoku-specific; need a different mechanism for Pentomino
#
# Wall time on H800: ~3-5 hours (depends on rollout length distribution under
# B-8's policy; with ~2.3 mean steps vs Sudoku's ~5, may be faster per step).

set -e
cd /root/autodl-tmp/world_model_termination_spa

N_TOTAL_STEPS=${N_TOTAL_STEPS:-200}
SFT_PATH=${SFT_PATH:-outputs/sft_pentomino_b8_augmented/final}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/rl_b8_v8_anchor}
N_PUZZLES_PER_BATCH=${N_PUZZLES_PER_BATCH:-4}
GROUP_SIZE=${GROUP_SIZE:-8}
LR=${LR:-1e-5}
KL_COEF=${KL_COEF:-0.05}
EVAL_EVERY=${EVAL_EVERY:-25}
SEED=${SEED:-42}
REWARD_VERSION=${REWARD_VERSION:-v8}
PROGRESS_BONUS=${PROGRESS_BONUS:-0.1}
CLASS_BALANCE_CAP=${CLASS_BALANCE_CAP:-5.0}
VIABILITY_KL_COEF=${VIABILITY_KL_COEF:-0.5}

mkdir -p "$OUTPUT_DIR"
LOG=logs/rl_b8_v8.log
> "$LOG"

echo "============================================================"
echo "  B-8 RL with v8 anchor on Pentomino-easy"
echo "============================================================"
echo "  SFT checkpoint:    $SFT_PATH  (B-8, Pass@1 stochastic 22.25%)"
echo "  Output dir:        $OUTPUT_DIR"
echo "  Env:               polyomino (5x4 board, {L, P, W, Y} pieces)"
echo "  Total RL steps:    $N_TOTAL_STEPS"
echo "  Puzzles/batch:     $N_PUZZLES_PER_BATCH"
echo "  Group size:        $GROUP_SIZE  (rollouts/puzzle)"
echo "  Effective batch:   $((N_PUZZLES_PER_BATCH * GROUP_SIZE)) rollouts/step"
echo "  Learning rate:     $LR"
echo "  KL coefficient:    $KL_COEF"
echo "  Reward version:    $REWARD_VERSION  (= v7 + viability-tag KL anchor)"
echo "  Progress bonus:    $PROGRESS_BONUS per valid step"
echo "  Class-balance cap: $CLASS_BALANCE_CAP"
echo "  Viability KL coef: $VIABILITY_KL_COEF"
echo "  Eval every:        $EVAL_EVERY steps (Pass@1 greedy on 30 puzzles)"
echo "============================================================"

bash scripts/_run_with_env.sh /root/miniconda3/bin/python -u src/training/rl_trainer_v6.py \
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
    --viability-kl-coef "$VIABILITY_KL_COEF" \
    2>&1 | tee "$LOG"

echo
echo "=== B-8 RL with v8 anchor done ==="
echo "Logs:        $LOG"
echo "Checkpoints: $OUTPUT_DIR/"
echo "Per-step JSONL: $OUTPUT_DIR/rl_log.jsonl"
