#!/bin/bash
# B-8 RL with v8.2 (dual-token anchor): Pentomino-easy RL with both >true and
# >false logprobs anchored at every viability position, regardless of which
# was sampled.
#
# Why v8.2:
#   v8 (single-token anchor) preserves logp of the SAMPLED viability token,
#   which keeps stochastic sampling on-distribution. But greedy argmax is
#   determined by the relative order of >true vs >false logprobs at each
#   viability position; if the unsampled token's logp drifts independently,
#   greedy can flip from True→False even while the anchor metric stays at 0.
#
#   B-8 RL with v8 anchor showed exactly this: stochastic per-batch solve
#   rate 50–84%, but greedy Pass@1=0% and solvable_acc=0.0 across 3
#   consecutive evals. The action policy improved dramatically; the greedy
#   viability head collapsed.
#
# v8.2 anchors logp(>true) AND logp(>false) at every viability position
# regardless of which was sampled, preserving the SFT relative ordering of
# the two tokens by construction. Greedy argmax behavior is preserved.
#
# Cost: extra forward passes on response_ids per PPO update for the dual-token
# logits at each viability position (small — only at 1-3 positions per response).
#
# Wall time on H800: similar to v8 (~3-5 hours for 200 steps).

set -e
cd /root/autodl-tmp/world_model_termination_spa

N_TOTAL_STEPS=${N_TOTAL_STEPS:-200}
SFT_PATH=${SFT_PATH:-outputs/sft_pentomino_b8_augmented/final}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/rl_b8_v8_2_dual_anchor}
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
LOG=logs/rl_b8_v8_2.log
> "$LOG"

echo "============================================================"
echo "  B-8 RL with v8.2 (dual-token anchor) on Pentomino-easy"
echo "============================================================"
echo "  SFT checkpoint:    $SFT_PATH  (B-8, Pass@1 stochastic 22.25%)"
echo "  Output dir:        $OUTPUT_DIR"
echo "  Env:               polyomino (5x4 board, {L, P, W, Y} pieces)"
echo "  Total RL steps:    $N_TOTAL_STEPS"
echo "  Effective batch:   $((N_PUZZLES_PER_BATCH * GROUP_SIZE)) rollouts/step"
echo "  Learning rate:     $LR"
echo "  KL coefficient:    $KL_COEF"
echo "  Reward version:    $REWARD_VERSION"
echo "  Viability KL coef: $VIABILITY_KL_COEF"
echo "  v8.2 dual anchor:  ENABLED (anchors logp(>true) AND logp(>false))"
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
    --dual-token-anchor \
    2>&1 | tee "$LOG"

echo
echo "=== B-8 RL with v8.2 done ==="
echo "Logs:        $LOG"
echo "Checkpoints: $OUTPUT_DIR/"
echo "Per-step JSONL: $OUTPUT_DIR/rl_log.jsonl"
