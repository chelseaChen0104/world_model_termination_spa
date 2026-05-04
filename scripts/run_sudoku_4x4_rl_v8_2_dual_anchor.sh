#!/bin/bash
# Phase 3 v8.2 RL on Sudoku 4×4: same recipe as Phase 3 v8, but with the
# dual-token anchor enabled (`--dual-token-anchor`). Anchors logp(>true) AND
# logp(>false) at every viability position regardless of which was sampled.
#
# Why v8.2 (after min-step sweep, 2026-05-01):
#   v8 single-token anchor causes bimodal confidence — at any state the model
#   is either very confident True or very confident False. The τ-sweep
#   confirmed this (τ ∈ {0.95..0.9999} all gave identical 55%/22% truncation
#   behavior). The min-step sweep then rejected the "premature in-training
#   kill" hypothesis (per-batch solve rate flat across min_step 0..4). The
#   remaining hypotheses for the −10pp eval Pass@1 cost:
#     1. eval-time gate fires too eagerly on borderline rollouts that would
#        recover in eval (where rollouts are uncapped)
#     2. solvable_acc calibration is corrupted by drifting unsampled-token logp
#     3. τ has no fine control because the bimodal mass is anchor-induced
#   v8.2 directly addresses (2) and (3) by anchoring both tokens; (1) follows
#   if the bimodality flattens.
#
# Starting checkpoint: same as Phase 3 v8 (outputs/rl_b5_phase2_continue/final).
# Output:              outputs/rl_b5_phase3_v8_2_dual_anchor
# Wall time on H800:   similar to v8, ~3-4 hr for 200 steps.

set -e
cd /root/autodl-tmp/world_model_termination_spa

N_TOTAL_STEPS=${N_TOTAL_STEPS:-200}
SFT_PATH=${SFT_PATH:-outputs/rl_b5_phase2_continue/final}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/rl_b5_phase3_v8_2_dual_anchor}
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
LOG=logs/rl_b5_phase3_v8_2.log
> "$LOG"

echo "============================================================"
echo "  RL Phase 3 v8.2 — Sudoku 4×4, dual-token anchor on Run A"
echo "============================================================"
echo "  SFT checkpoint:    $SFT_PATH  (= Run A endpoint, Pass@1 ~33%)"
echo "  Output dir:        $OUTPUT_DIR"
echo "  Env:               sudoku 4x4 easy"
echo "  Total RL steps:    $N_TOTAL_STEPS"
echo "  Puzzles/batch:     $N_PUZZLES_PER_BATCH"
echo "  Group size:        $GROUP_SIZE  (rollouts/puzzle)"
echo "  Effective batch:   $((N_PUZZLES_PER_BATCH * GROUP_SIZE)) rollouts/step"
echo "  Learning rate:     $LR"
echo "  KL coefficient:    $KL_COEF"
echo "  Reward version:    $REWARD_VERSION + dual-token anchor"
echo "  Progress bonus:    $PROGRESS_BONUS per valid step"
echo "  Class-balance cap: $CLASS_BALANCE_CAP"
echo "  Viability KL coef: $VIABILITY_KL_COEF"
echo "  v8.2 dual anchor:  ENABLED (anchors logp(>true) AND logp(>false))"
echo "  Eval every:        $EVAL_EVERY steps (Pass@1 greedy on 30 puzzles)"
echo "============================================================"

bash scripts/_run_with_env.sh python -u src/training/rl_trainer_v6.py \
    --env sudoku \
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
    --reward-version "$REWARD_VERSION" \
    --progress-bonus "$PROGRESS_BONUS" \
    --class-balance-cap "$CLASS_BALANCE_CAP" \
    --viability-kl-coef "$VIABILITY_KL_COEF" \
    --dual-token-anchor \
    2>&1 | tee "$LOG"

echo
echo "=== Phase 3 v8.2 done ==="
echo "Logs:        $LOG"
echo "Checkpoints: $OUTPUT_DIR/"
echo "Per-step JSONL: $OUTPUT_DIR/rl_log.jsonl"
