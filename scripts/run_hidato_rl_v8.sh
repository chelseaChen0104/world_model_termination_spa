#!/bin/bash
# B-H1 RL: PPO with v8 viability-tag KL anchor on the Hidato SFT checkpoint.
#
# Mirrors run_sudoku_4x4_rl_v8_phase3.sh (Sudoku Phase 3 v8 anchor) but for Hidato.
# Same hparams: lr=1e-5, kl=0.05, group=8, batch=4, v8 reward, viability KL coef 0.5.
#
# Starting point: outputs/sft_hidato_b_h1/final (Pass@1=??%, AUC=1.000 logprob discrimination).
# Output:         outputs/rl_b_h1_v8_anchor/
# Wall time:      ~3-4 hr on A800 for 200 RL steps.

set -e
cd /root/autodl-tmp/world_model_termination_spa

N_TOTAL_STEPS=${N_TOTAL_STEPS:-200}
SFT_PATH=${SFT_PATH:-outputs/sft_hidato_b_h1/final}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/rl_b_h1_v8_anchor}
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

mkdir -p "$OUTPUT_DIR" logs
LOG=logs/rl_b_h1_v8.log
: > "$LOG"

echo "============================================================" | tee -a "$LOG"
echo "  B-H1 RL with v8 viability-tag KL anchor (Hidato)" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo "  SFT checkpoint:    $SFT_PATH" | tee -a "$LOG"
echo "  Output dir:        $OUTPUT_DIR" | tee -a "$LOG"
echo "  Env:               hidato (puzzle bank: 8 hand-curated puzzles)" | tee -a "$LOG"
echo "  Total RL steps:    $N_TOTAL_STEPS" | tee -a "$LOG"
echo "  Effective batch:   $((N_PUZZLES_PER_BATCH * GROUP_SIZE)) rollouts/step" | tee -a "$LOG"
echo "  Learning rate:     $LR" | tee -a "$LOG"
echo "  KL coef:           $KL_COEF" | tee -a "$LOG"
echo "  Reward version:    $REWARD_VERSION (viability-tag KL anchor)" | tee -a "$LOG"
echo "  Viability KL coef: $VIABILITY_KL_COEF" | tee -a "$LOG"
echo "  Progress bonus:    $PROGRESS_BONUS per valid step" | tee -a "$LOG"
echo "  Eval every:        $EVAL_EVERY steps (Pass@1 greedy)" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

bash scripts/_run_with_env.sh /root/miniconda3/bin/python -u src/training/rl_trainer_v6.py \
    --env hidato \
    --sft-checkpoint "$SFT_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --n-total-steps "$N_TOTAL_STEPS" \
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
    --truncation-mode off \
    --prepend-current-state \
    --single-turn-eval \
    --max-response-tokens 512 \
    2>&1 | tee -a "$LOG"

echo | tee -a "$LOG"
echo "=== B-H1 RL complete ===" | tee -a "$LOG"
echo "Logs:        $LOG" | tee -a "$LOG"
echo "Checkpoint:  $OUTPUT_DIR/final/" | tee -a "$LOG"
