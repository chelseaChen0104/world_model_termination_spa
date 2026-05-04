#!/bin/bash
# Vanilla RL baseline: PPO directly from base Qwen2.5-1.5B-Instruct, no SFT.
#
# This is SPA Table 5's "Vanilla RL" row, mapped to our Sudoku 4×4-easy setup.
# The model has to bootstrap action-format compliance from format-reward alone
# (no SFT cold-start). Expected to be very weak — likely near-0% Pass@1.
# That's the *point*: it anchors the bottom of the comparison and isolates
# what SFT cold-start contributes vs PPO alone.
#
# No <solvable> reward components — only action format + solve reward.
# Reward version v6 (the older simple reward) is the closest match for a
# vanilla baseline; v7/v8 add termination-prediction reward shaping that
# would be unfair to compare here.
#
# Output:    outputs/baseline_vanilla_rl/
# Wall time on A800: ~3-4 hr for 200 steps.

set -e
cd /root/autodl-tmp/world_model_termination_spa

N_TOTAL_STEPS=${N_TOTAL_STEPS:-200}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/baseline_vanilla_rl}
N_PUZZLES_PER_BATCH=${N_PUZZLES_PER_BATCH:-4}
GROUP_SIZE=${GROUP_SIZE:-8}
LR=${LR:-1e-5}
KL_COEF=${KL_COEF:-0.05}
EVAL_EVERY=${EVAL_EVERY:-25}
SEED=${SEED:-42}

mkdir -p "$OUTPUT_DIR" logs
LOG=logs/baseline_vanilla_rl.log
: > "$LOG"

echo "============================================================" | tee -a "$LOG"
echo "  Vanilla RL baseline (SPA Table 5 row 1)" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"
echo "  Base model:       $BASE_MODEL  (NO SFT)" | tee -a "$LOG"
echo "  Output dir:       $OUTPUT_DIR" | tee -a "$LOG"
echo "  Env:              sudoku 4x4 easy" | tee -a "$LOG"
echo "  Total RL steps:   $N_TOTAL_STEPS" | tee -a "$LOG"
echo "  Effective batch:  $((N_PUZZLES_PER_BATCH * GROUP_SIZE)) rollouts/step" | tee -a "$LOG"
echo "  Reward version:   v6 (no termination-prediction reward)" | tee -a "$LOG"
echo "  Eval every:       $EVAL_EVERY steps (Pass@1 greedy on 30 puzzles)" | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

bash scripts/_run_with_env.sh /root/miniconda3/bin/python -u src/training/rl_trainer_v6.py \
    --env sudoku \
    --sft-checkpoint "$BASE_MODEL" \
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
    --reward-version v6 \
    --truncation-mode off \
    --skip-solvable-reward \
    --skip-prediction-tag \
    2>&1 | tee -a "$LOG"

echo | tee -a "$LOG"
echo "=== vanilla RL baseline done ===" | tee -a "$LOG"
echo "Logs:        $LOG" | tee -a "$LOG"
echo "Checkpoint:  $OUTPUT_DIR/final/" | tee -a "$LOG"
