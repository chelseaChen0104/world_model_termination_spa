#!/bin/bash
# Phase 3 RL on Sudoku 4×4: continue from Run A's endpoint with v8 viability-tag KL anchor.
#
# Run A (lr=1e-5, 500 steps, v6.1 reward) lifted Pass@1 from 6.67% → 33.33% but
# `solvable_acc` drifted from 0.62 → 0.51 (calibration regression). bp_recall stayed
# at 1.000 throughout. See doc/runs_ledger_2026-04-29.md for full eval trajectory.
#
# Phase 3 hypothesis: applying v8's <solvable>-tag KL anchor (coef 0.5) on top of
# Run A's checkpoint should restore calibration to ~0.95 (matching B-5 SFT) WITHOUT
# losing the Pass@1 gains. This validates the anchor on a known-working setup,
# decoupled from Pentomino's "Pass@1 stuck at 0%" confound.
#
# Usage:
#   bash scripts/run_sudoku_4x4_rl_v8_phase3.sh              # default: 200 steps from Run A's final
#   N_TOTAL_STEPS=300 bash scripts/run_sudoku_4x4_rl_v8_phase3.sh
#
# Default: 200 steps × 4 puzzles × 8 rollouts = 32 rollouts/step at lr=1e-5
# Wall time on H800: ~3-4 hours

set -e
cd /root/autodl-tmp/world_model_termination_spa

N_TOTAL_STEPS=${N_TOTAL_STEPS:-200}
SFT_PATH=${SFT_PATH:-outputs/rl_b5_phase2_continue/final}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/rl_b5_phase3_v8_anchor}
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
LOG=logs/rl_b5_phase3_v8.log
> "$LOG"

echo "============================================================"
echo "  RL Phase 3 ($REWARD_VERSION) — Sudoku 4×4, anchor on Run A"
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
echo "  Reward version:    $REWARD_VERSION (= v7 + viability/solvable-tag KL anchor)"
echo "  Progress bonus:    $PROGRESS_BONUS per valid step"
echo "  Class-balance cap: $CLASS_BALANCE_CAP"
echo "  Viability KL coef: $VIABILITY_KL_COEF (locks <solvable> calibration)"
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
    2>&1 | tee "$LOG"

echo
echo "=== Phase 3 done ==="
echo "Logs:        $LOG"
echo "Checkpoints: $OUTPUT_DIR/"
echo "Per-step JSONL: $OUTPUT_DIR/rl_log.jsonl"
