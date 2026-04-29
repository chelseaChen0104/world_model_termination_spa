#!/bin/bash
# 4x4 Sudoku end-to-end pipeline — replicates SPA paper's exact setup
# (4x4 grid, "easy" = 40% × 16 = 6 empty cells, matching SPA's "6 empty cells").
#
# Purpose: sanity check whether our recipe (single-step SFT + minimal tags +
# action-conditional <solvable>) can produce learned discrimination on the
# task SPA proved works. If yes → 9x9 is too hard for 1.5B, scale data/use RL.
# If no → recipe itself is broken regardless of difficulty.
#
# Pipeline stages, all sequential:
#   1. Generate LLM-policy data (1000 trajectories, 4x4 easy)
#   2. Reformat to single-step minimal (sudoku_minimal variant)
#   3. Filter post-BP (class balance for fair comparison with B-3)
#   4. Train SFT (3 epochs)
#   5. Eval (termination + threshold-based logprob sweep)
#
# Total wall time on H800: ~30-45 min (4x4 trajectories are very short).
#
# Outputs:
#   data/sudoku_4x4_llm_policy/                  raw trajectories (multi-turn parquet)
#   data/sudoku_4x4_llm_policy_minimal/          single-step minimal format
#   outputs/sft_sudoku_4x4_minimal_no_post_bp/   trained model
#   logs/sft_4x4.log, logs/eval_4x4.log
#
# Usage: bash scripts/run_4x4_pipeline.sh

set -e

REPO=/root/autodl-tmp/world_model_termination_spa
PYTHON=${PYTHON:-/root/miniconda3/bin/python}
N_TRAJ=${N_TRAJ:-1000}

cd "$REPO"

echo "============================================================"
echo "  4x4 Sudoku End-to-End Pipeline (SPA-paper replication)"
echo "============================================================"
echo "  Trajectories: $N_TRAJ"
echo "  Grid: 4x4, difficulty=easy (~6 empty cells, matches SPA)"
echo "============================================================"

echo
echo "=== STAGE 1: data gen ==="
> logs/4x4_datagen.log
$PYTHON -u -m src.data.llm_trajectory_generator \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --num-trajectories $N_TRAJ \
    --max-steps 12 \
    --grid-size 4 \
    --difficulty easy \
    --output-dir data/sudoku_4x4_llm_policy \
    --seed 42 \
    --temperature 0.7 \
    --val-split 0.2 \
    --device auto \
    --multi-turn --max-context-turns 10 \
    2>&1 | tee logs/4x4_datagen.log

echo
echo "=== STAGE 2: reformat to single-step minimal ==="
$PYTHON scripts/reformat_to_minimal.py \
    --input-dir data/sudoku_4x4_llm_policy \
    --output-dir data/sudoku_4x4_llm_policy_minimal

echo
echo "=== STAGE 3: filter post-BP (class balance) ==="
$PYTHON scripts/filter_post_bp.py \
    --input-dir data/sudoku_4x4_llm_policy_minimal

echo
echo "=== STAGE 4: SFT training ==="
> logs/sft_4x4.log
$PYTHON -u src/training/simple_sft_trainer.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --train_file data/sudoku_4x4_llm_policy_minimal/wm_train_no_post_bp.parquet \
    --val_file data/sudoku_4x4_llm_policy_minimal/wm_val_no_post_bp.parquet \
    --output_dir outputs/sft_sudoku_4x4_minimal_no_post_bp \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --max_length 1024 \
    --eval_steps 50 \
    --save_steps 100 \
    2>&1 | tee logs/sft_4x4.log

echo
echo "=== STAGE 5: eval (termination + logprob) ==="
> logs/eval_4x4.log
$PYTHON -u evaluate_rl.py \
    --env sudoku --metric termination --skip-rl \
    --sft-path outputs/sft_sudoku_4x4_minimal_no_post_bp/final \
    --eval-from-parquet data/sudoku_4x4_llm_policy_minimal/wm_val.parquet \
    --n-per-class 100 \
    --sample-outputs 0 \
    2>&1 | tee logs/eval_4x4.log

echo
$PYTHON -u evaluate_rl.py \
    --env sudoku --metric solvable-logprob --skip-rl \
    --sft-path outputs/sft_sudoku_4x4_minimal_no_post_bp/final \
    --eval-from-parquet data/sudoku_4x4_llm_policy_minimal/wm_val.parquet \
    --n-per-class 100 \
    --sample-outputs 0 \
    2>&1 | tee -a logs/eval_4x4.log

echo
echo "============================================================"
echo "  4x4 PIPELINE COMPLETE"
echo "============================================================"
echo "  Compare AUC vs 9x9 results:"
echo "    9x9 B-2:  AUC 0.468 (no discrimination)"
echo "    9x9 B-3:  AUC 0.462 (no discrimination)"
echo "    4x4:      see logs/eval_4x4.log"
echo "============================================================"
