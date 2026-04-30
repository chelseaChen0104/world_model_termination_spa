#!/bin/bash
# Run B-5: 4x4 single-step data + SPA hyperparameters.
# Mirror of B-4 (which applied SPA hparams to 9x9 data) but on 4x4.
#
# Hypothesis: does the 80x under-training story hold for 4x4 too, or
# did 4x4 already work in the baseline because the task is easier?
#
# Hparams (vs original 4x4 run that got AUC ~?):
#   learning_rate:           1e-5  → 1e-4   (10x larger)
#   gradient_accumulation:   8     → 4      (effective batch 32 → 16)
#   num_train_epochs:        3     → 5      (1.7x longer)
#
# Output: outputs/sft_sudoku_4x4_minimal_b5_spa_hparams/
# Wall time: ~5-8 min on H800 (4x4 dataset is small)

set -e
cd /root/autodl-tmp/world_model_termination_spa

> logs/sft_b5.log
echo "=== launching SFT Run B-5 (4x4 + SPA hyperparameters) ==="
bash scripts/_run_with_env.sh python -u src/training/simple_sft_trainer.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --train_file data/sudoku_4x4_llm_policy_minimal_spa_scale/wm_train_no_post_bp.parquet \
    --val_file data/sudoku_4x4_llm_policy_minimal_spa_scale/wm_val_no_post_bp.parquet \
    --output_dir outputs/sft_sudoku_4x4_minimal_b5_spa_hparams \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --max_length 1024 \
    --eval_steps 10 \
    --save_steps 200 \
    2>&1 | tee logs/sft_b5.log

echo
echo "=== launching B-5 evals (greedy + logprob) ==="
> logs/eval_b5.log
bash scripts/_run_with_env.sh python -u evaluate_rl.py \
    --env sudoku --metric termination --skip-rl \
    --sft-path outputs/sft_sudoku_4x4_minimal_b5_spa_hparams/final \
    --eval-from-parquet data/sudoku_4x4_llm_policy_minimal_spa_scale/wm_val.parquet \
    --n-per-class 100 --sample-outputs 0 --eval-temperature 0.0 \
    2>&1 | tee logs/eval_b5.log

echo
bash scripts/_run_with_env.sh python -u evaluate_rl.py \
    --env sudoku --metric solvable-logprob --skip-rl \
    --sft-path outputs/sft_sudoku_4x4_minimal_b5_spa_hparams/final \
    --eval-from-parquet data/sudoku_4x4_llm_policy_minimal_spa_scale/wm_val.parquet \
    --n-per-class 100 --sample-outputs 0 \
    2>&1 | tee -a logs/eval_b5.log

echo
echo "============================================================"
echo "  Run B-5 complete. Compare AUC vs:"
echo "    4x4 baseline (3 ep, lr=1e-5):       see logs/eval_4x4.log"
echo "    B-5      (5 ep, lr=1e-4):           see logs/eval_b5.log"
echo "    B-4 9x9  (5 ep, lr=1e-4, ck600):    see logs/eval_b4_ck600.log"
echo "============================================================"
