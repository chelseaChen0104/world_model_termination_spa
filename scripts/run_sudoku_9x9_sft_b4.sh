#!/bin/bash
# Run B-4: same data as B-3, but with SPA's published SFT hyperparameters.
#
# Tests whether B-3's "no learned discrimination" failure is just under-training.
# Closes the ~80x effective-gradient gap vs SPA in one experiment.
#
# Hyperparameters changed from B-3:
#   learning_rate:           1e-5  → 1e-4   (10x larger)
#   gradient_accumulation:   8     → 4      (effective batch 32 → 16)
#   num_train_epochs:        3     → 5      (1.7x longer)
#
# Total effective gradient signal:
#   B-3:  3 epochs × 78 updates × lr 1e-5 ≈ 0.0023 magnitude
#   B-4:  5 epochs × 155 updates × lr 1e-4 ≈ 0.0775 magnitude   (~33x more)
#
# Note: SPA also reports running with batch_size=16 and lr=1e-4 for SFT, with
# 5 epochs as the headline-result variant (Section 4.4, Table 5 of SPA paper).
#
# Output: outputs/sft_sudoku_minimal_b4_spa_hparams/
# Wall time: ~12-15 min on H800

set -e
cd /root/autodl-tmp/world_model_termination_spa

> logs/sft_b4.log
echo "=== launching SFT Run B-4 (SPA hyperparameters) ==="
bash scripts/_run_with_env.sh python -u src/training/simple_sft_trainer.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --train_file data/sudoku_llm_policy_minimal/wm_train_filtered_no_post_bp.parquet \
    --val_file data/sudoku_llm_policy_minimal/wm_val_filtered_no_post_bp.parquet \
    --output_dir outputs/sft_sudoku_minimal_b4_spa_hparams \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --max_length 2048 \
    --eval_steps 25 \
    --save_steps 200 \
    2>&1 | tee logs/sft_b4.log

echo
echo "=== launching B-4 evals (greedy + logprob) ==="
> logs/eval_b4.log
bash scripts/_run_with_env.sh python -u evaluate_rl.py \
    --env sudoku --metric termination --skip-rl \
    --sft-path outputs/sft_sudoku_minimal_b4_spa_hparams/final \
    --eval-from-parquet data/sudoku_llm_policy_minimal/wm_val_filtered.parquet \
    --n-per-class 100 --sample-outputs 0 --eval-temperature 0.0 \
    2>&1 | tee logs/eval_b4.log

echo
bash scripts/_run_with_env.sh python -u evaluate_rl.py \
    --env sudoku --metric solvable-logprob --skip-rl \
    --sft-path outputs/sft_sudoku_minimal_b4_spa_hparams/final \
    --eval-from-parquet data/sudoku_llm_policy_minimal/wm_val_filtered.parquet \
    --n-per-class 100 --sample-outputs 0 \
    2>&1 | tee -a logs/eval_b4.log

echo
echo "============================================================"
echo "  Run B-4 complete. Compare AUC vs:"
echo "    B-2 (3 epochs, lr=1e-5):  AUC 0.468"
echo "    B-3 (3 epochs, lr=1e-5):  AUC 0.462"
echo "    B-4 (5 epochs, lr=1e-4):  see logs/eval_b4.log"
echo "============================================================"
