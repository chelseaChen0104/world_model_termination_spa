#!/bin/bash
# Run B-7: pentomino-easy SFT (5×4 board, {L, P, W, Y} pieces).
# Mirror of B-5 hparams (SPA-scale): lr=1e-4, ep=5, bs=16, max_length=1024.
#
# Data: data/pentomino_easy_llm_policy_minimal/wm_train_no_post_bp.parquet
#       (2,964 samples, ~18.8% solvable / 81.2% BP — both classes present)
#
# Output: outputs/sft_pentomino_easy_b7_spa_hparams/
# Wall time on H800: ~30-60 min (smaller dataset than B-5)

set -e
cd /root/autodl-tmp/world_model_termination_spa

> logs/sft_b7.log
echo "=== launching SFT Run B-7 (pentomino-easy + SPA hyperparameters) ==="
bash scripts/_run_with_env.sh python -u src/training/simple_sft_trainer.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --train_file data/pentomino_easy_llm_policy_minimal/wm_train_no_post_bp.parquet \
    --val_file data/pentomino_easy_llm_policy_minimal/wm_val_no_post_bp.parquet \
    --output_dir outputs/sft_pentomino_easy_b7_spa_hparams \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --max_length 1024 \
    --eval_steps 25 \
    --save_steps 100000 \
    2>&1 | tee logs/sft_b7.log

echo
echo "============================================================"
echo "  B-7 SFT complete."
echo "  Eval (greedy on <viability>) coming next; logprob eval needs a"
echo "  small evaluate_rl.py patch to read <viability> instead of <solvable>."
echo "============================================================"
