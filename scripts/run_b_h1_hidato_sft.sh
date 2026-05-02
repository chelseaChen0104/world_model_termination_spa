#!/bin/bash
# Run B-H1: Hidato (Numbrix variant) SFT, the first Hidato model.
# Same SPA-scale hparams as B-5/B-7/B-8: lr=1e-4, ep=5, bs=16, max_length=1024.
#
# Data: data/hidato_b_h1_combined/wm_train_no_post_bp.parquet
#   = 9,627 LLM-policy samples + 2,400 (80 × 30) solution-path augmented
#   = 12,027 combined samples (80% solvable / 20% doom)
#
# Output: outputs/sft_hidato_b_h1/
# Wall time on H800: ~1.5-2 hours.

set -e
cd /root/autodl-tmp/world_model_termination_spa

> logs/sft_b_h1.log
echo "=== launching SFT B-H1 (Hidato, augmented) ==="
bash scripts/_run_with_env.sh python -u src/training/simple_sft_trainer.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --train_file data/hidato_b_h1_combined/wm_train_no_post_bp.parquet \
    --val_file data/hidato_b_h1_combined/wm_val_no_post_bp.parquet \
    --output_dir outputs/sft_hidato_b_h1 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --max_length 1024 \
    --eval_steps 25 \
    --save_steps 100000 \
    2>&1 | tee logs/sft_b_h1.log

echo
echo "============================================================"
echo "  B-H1 SFT complete."
echo "  Output: outputs/sft_hidato_b_h1/final/"
echo "  Next steps:"
echo "    1. logprob threshold sweep (evaluate_rl.py --tag-name solvable)"
echo "    2. sanity rollout test (Pass@1 stochastic on the SFT)"
echo "    3. RL with v8 anchor (run_rl_b_h1_v8.sh)"
echo "============================================================"
