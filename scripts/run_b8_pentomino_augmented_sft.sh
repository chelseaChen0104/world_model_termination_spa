#!/bin/bash
# Run B-8: pentomino-easy SFT trained on B-7 data + solution-path augmented samples.
#
# Compared to B-7:
#   - Same hparams (lr=1e-4, ep=5, bs=16, max_length=1024)
#   - Same env (5×4 / {L, P, W, Y})
#   - Different DATA: combine the existing 2964 LLM-policy samples with 720
#     repeated solution-path samples (72 unique × 10 oversample) covering
#     trajectory positions 0-3 uniformly. Tests whether late-stage SFT samples
#     resolve the "model can't generate coherent step 2/3 responses" problem
#     that drove Pass@1=0% on B-7.
#
# Data: data/pentomino_b8_combined/wm_train_no_post_bp.parquet (~3.7K samples)
# Output: outputs/sft_pentomino_b8_augmented/
# Wall time on H800: ~40-60 min

set -e
cd /root/autodl-tmp/world_model_termination_spa

> logs/sft_b8.log
echo "=== launching SFT Run B-8 (pentomino-easy + augmented late-stage data) ==="
bash scripts/_run_with_env.sh python -u src/training/simple_sft_trainer.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --train_file data/pentomino_b8_combined/wm_train_no_post_bp.parquet \
    --val_file data/pentomino_b8_combined/wm_val_no_post_bp.parquet \
    --output_dir outputs/sft_pentomino_b8_augmented \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --max_length 1024 \
    --eval_steps 25 \
    --save_steps 100000 \
    2>&1 | tee logs/sft_b8.log

echo
echo "============================================================"
echo "  B-8 SFT complete."
echo "  Output: outputs/sft_pentomino_b8_augmented/final/"
echo "  Compare against B-7 SFT (outputs/sft_pentomino_easy_b7_spa_hparams/final)"
echo "  Key checks:"
echo "    1. AUC stays >= 0.95 (calibration preserved)"
echo "    2. Sanity test: rollout-length distribution shifts toward step 2-3"
echo "    3. Pass@1 lifts off 0% (any non-zero is informative)"
echo "============================================================"
