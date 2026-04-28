#!/bin/bash
# Generate multi-turn SFT data using LLM policy on GPU (AutoDL)
#
# This script generates in-distribution Sudoku training data by having
# Qwen2.5-1.5B-Instruct play the game. Each trajectory becomes a multi-turn
# conversation with growing game history context.
#
# Run on a machine with CUDA GPU (e.g., AutoDL with A100/V100).
#
# Usage:
#   bash scripts/generate_llm_policy_data_gpu.sh
#
# Output:
#   data/sudoku_llm_policy/wm_train.parquet
#   data/sudoku_llm_policy/wm_val.parquet

set -e

echo "============================================="
echo "  LLM-Policy Multi-Turn SFT Data Generation"
echo "============================================="

# Check CUDA
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Generate data
python -m src.data.llm_trajectory_generator \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --num-trajectories 1280 \
    --max-steps 30 \
    --grid-size 9 \
    --difficulty easy \
    --output-dir data/sudoku_llm_policy \
    --seed 42 \
    --temperature 0.7 \
    --val-split 0.2 \
    --device auto \
    --multi-turn \
    --max-context-turns 10

echo ""
echo "============================================="
echo "  Data generation complete!"
echo "============================================="
echo ""
echo "Next step: train SFT model:"
echo "  python src/training/simple_sft_trainer.py \\"
echo "    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \\"
echo "    --train_file data/sudoku_llm_policy/wm_train.parquet \\"
echo "    --val_file data/sudoku_llm_policy/wm_val.parquet \\"
echo "    --output_dir outputs/sft_sudoku \\"
echo "    --num_train_epochs 3 \\"
echo "    --per_device_train_batch_size 4 \\"
echo "    --gradient_accumulation_steps 8 \\"
echo "    --learning_rate 1e-5 \\"
echo "    --max_length 4096"
