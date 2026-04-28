#!/bin/bash
# SFT Training Script for Termination Prediction
#
# Usage:
#   bash scripts/train_sft.sh [NUM_GPUS] [CONFIG]
#
# Example:
#   bash scripts/train_sft.sh 4 sft_termination

set -e

NUM_GPUS=${1:-1}
CONFIG=${2:-sft_termination}

echo "============================================================"
echo "Termination Prediction - SFT Training"
echo "============================================================"
echo "GPUs: $NUM_GPUS"
echo "Config: $CONFIG"
echo "============================================================"

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# Check if config exists
CONFIG_PATH="src/training/config/${CONFIG}.yaml"
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found: $CONFIG_PATH"
    exit 1
fi

# Run training
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Running distributed training on $NUM_GPUS GPUs..."
    torchrun --nproc_per_node=$NUM_GPUS \
        -m src.training.sft_trainer \
        --config-path=config \
        --config-name=$CONFIG
else
    echo "Running single GPU training..."
    python -m src.training.sft_trainer \
        --config-path=config \
        --config-name=$CONFIG
fi

echo "============================================================"
echo "Training Complete!"
echo "============================================================"
