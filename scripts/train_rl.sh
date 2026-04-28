#!/bin/bash
# RL Training Script for Termination Prediction
#
# Usage:
#   bash scripts/train_rl.sh [NUM_GPUS] [CONFIG] [SFT_CHECKPOINT]
#
# Example:
#   bash scripts/train_rl.sh 4 rl_termination outputs/sft_termination/checkpoint-best

set -e

NUM_GPUS=${1:-1}
CONFIG=${2:-rl_termination}
SFT_CHECKPOINT=${3:-outputs/sft_termination/checkpoint-best}

echo "============================================================"
echo "Termination Prediction - RL Training"
echo "============================================================"
echo "GPUs: $NUM_GPUS"
echo "Config: $CONFIG"
echo "SFT Checkpoint: $SFT_CHECKPOINT"
echo "============================================================"

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

# Check if SFT checkpoint exists
if [ ! -d "$SFT_CHECKPOINT" ]; then
    echo "Warning: SFT checkpoint not found: $SFT_CHECKPOINT"
    echo "Please run SFT training first or specify the correct path."
    echo "Using base model instead..."
fi

# Check if config exists
CONFIG_PATH="src/training/config/${CONFIG}.yaml"
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found: $CONFIG_PATH"
    exit 1
fi

# Run training
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Running distributed RL training on $NUM_GPUS GPUs..."
    torchrun --nproc_per_node=$NUM_GPUS \
        -m src.training.rl_trainer \
        --config-path=config \
        --config-name=$CONFIG \
        model.partial_pretrain=$SFT_CHECKPOINT
else
    echo "Running single GPU RL training..."
    python -m src.training.rl_trainer \
        --config-path=config \
        --config-name=$CONFIG \
        model.partial_pretrain=$SFT_CHECKPOINT
fi

echo "============================================================"
echo "RL Training Complete!"
echo "============================================================"
