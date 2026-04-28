#!/bin/bash
# Simple SFT Training Script for Termination Prediction
#
# This script uses the standalone trainer that doesn't require RAGEN/verl.
# For full distributed training with FSDP, use train_sft.sh instead.
#
# Usage:
#   bash scripts/train_sft_simple.sh [OPTIONS]
#
# Example:
#   bash scripts/train_sft_simple.sh --lora_r 32  # Train with LoRA
#   bash scripts/train_sft_simple.sh --num_train_epochs 5  # More epochs

set -e

# Activate virtual environment
source .venv/bin/activate

# Add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "============================================================"
echo "Termination Prediction - SFT Training (Simple)"
echo "============================================================"

# Default arguments
MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
TRAIN_FILE="data/termination_study_v2/wm_train.parquet"
VAL_FILE="data/termination_study_v2/wm_val.parquet"
OUTPUT_DIR="outputs/sft_termination"
MAX_LENGTH=2048
NUM_EPOCHS=3
BATCH_SIZE=4
GRAD_ACCUM=8
LEARNING_RATE=1e-5
LORA_R=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_name_or_path)
            MODEL_NAME="$2"
            shift 2
            ;;
        --train_file)
            TRAIN_FILE="$2"
            shift 2
            ;;
        --val_file)
            VAL_FILE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --max_length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --num_train_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --per_device_train_batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --gradient_accumulation_steps)
            GRAD_ACCUM="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --lora_r)
            LORA_R="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Model: $MODEL_NAME"
echo "Train file: $TRAIN_FILE"
echo "Val file: $VAL_FILE"
echo "Output dir: $OUTPUT_DIR"
echo "Max length: $MAX_LENGTH"
echo "Epochs: $NUM_EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Gradient accumulation: $GRAD_ACCUM"
echo "Learning rate: $LEARNING_RATE"
echo "LoRA rank: $LORA_R"
echo "============================================================"

# Run training
python3 src/training/simple_sft_trainer.py \
    --model_name_or_path "$MODEL_NAME" \
    --train_file "$TRAIN_FILE" \
    --val_file "$VAL_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --max_length "$MAX_LENGTH" \
    --num_train_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LEARNING_RATE" \
    --lora_r "$LORA_R" \
    --bf16 \
    --gradient_checkpointing

echo "============================================================"
echo "Training complete!"
echo "Model saved to: $OUTPUT_DIR/final"
echo "============================================================"
