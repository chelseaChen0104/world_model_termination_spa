"""
Simple SFT Trainer for Termination Prediction Study

A standalone trainer that doesn't require RAGEN/verl infrastructure.
Uses HuggingFace Trainer with custom data collation.

This trainer follows the SPA paper's loss function:
- Train on all tokens inside <think>...</think> and <answer>...</answer>
- Mask input prompt tokens (standard LLM SFT behavior)
"""

import os
import json
import torch
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import argparse


@dataclass
class SFTConfig:
    """Configuration for SFT training."""
    # Model
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # Data
    train_file: str = "data/termination_study_v2/wm_train.parquet"
    val_file: str = "data/termination_study_v2/wm_val.parquet"
    max_length: int = 2048

    # Training
    output_dir: str = "outputs/sft_termination"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"

    # LoRA (set lora_r > 0 to enable)
    lora_r: int = 0
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    save_total_limit: int = 3

    # Hardware
    bf16: bool = True
    gradient_checkpointing: bool = True


def load_and_process_data(file_path: str, tokenizer, max_length: int) -> Dataset:
    """Load parquet data and tokenize for SFT.

    The data format has:
    - prompt: list of messages (single-turn or multi-turn)
      Single-turn: [{"role": "system", ...}, {"role": "user", ...}]
      Multi-turn:  [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...},
                     {"role": "user", ...}, {"role": "assistant", ...}, ..., {"role": "user", ...}]
    - response: assistant response for the current (last) turn

    We concatenate prompt + response and create labels where:
    - All prompt tokens (including prior assistant turns) have label = -100 (ignored in loss)
    - Only the final response tokens have their actual token ids as labels
    """
    df = pd.read_parquet(file_path)

    processed_data = []
    for idx, row in df.iterrows():
        # Parse prompt (it's a list of message dicts)
        prompt_messages = row['prompt']
        if isinstance(prompt_messages, str):
            prompt_messages = json.loads(prompt_messages)
        # Handle numpy array from parquet
        if hasattr(prompt_messages, 'tolist'):
            prompt_messages = prompt_messages.tolist()

        # Build the full prompt text (supports system, user, and assistant roles)
        prompt_text = ""
        for msg in prompt_messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                prompt_text += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == 'user':
                prompt_text += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == 'assistant':
                # Prior assistant turns are part of the prompt context (multi-turn).
                # They get label = -100 since only the final response is trained on.
                prompt_text += f"<|im_start|>assistant\n{content}<|im_end|>\n"

        # Add assistant prefix for the current (final) turn
        prompt_text += "<|im_start|>assistant\n"

        # Response
        response_text = row['response'] + "<|im_end|>"

        # Full text
        full_text = prompt_text + response_text

        # Tokenize
        prompt_tokens = tokenizer(prompt_text, add_special_tokens=False)
        full_tokens = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_length)

        # Create labels: -100 for prompt, actual ids for response
        prompt_len = len(prompt_tokens['input_ids'])
        labels = [-100] * prompt_len + full_tokens['input_ids'][prompt_len:]

        # Ensure same length
        if len(labels) < len(full_tokens['input_ids']):
            labels = labels + [-100] * (len(full_tokens['input_ids']) - len(labels))
        elif len(labels) > len(full_tokens['input_ids']):
            labels = labels[:len(full_tokens['input_ids'])]

        processed_data.append({
            'input_ids': full_tokens['input_ids'],
            'attention_mask': full_tokens['attention_mask'],
            'labels': labels,
        })

    return Dataset.from_list(processed_data)


class SFTDataCollator:
    """Data collator for SFT that pads sequences and creates proper labels."""

    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Find max length in batch
        max_len = min(max(len(f['input_ids']) for f in features), self.max_length)

        batch = {
            'input_ids': [],
            'attention_mask': [],
            'labels': [],
        }

        for f in features:
            # Truncate if needed
            input_ids = f['input_ids'][:max_len]
            attention_mask = f['attention_mask'][:max_len]
            labels = f['labels'][:max_len]

            # Pad
            pad_len = max_len - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            labels = labels + [-100] * pad_len

            batch['input_ids'].append(input_ids)
            batch['attention_mask'].append(attention_mask)
            batch['labels'].append(labels)

        return {
            'input_ids': torch.tensor(batch['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(batch['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(batch['labels'], dtype=torch.long),
        }


def main():
    parser = argparse.ArgumentParser(description="Simple SFT Trainer for Termination Prediction")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--train_file", type=str, default="data/termination_study_v2/wm_train.parquet")
    parser.add_argument("--val_file", type=str, default="data/termination_study_v2/wm_val.parquet")
    parser.add_argument("--output_dir", type=str, default="outputs/sft_termination")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--lora_r", type=int, default=0, help="LoRA rank (0 to disable)")
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=500)
    args = parser.parse_args()

    print("=" * 60)
    print("Termination Prediction SFT Training")
    print("=" * 60)
    print(f"Model: {args.model_name_or_path}")
    print(f"Train file: {args.train_file}")
    print(f"Val file: {args.val_file}")
    print(f"Output dir: {args.output_dir}")
    print("=" * 60)

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
        use_bf16 = args.bf16
        attn_impl = "sdpa"
    elif torch.backends.mps.is_available():
        device = "mps"
        use_bf16 = False  # MPS doesn't fully support bf16
        attn_impl = "eager"  # Flash attention not available on MPS
    else:
        device = "cpu"
        use_bf16 = False
        attn_impl = "eager"

    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16 if device == "mps" else (torch.bfloat16 if use_bf16 else torch.float32),
        trust_remote_code=True,
        attn_implementation=attn_impl if device == "cuda" else None,
    )

    # Apply LoRA if enabled
    if args.lora_r > 0:
        print(f"Applying LoRA with r={args.lora_r}, alpha={args.lora_alpha}")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Load data
    print("Loading training data...")
    train_dataset = load_and_process_data(args.train_file, tokenizer, args.max_length)
    print(f"Train samples: {len(train_dataset)}")

    print("Loading validation data...")
    val_dataset = load_and_process_data(args.val_file, tokenizer, args.max_length)
    print(f"Val samples: {len(val_dataset)}")

    # Data collator
    data_collator = SFTDataCollator(tokenizer, max_length=args.max_length)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=use_bf16 and device == "cuda",
        fp16=device == "mps",  # Use fp16 for MPS
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0 if device == "mps" else 4,  # MPS has issues with multiprocessing
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(os.path.join(args.output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))

    print("\nTraining complete!")
    print(f"Model saved to: {os.path.join(args.output_dir, 'final')}")


if __name__ == "__main__":
    main()
