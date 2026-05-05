"""Q3 baseline: Learned progress-score SFT (paper §3.4).

Standard HuggingFace SFT on data emitted by progress_sft_prepare.py.
No custom losses (no L_viab/L_rank/L_state) — just CE on response tokens.
Same backbone (Qwen2.5-1.5B-Instruct), same epoch budget as SAVE.

Usage:
    python scripts/sudoku_scripts/progress_sft_train.py \\
        --train data/sudoku4/sft_progress/train.sft.jsonl \\
        --val   data/sudoku4/sft_progress/val.sft.jsonl \\
        --output_dir outputs/q3_progress_score
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed,
)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class ProgressSFTDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: List[Dict] = []
        with Path(path).open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        tok = self.tokenizer
        prompt = tok.apply_chat_template(
            s["messages"], tokenize=False, add_generation_prompt=True
        )
        full = prompt + s["response"] + "<|im_end|>"
        enc = tok(
            full,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )
        ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        prompt_end = len(prompt)
        resp_start = next(
            (i for i, (start, _e) in enumerate(offsets) if start >= prompt_end),
            len(ids),
        )
        labels = [-100] * len(ids)
        for i in range(resp_start, len(ids)):
            labels[i] = ids[i]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(ids), dtype=torch.long),
        }


def make_collator(pad_id: int):
    def collate(features):
        max_len = max(f["input_ids"].size(0) for f in features)
        B = len(features)
        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
        labels = torch.full((B, max_len), -100, dtype=torch.long)
        attn = torch.zeros((B, max_len), dtype=torch.long)
        for i, f in enumerate(features):
            L = f["input_ids"].size(0)
            input_ids[i, :L] = f["input_ids"]
            labels[i, :L] = f["labels"]
            attn[i, :L] = f["attention_mask"]
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}
    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--train", required=True, type=Path)
    ap.add_argument("--val", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--num_train_epochs", type=float, default=3.0)
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--per_device_train_batch_size", type=int, default=32)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--eval_steps", type=int, default=100)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    print(f"[init] {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=torch.bfloat16, local_files_only=True,
    )

    train_ds = ProgressSFTDataset(args.train, tokenizer, args.max_length)
    val_ds = ProgressSFTDataset(args.val, tokenizer, args.max_length)
    print(f"[init] train n={len(train_ds)} | val n={len(val_ds)}")

    targs = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="no",  # only save final manually after train()
        save_only_model=True,
        bf16=True,
        report_to=["tensorboard"],
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=make_collator(tokenizer.pad_token_id),
    )

    print("[train] starting")
    trainer.train()

    print(f"[done] save final to {args.output_dir / 'final'}")
    trainer.save_model(str(args.output_dir / "final"))
    tokenizer.save_pretrained(str(args.output_dir / "final"))


if __name__ == "__main__":
    main()
