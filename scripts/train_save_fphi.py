"""SAVE f_phi viability scorer SFT trainer.

Trains a HuggingFace causal-LM to predict, given (state, proposed action):
  <next_state>...</next_state>
  <viability>true|false</viability>
  <state_viable>true|false</state_viable>

Per `doc/SAVE_handoff.md` §3. Uses standard token-level cross-entropy on
the response tokens — this implicitly combines L_trans (next_state tokens),
L_viab (the viability true/false token), and L_state (state_viable token).
The pairwise ranking loss L_rank is NOT implemented in this Phase-1
trainer; if validation AUC plateaus low we'll add it.

Input: pre-materialized SFT JSONL produced by `scripts/save_sft_prepare.py`.
Each line is `{messages, response, next_viable, state_viable, set_mixed, ...}`.

Works for all three envs (sudoku4 / pentomino5x6 / hidato5x4) — env-specific
system prompts are baked into the materialized records by the prep script.

Usage:
  python scripts/train_save_fphi.py \\
    --train data/hidato5x4/sft/train.sft.jsonl \\
    --val   data/hidato5x4/sft/val.sft.jsonl \\
    --output_dir outputs/save_fphi_hidato5x4 \\
    --model Qwen/Qwen2.5-1.5B-Instruct \\
    --epochs 3 --batch 4 --grad_accum 8 --lr 1e-5
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


def _load_sft_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    if not out:
        raise ValueError(f"no records in {path}")
    return out


class SAVESFTDataset(Dataset):
    """Loads pre-materialized {messages, response, ...} records."""

    def __init__(self, records: List[Dict], tokenizer, max_length: int):
        self.records = records
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        r = self.records[idx]
        prompt_text = self.tok.apply_chat_template(
            r["messages"], tokenize=False, add_generation_prompt=True
        )
        full_text = prompt_text + r["response"] + self.tok.eos_token

        prompt_ids = self.tok(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tok(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]

        labels = list(full_ids)
        n_mask = min(len(prompt_ids), len(labels))
        for i in range(n_mask):
            labels[i] = -100

        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": [1] * len(full_ids),
        }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True,
                   help="train .sft.jsonl from save_sft_prepare.py")
    p.add_argument("--val", required=True,
                   help="val .sft.jsonl from save_sft_prepare.py")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_steps", type=int, default=30)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--logging_steps", type=int, default=5)
    p.add_argument("--eval_steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient_checkpointing", action="store_true")
    args = p.parse_args()

    print(f"=== SAVE f_phi SFT trainer ===")
    print(f"  train: {args.train}")
    print(f"  val:   {args.val}")
    print(f"  out:   {args.output_dir}")
    print(f"  model: {args.model}")

    print("Loading SFT records...")
    train_records = _load_sft_jsonl(args.train)
    val_records = _load_sft_jsonl(args.val)
    print(f"  train: {len(train_records)} samples")
    print(f"  val:   {len(val_records)} samples")
    nv_t = sum(1 for r in train_records if r.get("next_viable"))
    nv_f = len(train_records) - nv_t
    n_mixed = sum(1 for r in train_records if r.get("set_mixed"))
    n_decep = sum(1 for r in train_records if r.get("deceptive_pair_memberships"))
    print(f"  train next_viable T/F: {nv_t}/{nv_f}")
    print(f"  train samples in mixed sets: {n_mixed}")
    print(f"  train samples in deceptive pairs: {n_decep}")

    print("Loading tokenizer + model...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    train_ds = SAVESFTDataset(train_records, tok, args.max_length)
    val_ds = SAVESFTDataset(val_records, tok, args.max_length)

    print("\n--- sample 0 (train) ---")
    r0 = train_records[0]
    print(f"  sibling_set_id: {r0.get('sibling_set_id')}, candidate_id: {r0.get('candidate_id')}")
    print(f"  next_viable={r0['next_viable']}, state_viable={r0['state_viable']}, "
          f"class={r0.get('candidate_class')}, set_mixed={r0.get('set_mixed')}")
    s0 = train_ds[0]
    masked = sum(1 for x in s0["labels"] if x == -100)
    learn = len(s0["labels"]) - masked
    print(f"  total tokens={len(s0['input_ids'])}, masked={masked}, "
          f"learning on {learn} response tokens")

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="linear",
        bf16=True,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="no",
        seed=args.seed,
        report_to="none",
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tok, padding=True, label_pad_token_id=-100
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
        data_collator=collator,
    )

    print("\n=== Starting training ===")
    trainer.train()
    print("\n=== Saving final ===")
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tok.save_pretrained(final_dir)
    print(f"saved to: {final_dir}")


if __name__ == "__main__":
    main()
