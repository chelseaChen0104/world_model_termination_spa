"""SAVE f_phi SFT trainer v2 — token-weighted cross-entropy.

Same data, same Dataset, same simple format as v1 (train_save_fphi.py).
Only difference: per-token CE multiplies the loss at boolean-label positions
(`true` / `false`) by --bool-weight. This corrects the v1 imbalance where the
viability + state_viable tokens (2 tokens out of ~100 in the response) got
~1% of the gradient and the model never learned to predict viability well.

L_rank from the paper is NOT implemented here — per user directive 2026-05-04
we keep the trainer simple and only apply the token-weight fix. If AUC stays
low after this, L_rank is the next step.

Usage:
  python scripts/train_save_fphi_v2.py \\
    --train data/hidato5x4/sft/train.sft.jsonl \\
    --val   data/hidato5x4/sft/val.sft.jsonl \\
    --output_dir outputs/save_fphi_hidato5x4_v2 \\
    --model Qwen/Qwen2.5-1.5B-Instruct \\
    --epochs 3 --batch 4 --grad_accum 8 --lr 1e-5 \\
    --bool-weight 100.0
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
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


def _resolve_true_false_token_ids(tok) -> Tuple[int, int]:
    """First-token id for 'true' / 'false' as they appear right after '>'."""
    base = tok(">", add_special_tokens=False).input_ids
    after_true = tok(">true", add_special_tokens=False).input_ids
    after_false = tok(">false", add_special_tokens=False).input_ids
    if len(after_true) <= len(base) or len(after_false) <= len(base):
        # Fallback: standalone tokens
        return (tok("true", add_special_tokens=False).input_ids[0],
                tok("false", add_special_tokens=False).input_ids[0])
    return after_true[len(base)], after_false[len(base)]


class WeightedCETrainer(Trainer):
    """Trainer with token-weighted CE: positions whose label is true_id or
    false_id get their loss multiplied by `bool_weight`."""

    def __init__(self, *args, true_id: int = None, false_id: int = None,
                 bool_weight: float = 100.0, **kwargs):
        super().__init__(*args, **kwargs)
        assert true_id is not None and false_id is not None
        self._true_id = true_id
        self._false_id = false_id
        self._bool_weight = float(bool_weight)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # [B, T, V]

        # Standard next-token shift: logits[t] predicts labels[t+1].
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Per-token CE; -100 positions return 0 from CE (ignore_index).
        loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        per_tok_loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.shape)  # [B, T-1]

        # Position weights: 1.0 default, bool_weight where label ∈ {true_id, false_id}.
        bool_mask = (shift_labels == self._true_id) | (shift_labels == self._false_id)
        weights = torch.ones_like(per_tok_loss)
        weights = torch.where(bool_mask, weights * self._bool_weight, weights)

        # Mask out padding / -100 positions when normalizing.
        valid_mask = (shift_labels != -100).float()
        eff_weights = weights * valid_mask
        denom = eff_weights.sum().clamp_min(1.0)
        loss = (per_tok_loss * eff_weights).sum() / denom

        return (loss, outputs) if return_outputs else loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
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
    p.add_argument("--bool-weight", type=float, default=100.0,
                   help="Multiplier for CE loss at true/false label positions. "
                        "v1 was effectively 1.0 (uniform); v2 default is 100x to "
                        "rebalance against the long next_state token sequence.")
    args = p.parse_args()

    print(f"=== SAVE f_phi SFT trainer v2 (token-weighted CE) ===")
    print(f"  train: {args.train}")
    print(f"  val:   {args.val}")
    print(f"  out:   {args.output_dir}")
    print(f"  model: {args.model}")
    print(f"  bool_weight: {args.bool_weight}")

    print("Loading SFT records...")
    train_records = _load_sft_jsonl(args.train)
    val_records = _load_sft_jsonl(args.val)
    print(f"  train: {len(train_records)} samples")
    print(f"  val:   {len(val_records)} samples")

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

    true_id, false_id = _resolve_true_false_token_ids(tok)
    print(f"  true_id={true_id} ({tok.decode([true_id])!r}), "
          f"false_id={false_id} ({tok.decode([false_id])!r})")

    train_ds = SAVESFTDataset(train_records, tok, args.max_length)
    val_ds = SAVESFTDataset(val_records, tok, args.max_length)

    print("\n--- sample 0 (train) ---")
    r0 = train_records[0]
    print(f"  sibling_set_id: {r0.get('sibling_set_id')}, "
          f"candidate_id: {r0.get('candidate_id')}")
    print(f"  next_viable={r0['next_viable']}, "
          f"state_viable={r0['state_viable']}, "
          f"class={r0.get('candidate_class')}, "
          f"set_mixed={r0.get('set_mixed')}")
    s0 = train_ds[0]
    masked = sum(1 for x in s0["labels"] if x == -100)
    learn = len(s0["labels"]) - masked
    n_bool = sum(1 for x in s0["labels"] if x in (true_id, false_id))
    print(f"  total tokens={len(s0['input_ids'])}, masked={masked}, "
          f"learning on {learn} response tokens "
          f"({n_bool} of which are bool-weighted x{args.bool_weight})")

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

    trainer = WeightedCETrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
        data_collator=collator,
        true_id=true_id,
        false_id=false_id,
        bool_weight=args.bool_weight,
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
