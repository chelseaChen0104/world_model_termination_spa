"""SAVE f_phi SFT training entry point.

Run on autodl2:
    cd /root/autodl-tmp/world_model_termination_spa
    /root/miniconda3/bin/python scripts/sudoku_scripts/save_sft_train.py \\
        --train data/sudoku4/sft/train_balanced.sft.jsonl \\
        --val   data/sudoku4/sft/val_natural_calibration.sft.jsonl \\
        --output_dir outputs/save_sudoku4_f_phi \\
        --num_train_epochs 3 \\
        --per_device_batch_sets 8 \\
        --learning_rate 1e-5

Loss = L_trans + λ·L_viab + η·L_rank + μ·L_state, defaults λ=η=1.0, μ=0.5.
Implements per doc/SAVE_handoff.md §3.3.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

# Avoid HF Hub network calls on machines without external internet (e.g. autodl)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Allow importing from scripts/sudoku_scripts/ and scripts/
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # for save_schema if needed

from save_sft_dataset import (  # noqa: E402
    SaveSFTDataset,
    SaveSFTCollator,
    GroupedSampler,
)


# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------


def compute_save_loss(
    logits: torch.Tensor,           # [B, T, V]
    labels: torch.Tensor,           # [B, T] (-100 mask on prompt)
    viab_value_pos: torch.Tensor,   # [B] long
    state_value_pos: torch.Tensor,  # [B] long
    next_viable: torch.Tensor,      # [B] float
    state_viable: torch.Tensor,     # [B] float
    set_mixed: torch.Tensor,        # [B] bool
    sibling_set_ids: List[str],
    candidate_classes: List[str],
    true_token_id: int,
    false_token_id: int,
    lambda_viab: float,
    eta_rank: float,
    mu_state: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute the 4-component SAVE loss. Returns (total_loss, components)."""
    B, T, V = logits.shape
    device = logits.device

    # -- L_trans: standard causal LM loss on response tokens (labels masked to -100 on prompt) --
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    l_trans = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="mean",
    )

    # -- L_viab / L_state: BCE on (logit_true - logit_false) at the value-token slot --
    # IMPORTANT: viab_value_pos points at the *value token* in input_ids. The model's
    # prediction of that token comes from logits at position viab_value_pos - 1
    # (causal LM convention: logits[t] predict input_ids[t+1]).
    pred_pos_viab = viab_value_pos - 1
    pred_pos_state = state_value_pos - 1

    valid_viab_mask = pred_pos_viab >= 0
    valid_state_mask = pred_pos_state >= 0

    # Gather logit slices [B, V] at the prediction positions
    # We must clamp to valid range; we'll mask invalid samples after
    safe_pos_viab = pred_pos_viab.clamp(min=0, max=T - 1)
    safe_pos_state = pred_pos_state.clamp(min=0, max=T - 1)
    batch_idx = torch.arange(B, device=device)
    viab_logits = logits[batch_idx, safe_pos_viab, :]    # [B, V]
    state_logits = logits[batch_idx, safe_pos_state, :]  # [B, V]

    ell_viab = viab_logits[:, true_token_id] - viab_logits[:, false_token_id]   # [B]
    ell_state = state_logits[:, true_token_id] - state_logits[:, false_token_id]

    # BCE only over valid samples
    if valid_viab_mask.any():
        l_viab = F.binary_cross_entropy_with_logits(
            ell_viab[valid_viab_mask], next_viable[valid_viab_mask], reduction="mean"
        )
    else:
        l_viab = logits.new_zeros(())

    if valid_state_mask.any():
        l_state = F.binary_cross_entropy_with_logits(
            ell_state[valid_state_mask], state_viable[valid_state_mask], reduction="mean"
        )
    else:
        l_state = logits.new_zeros(())

    # -- L_rank: pairwise margin within mixed sibling sets --
    # For each mixed group, take ALL (viable, doomed) pairs of candidates,
    # accumulate -log σ(ℓ_viab[viable] - ℓ_viab[doomed]).
    groups: Dict[str, List[int]] = defaultdict(list)
    for i, sid in enumerate(sibling_set_ids):
        groups[sid].append(i)

    rank_terms = []
    for sid, idxs in groups.items():
        if not set_mixed[idxs[0]].item():
            continue
        viable_idxs = [i for i in idxs if candidate_classes[i] == "valid_viable"]
        doomed_idxs = [i for i in idxs if candidate_classes[i] == "valid_doomed"]
        if not viable_idxs or not doomed_idxs:
            continue
        ell_v = ell_viab[viable_idxs]   # [V_v]
        ell_d = ell_viab[doomed_idxs]   # [V_d]
        # All pairs (v, d): margin = ell_v - ell_d
        diffs = ell_v[:, None] - ell_d[None, :]     # [V_v, V_d]
        # -log σ(diffs) summed; we'll average across all groups+pairs
        rank_terms.append(F.binary_cross_entropy_with_logits(
            diffs, torch.ones_like(diffs), reduction="mean"
        ))

    if rank_terms:
        l_rank = torch.stack(rank_terms).mean()
    else:
        l_rank = logits.new_zeros(())

    total = l_trans + lambda_viab * l_viab + eta_rank * l_rank + mu_state * l_state

    return total, {
        "loss": total.detach().item(),
        "L_trans": l_trans.detach().item(),
        "L_viab": l_viab.detach().item(),
        "L_rank": l_rank.detach().item(),
        "L_state": l_state.detach().item(),
        "n_rank_groups": float(len(rank_terms)),
    }


# -----------------------------------------------------------------------------
# Custom Trainer
# -----------------------------------------------------------------------------


class SaveSFTTrainer(Trainer):
    """HF Trainer subclass with the 4-component SAVE loss + grouped sampler."""

    def __init__(
        self,
        *args,
        true_token_id: int,
        false_token_id: int,
        lambda_viab: float = 1.0,
        eta_rank: float = 1.0,
        mu_state: float = 0.5,
        sets_per_batch: int = 8,
        train_grouped_dataset: Optional[SaveSFTDataset] = None,
        eval_grouped_dataset: Optional[SaveSFTDataset] = None,
        collator_fn=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.true_token_id = true_token_id
        self.false_token_id = false_token_id
        self.lambda_viab = lambda_viab
        self.eta_rank = eta_rank
        self.mu_state = mu_state
        self.sets_per_batch = sets_per_batch
        self._train_grouped = train_grouped_dataset
        self._eval_grouped = eval_grouped_dataset
        self._collator_fn = collator_fn

    def get_train_dataloader(self) -> DataLoader:
        if self._train_grouped is None:
            return super().get_train_dataloader()
        sampler = GroupedSampler(
            self._train_grouped,
            sets_per_batch=self.sets_per_batch,
            shuffle=True,
            seed=self.args.seed,
        )
        return DataLoader(
            self._train_grouped,
            batch_sampler=sampler,
            collate_fn=self._collator_fn,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        ds = eval_dataset if eval_dataset is not None else self._eval_grouped
        if ds is None:
            return super().get_eval_dataloader(eval_dataset)
        sampler = GroupedSampler(
            ds, sets_per_batch=self.sets_per_batch, shuffle=False, seed=0
        )
        return DataLoader(
            ds,
            batch_sampler=sampler,
            collate_fn=self._collator_fn,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Forward pass
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        logits = outputs.logits

        loss, components = compute_save_loss(
            logits=logits,
            labels=inputs["labels"],
            viab_value_pos=inputs["viab_value_pos"],
            state_value_pos=inputs["state_value_pos"],
            next_viable=inputs["next_viable"],
            state_viable=inputs["state_viable"],
            set_mixed=inputs["set_mixed"],
            sibling_set_ids=inputs["sibling_set_ids"],
            candidate_classes=inputs["candidate_classes"],
            true_token_id=self.true_token_id,
            false_token_id=self.false_token_id,
            lambda_viab=self.lambda_viab,
            eta_rank=self.eta_rank,
            mu_state=self.mu_state,
        )

        # Log components when training (HF's log_metrics will pick these up via callback)
        if self.state.global_step % max(1, self.args.logging_steps) == 0 and model.training:
            for k, v in components.items():
                if k == "loss":
                    continue
                self.log({f"train/{k}": v})

        return (loss, outputs) if return_outputs else loss


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--train", required=True, type=Path)
    p.add_argument("--val", required=True, type=Path)
    p.add_argument("--output_dir", required=True, type=Path)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--per_device_batch_sets", type=int, default=8,
                   help="Number of sibling sets per batch (each ~4 samples)")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--lambda_viab", type=float, default=1.0)
    p.add_argument("--eta_rank", type=float, default=1.0)
    p.add_argument("--mu_state", type=float, default=0.5)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--gradient_checkpointing", action="store_true", default=False,
                   help="Enable gradient checkpointing (saves memory, costs ~30%% throughput)")
    p.add_argument("--optim", default="adamw_torch",
                   help="Optimizer name. Use 'paged_adamw_8bit' for 7B+ to fit in memory.")
    p.add_argument("--smoke_n_train", type=int, default=0,
                   help="If >0, truncate train set to first N records (for smoke test)")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"[init] loading tokenizer + model from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        local_files_only=True,
    )

    print(f"[init] loading datasets")
    train_ds = SaveSFTDataset(args.train, tokenizer, max_length=args.max_length)
    val_ds = SaveSFTDataset(args.val, tokenizer, max_length=args.max_length)
    if args.smoke_n_train > 0:
        train_ds.samples = train_ds.samples[: args.smoke_n_train]
        print(f"[smoke] truncated train to {len(train_ds.samples)} samples")

    print(f"[init] train n={len(train_ds)} | val n={len(val_ds)}")
    print(f"[init] true_token_id={train_ds.true_token_id} false_token_id={train_ds.false_token_id}")

    collator = SaveSFTCollator(pad_token_id=tokenizer.pad_token_id)

    targs = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=1,  # dummy; sampler controls true batch
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="no",  # only save final manually after train()
        save_only_model=True,
        bf16=args.bf16,
        report_to=["tensorboard"],
        seed=args.seed,
        remove_unused_columns=False,
        gradient_checkpointing=args.gradient_checkpointing,
        optim=args.optim,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
    )

    trainer = SaveSFTTrainer(
        model=model,
        args=targs,
        tokenizer=tokenizer,
        train_dataset=train_ds,  # required by Trainer; overridden by get_train_dataloader
        eval_dataset=val_ds,
        true_token_id=train_ds.true_token_id,
        false_token_id=train_ds.false_token_id,
        lambda_viab=args.lambda_viab,
        eta_rank=args.eta_rank,
        mu_state=args.mu_state,
        sets_per_batch=args.per_device_batch_sets,
        train_grouped_dataset=train_ds,
        eval_grouped_dataset=val_ds,
        collator_fn=collator,
    )

    print("[train] starting")
    trainer.train()

    print(f"[done] saving final to {args.output_dir / 'final'}")
    trainer.save_model(str(args.output_dir / "final"))
    tokenizer.save_pretrained(str(args.output_dir / "final"))


if __name__ == "__main__":
    main()
