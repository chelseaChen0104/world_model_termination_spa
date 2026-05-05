"""Dataset, GroupedSampler, and Collator for SAVE f_phi SFT training.

Loads samples produced by scripts/save_sft_prepare.py. Each sample is one
(sibling_set_id, candidate) pair; the dataset finds the token position of
the <viability> and <state_viable> value slots so the trainer can compute
L_viab / L_rank / L_state directly from logits.

Group-aware batching: GroupedSampler picks N sibling sets and yields all
their candidates as one batch, so L_rank can be computed within-group.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, Sampler


# Tag delimiters whose end-positions we need in the response
VIAB_OPEN = "<viability>"
STATE_OPEN = "<state_viable>"


@dataclass
class TagPositions:
    """Token indices in the response for the two BCE slots."""
    viab_value_idx: int   # token index of "true"/"false" inside <viability>
    state_value_idx: int  # token index of "true"/"false" inside <state_viable>


def _find_value_token_idx(response: str, tag_open: str, offsets: List[Tuple[int, int]]) -> int:
    """Return the index in `offsets` whose token starts at the char position
    immediately after `tag_open` in `response`. Raises ValueError if not found."""
    char_pos = response.index(tag_open) + len(tag_open)
    for i, (s, _e) in enumerate(offsets):
        if s == char_pos:
            return i
    # Fallback: tokenizer may merge the open tag with the value char; find the
    # token whose span CONTAINS char_pos and whose start is closest to it.
    for i, (s, e) in enumerate(offsets):
        if s <= char_pos < e:
            return i
    raise ValueError(f"Could not locate token for tag {tag_open!r} at char {char_pos}")


class SaveSFTDataset(Dataset):
    """Reads JSONL produced by save_sft_prepare.py.

    Each __getitem__ returns a dict ready for the collator:
      input_ids        — full tokenized prompt+response (LongTensor)
      labels           — same shape, -100 on prompt, target ids on response
      attention_mask   — all 1s up to length (LongTensor)
      viab_value_pos   — int, position of viability value token in input_ids
      state_value_pos  — int, position of state_viable value token in input_ids
      next_viable      — float (1.0 / 0.0)
      state_viable     — float (1.0 / 0.0)
      sibling_set_id   — str (group key for L_rank)
      candidate_class  — str ("valid_viable" / "valid_doomed" / ...)
      set_mixed        — bool
      candidate_id     — str
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer,
        max_length: int = 512,
    ):
        self.path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: List[Dict[str, Any]] = []
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))

        # Pre-compute token IDs for "true" / "false" so the loss code can use them.
        # We tokenize them in-context to handle BPE merges; expect single-token results.
        true_ids = tokenizer.encode("true", add_special_tokens=False)
        false_ids = tokenizer.encode("false", add_special_tokens=False)
        if len(true_ids) != 1 or len(false_ids) != 1:
            raise RuntimeError(
                f"Expected single-token 'true'/'false'; got true={true_ids} false={false_ids}. "
                "Loss code assumes 1 token per value; refactor if multi-token."
            )
        self.true_token_id = true_ids[0]
        self.false_token_id = false_ids[0]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        tokenizer = self.tokenizer

        # Render prompt via chat template (no assistant turn yet)
        prompt_str = tokenizer.apply_chat_template(
            s["messages"], tokenize=False, add_generation_prompt=True
        )
        response_str = s["response"]

        # Build the full conversation including the assistant turn so the
        # tokenizer's chat template appends the model-specific end-of-turn
        # markers automatically (Qwen: <|im_end|>; LLaMA-3.x: <|eot_id|>; etc.).
        full_messages = list(s["messages"]) + [
            {"role": "assistant", "content": response_str}
        ]
        full_str = tokenizer.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

        # Tokenize the FULL string with offset mapping so we can locate tag positions
        enc = tokenizer(
            full_str,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        attention_mask = [1] * len(input_ids)

        # Char offset where prompt ends / response begins
        prompt_char_end = len(prompt_str)

        # First token whose span starts at or after prompt_char_end is the first response token
        response_start_idx = next(
            (i for i, (start, _e) in enumerate(offsets) if start >= prompt_char_end),
            len(input_ids),
        )

        # Build labels: -100 over prompt, copy of input_ids over response
        labels = [-100] * len(input_ids)
        for i in range(response_start_idx, len(input_ids)):
            labels[i] = input_ids[i]

        # Find viability / state_viable value positions (offsets are over full_str)
        # The tag opens inside the response, so search there
        # We compute their char positions in full_str:
        viab_char_pos = full_str.index(VIAB_OPEN, prompt_char_end) + len(VIAB_OPEN)
        state_char_pos = full_str.index(STATE_OPEN, prompt_char_end) + len(STATE_OPEN)

        viab_value_pos = self._char_to_token_idx(offsets, viab_char_pos)
        state_value_pos = self._char_to_token_idx(offsets, state_char_pos)

        # Sanity: the token at those positions should be true_token_id or false_token_id.
        # If the response was truncated, position may be invalid; we then drop the sample
        # by signalling -1 (collator will skip).
        in_range = lambda p: p is not None and 0 <= p < len(input_ids)
        if not (in_range(viab_value_pos) and in_range(state_value_pos)):
            viab_value_pos = -1
            state_value_pos = -1

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "viab_value_pos": viab_value_pos,
            "state_value_pos": state_value_pos,
            "next_viable": float(s["next_viable"]),
            "state_viable": float(s["state_viable"]),
            "sibling_set_id": s["sibling_set_id"],
            "candidate_class": s["candidate_class"],
            "set_mixed": bool(s["set_mixed"]),
            "candidate_id": s["candidate_id"],
        }

    @staticmethod
    def _char_to_token_idx(offsets: List[Tuple[int, int]], char_pos: int) -> Optional[int]:
        for i, (s, e) in enumerate(offsets):
            if s == char_pos:
                return i
        for i, (s, e) in enumerate(offsets):
            if s <= char_pos < e:
                return i
        return None


class GroupedSampler(Sampler[List[int]]):
    """Yields BATCHES of indices: each batch packs N sibling sets' candidates.

    A 'batch' returned by this sampler is a List[int] of variable size
    (sum of group sizes). Use with `batch_sampler=...` (NOT `sampler=...`)
    in DataLoader.
    """

    def __init__(
        self,
        dataset: SaveSFTDataset,
        sets_per_batch: int,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.dataset = dataset
        self.sets_per_batch = sets_per_batch
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Group sample indices by sibling_set_id
        groups: Dict[str, List[int]] = defaultdict(list)
        for i, s in enumerate(dataset.samples):
            groups[s["sibling_set_id"]].append(i)
        # Sorted for reproducibility
        self.group_ids: List[str] = sorted(groups.keys())
        self.groups: Dict[str, List[int]] = groups

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch) if self.shuffle else None
        order = list(self.group_ids)
        if rng is not None:
            rng.shuffle(order)
        for i in range(0, len(order), self.sets_per_batch):
            chunk = order[i : i + self.sets_per_batch]
            indices: List[int] = []
            for gid in chunk:
                indices.extend(self.groups[gid])
            yield indices
        self.epoch += 1

    def __len__(self) -> int:
        return (len(self.group_ids) + self.sets_per_batch - 1) // self.sets_per_batch


@dataclass
class SaveSFTCollator:
    """Pads variable-length input_ids / labels / attention_mask, keeps metadata."""
    pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_len = max(f["input_ids"].size(0) for f in features)
        B = len(features)

        input_ids = torch.full((B, max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((B, max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)

        viab_pos = torch.tensor([f["viab_value_pos"] for f in features], dtype=torch.long)
        state_pos = torch.tensor([f["state_value_pos"] for f in features], dtype=torch.long)
        next_viable = torch.tensor([f["next_viable"] for f in features], dtype=torch.float)
        state_viable = torch.tensor([f["state_viable"] for f in features], dtype=torch.float)
        set_mixed = torch.tensor([f["set_mixed"] for f in features], dtype=torch.bool)

        for i, f in enumerate(features):
            L = f["input_ids"].size(0)
            input_ids[i, :L] = f["input_ids"]
            labels[i, :L] = f["labels"]
            attention_mask[i, :L] = f["attention_mask"]

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "viab_value_pos": viab_pos,
            "state_value_pos": state_pos,
            "next_viable": next_viable,
            "state_viable": state_viable,
            "set_mixed": set_mixed,
            "sibling_set_ids": [f["sibling_set_id"] for f in features],
            "candidate_classes": [f["candidate_class"] for f in features],
            "candidate_ids": [f["candidate_id"] for f in features],
        }
