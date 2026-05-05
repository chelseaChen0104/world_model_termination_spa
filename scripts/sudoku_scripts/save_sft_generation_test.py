"""Generation-time sanity check for trained SAVE f_phi.

Unlike save_sft_eval.py (which reads logits at fixed token positions and
never asks the model to actually generate text), this script makes the
model decode a free-form response and verifies:

  1. Schema parse rate — fraction of outputs matching the 3-tag template
  2. Transition exact-match — predicted <next_state> equals oracle T(s,a)
  3. Viability prediction — predicted <viability> matches oracle next_viable
  4. State-viable prediction — predicted <state_viable> matches oracle

Usage:
    python scripts/sudoku_scripts/save_sft_generation_test.py \\
        --checkpoint outputs/save_sudoku4_f_phi/final \\
        --eval data/sudoku4/sft/val_natural_calibration.sft.jsonl \\
        --n_samples 50
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


RESPONSE_PARSE_RE = re.compile(
    r"<next_state>\n(?P<next_state>.*?)\n</next_state>\s*"
    r"<viability>(?P<viability>true|false)</viability>\s*"
    r"<state_viable>(?P<state_viable>true|false)</state_viable>",
    re.DOTALL,
)


def parse_response(text: str) -> Optional[Dict[str, str]]:
    m = RESPONSE_PARSE_RE.search(text)
    if not m:
        return None
    return {
        "next_state": m.group("next_state"),
        "viability": m.group("viability"),
        "state_viable": m.group("state_viable"),
    }


def parse_oracle_response(response_str: str) -> Dict[str, str]:
    m = RESPONSE_PARSE_RE.search(response_str)
    if m is None:
        raise ValueError(f"Oracle response did not parse: {response_str[:200]}")
    return {
        "next_state": m.group("next_state"),
        "viability": m.group("viability"),
        "state_viable": m.group("state_viable"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--eval", required=True, type=Path)
    ap.add_argument("--n_samples", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_json", type=Path, default=None)
    args = ap.parse_args()

    print(f"[load] {args.checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Greedy decoding: pad on left so generated tokens align across batch
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, local_files_only=True,
    ).to("cuda")
    model.eval()

    print(f"[load] {args.eval}")
    samples: List[Dict] = []
    with args.eval.open() as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if args.n_samples > 0:
        # Reproducible random subset
        import random
        random.Random(args.seed).shuffle(samples)
        samples = samples[: args.n_samples]
    print(f"  evaluating on {len(samples)} samples")

    parsed_ok = 0
    next_state_match = 0
    viab_match = 0
    state_match = 0
    parse_failures: List[Tuple[str, str]] = []  # (prompt_excerpt, output)
    oracle_results: List[Dict] = []

    eos_id = tokenizer.eos_token_id
    im_end_id_seq = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    im_end_id = im_end_id_seq[0] if len(im_end_id_seq) == 1 else eos_id

    for i in range(0, len(samples), args.batch_size):
        batch = samples[i : i + args.batch_size]
        prompts = [
            tokenizer.apply_chat_template(s["messages"], tokenize=False, add_generation_prompt=True)
            for s in batch
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=im_end_id,
            )
        # Strip the prompt prefix; decode only the new tokens
        gen_tokens = out[:, enc["input_ids"].shape[1]:]
        gen_texts = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        for sample, gen in zip(batch, gen_texts):
            oracle = parse_oracle_response(sample["response"])
            parsed = parse_response(gen)
            row = {
                "sibling_set_id": sample["sibling_set_id"],
                "candidate_id": sample["candidate_id"],
                "oracle": oracle,
                "generated_raw": gen,
                "parsed": parsed,
            }
            if parsed is None:
                parse_failures.append((sample["candidate_id"], gen[:300]))
            else:
                parsed_ok += 1
                if parsed["next_state"].strip() == oracle["next_state"].strip():
                    next_state_match += 1
                if parsed["viability"] == oracle["viability"]:
                    viab_match += 1
                if parsed["state_viable"] == oracle["state_viable"]:
                    state_match += 1
            oracle_results.append(row)

    n = len(samples)
    metrics = {
        "n_samples": n,
        "schema_parse_rate": parsed_ok / n if n > 0 else float("nan"),
        "transition_exact_match": next_state_match / parsed_ok if parsed_ok > 0 else float("nan"),
        "viability_acc": viab_match / parsed_ok if parsed_ok > 0 else float("nan"),
        "state_viable_acc": state_match / parsed_ok if parsed_ok > 0 else float("nan"),
        "n_parse_failures": len(parse_failures),
    }

    print("\n=== GENERATION SANITY ===")
    print(json.dumps(metrics, indent=2))

    if parse_failures:
        print(f"\n=== Parse failures (first 3) ===")
        for cid, txt in parse_failures[:3]:
            print(f"--- {cid} ---")
            print(txt)
            print()

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(
            {"metrics": metrics, "results": oracle_results}, indent=2,
        ))
        print(f"[save] {args.output_json}")


if __name__ == "__main__":
    main()
