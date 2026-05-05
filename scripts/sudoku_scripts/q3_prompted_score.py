"""Q3 Prompted score-only baseline (paper §3.4 / Table 3).

Zero-shot scoring with the base Qwen2.5-1.5B-Instruct model (NOT the
SFT'd f_phi, NOT the RL'd policy π_θ — the unmodified pretrained model).
Given (state, action), the base model is prompted to output a single
scalar utility ∈ [0, 1]. For each deceptive pair, we score a+ and a-,
then pick the higher-scoring one. Non-viable selection rate is reported
on the same test pairs as Policy top-1 / Local progress / SAVE.

Usage:
    python scripts/sudoku_scripts/q3_prompted_score.py \\
        --base_model Qwen/Qwen2.5-1.5B-Instruct \\
        --eval data/sudoku4/test_natural_policy.jsonl \\
        --output_json outputs/save_sudoku4_f_phi/q3_prompted_score.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# Float in [0, 1] with optional leading 0
SCORE_RE = re.compile(r"\b(0(?:\.\d+)?|1(?:\.0+)?|\.\d+)\b")


SYSTEM_PROMPT = (
    "You are evaluating Sudoku 4×4 actions. Each row, column, and 2×2 box "
    "must contain digits 1–4 exactly once. Given a current state and a "
    "proposed placement, output a probability between 0 and 1 that this "
    "specific action leads to eventually solving the puzzle. Higher = "
    "more likely to succeed. Respond with ONLY a numeric probability."
)


def render_user_message(state_text: str, action_text: str) -> str:
    return (
        f"Current state:\n{state_text}\n\n"
        f"Proposed action: {action_text}\n\n"
        f"Probability of solving:"
    )


def parse_score(text: str) -> Optional[float]:
    """Find first numeric token in [0, 1]. Returns None if no parseable score."""
    m = SCORE_RE.search(text)
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    if 0.0 <= v <= 1.0:
        return v
    return None


def select_by_score(ap: Optional[float], am: Optional[float], rng=None) -> str:
    """Pick a_plus / a_minus / tie. Tie => random (paper convention).

    rng can be a random.Random instance for reproducibility.
    """
    import random as _r
    rng = rng or _r
    if ap is None and am is None:
        return "both_unparsed"
    if ap is None:
        return "a_minus"
    if am is None:
        return "a_plus"
    if ap > am: return "a_plus"
    if am > ap: return "a_minus"
    return rng.choice(["a_plus", "a_minus"])


def score_candidates(
    model, tokenizer, prompts: List[List[Dict]], batch_size: int = 8,
    max_new_tokens: int = 16,
) -> List[Tuple[Optional[float], str]]:
    """Score a list of (system, user) prompts; returns (parsed_score, raw_output) per prompt."""
    out_pairs = []
    eos_id = tokenizer.eos_token_id
    im_end_id_seq = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    im_end_id = im_end_id_seq[0] if len(im_end_id_seq) == 1 else eos_id

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        text = [
            tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            for p in batch
        ]
        enc = tokenizer(text, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=im_end_id,
            )
        gen_tokens = out[:, enc["input_ids"].shape[1]:]
        gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        for raw in gen_text:
            out_pairs.append((parse_score(raw), raw))
    return out_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--eval", required=True, type=Path,
                    help="RAW SAVE JSONL (with deceptive_pairs)")
    ap.add_argument("--output_json", type=Path, default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    print(f"[load] base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, local_files_only=True,
    ).to("cuda")
    model.eval()

    print(f"[load] {args.eval}")
    records = []
    with args.eval.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    print(f"  records: {len(records)}")

    # Build (pair_id, side, prompt) tuples
    job: List[Dict] = []
    for rec in records:
        cand_by_id = {c["candidate_id"]: c for c in rec["candidates"]}
        state_text = rec["state"]["state_text"]
        for pair in rec.get("deceptive_pairs", []):
            ap_c = cand_by_id.get(pair["a_plus_candidate_id"])
            am_c = cand_by_id.get(pair["a_minus_candidate_id"])
            if ap_c is None or am_c is None:
                continue
            if not (ap_c.get("local_valid") and am_c.get("local_valid")):
                continue
            pid = f"{rec['sibling_set_id']}/{pair['pair_id']}"
            for side, cand in (("a_plus", ap_c), ("a_minus", am_c)):
                msgs = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": render_user_message(state_text, cand["action_text"])},
                ]
                job.append({"pair_id": pid, "side": side, "messages": msgs})

    print(f"  scoring {len(job)} (pair, side) prompts ({len(job)//2} pairs)")
    results = score_candidates(model, tokenizer, [j["messages"] for j in job], batch_size=args.batch_size)

    # Aggregate per-pair
    by_pair: Dict[str, Dict[str, Tuple[Optional[float], str]]] = defaultdict(dict)
    for j, (score, raw) in zip(job, results):
        by_pair[j["pair_id"]][j["side"]] = (score, raw)

    # Compute pick + non-viable rate
    import random as _random
    rng = _random.Random(42)
    n = 0
    n_picked_minus = 0
    n_both_unparsed = 0
    n_one_unparsed = 0
    n_tie = 0
    n_distinct_scores = set()
    sample_outputs = []
    for pid, sides in sorted(by_pair.items()):  # sorted for determinism
        ap_score, ap_raw = sides.get("a_plus", (None, ""))
        am_score, am_raw = sides.get("a_minus", (None, ""))
        if ap_score is not None: n_distinct_scores.add(round(ap_score, 4))
        if am_score is not None: n_distinct_scores.add(round(am_score, 4))
        if ap_score == am_score and ap_score is not None: n_tie += 1
        pick = select_by_score(ap_score, am_score, rng=rng)
        n += 1
        if pick == "both_unparsed":
            n_both_unparsed += 1
        elif ap_score is None or am_score is None:
            n_one_unparsed += 1
        if pick == "a_minus":
            n_picked_minus += 1
        if len(sample_outputs) < 5:
            sample_outputs.append({
                "pair_id": pid,
                "a_plus_score": ap_score,
                "a_minus_score": am_score,
                "pick": pick,
                "a_plus_raw": ap_raw,
                "a_minus_raw": am_raw,
            })

    rate = n_picked_minus / n if n > 0 else float("nan")

    out = {
        "base_model": args.base_model,
        "n_pairs": n,
        "n_both_unparsed": n_both_unparsed,
        "n_one_unparsed": n_one_unparsed,
        "n_tie_pairs": n_tie,
        "n_distinct_scores_seen": len(n_distinct_scores),
        "non_viable_selection_rate": rate,
        "sample_outputs": sample_outputs,
    }
    print("\n=== Q3 PROMPTED SCORE-ONLY ===")
    print(json.dumps(out, indent=2))

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(out, indent=2))
        print(f"\n[save] {args.output_json}")


if __name__ == "__main__":
    main()
