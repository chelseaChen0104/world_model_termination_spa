"""Q3 baseline eval: Learned progress-score deceptive bench (paper §3.4).

Loads a trained progress g_ψ checkpoint, generates <progress>X</progress> for
each (a+, a-) candidate in the deceptive pairs of test_natural_policy.jsonl,
parses the float, and reports non-viable selection rate.

Usage:
    python scripts/sudoku_scripts/q3_progress_score_eval.py \\
        --checkpoint outputs/q3_progress_score/final \\
        --eval data/sudoku4/test_natural_policy.jsonl \\
        --output_json outputs/q3_progress_score/q3_progress_eval.json
"""
from __future__ import annotations

import argparse
import json
import os
import random as _random
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from progress_sft_prepare import SYSTEM_PROMPT, USER_TEMPLATE  # noqa: E402


PROGRESS_RE = re.compile(r"<progress>\s*(?P<v>-?\d+\.?\d*)\s*</progress>")


def parse_progress(text: str) -> Optional[float]:
    m = PROGRESS_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group("v"))
    except ValueError:
        return None


def select_by_score(ap: Optional[float], am: Optional[float], rng) -> str:
    if ap is None and am is None:
        return "both_unparsed"
    if ap is None:
        return "a_minus"
    if am is None:
        return "a_plus"
    if ap > am: return "a_plus"
    if am > ap: return "a_minus"
    return rng.choice(["a_plus", "a_minus"])


def render_messages(state_text: str, action_text: str) -> List[Dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(
            state_text=state_text, action_text=action_text
        )},
    ]


@torch.no_grad()
def generate_scores(model, tokenizer, jobs, batch_size=16, max_new_tokens=24):
    eos_id = tokenizer.eos_token_id
    im_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    eos_used = im_end[0] if len(im_end) == 1 else eos_id

    out: List[Tuple[Optional[float], str]] = []
    for i in range(0, len(jobs), batch_size):
        batch = jobs[i : i + batch_size]
        text = [
            tokenizer.apply_chat_template(j["messages"], tokenize=False, add_generation_prompt=True)
            for j in batch
        ]
        enc = tokenizer(text, return_tensors="pt", padding=True).to("cuda")
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_used,
        )
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for raw in decoded:
            out.append((parse_progress(raw), raw))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--eval", required=True, type=Path,
                    help="RAW SAVE JSONL with deceptive_pairs")
    ap.add_argument("--output_json", type=Path, default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    print(f"[load] {args.checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, local_files_only=True,
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

    jobs = []
    for rec in records:
        cand_by_id = {c["candidate_id"]: c for c in rec["candidates"]}
        st = rec["state"]["state_text"]
        for pair in rec.get("deceptive_pairs", []):
            ap_c = cand_by_id.get(pair["a_plus_candidate_id"])
            am_c = cand_by_id.get(pair["a_minus_candidate_id"])
            if ap_c is None or am_c is None: continue
            if not (ap_c.get("local_valid") and am_c.get("local_valid")): continue
            pid = f"{rec['sibling_set_id']}/{pair['pair_id']}"
            for side, cand in (("a_plus", ap_c), ("a_minus", am_c)):
                jobs.append({
                    "pair_id": pid, "side": side,
                    "messages": render_messages(st, cand["action_text"]),
                })
    print(f"  scoring {len(jobs)} prompts ({len(jobs)//2} pairs)")

    results = generate_scores(model, tokenizer, jobs, batch_size=args.batch_size)

    by_pair: Dict[str, Dict[str, Tuple[Optional[float], str]]] = defaultdict(dict)
    for j, (score, raw) in zip(jobs, results):
        by_pair[j["pair_id"]][j["side"]] = (score, raw)

    rng = _random.Random(42)
    n = 0
    n_picked_minus = 0
    n_tie = 0
    n_unparsed = 0
    distinct = set()
    samples = []
    for pid, sides in sorted(by_pair.items()):
        ap_score, ap_raw = sides.get("a_plus", (None, ""))
        am_score, am_raw = sides.get("a_minus", (None, ""))
        if ap_score is not None: distinct.add(round(ap_score, 4))
        if am_score is not None: distinct.add(round(am_score, 4))
        if ap_score is None or am_score is None: n_unparsed += 1
        if ap_score == am_score and ap_score is not None: n_tie += 1
        pick = select_by_score(ap_score, am_score, rng)
        n += 1
        if pick == "a_minus": n_picked_minus += 1
        if len(samples) < 5:
            samples.append({
                "pair_id": pid, "a_plus_score": ap_score, "a_minus_score": am_score,
                "pick": pick, "a_plus_raw": ap_raw, "a_minus_raw": am_raw,
            })

    rate = n_picked_minus / n if n > 0 else float("nan")
    out = {
        "checkpoint": str(args.checkpoint),
        "n_pairs": n,
        "n_unparsed_pairs": n_unparsed,
        "n_tie_pairs": n_tie,
        "n_distinct_scores": len(distinct),
        "non_viable_selection_rate": rate,
        "sample_outputs": samples,
    }
    print("\n=== Q3 LEARNED PROGRESS-SCORE EVAL ===")
    print(json.dumps(out, indent=2))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(out, indent=2))
        print(f"\n[save] {args.output_json}")


if __name__ == "__main__":
    main()
