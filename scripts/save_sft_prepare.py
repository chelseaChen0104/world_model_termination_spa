"""Convert SAVE sibling-set JSONL into per-candidate f_phi SFT samples.

Per handoff doc/SAVE_handoff.md §3:
  - one record (sibling set) -> K SFT samples (one per local_valid candidate)
  - skip candidates with local_valid=false (covers parse_invalid + local_invalid)
  - emit messages (system + user) and a target response string with three tags
  - preserve set_mixed flag (gates L_rank later) and deceptive_pair role (eval)

Env detection: reads the `env` field from the first record. System prompts
are env-keyed. Ports the Sudoku-only original (autodl2) to be env-agnostic
so Hidato + Pentomino can use the same pipeline.

Usage:
    python scripts/save_sft_prepare.py \\
        --input  data/hidato5x4/train_balanced.jsonl \\
        --output data/hidato5x4/sft/train.sft.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# NOTE: this script does NOT use save_schema.py's Pydantic models because
# the schema there is currently spec-strict for Sudoku state_struct +
# progress.features (per user directive 2026-05-04). This script needs to
# work for Hidato + Pentomino too. Field-shape validation lives in
# scripts/validate_dataset.py — run that separately on the input file
# before/after prep if you want structural validation.


SYSTEM_PROMPTS = {
    "sudoku4": (
        "You are a viability scorer for Sudoku 4×4. Given a current state "
        "and a proposed action, predict the next state, whether the next state "
        "remains recoverable (viable), and whether the current state itself is viable."
    ),
    "pentomino5x6": (
        "You are a viability scorer for Pentomino tiling on a 5×6 board. Given a "
        "current state and a proposed action (a piece placement), predict the next "
        "state, whether the resulting board can still be tiled with the remaining "
        "pieces (viable), and whether the current state itself is viable."
    ),
    "hidato5x4": (
        "You are a viability scorer for Hidato (number-path) puzzles on a 5×4 grid. "
        "Given a current state and a proposed action (a number placement), predict "
        "the next state, whether the resulting puzzle still has a valid completion "
        "(viable), and whether the current state itself is viable."
    ),
}


USER_TEMPLATE = (
    "Current state:\n"
    "{state_text}\n"
    "\n"
    "Proposed action: {action_text}\n"
    "\n"
    "Predict the next state, whether the next state is viable, and whether "
    "the current state is viable. Respond in the following format:\n"
    "<next_state>...</next_state>\n"
    "<viability>true|false</viability>\n"
    "<state_viable>true|false</state_viable>"
)

RESPONSE_TEMPLATE = (
    "<next_state>\n"
    "{next_state_text}\n"
    "</next_state>\n"
    "<viability>{viability}</viability>\n"
    "<state_viable>{state_viable}</state_viable>"
)

RESPONSE_PARSE_RE = re.compile(
    r"^<next_state>\n(?P<next_state>.*?)\n</next_state>\n"
    r"<viability>(?P<viability>true|false)</viability>\n"
    r"<state_viable>(?P<state_viable>true|false)</state_viable>$",
    re.DOTALL,
)


def render_user(state_text: str, action_text: str) -> str:
    return USER_TEMPLATE.format(state_text=state_text, action_text=action_text)


def render_response(next_state_text: str, next_viable: bool, state_viable: bool) -> str:
    return RESPONSE_TEMPLATE.format(
        next_state_text=next_state_text,
        viability="true" if next_viable else "false",
        state_viable="true" if state_viable else "false",
    )


def build_deceptive_lookup(record: dict) -> dict:
    """candidate_id -> List[{role, pair_id}] for deceptive_pairs.

    A candidate can belong to multiple deceptive pairs (e.g. one viable
    candidate paired against several doomed siblings), so we keep a list
    rather than a single (role, pair_id) tuple. pair.pair_id is only unique
    within a record (e.g. 'pair_000'), so we prepend sibling_set_id.
    """
    from collections import defaultdict
    lookup = defaultdict(list)
    sib_id = record["sibling_set_id"]
    for pair in record.get("deceptive_pairs", []) or []:
        global_pid = f"{sib_id}/{pair['pair_id']}"
        lookup[pair["a_plus_candidate_id"]].append(
            {"role": "a_plus", "pair_id": global_pid})
        lookup[pair["a_minus_candidate_id"]].append(
            {"role": "a_minus", "pair_id": global_pid})
    return lookup


def candidate_to_sample(record: dict, candidate: dict,
                         deceptive_lookup: dict, system_prompt: str):
    ns = candidate.get("next_state")
    if ns is None:
        return None
    next_state_text = ns.get("next_state_text")
    next_viable = ns.get("next_viable")
    if next_state_text is None or next_viable is None:
        return None
    state_text = record["state"]["state_text"]
    state_viable = record["state"]["state_viable"]
    user_msg = render_user(state_text, candidate["action_text"])
    response = render_response(
        next_state_text=next_state_text,
        next_viable=next_viable,
        state_viable=state_viable,
    )
    memberships = deceptive_lookup.get(candidate["candidate_id"], [])
    return {
        "sibling_set_id": record["sibling_set_id"],
        "candidate_id": candidate["candidate_id"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        "response": response,
        "next_viable": next_viable,
        "state_viable": state_viable,
        "candidate_class": candidate.get("candidate_class"),
        "set_mixed": record.get("set_stats", {}).get("mixed", False),
        "deceptive_pair_memberships": memberships,
    }


def _peek_env(input_path: Path) -> str:
    """Read first non-empty record's env field."""
    with input_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line).get("env")
    raise ValueError(f"no records in {input_path}")


def process_file(input_path: Path, output_path: Path) -> dict:
    env = _peek_env(input_path)
    if env not in SYSTEM_PROMPTS:
        raise ValueError(
            f"unknown env {env!r} from {input_path}; known: {list(SYSTEM_PROMPTS)}")
    system_prompt = SYSTEM_PROMPTS[env]

    n_records = 0
    n_mixed = 0
    n_deceptive_pairs = 0
    n_skipped = 0
    n_emitted = 0
    n_response_parse_fail = 0
    class_counts: Counter = Counter()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open() as fin, output_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_records += 1
            if rec.get("set_stats", {}).get("mixed"):
                n_mixed += 1
            n_deceptive_pairs += len(rec.get("deceptive_pairs", []) or [])
            deceptive_lookup = build_deceptive_lookup(rec)

            for cand in rec.get("candidates", []) or []:
                if not cand.get("local_valid"):
                    n_skipped += 1
                    continue
                sample = candidate_to_sample(rec, cand, deceptive_lookup, system_prompt)
                if sample is None:
                    n_skipped += 1
                    continue
                if not RESPONSE_PARSE_RE.match(sample["response"]):
                    n_response_parse_fail += 1
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                n_emitted += 1
                class_counts[cand.get("candidate_class")] += 1

    return {
        "input": str(input_path),
        "output": str(output_path),
        "env": env,
        "n_records": n_records,
        "n_mixed_records": n_mixed,
        "n_deceptive_pairs": n_deceptive_pairs,
        "n_skipped_invalid": n_skipped,
        "n_emitted_samples": n_emitted,
        "n_response_parse_fail": n_response_parse_fail,
        "class_counts": dict(class_counts),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    stats = process_file(args.input, args.output)

    print(f"=== {stats['input']} -> {stats['output']} ===")
    print(f"env:                 {stats['env']}")
    print(f"records read:        {stats['n_records']}")
    print(f"  mixed sibling sets: {stats['n_mixed_records']}"
          f" ({100 * stats['n_mixed_records'] / max(stats['n_records'], 1):.1f}%)")
    print(f"  deceptive pairs:    {stats['n_deceptive_pairs']}")
    print(f"samples emitted:     {stats['n_emitted_samples']}")
    print(f"skipped invalid:     {stats['n_skipped_invalid']}")
    print(f"response parse fail: {stats['n_response_parse_fail']}"
          f" {'(GOOD)' if stats['n_response_parse_fail'] == 0 else '(FIX REQUIRED)'}")
    print("class counts:")
    for cls, cnt in sorted(stats["class_counts"].items()):
        print(f"  {cls:20s} {cnt}")


if __name__ == "__main__":
    main()
