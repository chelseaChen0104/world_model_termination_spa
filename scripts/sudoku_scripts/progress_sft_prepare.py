"""Q3 baseline: Learned progress-score data prep (paper §3.4).

Same backbone, same input format, same SFT budget as SAVE — only the target
changes from viability {true,false} to a continuous progress score q(s,a).
We use compute_progress(next_state).local_progress_score (the same surface
formula used to construct deceptive pairs), so the baseline is supervised on
exactly the signal SAVE is supposed to *not* depend on.

Reads the RAW SAVE JSONL and emits one SFT sample per local_valid candidate.

Usage:
    python scripts/sudoku_scripts/progress_sft_prepare.py \\
        --input  data/sudoku4/train_balanced.jsonl \\
        --output data/sudoku4/sft_progress/train.sft.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # for save_schema, progress_sudoku4
from save_schema import SiblingSetRecord  # noqa: E402
from progress_sudoku4 import compute_progress  # noqa: E402


SYSTEM_PROMPT = (
    "You are a Sudoku 4×4 progress scorer. Given a current state and a "
    "proposed action, predict the local progress score of the resulting "
    "state. Higher progress means a more advanced board (more cells filled, "
    "fewer constraint violations). Respond with the score only."
)

USER_TEMPLATE = (
    "Current state:\n"
    "{state_text}\n"
    "\n"
    "Proposed action: {action_text}\n"
    "\n"
    "Predict the local progress score of the next state. Respond in the "
    "following format:\n"
    "<progress>NUMBER</progress>"
)

RESPONSE_TEMPLATE = "<progress>{score:.4f}</progress>"


def build_deceptive_lookup(record):
    lookup = defaultdict(list)
    for pair in record.deceptive_pairs:
        global_pid = f"{record.sibling_set_id}/{pair.pair_id}"
        lookup[pair.a_plus_candidate_id].append({"role": "a_plus", "pair_id": global_pid})
        lookup[pair.a_minus_candidate_id].append({"role": "a_minus", "pair_id": global_pid})
    return lookup


def candidate_to_sample(record, candidate, deceptive_lookup):
    if candidate.next_state is None:
        return None
    next_grid = candidate.next_state.next_state_struct.grid
    progress = compute_progress(next_grid)["local_progress_score"]
    user_msg = USER_TEMPLATE.format(
        state_text=record.state.state_text,
        action_text=candidate.action_text,
    )
    response = RESPONSE_TEMPLATE.format(score=progress)
    memberships = deceptive_lookup.get(candidate.candidate_id, [])
    return {
        "sibling_set_id": record.sibling_set_id,
        "candidate_id": candidate.candidate_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "response": response,
        "progress_target": progress,
        # Kept for downstream eval (deceptive bench needs viability label):
        "next_viable": candidate.next_state.next_viable,
        "candidate_class": candidate.candidate_class,
        "deceptive_pair_memberships": memberships,
    }


def process_file(input_path: Path, output_path: Path):
    n_records = 0
    n_emitted = 0
    n_skipped = 0
    progress_min = float("inf")
    progress_max = float("-inf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open() as fin, output_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = SiblingSetRecord.model_validate_json(line)
            n_records += 1
            lookup = build_deceptive_lookup(rec)
            for cand in rec.candidates:
                if not cand.local_valid:
                    n_skipped += 1
                    continue
                sample = candidate_to_sample(rec, cand, lookup)
                if sample is None:
                    n_skipped += 1
                    continue
                progress_min = min(progress_min, sample["progress_target"])
                progress_max = max(progress_max, sample["progress_target"])
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                n_emitted += 1
    return {
        "n_records": n_records,
        "n_emitted": n_emitted,
        "n_skipped": n_skipped,
        "progress_min": progress_min,
        "progress_max": progress_max,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    stats = process_file(args.input, args.output)
    print(f"=== {args.input} -> {args.output} ===")
    print(f"records:  {stats['n_records']}")
    print(f"emitted:  {stats['n_emitted']}")
    print(f"skipped:  {stats['n_skipped']}")
    print(f"progress range: [{stats['progress_min']:.4f}, {stats['progress_max']:.4f}]")


if __name__ == "__main__":
    main()
