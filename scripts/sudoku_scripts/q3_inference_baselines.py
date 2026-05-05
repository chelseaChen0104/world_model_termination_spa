"""Q3 deceptive-candidate benchmark — inference-only baselines for Sudoku.

Computes the "non-viable selection rate" (paper §3.4 / Table 3) for methods
that don't require additional training:

  - Policy top-1:    argmax_{a ∈ {a+,a-}} policy_eval_logprob(a)
  - Best-of-K:       same as Policy top-1 in this 2-action restriction
                     (BoK only differs from top-1 when sampling from the full
                      action space, which is Q4 territory — see Q4 rollout)
  - Local progress:  argmax_{a ∈ {a+,a-}} local_progress_score(T(s,a))
  - Oracle:          always picks a+ (non-viable rate = 0.0)
  - Local progress (sanity): should be 100.0 by construction

Reads the RAW SAVE JSONL (not the SFT-prepared one) because we need
candidate.logprobs.policy_eval_logprob and next_state.next_state_struct.grid.

Usage:
    python scripts/sudoku_scripts/q3_inference_baselines.py \\
        --eval data/sudoku4/test_natural_policy.jsonl \\
        --output_json outputs/save_sudoku4_f_phi/q3_baselines.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # for progress_sudoku4
from progress_sudoku4 import compute_progress  # noqa: E402


def select_by_score(
    a_plus_score: float, a_minus_score: float, tie_break: str = "a_plus"
) -> str:
    """Return 'a_plus' or 'a_minus' depending on which has higher score.

    Ties broken in favor of `tie_break` (default: a_plus, optimistic; alternative
    'a_minus' is pessimistic). For evaluating non-viable selection rate, the
    pessimistic tie-break is more honest (the method must actually prefer a+).
    """
    if a_plus_score > a_minus_score:
        return "a_plus"
    if a_minus_score > a_plus_score:
        return "a_minus"
    return tie_break


def evaluate_baselines(records: List[Dict], tie_break: str = "a_minus") -> Dict:
    """For each deceptive pair across all records, compute non-viable selection rate.

    Returns dict with per-method counts and rates.
    """
    pair_results = []  # (pair_global_id, top1_pick, progress_pick)
    n_pairs = 0
    n_skipped = 0

    for rec in records:
        # Index candidates by id
        cand_by_id: Dict[str, Dict] = {c["candidate_id"]: c for c in rec["candidates"]}
        for pair in rec.get("deceptive_pairs", []):
            n_pairs += 1
            a_plus = cand_by_id.get(pair["a_plus_candidate_id"])
            a_minus = cand_by_id.get(pair["a_minus_candidate_id"])
            if a_plus is None or a_minus is None:
                n_skipped += 1
                continue
            if not (a_plus.get("local_valid") and a_minus.get("local_valid")):
                n_skipped += 1
                continue

            # --- Policy top-1 score = policy_eval_logprob ---
            ap_lp = a_plus["logprobs"]["policy_eval_logprob"]
            am_lp = a_minus["logprobs"]["policy_eval_logprob"]
            top1_pick = select_by_score(ap_lp, am_lp, tie_break)

            # --- Local progress score = compute_progress(next_state_struct.grid) ---
            ap_grid = a_plus["next_state"]["next_state_struct"]["grid"]
            am_grid = a_minus["next_state"]["next_state_struct"]["grid"]
            ap_prog = compute_progress(ap_grid)["local_progress_score"]
            am_prog = compute_progress(am_grid)["local_progress_score"]
            prog_pick = select_by_score(ap_prog, am_prog, tie_break)

            pair_results.append({
                "pair_id": f"{rec['sibling_set_id']}/{pair['pair_id']}",
                "policy_top1_pick": top1_pick,
                "policy_top1_logprobs": {"a_plus": ap_lp, "a_minus": am_lp},
                "local_progress_pick": prog_pick,
                "local_progress_scores": {"a_plus": ap_prog, "a_minus": am_prog},
            })

    n_eval = len(pair_results)

    def non_viable_rate(picks: List[str]) -> float:
        if not picks: return float("nan")
        return sum(1 for p in picks if p == "a_minus") / len(picks)

    nv_top1 = non_viable_rate([r["policy_top1_pick"] for r in pair_results])
    nv_progress = non_viable_rate([r["local_progress_pick"] for r in pair_results])

    return {
        "n_deceptive_pairs": n_pairs,
        "n_evaluated": n_eval,
        "n_skipped": n_skipped,
        "tie_break": tie_break,
        "non_viable_selection_rate": {
            "policy_top1": nv_top1,
            "best_of_k": nv_top1,    # identical for {a+, a-} restriction
            "local_progress": nv_progress,
            "oracle_viability": 0.0,  # by definition
        },
        "pair_results": pair_results[:5],  # first 5 for inspection
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, type=Path,
                    help="RAW SAVE JSONL (with policy_eval_logprob + next_state_struct)")
    ap.add_argument("--tie_break", choices=["a_plus", "a_minus"], default="a_minus",
                    help="On exact tie, which side to pick (a_minus = pessimistic)")
    ap.add_argument("--output_json", type=Path, default=None)
    args = ap.parse_args()

    print(f"[load] {args.eval}")
    records = []
    with args.eval.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    print(f"  loaded {len(records)} records")

    metrics = evaluate_baselines(records, tie_break=args.tie_break)
    print("\n=== Q3 INFERENCE BASELINES ===")
    print(json.dumps(metrics, indent=2))

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(metrics, indent=2))
        print(f"\n[save] {args.output_json}")


if __name__ == "__main__":
    main()
