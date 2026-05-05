"""Compile paper Table 3 row (Sudoku) from individual baseline JSONs.

Aggregates non-viable selection rate from:
  - q3_baselines.json:        Policy top-1, Best-of-K, Local progress, Oracle
  - q3_prompted_score.json:   Prompted score-only
  - q3_progress_eval.json:    Learned progress-score
  - eval_test_q2.json:        SAVE (computed from deceptive_bench acc)

Usage:
    python scripts/sudoku_scripts/compile_q3_table.py \\
        --outputs_dir outputs/save_sudoku4_f_phi \\
        --progress_eval outputs/q3_progress_score/q3_progress_eval.json \\
        --output outputs/save_sudoku4_f_phi/q3_table_sudoku.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs_dir", required=True, type=Path)
    ap.add_argument("--progress_eval", type=Path, required=False)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()

    base = load(args.outputs_dir / "q3_baselines.json")
    prompted = load(args.outputs_dir / "q3_prompted_score.json")
    save_eval = load(args.outputs_dir / "eval_test_q2.json")
    progress = load(args.progress_eval) if args.progress_eval else None

    table = {
        "environment": "Sudoku 4×4",
        "n_pairs": base["n_evaluated"] if base else None,
        "non_viable_selection_rate": {
            "policy_top1": base["non_viable_selection_rate"]["policy_top1"] if base else None,
            "best_of_k": base["non_viable_selection_rate"]["best_of_k"] if base else None,
            "local_progress": base["non_viable_selection_rate"]["local_progress"] if base else None,
            "prompted_score_only": prompted["non_viable_selection_rate"] if prompted else None,
            "learned_progress_score": progress["non_viable_selection_rate"] if progress else None,
            "save": (
                1.0 - save_eval["deceptive_bench"]["acc"]
                if save_eval and save_eval.get("deceptive_bench", {}).get("acc") is not None
                else None
            ),
            "oracle_viability": base["non_viable_selection_rate"]["oracle_viability"] if base else None,
        },
        "diagnostics": {
            "prompted_n_distinct_scores": prompted["n_distinct_scores_seen"] if prompted else None,
            "prompted_n_tie_pairs": prompted["n_tie_pairs"] if prompted else None,
            "progress_n_distinct_scores": progress["n_distinct_scores"] if progress else None,
            "progress_n_tie_pairs": progress["n_tie_pairs"] if progress else None,
            "progress_n_unparsed": progress["n_unparsed_pairs"] if progress else None,
        },
    }

    print("=== TABLE 3 ROW: Sudoku 4×4 ===")
    print(f"{'Method':<30}{'Non-viable rate':>18}")
    print("-" * 48)
    rates = table["non_viable_selection_rate"]
    rows = [
        ("Policy top-1", rates["policy_top1"]),
        ("Best-of-K (K=8)", rates["best_of_k"]),
        ("Local progress heuristic", rates["local_progress"]),
        ("Prompted score-only", rates["prompted_score_only"]),
        ("Learned progress-score", rates["learned_progress_score"]),
        ("**SAVE**", rates["save"]),
        ("Oracle viability", rates["oracle_viability"]),
    ]
    for name, rate in rows:
        if rate is None:
            print(f"{name:<30}{'TBD':>18}")
        else:
            print(f"{name:<30}{rate * 100:>17.1f}%")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(table, indent=2))
    print(f"\n[save] {args.output}")


if __name__ == "__main__":
    main()
