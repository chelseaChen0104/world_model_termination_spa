"""Validate a SAVE-generated JSONL file against doc/data_generation_*.md
requirements. Reports per-record violations, field-consistency checks,
per-env shape checks, and distributional summaries.

Usage:
    python scripts/validate_dataset.py path/to/file.jsonl
    python scripts/validate_dataset.py path/to/file.jsonl --strict   # exit 1 on any violation

Per-env known deviations from the canonical Sudoku schema:
    - Pentomino: state_struct = {board: List[List[str]], remaining_pieces: List[str]};
                 action_struct = {piece, ori, row, col}
    - Hidato:    state_struct = {rows, cols, assignment: dict, puzzle_id};
                 action_struct = {row, col, value} but value MUST equal state.next_n
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


def _check(label: str, ok: bool, detail: str = "", warn: bool = False):
    """Print a check line; return ok."""
    icon = "  ✅" if ok else ("  ⚠️ " if warn else "  ❌")
    print(f"{icon} {label}" + (f"  — {detail}" if detail else ""))
    return ok


def validate_record_common(rec: Dict[str, Any], idx: int) -> List[str]:
    """Field-consistency checks shared across all envs. Returns list of violation strings."""
    violations = []

    # Required top-level fields
    required = ["schema", "env", "dataset_role", "split", "root_id",
                "trajectory_id", "sibling_set_id", "t", "state",
                "sampling_protocol", "candidates", "set_stats",
                "deceptive_pairs", "selection_criteria", "provenance"]
    for f in required:
        if f not in rec:
            violations.append(f"[{idx}] missing required field '{f}'")

    if rec.get("schema") != "save_sibling_set_v1.2":
        violations.append(f"[{idx}] schema field is not 'save_sibling_set_v1.2': {rec.get('schema')!r}")

    # State block
    state = rec.get("state", {})
    state_required = ["state_hash", "state_struct", "state_text",
                       "state_text_version", "state_viable", "state_is_goal",
                       "state_solver", "action_space_stats"]
    for f in state_required:
        if f not in state:
            violations.append(f"[{idx}] state missing '{f}'")

    if state.get("state_hash", "").startswith("sha1:") is False:
        violations.append(f"[{idx}] state_hash doesn't start with 'sha1:'")

    # Sampling protocol
    sp = rec.get("sampling_protocol", {})
    K_total = sp.get("K_total", 0)
    K_breakdown_sum = sum(sp.get(f"K_{k}", 0) for k in ("lt", "ht", "rand", "sol", "prt"))
    if K_breakdown_sum != K_total:
        violations.append(
            f"[{idx}] sampling_protocol K_total={K_total} but K_breakdown sum={K_breakdown_sum}"
        )

    # Candidates count vs set_stats
    candidates = rec.get("candidates", [])
    set_stats = rec.get("set_stats", {})
    if set_stats.get("num_candidates") != len(candidates):
        violations.append(
            f"[{idx}] set_stats.num_candidates={set_stats.get('num_candidates')} "
            f"!= len(candidates)={len(candidates)}"
        )

    # candidate_pool_size_after_dedup matches candidates length
    pool_after = sp.get("candidate_pool_size_after_dedup")
    if pool_after is not None and pool_after != len(candidates):
        violations.append(
            f"[{idx}] candidate_pool_size_after_dedup={pool_after} != len(candidates)={len(candidates)}"
        )

    # set_stats class counts must equal actual class counts
    class_counts = Counter(c.get("candidate_class") for c in candidates)
    expected_class_counts = {
        "num_parse_invalid": class_counts.get("parse_invalid", 0),
        "num_local_invalid": class_counts.get("local_invalid", 0),
        "num_valid_viable": class_counts.get("valid_viable", 0),
        "num_valid_doomed": class_counts.get("valid_doomed", 0),
        "num_goal_reaching": class_counts.get("goal_reaching", 0),
    }
    for k, expected in expected_class_counts.items():
        if set_stats.get(k) != expected:
            violations.append(
                f"[{idx}] set_stats.{k}={set_stats.get(k)} != actual={expected}"
            )

    # mixed flag
    n_v = expected_class_counts["num_valid_viable"]
    n_d = expected_class_counts["num_valid_doomed"]
    expected_mixed = (n_v > 0 and n_d > 0)
    if set_stats.get("mixed") != expected_mixed:
        violations.append(
            f"[{idx}] set_stats.mixed={set_stats.get('mixed')} but expected={expected_mixed} "
            f"(viable={n_v}, doomed={n_d})"
        )

    # Each candidate
    for ci, c in enumerate(candidates):
        c_required = ["candidate_id", "action_hash", "display_rank", "source",
                       "source_meta", "action_text", "action_text_canonical",
                       "action_struct", "logprobs", "parse_valid", "local_valid",
                       "transition_valid", "candidate_class", "eligible_for_viability_eval",
                       "next_state", "solver", "progress", "score_labels"]
        for f in c_required:
            if f not in c:
                violations.append(f"[{idx}.{ci}] candidate missing '{f}'")

        # Logprobs.policy_eval_logprob must always be a finite float
        lp = c.get("logprobs", {})
        peval = lp.get("policy_eval_logprob")
        if peval is None:
            violations.append(f"[{idx}.{ci}] policy_eval_logprob is None (must be finite float)")
        elif isinstance(peval, float) and (peval != peval or peval == float("inf") or peval == float("-inf")):
            violations.append(f"[{idx}.{ci}] policy_eval_logprob is not finite: {peval}")

        # source ∈ allowed set
        src = c.get("source")
        if src not in {"lt", "ht", "rand", "sol", "prt"}:
            violations.append(f"[{idx}.{ci}] source={src!r} not in allowed set")

        # candidate_class consistency with parse/local
        cls = c.get("candidate_class")
        parse_v = c.get("parse_valid")
        local_v = c.get("local_valid")
        if cls == "parse_invalid" and parse_v:
            violations.append(f"[{idx}.{ci}] class=parse_invalid but parse_valid=true")
        if cls == "local_invalid" and (not parse_v or local_v):
            violations.append(f"[{idx}.{ci}] class=local_invalid but parse_valid={parse_v} local_valid={local_v}")
        if cls in ("valid_viable", "valid_doomed", "goal_reaching") and not local_v:
            violations.append(f"[{idx}.{ci}] class={cls} but local_valid=false")

        # If valid (not parse/local invalid), next_state and solver must be populated
        if cls in ("valid_viable", "valid_doomed", "goal_reaching"):
            if c.get("next_state") is None:
                violations.append(f"[{idx}.{ci}] class={cls} but next_state is None")
            if c.get("solver") is None:
                violations.append(f"[{idx}.{ci}] class={cls} but solver is None")
            if c.get("progress") is None:
                violations.append(f"[{idx}.{ci}] class={cls} but progress is None")

        # source_meta.is_oracle_injected
        sm = c.get("source_meta", {})
        if src in ("sol", "prt") and not sm.get("is_oracle_injected"):
            violations.append(f"[{idx}.{ci}] source={src} but is_oracle_injected=false")
        if src in ("lt", "ht", "rand") and sm.get("is_oracle_injected"):
            violations.append(f"[{idx}.{ci}] source={src} but is_oracle_injected=true")

    # Deceptive pairs constraint
    cands_by_id = {c.get("candidate_id"): c for c in candidates}
    for dpi, dp in enumerate(rec.get("deceptive_pairs", [])):
        plus_id = dp.get("a_plus_candidate_id")
        minus_id = dp.get("a_minus_candidate_id")
        plus = cands_by_id.get(plus_id)
        minus = cands_by_id.get(minus_id)
        if plus is None or minus is None:
            violations.append(f"[{idx}.deceptive[{dpi}]] dangling candidate id reference")
            continue
        if plus.get("candidate_class") != "valid_viable":
            violations.append(f"[{idx}.deceptive[{dpi}]] a_plus class != valid_viable: {plus.get('candidate_class')}")
        if minus.get("candidate_class") != "valid_doomed":
            violations.append(f"[{idx}.deceptive[{dpi}]] a_minus class != valid_doomed: {minus.get('candidate_class')}")
        plus_prog = plus.get("progress", {}).get("local_progress_score")
        minus_prog = minus.get("progress", {}).get("local_progress_score")
        if plus_prog is not None and minus_prog is not None and minus_prog < plus_prog:
            violations.append(
                f"[{idx}.deceptive[{dpi}]] a_minus.progress < a_plus.progress (violates deceptive condition)"
            )

    return violations


def validate_record_env_specific(rec: Dict[str, Any], idx: int) -> List[str]:
    """Per-env shape and constraint checks."""
    violations = []
    env = rec.get("env")
    state = rec.get("state", {})
    state_struct = state.get("state_struct", {})
    state_text = state.get("state_text", "")

    # Doom-suffix leak guard (Pentomino/Hidato shouldn't have it; Sudoku doesn't render it)
    leak_phrases = ["— board now unsolvable", "board now unsolvable"]
    for leak in leak_phrases:
        if leak in state_text:
            violations.append(f"[{idx}] state_text contains doom-suffix leak: {leak!r}")
        for ci, c in enumerate(rec.get("candidates", [])):
            ns = c.get("next_state") or {}
            if leak in (ns.get("next_state_text") or ""):
                violations.append(f"[{idx}.{ci}] next_state_text contains doom-suffix leak")

    if env == "sudoku4":
        grid = state_struct.get("grid")
        if not isinstance(grid, list) or len(grid) != 4 or any(len(row) != 4 for row in grid):
            violations.append(f"[{idx}] sudoku state_struct.grid is not 4x4: {grid!r}")
        for r, row in enumerate(grid or []):
            for c, val in enumerate(row):
                if not isinstance(val, int) or val < 0 or val > 4:
                    violations.append(f"[{idx}] sudoku grid[{r}][{c}]={val} not in [0,4]")
        # Action shape
        for ci, cand in enumerate(rec.get("candidates", [])):
            asn = cand.get("action_struct")
            if asn is None: continue
            if not all(k in asn for k in ("row", "col", "value")):
                violations.append(f"[{idx}.{ci}] sudoku action_struct missing row/col/value: {asn!r}")
            if asn and not (1 <= asn.get("value", 0) <= 4):
                violations.append(f"[{idx}.{ci}] sudoku action.value={asn['value']} not in [1,4]")

    elif env == "pentomino5x4":
        board = state_struct.get("board")
        rem = state_struct.get("remaining_pieces")
        if not isinstance(board, list) or len(board) != 5 or any(len(row) != 4 for row in board):
            violations.append(f"[{idx}] pentomino state_struct.board is not 5x4")
        if not isinstance(rem, list):
            violations.append(f"[{idx}] pentomino state_struct.remaining_pieces is not list: {rem!r}")
        # Cells must be '.' or piece letter
        for r, row in enumerate(board or []):
            for c, val in enumerate(row):
                if not isinstance(val, str) or not (val == "." or val.isalpha()):
                    violations.append(f"[{idx}] pentomino board[{r}][{c}]={val!r} not '.' or letter")
        # Action shape
        for ci, cand in enumerate(rec.get("candidates", [])):
            asn = cand.get("action_struct")
            if asn is None: continue
            for f in ("piece", "ori", "row", "col"):
                if f not in asn:
                    violations.append(f"[{idx}.{ci}] pentomino action_struct missing '{f}'")

    elif env == "hidato5x4":
        for f in ("rows", "cols", "assignment"):
            if f not in state_struct:
                violations.append(f"[{idx}] hidato state_struct missing '{f}'")
        # Extract next_n from state_text — should be the value placed by all
        # candidates with local_valid=true.
        m = re.search(r"Next number to place:\s*(\d+)", state_text)
        if m:
            expected_n = int(m.group(1))
            for ci, cand in enumerate(rec.get("candidates", [])):
                if not cand.get("local_valid"):
                    continue
                asn = cand.get("action_struct") or {}
                actual_v = asn.get("value")
                if actual_v != expected_n:
                    violations.append(
                        f"[{idx}.{ci}] hidato local_valid candidate has value={actual_v} but next_n={expected_n}"
                    )
        # Action shape
        for ci, cand in enumerate(rec.get("candidates", [])):
            asn = cand.get("action_struct")
            if asn is None: continue
            for f in ("row", "col", "value"):
                if f not in asn:
                    violations.append(f"[{idx}.{ci}] hidato action_struct missing '{f}'")

    return violations


def validate_role_specific(rec: Dict[str, Any], idx: int) -> List[str]:
    """Role-specific: val/test must have no sol/prt candidates."""
    violations = []
    role = rec.get("dataset_role", "")
    if role in ("val_natural_calibration", "test_natural_policy"):
        for ci, c in enumerate(rec.get("candidates", [])):
            if c.get("source") in ("sol", "prt"):
                violations.append(
                    f"[{idx}.{ci}] role={role} but source={c.get('source')} (oracle leak)"
                )
            if (c.get("source_meta") or {}).get("is_oracle_injected"):
                violations.append(
                    f"[{idx}.{ci}] role={role} but is_oracle_injected=true"
                )
    return violations


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute distributional stats."""
    if not records:
        return {}
    n = len(records)
    env = records[0].get("env", "unknown")

    # Per-record summary
    n_mixed = sum(1 for r in records if r.get("set_stats", {}).get("mixed"))
    n_with_deceptive = sum(1 for r in records if r.get("deceptive_pairs"))
    total_deceptive = sum(len(r.get("deceptive_pairs", [])) for r in records)
    total_candidates = sum(len(r.get("candidates", [])) for r in records)
    total_pool_before = sum(r.get("sampling_protocol", {}).get("candidate_pool_size_before_dedup", 0) for r in records)
    total_pool_after = sum(r.get("sampling_protocol", {}).get("candidate_pool_size_after_dedup", 0) for r in records)

    # Per-candidate distributions
    class_counter: Counter = Counter()
    source_counter: Counter = Counter()
    next_viable_counter: Counter = Counter()
    state_hashes: set = set()

    for r in records:
        state_hashes.add(r.get("state", {}).get("state_hash", ""))
        for c in r.get("candidates", []):
            class_counter[c.get("candidate_class")] += 1
            source_counter[c.get("source")] += 1
            ns = c.get("next_state") or {}
            if "next_viable" in ns:
                next_viable_counter[ns["next_viable"]] += 1

    return {
        "env": env,
        "n_records": n,
        "n_mixed": n_mixed,
        "mixed_rate": n_mixed / n,
        "n_records_with_deceptive_pairs": n_with_deceptive,
        "total_deceptive_pairs": total_deceptive,
        "total_candidates": total_candidates,
        "avg_candidates_per_record": total_candidates / n,
        "total_pool_before_dedup": total_pool_before,
        "total_pool_after_dedup": total_pool_after,
        "dedup_loss_rate": (total_pool_before - total_pool_after) / max(1, total_pool_before),
        "candidate_class_distribution": dict(class_counter),
        "source_distribution": dict(source_counter),
        "next_viable_distribution": dict(next_viable_counter),
        "n_unique_states": len(state_hashes),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--strict", action="store_true",
                   help="Exit 1 if any violations found")
    p.add_argument("--max-violations-shown", type=int, default=20)
    args = p.parse_args()

    if not os.path.isfile(args.path):
        print(f"ERROR: file not found: {args.path}", file=sys.stderr)
        sys.exit(2)

    print(f"=== Validating: {args.path} ===")

    records = []
    with open(args.path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  ❌ JSON decode error: {e}")
                sys.exit(2)

    print(f"Loaded {len(records)} records")
    if not records:
        print("  ⚠️  empty file")
        sys.exit(0 if not args.strict else 1)

    print(f"\n--- Field consistency + per-env shape checks ---")
    all_violations = []
    for i, rec in enumerate(records):
        all_violations.extend(validate_record_common(rec, i))
        all_violations.extend(validate_record_env_specific(rec, i))
        all_violations.extend(validate_role_specific(rec, i))

    n_v = len(all_violations)
    if n_v == 0:
        _check("0 violations across all records", True)
    else:
        _check(f"{n_v} violations across {len(records)} records", False)
        for v in all_violations[:args.max_violations_shown]:
            print(f"     {v}")
        if n_v > args.max_violations_shown:
            print(f"     ... and {n_v - args.max_violations_shown} more")

    print(f"\n--- Distributional summary ---")
    summary = summarize(records)
    for k, v in summary.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")

    # Soft thresholds for paper-relevance
    print(f"\n--- Paper-relevance thresholds ---")
    role = records[0].get("dataset_role", "")
    if role == "train_balanced":
        _check(f"train_balanced mixed_rate >= 0.50 (Hidato) or 0.60 (Sudoku/Pentomino)",
               summary["mixed_rate"] >= 0.50,
               f"got {summary['mixed_rate']:.2%}", warn=True)
        # Spec §7.5: ≥100 deceptive pairs in train (relevant for full 1500, not smoke)
        if summary["total_deceptive_pairs"] < 5:
            _check("at least a few deceptive pairs in smoke", False, warn=True,
                   detail=f"got {summary['total_deceptive_pairs']}; ≥100 needed at full scale")
        else:
            _check("at least a few deceptive pairs in smoke", True,
                   detail=f"got {summary['total_deceptive_pairs']}")

    if args.strict and n_v > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
