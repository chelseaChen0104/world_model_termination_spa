"""Kakuro utilities — sum-subsets lookup, run extraction, constraint
propagation, solvability checking.

A Kakuro puzzle is an R×C grid where each cell is either:
  - Black with optional clues (right-clue = sum for the run starting at the
    cell to the right; down-clue = sum for the run starting at the cell below)
  - White (must be filled with a digit 1-9)

A "run" is a maximal sequence of horizontally or vertically consecutive white
cells. Each run is bounded on both sides by black cells (or the grid edge),
and has a clue from the black cell that "starts" the run (left-of-row or
above-of-column).

A puzzle is solvable iff there's an assignment of digits 1-9 to all white
cells such that:
  1. Every run sums to its clue
  2. Every run has all distinct digits

We use a partial-assignment representation where each white cell is either
empty (None) or has a fixed digit. Solvability check = "can the partial
assignment be extended to a full solution?"
"""
from __future__ import annotations
from typing import Optional
from itertools import combinations
from functools import lru_cache


# ----------------------------------------------------------------------------
# Sum-subsets lookup
# ----------------------------------------------------------------------------

@lru_cache(maxsize=None)
def sum_subsets(target: int, length: int) -> tuple:
    """All subsets of {1..9} of given length that sum to target.

    Returned as a tuple of frozensets (hashable, immutable).
    Example: sum_subsets(6, 2) -> ({1,5}, {2,4})  (note: {3,3} excluded; distinct)
    """
    out = []
    for combo in combinations(range(1, 10), length):
        if sum(combo) == target:
            out.append(frozenset(combo))
    return tuple(out)


def union_sum_subsets(target: int, length: int) -> frozenset:
    """Union of digits that appear in ANY valid sum-subset for (target, length).
    A digit not in this union cannot appear anywhere in such a run."""
    subsets = sum_subsets(target, length)
    if not subsets:
        return frozenset()
    return frozenset().union(*subsets)


# ----------------------------------------------------------------------------
# Run extraction from a grid layout
# ----------------------------------------------------------------------------

def extract_runs(rows: int, cols: int, cells: dict) -> list:
    """Extract all runs from a Kakuro grid.

    Args:
        rows, cols: grid dimensions
        cells: dict mapping (r, c) -> cell description, where cell desc is one of:
            {"type": "black", "right_clue": int|None, "down_clue": int|None}
            {"type": "white"}

    Returns:
        list of run dicts: {"direction": "row"|"col", "cells": [(r,c), ...],
                            "clue": int, "length": int}
    """
    runs = []

    # Row runs: scan each row for sequences of white cells, bounded by black/edge.
    for r in range(rows):
        c = 0
        while c < cols:
            cell = cells.get((r, c))
            if cell and cell["type"] == "black" and cell.get("right_clue") is not None:
                # Run starts at (r, c+1)
                run_start = c + 1
                cc = run_start
                while cc < cols and cells.get((r, cc)) and cells[(r, cc)]["type"] == "white":
                    cc += 1
                run_cells = [(r, cc2) for cc2 in range(run_start, cc)]
                if run_cells:
                    runs.append({
                        "direction": "row",
                        "cells": run_cells,
                        "clue": cell["right_clue"],
                        "length": len(run_cells),
                    })
                c = cc
            else:
                c += 1

    # Column runs
    for c in range(cols):
        r = 0
        while r < rows:
            cell = cells.get((r, c))
            if cell and cell["type"] == "black" and cell.get("down_clue") is not None:
                run_start = r + 1
                rr = run_start
                while rr < rows and cells.get((rr, c)) and cells[(rr, c)]["type"] == "white":
                    rr += 1
                run_cells = [(rr2, c) for rr2 in range(run_start, rr)]
                if run_cells:
                    runs.append({
                        "direction": "col",
                        "cells": run_cells,
                        "clue": cell["down_clue"],
                        "length": len(run_cells),
                    })
                r = rr
            else:
                r += 1

    return runs


# ----------------------------------------------------------------------------
# Constraint propagation
# ----------------------------------------------------------------------------

def candidate_digits(white_cells: list, runs: list, assignment: dict) -> dict:
    """For each white cell, compute the set of digits that COULD be placed
    consistent with the partial `assignment` and the run constraints.

    A digit d is a candidate for cell (r, c) iff:
      - d is not already used by another cell in any run containing (r, c)
        that has assignment
      - d appears in some valid sum-subset for each run containing (r, c)
        that's compatible with the partial assignment of that run

    Returns: dict mapping (r, c) -> frozenset of candidate digits.
    Empty frozenset = cell has no valid digit → puzzle is unsolvable.
    """
    # Index runs by cell
    cell_to_runs = {cell: [] for cell in white_cells}
    for run in runs:
        for cell in run["cells"]:
            cell_to_runs[cell].append(run)

    candidates = {}
    for cell in white_cells:
        if cell in assignment:
            candidates[cell] = frozenset({assignment[cell]})
            continue
        # Start with all digits 1-9
        cand = set(range(1, 10))
        for run in cell_to_runs[cell]:
            # 1. Exclude digits already used in this run
            used_in_run = set()
            for c2 in run["cells"]:
                if c2 != cell and c2 in assignment:
                    used_in_run.add(assignment[c2])
            cand -= used_in_run

            # 2. Restrict to digits that appear in some valid sum-subset for
            # this run's clue, given the partial assignment of other run cells.
            subsets_for_run = sum_subsets(run["clue"], run["length"])
            valid_sets = []
            for ss in subsets_for_run:
                # Check compatibility with partial assignment of other cells in run
                assigned_in_run = {assignment[c2] for c2 in run["cells"]
                                   if c2 != cell and c2 in assignment}
                if assigned_in_run.issubset(ss):
                    valid_sets.append(ss)
            if not valid_sets:
                cand = set()
                break
            allowed_for_cell = set().union(*[ss - assigned_in_run for ss in valid_sets
                                             if assigned_in_run.issubset(ss)])
            # Re-compute assigned_in_run since we used it inline above
            assigned_in_run = {assignment[c2] for c2 in run["cells"]
                               if c2 != cell and c2 in assignment}
            allowed_for_cell = set()
            for ss in valid_sets:
                allowed_for_cell |= (ss - assigned_in_run)
            cand &= allowed_for_cell
        candidates[cell] = frozenset(cand)

    return candidates


def is_solvable(rows: int, cols: int, cells: dict, assignment: dict,
                max_backtrack: int = 100_000) -> tuple:
    """Determine whether the partial `assignment` can be extended to a full
    valid solution. Returns (solvable, reason) where reason is None on success
    or a short diagnostic string on failure.

    Uses constraint propagation + bounded backtracking with MRV heuristic
    (most-constrained variable first).
    """
    runs = extract_runs(rows, cols, cells)
    white_cells = [pos for pos, c in cells.items() if c["type"] == "white"]

    # Validate existing assignment first
    for run in runs:
        digits = []
        for c in run["cells"]:
            if c in assignment:
                d = assignment[c]
                if d in digits:
                    return (False, f"duplicate {d} in run at {run['cells'][0]}")
                digits.append(d)
        # Check sum constraint if run is fully assigned
        if len(digits) == run["length"]:
            if sum(digits) != run["clue"]:
                return (False, f"run at {run['cells'][0]} sums to {sum(digits)}, expected {run['clue']}")

    candidates = candidate_digits(white_cells, runs, assignment)
    for cell, cand in candidates.items():
        if cell not in assignment and not cand:
            return (False, f"no valid digit for cell {cell}")

    # Backtrack
    counter = [0]

    def backtrack(asn: dict) -> bool:
        counter[0] += 1
        if counter[0] > max_backtrack:
            return False
        # Pick most-constrained empty cell
        empty = [c for c in white_cells if c not in asn]
        if not empty:
            return True  # full assignment — success
        cands = candidate_digits(white_cells, runs, asn)
        if any(c not in asn and not cands[c] for c in white_cells):
            return False
        cell = min(empty, key=lambda c: len(cands[c]))
        for d in sorted(cands[cell]):
            asn_new = dict(asn)
            asn_new[cell] = d
            # Re-propagate (cheap check that the fix doesn't break a run sum)
            broke = False
            for run in runs:
                if cell in run["cells"]:
                    digits_in_run = [asn_new[c2] for c2 in run["cells"] if c2 in asn_new]
                    if len(digits_in_run) == run["length"]:
                        if sum(digits_in_run) != run["clue"]:
                            broke = True; break
                    elif sum(digits_in_run) >= run["clue"]:
                        # Already exceeds clue with cells left to fill, but allowed
                        # if exactly clue matches when complete; for partial, just
                        # check we haven't gone over
                        if sum(digits_in_run) > run["clue"]:
                            broke = True; break
            if broke:
                continue
            if backtrack(asn_new):
                return True
        return False

    if backtrack(dict(assignment)):
        return (True, None)
    if counter[0] > max_backtrack:
        return (False, f"backtrack exceeded {max_backtrack} steps")
    return (False, "no valid completion")


# ----------------------------------------------------------------------------
# Quick smoke test (run with: python -m src.environments.kakuro_utils)
# ----------------------------------------------------------------------------

def _smoke():
    """Tiny sanity test on a known-solvable 2-cell puzzle."""
    print("Sum subsets:")
    print(f"  sum=6, len=2: {sum_subsets(6, 2)}")
    print(f"  sum=10, len=3: {sum_subsets(10, 3)}")
    print(f"  sum=45, len=9: {sum_subsets(45, 9)}")  # = (1,2,...,9)

    # Tiny 2x3 puzzle:
    #   #     #(3↓)  #(7↓)
    #   #(10→) .      .
    # Single row run of length 2 with clue 10. Single col runs of length 1
    # are degenerate (clue=digit), which we skip in extract_runs by requiring
    # cells past the black to be white.
    cells = {
        (0, 0): {"type": "black"},
        (0, 1): {"type": "black", "down_clue": 3},
        (0, 2): {"type": "black", "down_clue": 7},
        (1, 0): {"type": "black", "right_clue": 10},
        (1, 1): {"type": "white"},
        (1, 2): {"type": "white"},
    }
    runs = extract_runs(2, 3, cells)
    print("\nExtracted runs:")
    for r in runs:
        print(f"  {r['direction']} run @ {r['cells']}, clue={r['clue']}, len={r['length']}")

    # Empty assignment → solvable (3,7 satisfies row=10, col 1 = 3, col 2 = 7)
    ok, reason = is_solvable(2, 3, cells, {})
    print(f"\nEmpty assignment solvable: {ok} ({reason})")

    # With (1,1) = 4 → row needs (1,2)=6 to sum 10, but col 2 needs 7 → unsolvable
    ok, reason = is_solvable(2, 3, cells, {(1, 1): 4})
    print(f"Partial assignment {{(1,1)=4}} solvable: {ok} ({reason})")

    # With (1,1) = 3 → row needs (1,2)=7, col 1 needs 3, col 2 needs 7 → solvable
    ok, reason = is_solvable(2, 3, cells, {(1, 1): 3})
    print(f"Partial assignment {{(1,1)=3}} solvable: {ok} ({reason})")


if __name__ == "__main__":
    _smoke()
