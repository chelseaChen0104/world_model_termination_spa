"""Local progress formula for Sudoku 4×4 (sudoku_local_progress_v1).

Per spec §5.2:
    score = filled_normalized - 0.1 * (rows_violated + cols_violated + boxes_violated)

This is the handcrafted progress signal that competes with viability in
SAVE's deceptive-candidate benchmark. CRITICAL: the formula must depend
ONLY on surface features of the grid. It must NOT consult the solver.

Public surface:
    compute_progress(grid) -> dict with:
        - formula_id, formula_spec
        - local_progress_score
        - features: filled_cells, filled_normalized,
                    rows_violated, cols_violated, boxes_violated, constraint_violations
"""
from __future__ import annotations

from typing import List, Dict


FORMULA_ID = "sudoku_local_progress_v1"
FORMULA_SPEC = "filled_normalized - 0.1 * (rows_violated + cols_violated + boxes_violated)"


def _has_duplicate_nonzero(values) -> bool:
    """True if some non-zero value appears more than once. Empty (0) cells ignored."""
    seen = set()
    for v in values:
        if v == 0:
            continue
        if v in seen:
            return True
        seen.add(v)
    return False


def count_rows_with_duplicates(grid: List[List[int]]) -> int:
    return sum(1 for row in grid if _has_duplicate_nonzero(row))


def count_cols_with_duplicates(grid: List[List[int]]) -> int:
    n = len(grid)
    return sum(1 for c in range(n) if _has_duplicate_nonzero(grid[r][c] for r in range(n)))


def count_boxes_with_duplicates(grid: List[List[int]]) -> int:
    """For 4×4: four 2×2 boxes."""
    boxes_with_dup = 0
    for box_r0 in (0, 2):
        for box_c0 in (0, 2):
            cells = [grid[box_r0 + dr][box_c0 + dc]
                     for dr in (0, 1) for dc in (0, 1)]
            if _has_duplicate_nonzero(cells):
                boxes_with_dup += 1
    return boxes_with_dup


def compute_progress(grid: List[List[int]]) -> Dict:
    """Return the progress dict for `grid` per save_sibling_set_v1.2 schema."""
    n = len(grid)
    filled_cells = sum(1 for row in grid for cell in row if cell != 0)
    filled_normalized = filled_cells / float(n * n)
    rows_violated = count_rows_with_duplicates(grid)
    cols_violated = count_cols_with_duplicates(grid)
    boxes_violated = count_boxes_with_duplicates(grid)
    constraint_violations = rows_violated + cols_violated + boxes_violated

    score = filled_normalized - 0.1 * constraint_violations

    return {
        "formula_id": FORMULA_ID,
        "formula_spec": FORMULA_SPEC,
        "local_progress_score": score,
        "features": {
            "filled_cells": filled_cells,
            "filled_normalized": filled_normalized,
            "constraint_violations": constraint_violations,
            "rows_violated": rows_violated,
            "cols_violated": cols_violated,
            "boxes_violated": boxes_violated,
        },
    }


# --- Smoke test ---

def _smoke():
    # 1) Empty grid: score = 0
    empty = [[0]*4 for _ in range(4)]
    p = compute_progress(empty)
    assert p["local_progress_score"] == 0.0
    assert p["features"]["filled_cells"] == 0
    assert p["features"]["constraint_violations"] == 0
    print(f"  [1] empty grid: score={p['local_progress_score']}")

    # 2) Fully solved grid: score = 1.0 (no violations)
    full = [[1,2,3,4],[3,4,1,2],[2,1,4,3],[4,3,2,1]]
    p = compute_progress(full)
    assert p["local_progress_score"] == 1.0
    assert p["features"]["filled_cells"] == 16
    assert p["features"]["constraint_violations"] == 0
    print(f"  [2] solved grid: score={p['local_progress_score']}")

    # 3) Half-filled, no violations: score = 8/16 = 0.5
    partial = [[1,2,3,4],[3,4,1,2],[0,0,0,0],[0,0,0,0]]
    p = compute_progress(partial)
    assert p["local_progress_score"] == 0.5
    assert p["features"]["filled_cells"] == 8
    print(f"  [3] half-filled: score={p['local_progress_score']}")

    # 4) Violating row + box: row 0 has two 1s, AND those 1s are in the same 2x2 box
    bad_row = [[1,1,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    p = compute_progress(bad_row)
    # filled=2 → normalized=0.125; rows_violated=1, cols_violated=0, boxes_violated=1
    # score = 0.125 - 0.1 * 2 = -0.075
    assert abs(p["local_progress_score"] - (-0.075)) < 1e-9
    assert p["features"]["rows_violated"] == 1
    assert p["features"]["cols_violated"] == 0
    assert p["features"]["boxes_violated"] == 1
    print(f"  [4] bad row+box: rows_v={p['features']['rows_violated']} "
          f"box_v={p['features']['boxes_violated']} "
          f"score={p['local_progress_score']:.4f}")

    # 5) The deceptive-pair scenario sketch: a placement that fills a cell
    # (raises filled_normalized) but hits no immediate violation. Local
    # progress goes UP even if the resulting state is globally doomed.
    # This is exactly the property §3.4 of the paper exploits.
    good_state = [[1,2,3,4],[3,4,1,2],[2,1,0,0],[0,0,0,0]]  # filled=10, no violations
    p_good = compute_progress(good_state)
    deceptive_filled = [[1,2,3,4],[3,4,1,2],[2,1,4,0],[0,0,0,0]]  # filled=11, no violation
    p_dec = compute_progress(deceptive_filled)
    assert p_dec["local_progress_score"] > p_good["local_progress_score"]
    print(f"  [5] deceptive sketch: pre={p_good['local_progress_score']:.4f} "
          f"-> post={p_dec['local_progress_score']:.4f} "
          f"(progress goes UP regardless of viability)")

    # 6) Constraint-violation detection across all axes
    # Two 1s in column 0, two 2s in column 1, two 3s in box 0
    violations = [[1,2,0,0],[1,2,0,0],[3,3,0,0],[0,0,0,0]]
    p = compute_progress(violations)
    assert p["features"]["cols_violated"] == 2  # cols 0 and 1
    assert p["features"]["rows_violated"] == 1  # row 2 has two 3s
    assert p["features"]["boxes_violated"] == 2  # box 0 has 1,1,2,2; box 2 has 3,3
    print(f"  [6] multi-axis violations: rows={p['features']['rows_violated']} "
          f"cols={p['features']['cols_violated']} "
          f"boxes={p['features']['boxes_violated']}")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    _smoke()
