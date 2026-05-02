"""Hidato puzzle bank — hand-curated puzzles with known solutions.

Each puzzle is built by:
1. Designing a complete Hamiltonian path through the grid (the `solution`).
2. Picking a subset of cells to expose as `givens`.
3. The remaining cells are empty for the model to fill.

All paths are 4-connected (orthogonal-only adjacency), matching the Numbrix
variant of Hidato implemented in `hidato.py`.

Schema:
    {
        "id": str,
        "rows": int,
        "cols": int,
        "givens": dict[(r, c) -> int],   # pre-filled cells (subset of solution)
        "solution": dict[(r, c) -> int], # full solution (for validation/oracle)
    }

To add a new puzzle, write the solution as a 2D grid string for clarity, then
let the helper convert.
"""
from __future__ import annotations
from typing import List, Dict, Any


def _solution_from_grid(grid_str: str) -> dict:
    """Parse a multi-line grid string into a (r, c) -> int dict.

    Example input:
        "1 2 3\\n6 5 4\\n7 8 9"
    Returns: {(0,0):1, (0,1):2, (0,2):3, (1,0):6, ...}
    """
    out = {}
    for r, row in enumerate(grid_str.strip().split("\n")):
        for c, val in enumerate(row.split()):
            out[(r, c)] = int(val)
    return out


def _make_puzzle(id_: str, rows: int, cols: int, solution_grid: str,
                  given_cells: list) -> dict:
    """Build a puzzle from a solution grid and a list of (r, c) cells to expose."""
    solution = _solution_from_grid(solution_grid)
    assert len(solution) == rows * cols, (
        f"{id_}: solution has {len(solution)} cells, expected {rows*cols}"
    )
    # Validate the solution is a valid Hidato path:
    from .hidato_utils import verify_solution
    ok, reason = verify_solution(rows, cols, solution)
    assert ok, f"{id_}: solution invalid: {reason}"
    givens = {pos: solution[pos] for pos in given_cells}
    return {
        "id": id_,
        "rows": rows,
        "cols": cols,
        "givens": givens,
        "solution": solution,
    }


# ---- 3×3 puzzles (9 cells, ~7 to fill) ----

# Snake from top-left to bottom-right: 1→2→3→6→5→4→7→8→9
PUZZLE_3x3_SNAKE = _make_puzzle(
    "3x3_snake", 3, 3,
    """
    1 2 3
    6 5 4
    7 8 9
    """,
    given_cells=[(0, 0), (2, 2)],  # just 1 and 9
)

# U-shape: 1→8→7 ↓ 2→3→ ↓ 9 →4→5→6
# Path: (0,0)→(1,0)→(2,0)→(2,1)→(2,2)→(1,2)→(0,2)→(0,1)→(1,1)
# Numbers:    1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9
PUZZLE_3x3_U = _make_puzzle(
    "3x3_u", 3, 3,
    """
    1 8 7
    2 9 6
    3 4 5
    """,
    given_cells=[(0, 0), (1, 1)],  # 1 and the center
)

# Spiral inward: 1→2→3 ↓ 4 ↓ 5 ↓ 6 ← 7 ← 8 → 9
# Path: (0,0)(0,1)(0,2)(1,2)(2,2)(2,1)(2,0)(1,0)(1,1)
PUZZLE_3x3_SPIRAL = _make_puzzle(
    "3x3_spiral", 3, 3,
    """
    1 2 3
    8 9 4
    7 6 5
    """,
    given_cells=[(0, 0), (2, 0)],  # 1 and 7
)


# ---- 4×4 puzzles (16 cells, ~12 to fill) ----

# Boustrophedon (back-and-forth) path
PUZZLE_4x4_BOUSTROPHEDON = _make_puzzle(
    "4x4_boustrophedon", 4, 4,
    """
    1  2  3  4
    8  7  6  5
    9 10 11 12
    16 15 14 13
    """,
    given_cells=[(0, 0), (3, 0)],  # 1 and 16
)

# Spiral inward
# (0,0)(0,1)(0,2)(0,3)(1,3)(2,3)(3,3)(3,2)(3,1)(3,0)(2,0)(1,0)(1,1)(1,2)(2,2)(2,1)
# Numbers: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
PUZZLE_4x4_SPIRAL = _make_puzzle(
    "4x4_spiral", 4, 4,
    """
    1  2  3  4
    12 13 14 5
    11 16 15 6
    10 9  8  7
    """,
    given_cells=[(0, 0), (1, 1), (3, 3)],  # 1, 13, 7
)


# ---- 4×3 (rectangular) puzzle (12 cells, ~8 to fill) ----

# Snake across rows
PUZZLE_4x3_SNAKE = _make_puzzle(
    "4x3_snake", 4, 3,
    """
     1  2  3
     6  5  4
     7  8  9
    12 11 10
    """,
    given_cells=[(0, 0), (3, 0)],  # 1 and 12
)


# ---- 5×3 puzzle (15 cells) ----

PUZZLE_5x3_SNAKE = _make_puzzle(
    "5x3_snake", 5, 3,
    """
     1  2  3
     6  5  4
     7  8  9
    12 11 10
    13 14 15
    """,
    given_cells=[(0, 0), (2, 1), (4, 2)],  # 1, 8, 15
)


# ---- 5×4 puzzle (20 cells) ----

# Larger boustrophedon — 4 rows snaking
PUZZLE_5x4_SNAKE = _make_puzzle(
    "5x4_snake", 5, 4,
    """
     1  2  3  4
     8  7  6  5
     9 10 11 12
    16 15 14 13
    17 18 19 20
    """,
    given_cells=[(0, 0), (2, 0), (4, 0)],  # 1, 9, 17
)


# ---- Master list ----

PUZZLES: List[Dict[str, Any]] = [
    PUZZLE_3x3_SNAKE,
    PUZZLE_3x3_U,
    PUZZLE_3x3_SPIRAL,
    PUZZLE_4x4_BOUSTROPHEDON,
    PUZZLE_4x4_SPIRAL,
    PUZZLE_4x3_SNAKE,
    PUZZLE_5x3_SNAKE,
    PUZZLE_5x4_SNAKE,
]


def get_puzzle(name_or_idx) -> dict:
    """Look up by name or index."""
    if isinstance(name_or_idx, int):
        return PUZZLES[name_or_idx]
    for p in PUZZLES:
        if p["id"] == name_or_idx:
            return p
    raise KeyError(f"No puzzle named {name_or_idx!r}")


def _validate_all():
    """Verify each puzzle: solution is a valid Hamiltonian path AND the given
    state is solvable (i.e., the partial puzzle has at least one valid
    completion — namely, the recorded solution)."""
    from .hidato_utils import is_solvable, verify_solution
    for p in PUZZLES:
        # Solution must be a valid Hamiltonian path
        ok, reason = verify_solution(p["rows"], p["cols"], p["solution"])
        assert ok, f"{p['id']}: solution invalid: {reason}"
        # Given state must be solvable
        ok, reason = is_solvable(p["rows"], p["cols"], dict(p["givens"]))
        assert ok, f"{p['id']}: given state unsolvable: {reason}"
        n_givens = len(p["givens"])
        n_empty = p["rows"] * p["cols"] - n_givens
        print(f"  {p['id']:>22s}: {p['rows']}x{p['cols']}, {n_givens} givens, {n_empty} empty cells ✓")


if __name__ == "__main__":
    print(f"Validating {len(PUZZLES)} Hidato puzzles in bank...")
    _validate_all()
    print("All puzzles valid.")
