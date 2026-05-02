"""Hand-curated Kakuro puzzle bank.

Each puzzle is a small grid (3-5 white-cell area) with:
- A grid layout: which cells are black, which are white
- Clues on relevant black cells (right_clue / down_clue)
- A known solution (digit at each white cell), used for validation/testing

The puzzles are simple enough that the solvability checker resolves them quickly,
but rich enough that there's a real predictive gap (some partial assignments
that look valid are actually doomed).

To add a new puzzle: extend PUZZLES with a dict matching the schema below.

Schema:
    {
        "id": str,                 # human-readable identifier
        "rows": int,               # grid height
        "cols": int,               # grid width
        "cells": list[tuple],      # one entry per cell, ordered by (row, col)
            # tuple = (r, c, type, ...) where:
            #   type='black': (r, c, 'black', right_clue or None, down_clue or None)
            #   type='white': (r, c, 'white')
        "solution": dict,          # {(r, c): digit} for all white cells
    }

Notes:
- Rows/cols are 0-indexed.
- "right_clue" on a black cell at (r, c) is the sum for the run of WHITE cells
  starting at (r, c+1) and continuing right until hitting a black cell or edge.
- "down_clue" on a black cell at (r, c) is similar but for the column run
  starting at (r+1, c).
- Black cells without clues just exist as walls (e.g., cells in the corner
  not bounding any run).
"""
from __future__ import annotations
from typing import List, Dict, Any


def _b(r: int, c: int, right=None, down=None) -> tuple:
    """Black cell shorthand."""
    return (r, c, "black", right, down)


def _w(r: int, c: int) -> tuple:
    """White cell shorthand."""
    return (r, c, "white")


# ---- Tiny puzzles (2-3 white cells) — for smoke testing ----

PUZZLE_TINY_1 = {
    "id": "tiny_1",
    "rows": 2,
    "cols": 3,
    # Layout (· = black, _ = white, [3↓] = down-clue 3, [10→] = right-clue 10):
    #   ·    [3↓]  [7↓]
    #   [10→] _    _
    "cells": [
        _b(0, 0), _b(0, 1, down=3), _b(0, 2, down=7),
        _b(1, 0, right=10), _w(1, 1), _w(1, 2),
    ],
    "solution": {(1, 1): 3, (1, 2): 7},
}


# ---- Small puzzles (3-5 white cells, single run shape) ----

# A "+" of 4 cells: row run + column run share a center cell
PUZZLE_PLUS_4 = {
    "id": "plus_4",
    "rows": 3,
    "cols": 3,
    # ·     [12↓]  ·
    # [10→] _      ·         (only 1 white cell in row, run = 1)
    # ·     [3↓]   ·         actually let's make it row of 2:
    # Use a 3x4 layout instead:
    #
    # Trying again — a true plus:
    #
    # ·     [13↓]   ·     ·
    # [11→] _       ·     ·
    # ·     [4↓]    ·     ·
    #
    # No, build a proper plus:
    #   ·       ·       [11↓]   ·
    #   ·       [9→]    _       _    (row run length 2 at clue 9)
    #   [10→]   _       _       ·    (row run length 2 at clue 10)
    #   ·       _       ·       ·    (col run length 2 at clue 11)
    #
    # Hmm, this is getting complex. Let me simplify:
    #
    # 3x3 puzzle, 4 white cells in two rows:
    #   ·       [9↓]   [12↓]
    #   [11→]   _      _      → row sum 11 in 2 cells
    #   [10→]   _      _      → row sum 10 in 2 cells
    # column sums: col 1 sum 9 (in 2 cells), col 2 sum 12 (in 2 cells)
    #
    # Try: row 1 = (8, 3) sum 11, row 2 = (1, 9) sum 10
    # col 1 = (8+1)=9 ✓, col 2 = (3+9)=12 ✓. Valid!
    "cells": [
        _b(0, 0), _b(0, 1, down=9), _b(0, 2, down=12),
        _b(1, 0, right=11), _w(1, 1), _w(1, 2),
        _b(2, 0, right=10), _w(2, 1), _w(2, 2),
    ],
    "solution": {(1, 1): 8, (1, 2): 3, (2, 1): 1, (2, 2): 9},
}


# ---- Medium puzzles (6-10 white cells) ----

# A 3x4 puzzle with 6 white cells in 2 row runs + 3 col runs
PUZZLE_MED_6 = {
    "id": "med_6",
    "rows": 4,
    "cols": 4,
    #   ·       [10↓]   [11↓]   [4↓]
    #   [16→]   _       _       _
    #   [10→]   _       _       _
    #   ·       ·       ·       ·       (extra row not used; can crop)
    #
    # Constraints:
    #   row 1 sum 16 in 3 cells = e.g., 1+6+9, 1+7+8, 2+5+9, 2+6+8, 3+4+9, 3+5+8, 3+6+7, 4+5+7, 4+3+9, etc.
    #   row 2 sum 10 in 3 cells = e.g., 1+2+7, 1+3+6, 1+4+5, 2+3+5
    #   col 1 sum 10 in 2 cells = (1,9), (2,8), (3,7), (4,6)
    #   col 2 sum 11 in 2 cells = (2,9), (3,8), (4,7), (5,6)
    #   col 3 sum 4 in 2 cells = (1,3)
    # Solution: col 3 must be (1,3) some order. Row 2 has col-3 cell = 1 or 3.
    # Try: col 3 = (3, 1) → cells (1,3)=3, (2,3)=1
    # row 1 sum 16 with (1,3)=3 → other two sum 13: (4,9), (5,8), (6,7)
    # row 2 sum 10 with (2,3)=1 → other two sum 9: (1,8), (2,7), (3,6), (4,5)
    # col 1 sum 10 → (a,b) with a+b=10
    # col 2 sum 11 → (c,d) with c+d=11
    # row 1: a + c + 3 = 16 → a + c = 13
    # row 2: b + d + 1 = 10 → b + d = 9
    # col 1: a + b = 10
    # col 2: c + d = 11
    # From a+c=13 and a+b=10: c-b=3
    # From c+d=11 and b+d=9: c-b=2
    # Contradiction. So col 3 = (1, 3) doesn't work that way.
    # Try col 3 = (3, 1):  same as above
    # Try col 3 = (1, 3): cells (1,3)=1, (2,3)=3
    # row 1 sum 16 with (1,3)=1 → other two sum 15: (6,9), (7,8)
    # row 2 sum 10 with (2,3)=3 → other two sum 7: (1,6), (2,5), (3,4)
    # col 1 a+b=10, col 2 c+d=11
    # row 1: a + c + 1 = 16 → a + c = 15  (so {(a,c)} = {(6,9),(7,8),(8,7),(9,6)})
    # row 2: b + d + 3 = 10 → b + d = 7   ({(1,6),(2,5),(3,4),(4,3),(5,2),(6,1)})
    # col 1: a + b = 10, col 2: c + d = 11
    # From a+c=15, a+b=10: c-b=5. From b+d=7, c+d=11: c-b=4. Contradiction.
    #
    # Hmm. Let me try a different layout — maybe my constraints are over-determined.
    # Switch col 3 clue: change from 4 to 7 (more flexible).
    #
    # New constraints:
    #   row 1 sum 16, row 2 sum 10
    #   col 1 sum 10, col 2 sum 11, col 3 sum 7 (e.g., (1,6), (2,5), (3,4))
    # Try col 3 = (3, 4): row 1 has 3 in col 3 → row 1 sum (3+a+b)=16 → a+b=13 in cols 1,2
    #   row 2 has 4 in col 3 → row 2 sum (4+c+d)=10 → c+d=6 in cols 1,2
    #   col 1 sum 10: a+c=10; col 2 sum 11: b+d=11
    #   From a+b=13, a+c=10 → b-c=3
    #   From c+d=6, b+d=11 → b-c=5. Contradiction.
    # Try col 3 = (4, 3): a+b=12; c+d=7
    #   col1: a+c=10; col2: b+d=11
    #   b-c = 12-10=2 vs 11-7=4. No.
    # Try col 3 = (2, 5): a+b=14; c+d=5
    #   col1: a+c=10; col2: b+d=11
    #   b-c=14-10=4 vs 11-5=6. No.
    # Try col 3 = (5, 2): a+b=11; c+d=8
    #   b-c = 11-10=1 vs 11-8=3. No.
    # Try col 3 = (1, 6): a+b=15; c+d=4
    #   b-c=15-10=5 vs 11-4=7. No.
    # Try col 3 = (6, 1): a+b=10; c+d=9
    #   b-c=10-10=0 vs 11-9=2. No.
    #
    # The contradictions repeat: (b-c) computed two ways doesn't match.
    # System has 4 vars (a,b,c,d) and 4 constraints — but they need to be
    # algebraically consistent. Let me solve the system symbolically:
    #   row 1: a + c + col3[0] = 16  →  a + c = 16 - col3[0]
    #   row 2: b + d + col3[1] = 10  →  b + d = 10 - col3[1]
    #   col 1: a + b = 10
    #   col 2: c + d = 11
    # Sum row constraints: a+b+c+d = 26 - (col3[0]+col3[1]) = 26 - col3_sum
    # Sum col constraints: a+b+c+d = 21
    # → 26 - col3_sum = 21 → col3_sum = 5
    # So col 3's sum MUST equal 5 for the system to have a solution.
    #
    # Set col 3 sum to 5:
    "cells": [
        _b(0, 0), _b(0, 1, down=10), _b(0, 2, down=11), _b(0, 3, down=5),
        _b(1, 0, right=16), _w(1, 1), _w(1, 2), _w(1, 3),
        _b(2, 0, right=10), _w(2, 1), _w(2, 2), _w(2, 3),
    ],
    # Solve: col 3 sum 5 = (1,4) or (2,3)
    # Try col 3 = (1, 4): a+c=15, b+d=6, a+b=10, c+d=11
    #   a + c = 15, a + b = 10 → c - b = 5
    #   c + d = 11, b + d = 6 → c - b = 5 ✓
    #   Pick a = 6, b = 4, c = 9, d = 2 → check: 6+4=10 ✓, 9+2=11 ✓, 6+9+1=16 ✓, 4+2+4=10 ✓
    "solution": {
        (1, 1): 6, (1, 2): 9, (1, 3): 1,
        (2, 1): 4, (2, 2): 2, (2, 3): 4,  # uh oh — col 3 has 1 and 4 = OK, but row 2 has 4 twice!
    },
}

# The constraint check: row 2 = 4 + 2 + 4 = 10 ✓ but has duplicate 4. Invalid.
# Need to find solution where (1,3)=1, (2,3)=4 AND row 2 has distinct digits.
# Row 2: b + 2 + 4 = 10 → b = 4. So row 2 = (4, 2, 4) — duplicate.
# Try col 3 = (4, 1): a+c=12, b+d=9, a+b=10, c+d=11
#   c-b = 12-10=2 vs 11-9=2 ✓
#   Pick a=4, b=6, c=8, d=3 → row 1=(4,8,4) duplicate; row 2=(6,3,1)=10✓, (4,1) col=4+1=5 ✓, col1=4+6=10 ✓, col2=8+3=11 ✓
#   But row 1 duplicate (4 appears in col 1 and col 3).
# Try a=3, b=7, c=9, d=2: row 1=(3,9,4) sums to 16 ✓ all distinct ✓
#   row 2=(7,2,1) sums to 10 ✓ all distinct ✓
#   col 1 = 3+7=10 ✓, col 2 = 9+2=11 ✓, col 3 = 4+1=5 ✓
PUZZLE_MED_6["solution"] = {
    (1, 1): 3, (1, 2): 9, (1, 3): 4,
    (2, 1): 7, (2, 2): 2, (2, 3): 1,
}


# ---- Larger puzzles (5x5+ with ~10 white cells) ----

# A 4x5 puzzle with 8 white cells
PUZZLE_LARGE_8 = {
    "id": "large_8",
    "rows": 5,
    "cols": 5,
    # Layout:
    #   ·       ·       [11↓]   [12↓]   ·
    #   ·       [10→]   _       _       ·          (row sum 10, len 2)
    #   [16→]   _       _       _       ·          (row sum 16, len 3)
    #   [10→]   _       _       ·       ·          (row sum 10, len 2)
    #   ·       ·       ·       ·       ·
    # col sums: col 1 = 2 cells, col 2 = 3 cells (11), col 3 = 2 cells (12), col 4 = 1 cell
    # Wait, let me recount:
    #   col 1 (cells at (2,1),(3,1)): 2 cells, no clue from above (row 1 col 1 is black with no down_clue)
    #   col 2 (cells at (1,2),(2,2),(3,2)): 3 cells, clue 11 at (0,2)
    #   col 3 (cells at (1,3),(2,3)): 2 cells, clue 12 at (0,3)
    # Hmm col 1 has no clue, that's a bug. Need a black cell with down_clue at (1,1).
    # Re-do layout:
    "cells": [
        # row 0
        _b(0, 0), _b(0, 1), _b(0, 2, down=11), _b(0, 3, down=12), _b(0, 4),
        # row 1
        _b(1, 0), _b(1, 1, right=10, down=None), _w(1, 2), _w(1, 3), _b(1, 4),
        # row 2
        _b(2, 0, right=16, down=None), _w(2, 1, ), _w(2, 2), _w(2, 3), _b(2, 4),
        # ...this is getting complex with my shorthand; rebuilding cleanly below
    ],
    "solution": {},
}

# Let me just remove the broken large puzzle for now and keep the working ones.
# We'll add more puzzles after the env is tested.
del PUZZLE_LARGE_8


# ---- Master puzzle list ----

PUZZLES: List[Dict[str, Any]] = [
    PUZZLE_TINY_1,
    PUZZLE_PLUS_4,
    PUZZLE_MED_6,
]


def get_puzzle(name_or_idx) -> dict:
    """Look up by name or index."""
    if isinstance(name_or_idx, int):
        return PUZZLES[name_or_idx]
    for p in PUZZLES:
        if p["id"] == name_or_idx:
            return p
    raise KeyError(f"No puzzle named {name_or_idx!r}")


def cells_to_dict(cells: list) -> dict:
    """Convert the flat (r, c, type, ...) list to a dict mapping (r,c) -> attrs."""
    out = {}
    for entry in cells:
        r, c, typ, *rest = entry
        if typ == "black":
            right_clue = rest[0] if len(rest) >= 1 else None
            down_clue = rest[1] if len(rest) >= 2 else None
            out[(r, c)] = {"type": "black", "right_clue": right_clue, "down_clue": down_clue}
        elif typ == "white":
            out[(r, c)] = {"type": "white"}
    return out


def _validate_all():
    """Verify each puzzle's solution actually satisfies the constraints."""
    from .kakuro_utils import is_solvable, extract_runs
    for p in PUZZLES:
        cells = cells_to_dict(p["cells"])
        runs = extract_runs(p["rows"], p["cols"], cells)
        solution = p["solution"]
        # Check: every white cell has a digit in solution
        white_cells = {pos for pos, c in cells.items() if c["type"] == "white"}
        if set(solution.keys()) != white_cells:
            raise AssertionError(f"{p['id']}: solution does not cover all white cells")
        # Verify against constraints
        for run in runs:
            digits = [solution[c] for c in run["cells"]]
            assert sum(digits) == run["clue"], f"{p['id']}: run {run['cells']} sums to {sum(digits)}, expected {run['clue']}"
            assert len(set(digits)) == len(digits), f"{p['id']}: run {run['cells']} has duplicates {digits}"
        # Verify the empty-board state is solvable (sanity check on solver)
        ok, reason = is_solvable(p["rows"], p["cols"], cells, {})
        assert ok, f"{p['id']}: solver says empty board unsolvable: {reason}"
        print(f"  {p['id']}: ✓ ({len(white_cells)} white cells)")


if __name__ == "__main__":
    print(f"Validating {len(PUZZLES)} puzzles in bank...")
    _validate_all()
    print("All puzzles valid.")
