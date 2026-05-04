"""SAVE-side Sudoku 4×4 environment helpers.

Wraps grid state + actions in the format used by the SAVE data generator,
WITHOUT touching src/environments/sudoku*.py (additivity contract per
CLAUDE.md "## SAVE Data Pipeline").

What's in here:
  - render_state_b5(grid)              -> string in B-5 training format
                                         (with `|` and `---` separators)
  - render_state_save(grid)            -> string in SAVE's "row1: ..." format
                                         (kept for forward compatibility)
  - parse_action_text(text)            -> ActionStruct or None
  - canonical_action(action_struct)    -> "R{r}C{c}={v}" canonical key
  - action_text(action_struct)         -> "place V at row R col C"
  - is_local_valid(grid, action)       -> bool
  - apply_action(grid, action)         -> new grid (deep-copied)
  - state_hash(grid)                   -> "sha1:{40 hex chars}"
  - generate_root_puzzle(seed, n_empty=10) -> grid (4×4 list of lists)
  - enumerate_legal_actions(grid)      -> list[ActionStruct] of all legal placements

Action coordinates are 1-indexed at the user-facing level (matches SAVE
schema action_struct.{row, col, value}). Internally we convert to
0-indexed for grid indexing.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from typing import Optional, List, Tuple


# --- Constants ---

GRID_N = 4
ENV_VERSION = "sudoku4_env_v1_6empty"
TEXT_VERSION_B5 = "sudoku_text_b5_compat_v1"
TEXT_VERSION_SAVE = "sudoku_text_v1"


@dataclass(frozen=True)
class ActionStruct:
    """1-indexed action: place `value` at (`row`, `col`)."""
    row: int
    col: int
    value: int

    @property
    def row0(self) -> int:
        return self.row - 1

    @property
    def col0(self) -> int:
        return self.col - 1

    def to_dict(self) -> dict:
        return {"row": self.row, "col": self.col, "value": self.value}


# --- Renderers ---

def render_state_b5(grid: List[List[int]]) -> str:
    """Render grid in the EXACT format rl_b5 saw during SFT training.

    Format (verified from data/sudoku_4x4_llm_policy_minimal_spa_scale samples):
        3 2 | 1 4
        1 . | 3 .
        ---------
        . . | 2 1
        . 1 | . 3

    Empty cells are '.'; values 1-4 shown as digits; ' | ' between col 1-2 and
    col 3-4 of each row; '---------' line between row 2 and row 3 of the grid.
    Note: NO 'Current state:' prefix (the user message appends that).
    """
    def cell(v: int) -> str:
        return "." if v == 0 else str(v)

    rows = []
    for r in range(GRID_N):
        left = " ".join(cell(grid[r][c]) for c in (0, 1))
        right = " ".join(cell(grid[r][c]) for c in (2, 3))
        rows.append(f"{left} | {right}")
    # Insert horizontal separator between row 1 and row 2 (after the top 2x2 box)
    rows.insert(2, "-" * 9)
    return "\n".join(rows)


def render_state_save(grid: List[List[int]]) -> str:
    """Render grid in the SAVE spec's `sudoku_text_v1` format.

        row1: 1 2 3 4
        row2: 3 4 . .
        row3: . . 4 3
        row4: 4 3 . .

    Kept for forward compatibility / debugging; B-5 format is the active one.
    """
    def cell(v: int) -> str:
        return "." if v == 0 else str(v)
    return "\n".join(
        f"row{r+1}: " + " ".join(cell(grid[r][c]) for c in range(GRID_N))
        for r in range(GRID_N)
    )


# --- Action utilities ---

_ACTION_RE = re.compile(
    r"place\s+(?P<value>[1-9])\s+at\s+row\s+(?P<row>[1-9])\s+col(?:umn)?\s+(?P<col>[1-9])",
    re.IGNORECASE,
)


def parse_action_text(text: str) -> Optional[ActionStruct]:
    """Parse 'place 1 at row 2 col 3' (or 'column 3') -> ActionStruct.

    Returns None if no match. Does NOT validate against current grid;
    use is_local_valid for that.
    """
    if not text:
        return None
    m = _ACTION_RE.search(text)
    if not m:
        return None
    row = int(m.group("row"))
    col = int(m.group("col"))
    value = int(m.group("value"))
    if not (1 <= row <= GRID_N and 1 <= col <= GRID_N and 1 <= value <= GRID_N):
        return None
    return ActionStruct(row=row, col=col, value=value)


def canonical_action(action: ActionStruct) -> str:
    """Compact canonical form for dedup. e.g. R2C3=1 ."""
    return f"R{action.row}C{action.col}={action.value}"


def action_text(action: ActionStruct) -> str:
    """Natural-language form matching B-5's action format."""
    return f"place {action.value} at row {action.row} col {action.col}"


def action_hash(action: ActionStruct) -> str:
    """SHA-1 of canonical form, prefixed `sha1:`."""
    return "sha1:" + hashlib.sha1(canonical_action(action).encode()).hexdigest()


# --- Legality + transition ---

def is_local_valid(grid: List[List[int]], action: ActionStruct) -> bool:
    """Is this placement legal in the current grid?

    Legal = cell is empty AND value doesn't conflict with row/col/2x2-box.
    Does NOT check whether the resulting grid has a global completion;
    that's the solver's job (next_viable).
    """
    r, c, v = action.row0, action.col0, action.value
    if not (0 <= r < GRID_N and 0 <= c < GRID_N and 1 <= v <= GRID_N):
        return False
    if grid[r][c] != 0:
        return False
    # Row/col conflict
    for k in range(GRID_N):
        if grid[r][k] == v or grid[k][c] == v:
            return False
    # 2x2 box conflict
    box_r0 = (r // 2) * 2
    box_c0 = (c // 2) * 2
    for dr in (0, 1):
        for dc in (0, 1):
            if grid[box_r0 + dr][box_c0 + dc] == v:
                return False
    return True


def apply_action(grid: List[List[int]], action: ActionStruct) -> List[List[int]]:
    """Return a new grid with the action applied. Does NOT validate."""
    new_grid = [row[:] for row in grid]
    new_grid[action.row0][action.col0] = action.value
    return new_grid


def is_goal(grid: List[List[int]]) -> bool:
    """Grid is fully filled with no constraint violations."""
    for r in range(GRID_N):
        for c in range(GRID_N):
            if grid[r][c] == 0:
                return False
    # Use solver-side consistency check via constraint enumeration
    for r in range(GRID_N):
        if set(grid[r]) != {1, 2, 3, 4}:
            return False
    for c in range(GRID_N):
        col = [grid[r][c] for r in range(GRID_N)]
        if set(col) != {1, 2, 3, 4}:
            return False
    for box_r0 in (0, 2):
        for box_c0 in (0, 2):
            box = [grid[box_r0 + dr][box_c0 + dc]
                   for dr in (0, 1) for dc in (0, 1)]
            if set(box) != {1, 2, 3, 4}:
                return False
    return True


def enumerate_legal_actions(grid: List[List[int]]) -> List[ActionStruct]:
    """All (row, col, value) placements that are locally legal at this grid."""
    out = []
    for r in range(GRID_N):
        for c in range(GRID_N):
            if grid[r][c] != 0:
                continue
            for v in range(1, GRID_N + 1):
                a = ActionStruct(row=r + 1, col=c + 1, value=v)
                if is_local_valid(grid, a):
                    out.append(a)
    return out


# --- State hashing + puzzle generation ---

def state_hash(grid: List[List[int]]) -> str:
    """SHA-1 hash of canonical grid JSON, prefixed `sha1:`. For dedup + leakage check."""
    canonical = json.dumps(grid, separators=(",", ":"), sort_keys=False)
    return "sha1:" + hashlib.sha1(canonical.encode()).hexdigest()


def _shuffle(seq, rng):
    seq = list(seq)
    rng.shuffle(seq)
    return seq


def _fill_random_complete(rng) -> List[List[int]]:
    """Generate a random fully-solved 4×4 Sudoku grid via backtracking with shuffled values."""
    grid = [[0] * GRID_N for _ in range(GRID_N)]

    def fill_recursive(idx: int) -> bool:
        if idx == GRID_N * GRID_N:
            return True
        r, c = idx // GRID_N, idx % GRID_N
        for v in _shuffle((1, 2, 3, 4), rng):
            a = ActionStruct(row=r + 1, col=c + 1, value=v)
            if is_local_valid(grid, a):
                grid[r][c] = v
                if fill_recursive(idx + 1):
                    return True
                grid[r][c] = 0
        return False

    ok = fill_recursive(0)
    if not ok:
        raise RuntimeError("Failed to fill a 4x4 Sudoku — should never happen")
    return grid


def generate_root_puzzle(seed: int, n_empty: int = 6) -> List[List[int]]:
    """Generate a 4x4 Sudoku puzzle with `n_empty` cells removed.

    Per plan_2026-05-03_save_data_generation.md decision #2 (revised
    2026-05-03 after sanity-check diagnostic): n_empty=6 matches rl_b5's
    actual evaluation distribution (SudokuEnv default difficulty="easy"
    removes 0.4×16 ≈ 6 cells). At n_empty=10 rl_b5 hits 0% greedy Pass@1;
    at n_empty=6 it hits ~80%. The earlier n_empty=10 directive was based
    on a misreading of B-5's setup.

    The result is guaranteed to admit at least one solution (by construction:
    we remove cells from a known-solved grid). Solution uniqueness is NOT
    enforced — 4×4 Sudoku frequently has multiple completions; the solver
    will report num_solutions accurately.
    """
    if not (0 <= n_empty <= GRID_N * GRID_N):
        raise ValueError(f"n_empty must be in [0, {GRID_N*GRID_N}], got {n_empty}")
    rng = random.Random(seed)
    grid = _fill_random_complete(rng)
    cells = [(r, c) for r in range(GRID_N) for c in range(GRID_N)]
    rng.shuffle(cells)
    for r, c in cells[:n_empty]:
        grid[r][c] = 0
    return grid


# --- Smoke test ---

def _smoke():
    # 1) render_state_b5 matches an example known to be in the existing data
    g = [
        [3, 2, 1, 4],
        [1, 0, 3, 0],
        [0, 0, 2, 1],
        [0, 1, 0, 3],
    ]
    rendered = render_state_b5(g)
    expected = "3 2 | 1 4\n1 . | 3 .\n---------\n. . | 2 1\n. 1 | . 3"
    assert rendered == expected, f"\nGOT:\n{rendered}\n\nEXPECTED:\n{expected}"
    print("  [1] render_state_b5 matches B-5 sample format")

    # 2) render_state_save (alternative format, sanity)
    save_render = render_state_save(g)
    assert save_render.startswith("row1: 3 2 1 4")
    print("  [2] render_state_save format ok")

    # 3) parse_action_text round trip
    a = ActionStruct(row=2, col=3, value=1)
    txt = action_text(a)
    parsed = parse_action_text(txt)
    assert parsed == a
    parsed2 = parse_action_text("place 1 at row 2 column 3")  # 'column' variant
    assert parsed2 == a
    bad = parse_action_text("nonsense")
    assert bad is None
    print(f"  [3] parse round-trip OK: '{txt}' -> {a}")

    # 4) canonical action + hash
    assert canonical_action(a) == "R2C3=1"
    h = action_hash(a)
    assert h.startswith("sha1:") and len(h) == 5 + 40
    print(f"  [4] canonical='{canonical_action(a)}', hash={h}")

    # 5) is_local_valid + apply_action
    assert is_local_valid(g, ActionStruct(row=2, col=2, value=4)) is True
    assert is_local_valid(g, ActionStruct(row=2, col=2, value=3)) is False  # row conflict
    g2 = apply_action(g, ActionStruct(row=2, col=2, value=4))
    assert g2[1][1] == 4 and g[1][1] == 0  # immutability of input
    print("  [5] is_local_valid + apply_action OK")

    # 6) enumerate_legal_actions
    legal = enumerate_legal_actions(g)
    assert all(is_local_valid(g, a) for a in legal)
    assert len(legal) > 0
    print(f"  [6] enumerate_legal_actions: {len(legal)} legal placements")

    # 7) is_goal
    full = [[1, 2, 3, 4], [3, 4, 1, 2], [2, 1, 4, 3], [4, 3, 2, 1]]
    assert is_goal(full)
    assert not is_goal(g)
    print("  [7] is_goal OK")

    # 8) state_hash deterministic
    h1 = state_hash(g)
    h2 = state_hash(g)
    assert h1 == h2 and h1.startswith("sha1:")
    print(f"  [8] state_hash deterministic: {h1}")

    # 9) generate_root_puzzle: 10 empty, deterministic given seed
    p1 = generate_root_puzzle(seed=42, n_empty=10)
    p2 = generate_root_puzzle(seed=42, n_empty=10)
    assert p1 == p2
    n_zero = sum(1 for r in p1 for c in r if c == 0)
    assert n_zero == 10, f"expected 10 empty, got {n_zero}"
    print(f"  [9] generate_root_puzzle(seed=42, n_empty=10) deterministic, "
          f"empty={n_zero}/16")
    print("Sample puzzle:")
    for line in render_state_b5(p1).split("\n"):
        print("    " + line)

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    _smoke()
