"""SAVE-side Hidato 5×4 environment helpers.

Wraps state/action utilities for the SAVE data generator. Does NOT modify
src/environments/hidato*.py.

Action representation: 1-indexed `place {N} at row {R} col {C}` where N must
equal the next sequential number (state.next_n). Adjacency to the previous
number (anchor_cell) is enforced by is_local_valid.

Public surface:
    render_state_hidato(state)               -> string
    parse_action_text(text)                  -> ActionStruct or None
    canonical_action(action)                 -> "R2C3=5"
    action_text(action)                      -> "place 5 at row 2 col 3"
    is_local_valid(state, action)            -> bool
    apply_action(state, action)              -> new state dict
    is_goal(state)                           -> bool
    enumerate_legal_actions(state)           -> list[ActionStruct]
    state_hash(state)                        -> "sha1:..."
    get_root_puzzle(seed)                    -> state dict (picks from puzzle bank)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Use the algorithmically-expanded 200-puzzle bank (locked 2026-05-04 per
# data_generation_hidato.md §3). Falls back to the legacy 8-puzzle bank if the
# expanded bank is missing (e.g., on a fresh machine before bank is generated).
_EXPANDED_BANK_PATH = os.path.join(_REPO_ROOT, "data", "hidato_bank_5x4_v2")
if os.path.isfile(os.path.join(_EXPANDED_BANK_PATH, "bank.py")):
    _spec_path = os.path.join(_EXPANDED_BANK_PATH, "bank.py")
    import importlib.util
    _spec = importlib.util.spec_from_file_location("hidato_bank_5x4_v2", _spec_path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    _BANK = _mod.PUZZLES
    _BANK_SOURCE = "hidato_bank_5x4_v2"
else:
    from src.environments.hidato_puzzle_bank import PUZZLES as _BANK
    _BANK_SOURCE = "legacy_8_puzzle"


ENV_VERSION = f"hidato5x4_env_v2_200puzzles" if _BANK_SOURCE == "hidato_bank_5x4_v2" else "hidato5x4_env_v1_8puzzles"
TEXT_VERSION = "hidato_text_v1"


@dataclass(frozen=True)
class ActionStruct:
    """1-indexed action: place `value` at (`row`, `col`)."""
    row: int
    col: int
    value: int

    @property
    def row0(self) -> int: return self.row - 1
    @property
    def col0(self) -> int: return self.col - 1

    def to_dict(self) -> dict:
        return {"row": self.row, "col": self.col, "value": self.value}


# --- State convenience ---

def _next_n(state: dict) -> Optional[int]:
    """Return the next number to place: smallest k in [1, R*C] not yet in assignment."""
    R, C = state["rows"], state["cols"]
    placed_values = set(state["assignment"].values())
    for k in range(1, R * C + 1):
        if k not in placed_values:
            return k
    return None  # all placed


def _anchor_cell(state: dict, target_n: Optional[int] = None) -> Optional[Tuple[int, int]]:
    """Return the (r, c) cell holding (target_n - 1). target_n defaults to state's next_n."""
    if target_n is None:
        target_n = _next_n(state)
    if target_n is None or target_n <= 1:
        return None
    for (r, c), v in state["assignment"].items():
        if v == target_n - 1:
            return (r, c)
    return None


def _adjacent_cells(r: int, c: int, R: int, C: int) -> List[Tuple[int, int]]:
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < R and 0 <= nc < C:
            out.append((nr, nc))
    return out


# --- Renderer ---

def render_state_hidato(state: dict) -> str:
    """Render Hidato state in the format used by HidatoEnv (minus history lines).

    Example:
        Hidato puzzle (5x4):
         1 .. .. ..
         2 .. .. ..
         9 .. .. ..
        .. .. .. ..
        17 .. .. ..

        Already placed: [1, 2, 9, 17]
        Next number to place: 3
        Must be adjacent to 2 at row 2 col 1.
    """
    R, C = state["rows"], state["cols"]
    asn = state["assignment"]
    n_total = R * C
    width = max(2, len(str(n_total)))   # cell width: ".." or right-padded number

    lines = [f"Hidato puzzle ({R}x{C}):"]
    for r in range(R):
        cells = []
        for c in range(C):
            v = asn.get((r, c))
            if v is None:
                cells.append(".." if width == 2 else "." * width)
            else:
                cells.append(str(v).rjust(width))
        lines.append(" ".join(cells))
    lines.append("")

    placed_sorted = sorted(asn.values())
    lines.append(f"Already placed: {placed_sorted}")

    nxt = _next_n(state)
    if nxt is None:
        lines.append("All numbers placed.")
    else:
        lines.append(f"Next number to place: {nxt}")
        anchor = _anchor_cell(state, nxt)
        if anchor is None:
            lines.append(f"(Number {nxt - 1} is not on the board; place {nxt} anywhere empty.)")
        else:
            lines.append(f"Must be adjacent to {nxt - 1} at row {anchor[0] + 1} col {anchor[1] + 1}.")

    return "\n".join(lines)


# --- Action utilities ---

_ACTION_RE = re.compile(
    r"place\s+(?P<value>\d+)\s+at\s+row\s+(?P<row>\d+)\s+col(?:umn)?\s+(?P<col>\d+)",
    re.IGNORECASE,
)


def parse_action_text(text: str) -> Optional[ActionStruct]:
    if not text:
        return None
    m = _ACTION_RE.search(text)
    if not m:
        return None
    try:
        value = int(m.group("value"))
        row = int(m.group("row"))
        col = int(m.group("col"))
    except ValueError:
        return None
    if value < 1 or row < 1 or col < 1 or row > 99 or col > 99 or value > 99:
        return None
    return ActionStruct(row=row, col=col, value=value)


def canonical_action(a: ActionStruct) -> str:
    return f"R{a.row}C{a.col}={a.value}"


def action_text(a: ActionStruct) -> str:
    return f"place {a.value} at row {a.row} col {a.col}"


def action_hash(a: ActionStruct) -> str:
    return "sha1:" + hashlib.sha1(canonical_action(a).encode()).hexdigest()


# --- Legality + transition ---

def is_local_valid(state: dict, action: ActionStruct) -> bool:
    """Local legality: action.value == state.next_n AND (action.row, action.col)
    is an empty cell adjacent to the predecessor's cell (or anywhere empty if
    predecessor doesn't exist on board)."""
    R, C = state["rows"], state["cols"]
    r0, c0 = action.row0, action.col0
    if not (0 <= r0 < R and 0 <= c0 < C):
        return False
    nxt = _next_n(state)
    if nxt is None:
        return False  # all placed already
    if action.value != nxt:
        return False
    if (r0, c0) in state["assignment"]:
        return False
    anchor = _anchor_cell(state, nxt)
    if anchor is None:
        # Predecessor not on board (e.g. placing 1 with no givens) — any empty
        # cell is legal.
        return True
    return (r0, c0) in _adjacent_cells(*anchor, R, C)


def apply_action(state: dict, action: ActionStruct) -> dict:
    """Return a new state with the action applied. Does NOT validate."""
    new_asn = dict(state["assignment"])
    new_asn[(action.row0, action.col0)] = action.value
    return {
        "rows": state["rows"],
        "cols": state["cols"],
        "assignment": new_asn,
    }


def is_goal(state: dict) -> bool:
    R, C = state["rows"], state["cols"]
    return len(state["assignment"]) == R * C


def enumerate_legal_actions(state: dict) -> List[ActionStruct]:
    """All legal placements at current state."""
    R, C = state["rows"], state["cols"]
    nxt = _next_n(state)
    if nxt is None:
        return []
    anchor = _anchor_cell(state, nxt)
    if anchor is None:
        # No predecessor on board — any empty cell is legal
        candidates = [(r, c) for r in range(R) for c in range(C)
                      if (r, c) not in state["assignment"]]
    else:
        candidates = [(r, c) for (r, c) in _adjacent_cells(*anchor, R, C)
                      if (r, c) not in state["assignment"]]
    return [ActionStruct(row=r + 1, col=c + 1, value=nxt) for (r, c) in candidates]


# --- State hash + puzzle ---

def state_hash(state: dict) -> str:
    """SHA-1 of canonical assignment representation."""
    items = sorted([(f"{r},{c}", v) for (r, c), v in state["assignment"].items()])
    canonical = json.dumps([state["rows"], state["cols"], items])
    return "sha1:" + hashlib.sha1(canonical.encode()).hexdigest()


def get_root_puzzle(seed: int) -> dict:
    """Pick a puzzle deterministically from the 8-puzzle bank.

    Returns state dict ready for SAVE generator: includes initial `assignment`
    (the puzzle's givens) and rows/cols.
    """
    if not _BANK:
        raise ValueError("Puzzle bank is empty")
    p = _BANK[seed % len(_BANK)]
    return {
        "rows": p["rows"],
        "cols": p["cols"],
        "assignment": dict(p["givens"]),
        "puzzle_id": p["id"],   # extra metadata (not part of schema state_struct)
    }


# --- Smoke ---

def _smoke():
    # 1) Render initial state of 5x4_snake
    state = get_root_puzzle(seed=7)   # 5x4_snake (last in bank)
    assert state["rows"] == 5 and state["cols"] == 4
    assert len(state["assignment"]) == 3   # 1, 9, 17 are givens
    txt = render_state_hidato(state)
    assert "Hidato puzzle (5x4):" in txt
    assert "Already placed:" in txt
    assert "Next number to place: 2" in txt   # 1 is given; next is 2
    print(f"  [1] get_root_puzzle(7) loads 5x4_snake; render OK")
    print("       --- render preview ---")
    for line in txt.split("\n"):
        print("       " + line)

    # 2) Parse action
    a = ActionStruct(row=2, col=1, value=2)
    txt2 = action_text(a)
    assert parse_action_text(txt2) == a
    print(f"  [2] parse round-trip OK: '{txt2}'")

    # 3) is_local_valid: place 2 at (2,1) which is row=2, col=1 (1-indexed)
    #    Anchor cell (where 1 lives in 5x4_snake) = (0, 0). Adjacent cells: (0,1),(1,0).
    #    So (2,1) which is (1,0) 0-indexed should be valid.
    a_valid = ActionStruct(row=2, col=1, value=2)   # cell (1, 0)
    a_invalid_far = ActionStruct(row=4, col=4, value=2)   # not adjacent
    a_invalid_value = ActionStruct(row=2, col=1, value=3)  # wrong value
    assert is_local_valid(state, a_valid)
    assert not is_local_valid(state, a_invalid_far)
    assert not is_local_valid(state, a_invalid_value)
    print(f"  [3] is_local_valid: valid={is_local_valid(state, a_valid)} "
          f"far={is_local_valid(state, a_invalid_far)} "
          f"wrong_value={is_local_valid(state, a_invalid_value)}")

    # 4) apply_action + immutability
    new_state = apply_action(state, a_valid)
    assert (1, 0) in new_state["assignment"] and new_state["assignment"][(1, 0)] == 2
    assert (1, 0) not in state["assignment"]   # original unchanged
    print(f"  [4] apply_action OK; original state immutable")

    # 5) enumerate_legal_actions
    legal = enumerate_legal_actions(state)
    assert len(legal) >= 1
    assert all(is_local_valid(state, a) for a in legal)
    print(f"  [5] enumerate_legal_actions: {len(legal)} legal placements: "
          f"{[(a.row, a.col, a.value) for a in legal]}")

    # 6) is_goal
    assert not is_goal(state)
    full_state = {"rows": 2, "cols": 2,
                  "assignment": {(0,0): 1, (0,1): 4, (1,0): 2, (1,1): 3}}
    assert is_goal(full_state)
    print(f"  [6] is_goal OK")

    # 7) state_hash deterministic + sensitive to assignment
    h1 = state_hash(state)
    h2 = state_hash(state)
    h3 = state_hash(new_state)
    assert h1 == h2 and h1 != h3
    print(f"  [7] state_hash: {h1[:20]}... (different from post-action hash ✓)")

    # 8) Cycle through all 8 puzzles
    for seed in range(8):
        p = get_root_puzzle(seed=seed)
        assert "puzzle_id" in p
        legal = enumerate_legal_actions(p)
        assert len(legal) >= 1, f"puzzle {p['puzzle_id']} has no legal actions"
    print(f"  [8] All 8 bank puzzles load + have legal actions")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    _smoke()
