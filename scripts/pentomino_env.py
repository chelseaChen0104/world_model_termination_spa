"""SAVE-side Pentomino environment helpers (board-size agnostic).

Wraps state/action utilities for the SAVE data generator. Reuses
PIECE_ORIENTATIONS and placement_cells from polyomino_utils WITHOUT
modifying it. Render format is the B-8/no-leak format used at training:
no "Last action: ... — board now unsolvable (...)" suffix. The doom-suffix
data leak that affected B-7/B-8 RL is explicitly avoided here.

Current target is **5×6 with 6-piece subsets** (172 valid configurations).
Functions are board-size agnostic; pass `(board, remaining_pieces)` and
the dimensions are derived from `len(board)` / `len(board[0])`.

Public surface:
    render_state_b8(board, remaining_pieces)        -> string (no Last action line)
    parse_action_text(text)                          -> ActionStruct or None
    canonical_action(action)                         -> "L:0:R1C1"-style key
    action_text(action)                              -> "place L ori=0 at row 1 col 1"
    is_local_valid(board, remaining_pieces, action)  -> bool
    apply_action(board, remaining_pieces, action)    -> (new_board, new_remaining)
    is_goal(board)                                   -> bool
    enumerate_legal_actions(board, remaining_pieces) -> list[ActionStruct]
    state_hash(board)                                -> "sha1:..."
    empty_board(h, w)                                -> List[List[str]]
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.environments.polyomino_utils import (
    PIECE_ORIENTATIONS, placement_cells, fits_on_board,
)


ENV_VERSION = "pentomino_env_v2"
TEXT_VERSION = "pentomino_text_b8_v1"  # B-8 board format minus the doom-suffix


@dataclass(frozen=True)
class ActionStruct:
    """1-indexed action: place `piece` at orientation `ori` with anchor at (`row`, `col`)."""
    piece: str
    ori: int
    row: int
    col: int

    @property
    def row0(self) -> int:
        return self.row - 1

    @property
    def col0(self) -> int:
        return self.col - 1

    def to_dict(self) -> dict:
        return {"piece": self.piece, "ori": self.ori, "row": self.row, "col": self.col}


# --- Renderer ---

def render_state_b8(board: List[List[str]], remaining_pieces: List[str]) -> str:
    """Render the board in the B-8 training format MINUS the doom-suffix leak.

        Current board (5x4):
        . . . .
        . . . .
        . . . .
        . . . .
        . . . .

        Remaining pieces: L, P, W, Y

    Notes:
    - Always shows "Current board (HxW)" header
    - 4-connectivity adjacency (single space separators between cells)
    - Empty cells '.', filled cells = piece letter
    - "Remaining pieces" line follows the board after a blank line
    - **No "Last action:" line** — that's where the doom-suffix leak lived;
      SAVE state rendering omits it entirely (state is rendered fresh each turn)
    """
    h = len(board)
    w = len(board[0]) if h else 0
    rows = [f"Current board ({h}x{w}):"]
    for r in range(h):
        rows.append(" ".join(board[r][c] for c in range(w)))
    rows.append("")
    if remaining_pieces:
        rows.append(f"Remaining pieces: {', '.join(remaining_pieces)}")
    else:
        rows.append("Remaining pieces: (none)")
    return "\n".join(rows)


# --- Action utilities ---

_ACTION_RE = re.compile(
    r"place\s+(?P<piece>[A-Z])\s+ori\s*=\s*(?P<ori>\d+)\s+at\s+row\s+(?P<row>\d+)\s+col(?:umn)?\s+(?P<col>\d+)",
    re.IGNORECASE,
)


def parse_action_text(text: str) -> Optional[ActionStruct]:
    """Parse 'place L ori=0 at row 1 col 1' (or 'column 1') -> ActionStruct."""
    if not text:
        return None
    m = _ACTION_RE.search(text)
    if not m:
        return None
    piece = m.group("piece").upper()
    try:
        ori = int(m.group("ori"))
        row = int(m.group("row"))
        col = int(m.group("col"))
    except ValueError:
        return None
    if piece not in PIECE_ORIENTATIONS:
        return None
    if not (0 <= ori < len(PIECE_ORIENTATIONS[piece])):
        return None
    if not (1 <= row <= 12 and 1 <= col <= 12):
        return None
    return ActionStruct(piece=piece, ori=ori, row=row, col=col)


def canonical_action(action: ActionStruct) -> str:
    """Compact canonical form for dedup. e.g. L:0:R1C1 ."""
    return f"{action.piece}:{action.ori}:R{action.row}C{action.col}"


def action_text(action: ActionStruct) -> str:
    return f"place {action.piece} ori={action.ori} at row {action.row} col {action.col}"


def action_hash(action: ActionStruct) -> str:
    return "sha1:" + hashlib.sha1(canonical_action(action).encode()).hexdigest()


# --- Legality + transition ---

def is_local_valid(board: List[List[str]], remaining_pieces: List[str], action: ActionStruct) -> bool:
    """Local legality:
    - piece is still in remaining_pieces
    - placement cells are all in-bounds
    - placement cells are all currently empty
    """
    if action.piece not in remaining_pieces:
        return False
    cells = placement_cells(action.piece, action.ori, action.row0, action.col0)
    if cells is None:
        return False
    return fits_on_board(cells, len(board), len(board[0]) if board else 0, board)


def apply_action(board: List[List[str]], remaining_pieces: List[str],
                  action: ActionStruct) -> Tuple[List[List[str]], List[str]]:
    """Return new (board, remaining_pieces) with the action applied.

    Does NOT validate legality; call is_local_valid first if needed.
    """
    new_board = [row[:] for row in board]
    cells = placement_cells(action.piece, action.ori, action.row0, action.col0)
    for r, c in cells:
        new_board[r][c] = action.piece
    new_remaining = [p for p in remaining_pieces if p != action.piece]
    return new_board, new_remaining


def is_goal(board: List[List[str]]) -> bool:
    """Board is fully tiled (no '.' cells)."""
    return all(c != "." for row in board for c in row)


def enumerate_legal_actions(board: List[List[str]],
                             remaining_pieces: List[str]) -> List[ActionStruct]:
    """All legal placements at current state."""
    out = []
    h = len(board)
    w = len(board[0]) if h else 0
    for piece in remaining_pieces:
        for ori_id, ori in enumerate(PIECE_ORIENTATIONS[piece]):
            for ar in range(h):
                for ac in range(w):
                    cells = placement_cells(piece, ori_id, ar, ac)
                    if cells and fits_on_board(cells, h, w, board):
                        out.append(ActionStruct(piece=piece, ori=ori_id, row=ar + 1, col=ac + 1))
    return out


# --- State hashing + puzzle ---

def state_hash(board: List[List[str]], remaining_pieces: List[str]) -> str:
    """SHA-1 of canonical state representation (board + remaining)."""
    canonical = json.dumps([board, list(remaining_pieces)], separators=(",", ":"))
    return "sha1:" + hashlib.sha1(canonical.encode()).hexdigest()


def empty_board(h: int, w: int) -> List[List[str]]:
    """Helper: an empty H×W board with all cells '.'."""
    return [["."] * w for _ in range(h)]


# --- Smoke test ---

def _smoke():
    # 1) Initial render of empty 5×6 with FILNPT (one of the 172 valid 6-piece subsets)
    board = empty_board(5, 6)
    remaining = ["F", "I", "L", "N", "P", "T"]
    rendered = render_state_b8(board, remaining)
    expected_lines = [
        "Current board (5x6):",
        ". . . . . .",
        ". . . . . .",
        ". . . . . .",
        ". . . . . .",
        ". . . . . .",
        "",
        "Remaining pieces: F, I, L, N, P, T",
    ]
    assert rendered == "\n".join(expected_lines), f"\nGOT:\n{rendered}"
    print("  [1] empty 5×6 FILNPT render matches expected format")

    # 2) parse_action_text round-trip
    a = ActionStruct(piece="L", ori=0, row=1, col=1)
    txt = action_text(a)
    parsed = parse_action_text(txt)
    assert parsed == a, f"got {parsed}, expected {a}"
    parsed2 = parse_action_text("place L ori=0 at row 1 column 1")
    assert parsed2 == a
    print(f"  [2] parse round-trip OK: '{txt}'")

    # 3) Canonical action + hash
    assert canonical_action(a) == "L:0:R1C1"
    h = action_hash(a)
    assert h.startswith("sha1:") and len(h) == 5 + 40
    print(f"  [3] canonical='{canonical_action(a)}', hash={h[:20]}...")

    # 4) is_local_valid + apply_action
    assert is_local_valid(board, remaining, a) is True
    new_board, new_remaining = apply_action(board, remaining, a)
    assert "L" not in new_remaining
    assert any(c == "L" for row in new_board for c in row)
    assert all(c == "." for row in board for c in row)
    assert "L" in remaining
    print("  [4] is_local_valid + apply_action OK; input immutability preserved")

    # 5) Bad placement (out of bounds)
    bad_action = ActionStruct(piece="L", ori=0, row=20, col=20)
    assert not is_local_valid(board, remaining, bad_action)
    print("  [5] out-of-bounds placement rejected")

    # 6) Piece not in remaining (after L is placed)
    a_l_again = ActionStruct(piece="L", ori=0, row=2, col=3)
    assert not is_local_valid(new_board, new_remaining, a_l_again)
    print("  [6] cannot re-place a piece already used")

    # 7) enumerate_legal_actions returns valid actions
    legal = enumerate_legal_actions(board, remaining)
    assert all(is_local_valid(board, remaining, a) for a in legal)
    print(f"  [7] {len(legal)} legal actions on empty 5×6 FILNPT")

    # 8) is_goal
    assert not is_goal(board)
    full = [["L"] * 6 for _ in range(5)]
    assert is_goal(full)
    print("  [8] is_goal OK")

    # 9) state_hash deterministic + collision-free for different states
    h1 = state_hash(board, remaining)
    h2 = state_hash(board, remaining)
    h3 = state_hash(new_board, new_remaining)
    assert h1 == h2 and h1 != h3
    print("  [9] state_hash deterministic + distinguishes states")

    # 10) Backward compat: 5×4 board still works
    b4 = empty_board(5, 4)
    rendered_4 = render_state_b8(b4, ["L", "P", "W", "Y"])
    assert "Current board (5x4):" in rendered_4
    print("  [10] legacy 5×4 board render still works")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    _smoke()
