"""Polyomino (pentomino) tiling environment for termination prediction.

Implements the env API laid out in doc/spec_2026-04-29_pentomino.md.

State: 2D board (chars) + remaining pieces.
Action: "place {piece} ori={K} at row {R} col {C}" — anchor lands at (R-1, C-1).
Termination: solved (board fully tiled), deadlock (oracle says no tiling possible),
             or step limit hit.

Compatible with `BaseTerminationEnv` (info dict mirrors SudokuEnv conventions).
"""
from __future__ import annotations
import re
import random
from typing import Tuple, Dict, List, Optional, Any

from src.environments.base import BaseTerminationEnv
from src.environments.polyomino_utils import (
    ALL_PIECES,
    PIECE_ORIENTATIONS,
    placement_cells,
    fits_on_board,
    PolyominoSolvabilityChecker,
)


# ----------------------------------------------------------------------------
# Action parsing — "place {piece} ori={K} at row {R} col {C}"
# ----------------------------------------------------------------------------

# Accept a few near-equivalent formats for robustness with LLM outputs:
#   place L ori=2 at row 1 col 1
#   place L orientation 2 at row 1 col 1
#   place L (ori=2) at (1, 1)
_ACTION_PATTERNS = [
    re.compile(
        r"place\s+([A-Za-z])\s*(?:ori(?:entation)?\s*[=:]?\s*|\(ori\s*=\s*)?(\d+)"
        r"\)?\s*(?:at|@)?\s*(?:row\s*|\(\s*)?(\d+)\s*(?:,|col\s*)\s*(\d+)\)?",
        re.IGNORECASE,
    ),
]


def parse_action(action: str) -> Optional[Tuple[str, int, int, int]]:
    """Return (piece_letter, ori_id, row_0idx, col_0idx) or None.

    All inputs are 1-indexed (matching the rendered board); we convert to 0-indexed here.
    """
    s = action.strip()
    for pat in _ACTION_PATTERNS:
        m = pat.search(s)
        if m:
            piece = m.group(1).upper()
            try:
                ori_id = int(m.group(2))
                r = int(m.group(3)) - 1
                c = int(m.group(4)) - 1
            except ValueError:
                continue
            return piece, ori_id, r, c
    return None


# ----------------------------------------------------------------------------
# Env
# ----------------------------------------------------------------------------

class PolyominoEnv(BaseTerminationEnv):
    """Pentomino tiling env. Default config: 5×4 easy variant with {L, P, W, Y}.

    Args:
        board_h, board_w: board dimensions
        piece_set: tuple of piece letters in the puzzle. Each used exactly once.
        max_steps: cap on actions (defaults to 2 × len(piece_set), generous safety margin)
        max_dlx_depth: backtracking depth limit for the oracle
    """

    def __init__(
        self,
        board_h: int = 5,
        board_w: int = 4,
        piece_set: Tuple[str, ...] = ('L', 'P', 'W', 'Y'),
        max_steps: Optional[int] = None,
        max_dlx_depth: int = 50,
    ):
        # Validate inputs
        for p in piece_set:
            if p not in PIECE_ORIENTATIONS:
                raise ValueError(f"unknown piece letter: {p}")
        if board_h * board_w != 5 * len(piece_set):
            raise ValueError(
                f"board area {board_h}×{board_w}={board_h*board_w} must equal "
                f"5 × {len(piece_set)} = {5*len(piece_set)} (one cell per piece × 5)"
            )

        self.board_h = board_h
        self.board_w = board_w
        self.initial_pieces = tuple(piece_set)
        self.max_steps = max_steps if max_steps is not None else 2 * len(piece_set)
        self.checker = PolyominoSolvabilityChecker(max_depth=max_dlx_depth)

        # State
        self.board: List[List[str]] = []
        self.remaining_pieces: List[str] = []
        self.placed: List[Tuple[str, int, int, int]] = []  # (piece, ori, anchor_r, anchor_c)
        self.num_steps = 0
        self.last_action_feedback = ""
        self._last_solvability: Optional[bool] = None

    # ------------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> str:
        """Reset board to empty. Seed only affects piece-order randomization in get_all_actions."""
        if seed is not None:
            random.seed(seed)
        self.board = [['.'] * self.board_w for _ in range(self.board_h)]
        self.remaining_pieces = list(self.initial_pieces)
        self.placed = []
        self.num_steps = 0
        self.last_action_feedback = ""
        self._last_solvability = None
        return self.render()

    def step(self, action) -> Tuple[str, float, bool, Dict[str, Any]]:
        self.num_steps += 1
        prev_solvable, _ = self.check_solvability()

        # Parse
        if isinstance(action, int):
            all_acts = self.get_all_actions()
            if 0 <= action < len(all_acts):
                action_str = all_acts[action]
            else:
                return self._invalid_step(f"action index {action} out of range")
        else:
            action_str = str(action)

        parsed = parse_action(action_str)
        if parsed is None:
            return self._invalid_step(f"could not parse: '{action_str}'")
        piece, ori_id, r, c = parsed

        if piece not in self.remaining_pieces:
            return self._invalid_step(f"piece {piece} is not in remaining set {self.remaining_pieces}")

        cells = placement_cells(piece, ori_id, r, c)
        if cells is None:
            return self._invalid_step(f"invalid orientation {ori_id} for piece {piece}")

        if not fits_on_board(cells, self.board_h, self.board_w, self.board):
            return self._invalid_step(
                f"placement of {piece} ori={ori_id} at ({r+1},{c+1}) doesn't fit / overlaps"
            )

        # Place
        for ar, ac in cells:
            self.board[ar][ac] = piece
        self.remaining_pieces.remove(piece)
        self.placed.append((piece, ori_id, r, c))
        self._last_solvability = None  # invalidate cache

        new_solvable, reason = self.check_solvability()
        is_breaking_point = bool(prev_solvable and not new_solvable)

        # Solved if all cells occupied AND no remaining pieces
        solved = (
            len(self.remaining_pieces) == 0 and
            all(self.board[rr][cc] != '.' for rr in range(self.board_h) for cc in range(self.board_w))
        )

        done = solved or (self.num_steps >= self.max_steps) or (not new_solvable)
        reward = 1.0 if solved else (0.0 if new_solvable else -0.5)

        self.last_action_feedback = (
            f"placed {piece} ori={ori_id} at row {r+1} col {c+1}"
            + (" — solved!" if solved else "")
            + ("" if new_solvable or solved else f" — board now unsolvable ({reason})")
        )

        info = {
            "action_is_valid": True,
            "action_is_effective": True,
            "success": solved,
            "is_solvable": bool(new_solvable),
            "deadlock_type": reason if not new_solvable else None,
            "is_breaking_point": is_breaking_point,
            "action_name": f"place {piece} ori={ori_id} at row {r+1} col {c+1}",
            "remaining_pieces": list(self.remaining_pieces),
            "step": self.num_steps,
        }
        return self.render(), reward, done, info

    def render(self) -> str:
        rows = []
        rows.append(f"Current board ({self.board_h}x{self.board_w}):")
        for r in range(self.board_h):
            rows.append(' '.join(self.board[r]))
        rows.append("")
        rows.append(f"Remaining pieces: {', '.join(self.remaining_pieces) if self.remaining_pieces else '(none)'}")
        if self.last_action_feedback:
            rows.append(f"Last action: {self.last_action_feedback}")
        return '\n'.join(rows)

    def check_solvability(self) -> Tuple[bool, Optional[str]]:
        return self.checker.check_solvability(self.board, self.remaining_pieces)

    def get_all_actions(self) -> List[str]:
        """All valid placements at the current state — '(piece, ori, anchor_r, anchor_c)' tuples
        formatted as action strings."""
        actions = []
        for piece in self.remaining_pieces:
            for ori_id, ori in enumerate(PIECE_ORIENTATIONS[piece]):
                for ar in range(self.board_h):
                    for ac in range(self.board_w):
                        cells = placement_cells(piece, ori_id, ar, ac)
                        if cells is None:
                            continue
                        if fits_on_board(cells, self.board_h, self.board_w, self.board):
                            actions.append(f"place {piece} ori={ori_id} at row {ar+1} col {ac+1}")
        return actions

    def get_state_info(self) -> Dict[str, Any]:
        is_solvable, reason = self.check_solvability()
        return {
            "board_h": self.board_h,
            "board_w": self.board_w,
            "remaining_pieces": list(self.remaining_pieces),
            "placed": list(self.placed),
            "num_steps": self.num_steps,
            "is_solvable": bool(is_solvable),
            "deadlock_type": reason if not is_solvable else None,
            "n_valid_actions": len(self.get_all_actions()),
        }

    # ------------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------------

    def _invalid_step(self, msg: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Return result for an invalid action: penalty -0.1, episode continues."""
        self.last_action_feedback = f"Invalid: {msg}"
        is_solvable, reason = self.check_solvability()
        info = {
            "action_is_valid": False,
            "action_is_effective": False,
            "success": False,
            "is_solvable": bool(is_solvable),
            "deadlock_type": reason if not is_solvable else None,
            "is_breaking_point": False,
            "action_name": "invalid",
            "error": msg,
            "remaining_pieces": list(self.remaining_pieces),
            "step": self.num_steps,
        }
        return self.render(), -0.1, False, info


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------

def _self_test():
    env = PolyominoEnv()  # 5×4, {L, P, W, Y}
    obs = env.reset(seed=42)
    print("Initial observation:")
    print(obs)
    print()

    n_actions = len(env.get_all_actions())
    print(f"Valid actions from empty board: {n_actions}")

    # First action: place L ori=0 at corner
    obs, r, done, info = env.step("place L ori=0 at row 1 col 1")
    print("After placing L ori=0 at (1,1):")
    print(obs)
    print(f"  reward={r}, done={done}, is_solvable={info['is_solvable']}, "
          f"is_bp={info['is_breaking_point']}, n_valid_next={env.get_state_info()['n_valid_actions']}")
    print()

    # Try an invalid placement (overlapping)
    obs, r, done, info = env.step("place P ori=0 at row 1 col 1")  # would overlap L
    assert not info['action_is_valid'], "should be invalid (overlap)"
    print(f"Invalid action correctly rejected: {info.get('error')}")

    # Try unparseable
    obs, r, done, info = env.step("place a piece somewhere")
    assert not info['action_is_valid']
    print(f"Unparseable action correctly rejected.")

    print("\nAll PolyominoEnv self-tests passed.")


if __name__ == "__main__":
    _self_test()
