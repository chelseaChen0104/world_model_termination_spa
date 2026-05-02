"""HidatoEnv — environment for the Hidato (Numbrix variant) puzzle.

A grid of integers in [1, R*C], some pre-filled (givens), the rest to be
filled by the model in sequence so that consecutive numbers are orthogonally
adjacent.

Mirrors the API conventions of PolyominoEnv / SudokuEnv:
  - reset(seed) returns rendered str
  - step(action_str) returns (obs, reward, done, info) where info has
    is_solvable, action_is_valid, success, etc.
  - render(), check_solvability(), get_all_actions(), get_state_info()
"""
from __future__ import annotations
import random
import re
from typing import Tuple, Dict, List, Optional, Any

from .base import BaseTerminationEnv
from .hidato_utils import adjacent_cells, is_solvable, verify_solution
from . import hidato_puzzle_bank as bank


# Action regex: 'place {N} at row {R} col {C}' (1-indexed)
_RE_ACTION = re.compile(
    r"place\s+(\d+)\s+at\s+row\s+(\d+)\s+col\s+(\d+)",
    re.IGNORECASE,
)


def parse_action(action_str: str) -> Optional[tuple]:
    """Returns (n, r_0idx, c_0idx) or None if unparseable."""
    m = _RE_ACTION.search(action_str)
    if not m:
        return None
    n, r1, c1 = map(int, m.groups())
    return (n, r1 - 1, c1 - 1)


class HidatoEnv(BaseTerminationEnv):
    """Hidato (Numbrix variant) environment.

    The puzzle is selected from the puzzle bank by `seed`. The model places
    numbers 1, 2, 3, ..., R*C in sequence; each must go in a cell adjacent
    to the previous (or any cell if 1 was not given). Givens are pre-filled
    and don't need to be placed.
    """

    def __init__(self, puzzle_bank: Optional[List[dict]] = None,
                 max_steps: Optional[int] = None):
        self.puzzle_bank = puzzle_bank if puzzle_bank is not None else bank.PUZZLES
        if not self.puzzle_bank:
            raise ValueError("Empty puzzle bank")
        self.max_steps = max_steps  # set during reset based on the puzzle

        # State (set on reset)
        self.puzzle: Optional[dict] = None
        self.rows: int = 0
        self.cols: int = 0
        self.assignment: Dict[tuple, int] = {}   # (r, c) -> int
        self.num_steps: int = 0
        self.last_action_feedback: str = ""
        self._last_solvability: Optional[Tuple[bool, Optional[str]]] = None

    # --- BaseTerminationEnv interface --------------------------------------

    def reset(self, seed: Optional[int] = None) -> str:
        rng = random.Random(seed)
        self.puzzle = self.puzzle_bank[rng.randrange(len(self.puzzle_bank))]
        self.rows = self.puzzle["rows"]
        self.cols = self.puzzle["cols"]
        self.assignment = dict(self.puzzle["givens"])
        self.num_steps = 0
        self.last_action_feedback = ""
        self._last_solvability = None
        if self.max_steps is None:
            # default: 2 × number of empty cells (generous safety margin)
            n_empty = self.rows * self.cols - len(self.assignment)
            self.max_steps = max(8, 2 * n_empty)
        return self.render()

    def step(self, action) -> Tuple[str, float, bool, Dict[str, Any]]:
        self.num_steps += 1
        prev_solvable, _ = self.check_solvability()

        action_str = str(action)
        parsed = parse_action(action_str)
        if parsed is None:
            return self._invalid_step(f"could not parse: '{action_str}'")
        n, r, c = parsed

        # Bounds
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return self._invalid_step(f"cell ({r+1},{c+1}) out of bounds")
        # Cell must be empty
        if (r, c) in self.assignment:
            return self._invalid_step(
                f"cell ({r+1},{c+1}) already has number {self.assignment[(r, c)]}"
            )
        # n must be in valid range
        n_cells = self.rows * self.cols
        if not (1 <= n <= n_cells):
            return self._invalid_step(f"number {n} out of range [1, {n_cells}]")
        # n must not already be placed
        if n in self.assignment.values():
            return self._invalid_step(f"number {n} is already placed")
        # n must equal next-required-number = max-currently-placed + 1, OR
        # n must equal min-unplaced (if there's a "hole" before max)
        next_required = self._next_required_number()
        if n != next_required:
            return self._invalid_step(
                f"number {n} not next; expected {next_required}"
            )
        # If n > 1, must be adjacent to where (n-1) is
        if n > 1:
            prev_cell = self._locate(n - 1)
            if prev_cell is None:
                return self._invalid_step(f"predecessor {n-1} is not on the board")
            if (r, c) not in adjacent_cells(*prev_cell, self.rows, self.cols):
                return self._invalid_step(
                    f"cell ({r+1},{c+1}) not adjacent to {n-1} at "
                    f"({prev_cell[0]+1},{prev_cell[1]+1})"
                )
        # If n+1 is already placed (a future given), it must be adjacent too
        if n + 1 in self.assignment.values():
            next_cell = self._locate(n + 1)
            if (r, c) not in adjacent_cells(*next_cell, self.rows, self.cols):
                return self._invalid_step(
                    f"cell ({r+1},{c+1}) not adjacent to next-given {n+1} at "
                    f"({next_cell[0]+1},{next_cell[1]+1})"
                )

        # Apply
        self.assignment[(r, c)] = n
        self._last_solvability = None  # invalidate cache

        new_solvable, reason = self.check_solvability()
        is_breaking_point = bool(prev_solvable and not new_solvable)

        # Solved if all cells assigned AND solution validates
        solved = (len(self.assignment) == n_cells)
        if solved:
            ok, _ = verify_solution(self.rows, self.cols, self.assignment)
            solved = ok

        done = solved or (self.num_steps >= self.max_steps) or (not new_solvable)
        reward = 1.0 if solved else (0.0 if new_solvable else -0.5)

        self.last_action_feedback = (
            f"placed {n} at row {r+1} col {c+1}"
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
            "action_name": f"place {n} at row {r+1} col {c+1}",
            "step": self.num_steps,
            "next_required": self._next_required_number(),
        }
        return self.render(), reward, done, info

    def render(self) -> str:
        rows = []
        rows.append(f"Hidato puzzle ({self.rows}x{self.cols}):")
        # Compute width per cell from largest possible number
        n_cells = self.rows * self.cols
        cell_w = max(2, len(str(n_cells)))
        for r in range(self.rows):
            row_strs = []
            for c in range(self.cols):
                if (r, c) in self.assignment:
                    row_strs.append(str(self.assignment[(r, c)]).rjust(cell_w))
                else:
                    row_strs.append("." * cell_w)
            rows.append(" ".join(row_strs))
        rows.append("")
        rows.append("Rules: place each remaining number 1..N in sequence such that")
        rows.append("consecutive numbers are orthogonally adjacent (share an edge).")
        # State summary
        next_n = self._next_required_number()
        if next_n is not None and next_n <= n_cells:
            placed = sorted(self.assignment.values())
            rows.append(f"Already placed: {placed}")
            rows.append(f"Next number to place: {next_n}")
            if next_n > 1:
                prev_pos = self._locate(next_n - 1)
                if prev_pos is not None:
                    rows.append(
                        f"Must be adjacent to {next_n-1} at "
                        f"row {prev_pos[0]+1} col {prev_pos[1]+1}."
                    )
        else:
            rows.append("All numbers placed.")
        if self.last_action_feedback:
            rows.append(f"Last action: {self.last_action_feedback}")
        return "\n".join(rows)

    def check_solvability(self) -> Tuple[bool, Optional[str]]:
        if self._last_solvability is not None:
            return self._last_solvability
        ok, reason = is_solvable(self.rows, self.cols, dict(self.assignment))
        self._last_solvability = (ok, reason)
        return self._last_solvability

    def get_all_actions(self) -> List[str]:
        """Return list of all valid 'place {n} at row {r} col {c}' action
        strings for the current state. The next-required-number is determined
        automatically; the model just picks the cell."""
        next_n = self._next_required_number()
        if next_n is None or next_n > self.rows * self.cols:
            return []
        n_cells = self.rows * self.cols
        if next_n in self.assignment.values():
            # next number is already a given (rare edge case after sequential placement)
            return []
        candidate_cells = self._candidate_cells_for(next_n)
        return [
            f"place {next_n} at row {r+1} col {c+1}"
            for (r, c) in candidate_cells
        ]

    def get_state_info(self) -> Dict[str, Any]:
        n_cells = self.rows * self.cols
        return {
            "puzzle_id": self.puzzle["id"] if self.puzzle else None,
            "rows": self.rows,
            "cols": self.cols,
            "n_filled": len(self.assignment),
            "n_empty": n_cells - len(self.assignment),
            "next_required": self._next_required_number(),
            "num_steps": self.num_steps,
        }

    # --- internal helpers --------------------------------------------------

    def _next_required_number(self) -> Optional[int]:
        """Smallest integer 1..R*C not yet placed."""
        n_cells = self.rows * self.cols
        placed = set(self.assignment.values())
        for k in range(1, n_cells + 1):
            if k not in placed:
                return k
        return None

    def _locate(self, n: int) -> Optional[tuple]:
        for pos, v in self.assignment.items():
            if v == n:
                return pos
        return None

    def _candidate_cells_for(self, n: int) -> List[tuple]:
        """Cells where number n could be placed *legally given the immediate
        neighbor constraints* (n must be adjacent to n-1 if placed, and to
        n+1 if it's already placed)."""
        empty = [(r, c) for r in range(self.rows) for c in range(self.cols)
                 if (r, c) not in self.assignment]
        if n == 1:
            # First number — restrict only by n+1 if that's already placed
            adj_to_next = self._locate(n + 1)
            if adj_to_next is not None:
                neighbors = adjacent_cells(*adj_to_next, self.rows, self.cols)
                empty = [c for c in empty if c in neighbors]
            return empty
        # Else: must be adjacent to n-1
        prev_cell = self._locate(n - 1)
        if prev_cell is None:
            return []
        candidates = [c for c in empty
                      if c in adjacent_cells(*prev_cell, self.rows, self.cols)]
        # Also require adjacency to n+1 if placed
        next_cell = self._locate(n + 1)
        if next_cell is not None:
            next_neighbors = adjacent_cells(*next_cell, self.rows, self.cols)
            candidates = [c for c in candidates if c in next_neighbors]
        return candidates

    def _invalid_step(self, reason: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        self.last_action_feedback = f"INVALID: {reason}"
        info = {
            "action_is_valid": False,
            "action_is_effective": False,
            "success": False,
            "is_solvable": True,  # state didn't change; assume previous solvability
            "deadlock_type": None,
            "is_breaking_point": False,
            "action_name": "INVALID",
            "step": self.num_steps,
            "next_required": self._next_required_number(),
        }
        return self.render(), -0.1, False, info


# ----------------------------------------------------------------------------
# Smoke test (run with: python -m src.environments.hidato)
# ----------------------------------------------------------------------------

def _smoke():
    env = HidatoEnv()
    obs = env.reset(seed=0)
    print("=== Initial state ===")
    print(obs)
    print(f"\nState info: {env.get_state_info()}")
    print(f"\nValid actions: {env.get_all_actions()}")

    # Try a step
    actions = env.get_all_actions()
    if actions:
        a = actions[0]
        print(f"\n=== Stepping with: {a} ===")
        obs, reward, done, info = env.step(a)
        print(obs)
        print(f"\nreward={reward}, done={done}, info={info}")


if __name__ == "__main__":
    _smoke()
