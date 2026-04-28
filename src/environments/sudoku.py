"""
Sudoku Environment for Termination Prediction

Implements BaseTerminationEnv for Sudoku with solvability detection.
Unlike Sokoban where simple pattern-matching detects deadlocks,
Sudoku solvability requires constraint propagation + backtracking,
making it the strongest testbed for LLM-based termination prediction.

Breaking point: a placement that transitions the grid from solvable to unsolvable.
This happens when a wrong number eliminates all candidates for some other cell(s),
creating a hidden constraint violation.
"""

import random
import re
import numpy as np
from typing import Tuple, Dict, List, Optional, Any

from src.environments.base import BaseTerminationEnv
from src.environments.sudoku_utils import (
    generate_sudoku_puzzle,
    is_valid_placement,
    get_valid_numbers,
    find_conflicts,
    is_solved,
    format_grid,
    SudokuSolvabilityChecker,
)


class SudokuEnv(BaseTerminationEnv):
    """Sudoku environment with solvability checking for termination prediction.

    Action format: "place N at row R col C" or "R,C,N"
    State: text grid with box separators

    Key difference from Sokoban: unsolvable states are NOT obvious from
    visual inspection. A wrong placement may look valid (no immediate conflict)
    but make the puzzle unsolvable through hidden constraint violations.
    """

    # All possible actions: place 1-9 at any of 81 cells
    # But for trajectory generation, we use a simplified action space
    ACTION_LOOKUP = {}  # Populated dynamically

    def __init__(
        self,
        grid_size: int = 9,
        difficulty: str = "easy",
        max_steps: int = 81,
        render_format: str = "simple",
        max_backtrack_depth: int = 30,
    ):
        self.grid_size = grid_size
        self.difficulty = difficulty
        self.max_steps = max_steps
        self.render_format = render_format

        self.current_grid = None
        self.initial_grid = None
        self.solution_grid = None
        self.num_steps = 0
        self.last_action_feedback = ""

        self.solvability_checker = SudokuSolvabilityChecker(
            max_backtrack_depth=max_backtrack_depth
        )

        # Cache solvability result (expensive to compute)
        self._last_solvability: Optional[Tuple[bool, Optional[str]]] = None

    def reset(self, seed: Optional[int] = None) -> str:
        """Reset with a new random puzzle."""
        self.initial_grid, self.solution_grid = generate_sudoku_puzzle(
            grid_size=self.grid_size,
            difficulty=self.difficulty,
            seed=seed,
        )
        self.current_grid = self.initial_grid.copy()
        self.num_steps = 0
        self.last_action_feedback = ""
        self._last_solvability = None
        return self.render()

    def step(self, action) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Execute a placement action.

        Action can be:
        - int index into get_all_actions() list
        - string like "place 5 at row 2 col 3" or "2,3,5"
        """
        self.num_steps += 1
        prev_solvable, _ = self.check_solvability()

        # Parse action
        if isinstance(action, int):
            # Index into all_actions
            all_acts = self.get_all_actions()
            if 0 <= action < len(all_acts):
                action_str = all_acts[action]
            else:
                return self._invalid_step(f"Action index {action} out of range")
        else:
            action_str = str(action)

        success, row, col, num, error_msg = self._parse_action(action_str)

        if not success:
            return self._invalid_step(error_msg)

        # Check if cell is modifiable
        if self.initial_grid[row, col] != 0:
            return self._invalid_step(f"Cannot modify initial cell at ({row+1},{col+1})")

        # Check if cell already has a value
        if self.current_grid[row, col] != 0:
            return self._invalid_step(
                f"Cell ({row+1},{col+1}) already has value {self.current_grid[row, col]}"
            )

        # Check Sudoku rule validity (no immediate conflict)
        if not is_valid_placement(self.current_grid, row, col, num):
            return self._conflict_step(row, col, num)

        # Place the number
        self.current_grid[row, col] = num
        self._last_solvability = None  # Invalidate cache

        # Check solvability after placement
        new_solvable, reason = self.check_solvability()
        is_breaking_point = prev_solvable and not new_solvable

        # Check if correct according to solution
        correct = (num == self.solution_grid[row, col])
        solved = is_solved(self.current_grid)

        # Determine reward
        if solved:
            reward = 10.0
            self.last_action_feedback = f"Placed {num} at ({row+1},{col+1}). Puzzle solved!"
        elif correct:
            reward = 1.0
            self.last_action_feedback = f"Placed {num} at ({row+1},{col+1}). Correct!"
        elif is_breaking_point:
            reward = -1.0
            self.last_action_feedback = f"Placed {num} at ({row+1},{col+1}). Puzzle now unsolvable!"
        else:
            reward = 0.5  # Valid but not matching solution
            self.last_action_feedback = f"Placed {num} at ({row+1},{col+1}). Valid placement."

        done = solved or (self.num_steps >= self.max_steps)

        info = {
            "action_is_valid": True,
            "action_is_effective": True,
            "success": solved,
            "is_solvable": new_solvable,
            "deadlock_type": reason if not new_solvable else None,
            "is_breaking_point": is_breaking_point,
            "correct_placement": correct,
            "cells_remaining": int(np.count_nonzero(self.current_grid == 0)),
            "action_name": action_str,
        }

        return self.render(), reward, done, info

    def render(self) -> str:
        """Return current grid as text."""
        return format_grid(self.current_grid, self.initial_grid)

    def check_solvability(self) -> Tuple[bool, Optional[str]]:
        """Check if current grid can be completed to a valid solution."""
        if self._last_solvability is not None:
            return self._last_solvability
        result = self.solvability_checker.check_solvability(self.current_grid)
        self._last_solvability = result
        return result

    def get_all_actions(self) -> list:
        """Return list of valid action strings for random play.

        Returns actions for all empty cells with all valid numbers.
        """
        actions = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.current_grid[r, c] == 0 and self.initial_grid[r, c] == 0:
                    valid_nums = get_valid_numbers(self.current_grid, r, c)
                    for num in sorted(valid_nums):
                        actions.append(f"place {num} at row {r+1} col {c+1}")
        return actions

    def get_state_info(self) -> Dict[str, Any]:
        """Return structured state metadata."""
        is_solvable, reason = self.check_solvability()
        filled = int(np.count_nonzero(self.current_grid))
        initial_filled = int(np.count_nonzero(self.initial_grid))
        conflicts = find_conflicts(self.current_grid)

        return {
            "grid_size": self.grid_size,
            "difficulty": self.difficulty,
            "cells_filled": filled,
            "cells_initial": initial_filled,
            "cells_placed": filled - initial_filled,
            "cells_remaining": self.grid_size * self.grid_size - filled,
            "is_solvable": is_solvable,
            "unsolvable_reason": reason,
            "num_conflicts": len(conflicts),
            "num_steps": self.num_steps,
        }

    def _parse_action(self, action: str) -> Tuple[bool, int, int, int, str]:
        """Parse action string. Returns (success, row, col, num, error_msg)."""
        action = action.strip().lower()

        patterns = [
            r'place\s+(\d+)\s+at\s+row\s+(\d+)\s+col\s+(\d+)',
            r'place\s+(\d+)\s+at\s+\((\d+),\s*(\d+)\)',
            r'place\s+(\d+)\s+at\s+(\d+),\s*(\d+)',
            r'(\d+)\s+at\s+(\d+),\s*(\d+)',
            r'\((\d+),\s*(\d+),\s*(\d+)\)',
            r'(\d+),\s*(\d+),\s*(\d+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, action)
            if match:
                groups = match.groups()
                if len(groups) == 3:
                    if 'place' in action or 'at' in action:
                        num, row, col = map(int, groups)
                    else:
                        row, col, num = map(int, groups)

                    row -= 1  # Convert to 0-indexed
                    col -= 1

                    if not (0 <= row < self.grid_size):
                        return False, -1, -1, -1, f"Row {row+1} out of range"
                    if not (0 <= col < self.grid_size):
                        return False, -1, -1, -1, f"Col {col+1} out of range"
                    if not (1 <= num <= self.grid_size):
                        return False, -1, -1, -1, f"Number {num} out of range"

                    return True, row, col, num, ""

        return False, -1, -1, -1, f"Could not parse: '{action}'"

    def _invalid_step(self, error: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Return result for an invalid action."""
        self.last_action_feedback = f"Invalid: {error}"
        is_solvable, reason = self.check_solvability()
        return self.render(), -0.1, False, {
            "action_is_valid": False,
            "action_is_effective": False,
            "success": False,
            "is_solvable": is_solvable,
            "deadlock_type": reason if not is_solvable else None,
            "error": error,
            "action_name": "invalid",
        }

    def _conflict_step(self, row, col, num) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Return result for a conflicting placement."""
        conflict_reasons = []
        if num in self.current_grid[row, :]:
            conflict_reasons.append(f"row {row+1}")
        if num in self.current_grid[:, col]:
            conflict_reasons.append(f"col {col+1}")

        box_size = int(np.sqrt(self.grid_size))
        br = (row // box_size) * box_size
        bc = (col // box_size) * box_size
        if num in self.current_grid[br:br + box_size, bc:bc + box_size]:
            conflict_reasons.append(f"box")

        reason = ", ".join(conflict_reasons)
        self.last_action_feedback = f"Conflict: {num} already in {reason}"

        is_solvable, sol_reason = self.check_solvability()
        return self.render(), -0.1, False, {
            "action_is_valid": False,
            "action_is_effective": False,
            "success": False,
            "is_solvable": is_solvable,
            "deadlock_type": sol_reason if not is_solvable else None,
            "error": f"{num} conflicts with {reason}",
            "action_name": f"place {num} at row {row+1} col {col+1}",
        }


if __name__ == "__main__":
    # Test the environment
    env = SudokuEnv(grid_size=9, difficulty="easy", max_steps=50)

    print("Testing SudokuEnv")
    print("=" * 50)

    obs = env.reset(seed=42)
    print("Initial state:")
    print(obs)
    print(f"\nState info: {env.get_state_info()}")

    # Test some actions
    actions = env.get_all_actions()
    print(f"\nAvailable actions: {len(actions)}")
    print(f"First 5: {actions[:5]}")

    # Take a few random actions
    for i in range(5):
        if not actions:
            break
        action = random.choice(actions)
        print(f"\nAction: {action}")
        obs, reward, done, info = env.step(action)
        print(f"Reward: {reward}, Done: {done}")
        print(f"Info: {info}")
        if done:
            break
        actions = env.get_all_actions()

    print(f"\nFinal state info: {env.get_state_info()}")
