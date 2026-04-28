"""
Sudoku Utilities and Solvability Checker

Provides:
1. Puzzle generation (random solvable puzzles)
2. Valid placement checking
3. Conflict detection
4. Grid formatting
5. SudokuSolvabilityChecker — the key component for termination prediction

The solvability checker determines whether a partially-filled Sudoku grid
can still be completed to a valid solution. Unlike Sokoban where simple
pattern-matching detects deadlocks, Sudoku solvability is NP-complete and
requires constraint propagation + bounded backtracking.
"""

import numpy as np
import random
from typing import Set, Tuple, List, Dict, Optional
from copy import deepcopy


def get_box_size(grid_size: int) -> int:
    return int(np.sqrt(grid_size))


def get_box_index(row: int, col: int, grid_size: int) -> Tuple[int, int]:
    box_size = get_box_size(grid_size)
    return row // box_size, col // box_size


def is_valid_placement(grid: np.ndarray, row: int, col: int, num: int) -> bool:
    """Check if placing num at (row, col) is valid per Sudoku rules."""
    grid_size = grid.shape[0]
    box_size = get_box_size(grid_size)

    if num in grid[row, :]:
        return False
    if num in grid[:, col]:
        return False

    box_row, box_col = get_box_index(row, col, grid_size)
    br = box_row * box_size
    bc = box_col * box_size
    if num in grid[br:br + box_size, bc:bc + box_size]:
        return False

    return True


def get_valid_numbers(grid: np.ndarray, row: int, col: int) -> Set[int]:
    """Get all valid numbers for cell (row, col)."""
    if grid[row, col] != 0:
        return set()

    grid_size = grid.shape[0]
    candidates = set(range(1, grid_size + 1))

    candidates -= set(grid[row, :]) - {0}
    candidates -= set(grid[:, col]) - {0}

    box_size = get_box_size(grid_size)
    box_row, box_col = get_box_index(row, col, grid_size)
    br = box_row * box_size
    bc = box_col * box_size
    candidates -= set(grid[br:br + box_size, bc:bc + box_size].flatten()) - {0}

    return candidates


def find_conflicts(grid: np.ndarray) -> List[Tuple[int, int]]:
    """Find all cells involved in conflicts (duplicates in row/col/box)."""
    grid_size = grid.shape[0]
    box_size = get_box_size(grid_size)
    conflict_cells = set()

    for i in range(grid_size):
        # Row
        row = grid[i, :]
        for num in range(1, grid_size + 1):
            positions = np.where(row == num)[0]
            if len(positions) > 1:
                for p in positions:
                    conflict_cells.add((i, p))

        # Column
        col = grid[:, i]
        for num in range(1, grid_size + 1):
            positions = np.where(col == num)[0]
            if len(positions) > 1:
                for p in positions:
                    conflict_cells.add((p, i))

    # Box
    for br in range(box_size):
        for bc in range(box_size):
            sr, sc = br * box_size, bc * box_size
            box = grid[sr:sr + box_size, sc:sc + box_size]
            for num in range(1, grid_size + 1):
                positions = np.argwhere(box == num)
                if len(positions) > 1:
                    for pos in positions:
                        conflict_cells.add((sr + pos[0], sc + pos[1]))

    return sorted(conflict_cells)


def is_solved(grid: np.ndarray) -> bool:
    """Check if the puzzle is completely and correctly solved."""
    if np.any(grid == 0):
        return False
    return len(find_conflicts(grid)) == 0


def format_grid(grid: np.ndarray, initial_grid: Optional[np.ndarray] = None) -> str:
    """Format grid as text with box separators."""
    grid_size = grid.shape[0]
    box_size = get_box_size(grid_size)
    lines = []

    for i, row in enumerate(grid):
        if i > 0 and i % box_size == 0:
            lines.append("-" * (grid_size * 2 + box_size - 1))

        parts = []
        for j, val in enumerate(row):
            if j > 0 and j % box_size == 0:
                parts.append("|")
            if val == 0:
                parts.append(".")
            elif initial_grid is not None and initial_grid[i, j] != 0:
                parts.append(str(val))  # Initial cell
            else:
                parts.append(str(val))
        lines.append(" ".join(parts))

    return "\n".join(lines)


def generate_sudoku_puzzle(
    grid_size: int = 9,
    difficulty: str = "easy",
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a Sudoku puzzle with a unique solution.

    Returns (puzzle_grid, solution_grid).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    solution = np.zeros((grid_size, grid_size), dtype=int)

    def fill(grid):
        empty = list(zip(*np.where(grid == 0)))
        if not empty:
            return True
        row, col = empty[0]
        nums = list(range(1, grid_size + 1))
        random.shuffle(nums)
        for num in nums:
            if is_valid_placement(grid, row, col, num):
                grid[row, col] = num
                if fill(grid):
                    return True
                grid[row, col] = 0
        return False

    fill(solution)

    puzzle = solution.copy()
    remove_fracs = {"easy": 0.4, "medium": 0.5, "hard": 0.6}
    n_remove = int(grid_size * grid_size * remove_fracs.get(difficulty, 0.4))

    cells = [(i, j) for i in range(grid_size) for j in range(grid_size)]
    random.shuffle(cells)
    for i in range(n_remove):
        r, c = cells[i]
        puzzle[r, c] = 0

    return puzzle, solution


class SudokuSolvabilityChecker:
    """Determines if a partially-filled Sudoku grid can be completed.

    Strategy:
    1. Compute candidate sets for all empty cells
    2. If any cell has 0 candidates -> unsolvable
    3. Constraint propagation (naked singles, hidden singles)
    4. If contradiction found -> unsolvable
    5. Bounded backtracking (depth limit) for definitive answer

    This is the key component that makes Sudoku interesting for
    termination prediction — unlike Sokoban's simple pattern matching,
    this requires genuine reasoning about constraint satisfaction.
    """

    def __init__(self, max_backtrack_depth: int = 30):
        self.max_backtrack_depth = max_backtrack_depth

    def check_solvability(self, grid: np.ndarray) -> Tuple[bool, Optional[str]]:
        """Check if the grid can be completed to a valid solution.

        Returns:
            (is_solvable, reason_if_not)
        """
        grid_size = grid.shape[0]

        # Check for existing conflicts (duplicates in row/col/box)
        conflicts = find_conflicts(grid)
        if conflicts:
            return False, "existing_conflict"

        # Compute candidates for all empty cells
        candidates = {}
        for r in range(grid_size):
            for c in range(grid_size):
                if grid[r, c] == 0:
                    cands = get_valid_numbers(grid, r, c)
                    if len(cands) == 0:
                        return False, "zero_candidates"
                    candidates[(r, c)] = cands

        if not candidates:
            # All cells filled, no conflicts
            return True, None

        # Constraint propagation
        work_grid = grid.copy()
        work_candidates = {k: set(v) for k, v in candidates.items()}

        solvable = self._propagate(work_grid, work_candidates)
        if not solvable:
            return False, "constraint_propagation"

        # If propagation solved it, done
        if not work_candidates:
            return True, None

        # Bounded backtracking for remaining cells
        result = self._backtrack(work_grid, work_candidates, depth=0)
        if result:
            return True, None
        else:
            return False, "no_solution"

    def _propagate(self, grid: np.ndarray, candidates: Dict) -> bool:
        """Apply constraint propagation until no more progress.

        Returns False if a contradiction is found.
        """
        changed = True
        grid_size = grid.shape[0]

        while changed:
            changed = False

            # Naked singles: cells with exactly one candidate
            to_place = []
            for (r, c), cands in list(candidates.items()):
                if len(cands) == 0:
                    return False  # Contradiction
                if len(cands) == 1:
                    to_place.append((r, c, next(iter(cands))))

            for r, c, num in to_place:
                if (r, c) not in candidates:
                    continue
                grid[r, c] = num
                del candidates[(r, c)]
                changed = True

                # Update peers
                if not self._eliminate_peers(grid, candidates, r, c, num):
                    return False

            # Hidden singles: number that can only go in one cell in a unit
            for unit_cells in self._get_all_units(grid_size):
                empty_in_unit = [(r, c) for r, c in unit_cells if (r, c) in candidates]
                if not empty_in_unit:
                    continue

                for num in range(1, grid_size + 1):
                    # Skip if already placed in this unit
                    if any(grid[r, c] == num for r, c in unit_cells):
                        continue

                    possible = [(r, c) for r, c in empty_in_unit
                                if num in candidates.get((r, c), set())]

                    if len(possible) == 0:
                        return False  # Contradiction: number has nowhere to go
                    if len(possible) == 1:
                        r, c = possible[0]
                        if len(candidates.get((r, c), set())) > 1:
                            # Place this number
                            grid[r, c] = num
                            del candidates[(r, c)]
                            changed = True
                            if not self._eliminate_peers(grid, candidates, r, c, num):
                                return False

        return True

    def _eliminate_peers(self, grid: np.ndarray, candidates: Dict,
                         row: int, col: int, num: int) -> bool:
        """Remove num from candidates of all peers of (row, col).

        Returns False if contradiction found.
        """
        grid_size = grid.shape[0]
        peers = self._get_peers(row, col, grid_size)

        for pr, pc in peers:
            if (pr, pc) in candidates and num in candidates[(pr, pc)]:
                candidates[(pr, pc)].discard(num)
                if len(candidates[(pr, pc)]) == 0:
                    return False  # Contradiction

        return True

    def _get_peers(self, row: int, col: int, grid_size: int) -> Set[Tuple[int, int]]:
        """Get all peer cells (same row, column, or box)."""
        box_size = get_box_size(grid_size)
        peers = set()

        # Same row
        for c in range(grid_size):
            if c != col:
                peers.add((row, c))

        # Same column
        for r in range(grid_size):
            if r != row:
                peers.add((r, col))

        # Same box
        br = (row // box_size) * box_size
        bc = (col // box_size) * box_size
        for r in range(br, br + box_size):
            for c in range(bc, bc + box_size):
                if (r, c) != (row, col):
                    peers.add((r, c))

        return peers

    def _get_all_units(self, grid_size: int) -> List[List[Tuple[int, int]]]:
        """Get all rows, columns, and boxes as lists of cell coordinates."""
        box_size = get_box_size(grid_size)
        units = []

        # Rows
        for r in range(grid_size):
            units.append([(r, c) for c in range(grid_size)])

        # Columns
        for c in range(grid_size):
            units.append([(r, c) for r in range(grid_size)])

        # Boxes
        for br in range(box_size):
            for bc in range(box_size):
                sr, sc = br * box_size, bc * box_size
                units.append([(sr + r, sc + c) for r in range(box_size) for c in range(box_size)])

        return units

    def _backtrack(self, grid: np.ndarray, candidates: Dict, depth: int) -> bool:
        """Bounded backtracking to check solvability.

        Returns True if a solution exists.
        """
        if depth >= self.max_backtrack_depth:
            # Exceeded depth limit — assume solvable (conservative)
            return True

        if not candidates:
            return True  # All cells filled

        # Pick cell with fewest candidates (MRV heuristic)
        min_cell = min(candidates.keys(), key=lambda k: len(candidates[k]))
        min_cands = candidates[min_cell]
        r, c = min_cell

        for num in list(min_cands):
            # Try placing num
            new_grid = grid.copy()
            new_candidates = {k: set(v) for k, v in candidates.items()}

            new_grid[r, c] = num
            del new_candidates[(r, c)]

            # Propagate
            if self._eliminate_peers(new_grid, new_candidates, r, c, num):
                if self._propagate(new_grid, new_candidates):
                    if self._backtrack(new_grid, new_candidates, depth + 1):
                        return True

        return False  # No valid placement works
