"""Instrumented Pentomino tiling solver for SAVE data generation.

Board-size agnostic: caller passes `board_h` and `board_w` to the constructor.
Used for 5×6 (current target — 6-piece subsets, 172 valid configs); also
works for any other rectangular board where pentominoes can tile.

Wraps `src.environments.polyomino_utils` PIECE_ORIENTATIONS + placement_cells
+ fits_on_board WITHOUT touching them (additivity). Implements its own
backtracking search with instrumentation per `save_sibling_set_v1.2` schema:
solvable, num_solutions, nodes, backtracks, solution_depth, solve_time_ms,
solution_path. Runs exhaustively even on unsolvable states (so `nodes` is
finite-comparable).

Public surface:
    PentominoSolver(board_h, board_w).solve(board, remaining_pieces)
    PentominoSolver(...).is_viable(board, remaining_pieces)
    PentominoSolver(...).find_one_solution(board, remaining_pieces)
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.environments.polyomino_utils import (
    PIECE_ORIENTATIONS, placement_cells, fits_on_board,
)


SOLVER_VERSION = "pentomino_solver_v2"

# A placement = (piece, ori_id, anchor_r, anchor_c) with row/col 0-indexed.
Placement = Tuple[str, int, int, int]


@dataclass
class SolverResult:
    solvable: bool
    num_solutions: int
    nodes: int
    backtracks: int
    solution_depth: Optional[int]
    solve_time_ms: float
    solution_path: Optional[List[Placement]] = None
    solver_version: str = SOLVER_VERSION


def _connected_components(board, h, w) -> list:
    """Sizes of connected '.' regions (4-adjacency)."""
    visited = [[False] * w for _ in range(h)]
    sizes = []
    for r in range(h):
        for c in range(w):
            if board[r][c] != '.' or visited[r][c]:
                continue
            size = 0
            stack = [(r, c)]
            visited[r][c] = True
            while stack:
                rr, cc = stack.pop()
                size += 1
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr][nc] and board[nr][nc] == '.':
                        visited[nr][nc] = True
                        stack.append((nr, nc))
            sizes.append(size)
    return sizes


class PentominoSolver:
    """Instrumented exhaustive Pentomino tiling solver.

    Uses MRV-style heuristic: pick the empty cell with fewest legal placements
    covering it, place a piece there, recurse. By default explores up to
    `solution_cap` solutions (so unsolvable states still get a meaningful node
    count and solvable states report num_solutions accurately for "uniqueness").
    """

    def __init__(self, board_h: int, board_w: int,
                 solution_cap: int = 8, node_cap: int = 500_000):
        self.board_h = board_h
        self.board_w = board_w
        self.solution_cap = solution_cap
        self.node_cap = node_cap

    def solve(self, board: List[List[str]], remaining_pieces: List[str]) -> SolverResult:
        """Run exhaustive search; return SolverResult.

        `board`: list of lists of strings; '.' for empty, a piece-letter for filled.
        `remaining_pieces`: list of pieces still to place.
        """
        h, w = self.board_h, self.board_w
        b = [row[:] for row in board]

        t0 = time.perf_counter()

        # Quick checks
        empty_count = sum(1 for r in range(h) for c in range(w) if b[r][c] == '.')
        if empty_count != 5 * len(remaining_pieces):
            return SolverResult(
                solvable=False, num_solutions=0, nodes=0, backtracks=0,
                solution_depth=None,
                solve_time_ms=(time.perf_counter() - t0) * 1000.0,
            )
        if not remaining_pieces:
            return SolverResult(
                solvable=True, num_solutions=1, nodes=0, backtracks=0,
                solution_depth=0,
                solve_time_ms=(time.perf_counter() - t0) * 1000.0,
                solution_path=[],
            )
        for sz in _connected_components(b, h, w):
            if sz % 5 != 0:
                return SolverResult(
                    solvable=False, num_solutions=0, nodes=0, backtracks=0,
                    solution_depth=None,
                    solve_time_ms=(time.perf_counter() - t0) * 1000.0,
                )

        nodes = [0]
        backtracks = [0]
        first_solution: List[Placement] = []
        n_solutions = [0]
        trail: List[Placement] = []

        def pick_cell_to_cover() -> Optional[Tuple[int, int]]:
            """Find the empty cell most-constrained: occurring in the fewest legal placements."""
            empties = [(r, c) for r in range(h) for c in range(w) if b[r][c] == '.']
            if not empties:
                return None  # board full
            # MRV approximation: just take the topmost-leftmost empty cell. Cheap and fine for small boards.
            return empties[0]

        def piece_placements_covering(target_r: int, target_c: int, pieces: List[str]) -> list:
            """All (piece, ori_id, anchor_r, anchor_c, cells) that cover (target_r, target_c)."""
            out = []
            for piece in pieces:
                for ori_id, ori in enumerate(PIECE_ORIENTATIONS[piece]):
                    for dr, dc in ori:
                        anchor_r = target_r - dr
                        anchor_c = target_c - dc
                        cells = placement_cells(piece, ori_id, anchor_r, anchor_c)
                        if cells and fits_on_board(cells, h, w, b):
                            out.append((piece, ori_id, anchor_r, anchor_c, cells))
            return out

        def search(remaining: List[str]) -> bool:
            """Returns True iff search should continue (caps not exhausted)."""
            nodes[0] += 1
            if nodes[0] > self.node_cap:
                return False
            if not remaining:
                # board should be empty-free
                if all(b[r][c] != '.' for r in range(h) for c in range(w)):
                    n_solutions[0] += 1
                    if n_solutions[0] == 1:
                        first_solution.extend(trail)
                    if n_solutions[0] >= self.solution_cap:
                        return False
                return True
            target = pick_cell_to_cover()
            if target is None:
                return True  # unreachable but defensive
            tr, tc = target
            placements = piece_placements_covering(tr, tc, remaining)
            if not placements:
                backtracks[0] += 1
                return True
            for piece, ori_id, ar, ac, cells in placements:
                for cr, cc in cells:
                    b[cr][cc] = piece
                trail.append((piece, ori_id, ar, ac))
                cont = search([p for p in remaining if p != piece])
                trail.pop()
                for cr, cc in cells:
                    b[cr][cc] = '.'
                if not cont:
                    return False
            backtracks[0] += 1
            return True

        search(list(remaining_pieces))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        solvable = n_solutions[0] > 0
        return SolverResult(
            solvable=solvable,
            num_solutions=n_solutions[0],
            nodes=nodes[0],
            backtracks=backtracks[0],
            solution_depth=len(first_solution) if solvable else None,
            solve_time_ms=elapsed_ms,
            solution_path=first_solution if solvable else None,
        )

    def is_viable(self, board: List[List[str]], remaining_pieces: List[str]) -> bool:
        original = self.solution_cap
        try:
            self.solution_cap = 1
            return self.solve(board, remaining_pieces).solvable
        finally:
            self.solution_cap = original

    def find_one_solution(self, board: List[List[str]], remaining_pieces: List[str]) -> Optional[List[Placement]]:
        original = self.solution_cap
        try:
            self.solution_cap = 1
            return self.solve(board, remaining_pieces).solution_path
        finally:
            self.solution_cap = original


# --- Smoke test ---

def _smoke():
    s = PentominoSolver(board_h=5, board_w=6)

    # 1) Empty 5x6 board with one of the 172 valid 6-piece subsets (FILNPT is one)
    empty = [['.'] * 6 for _ in range(5)]
    pieces = ['F', 'I', 'L', 'N', 'P', 'T']
    r = s.solve(empty, pieces)
    assert r.solvable and r.num_solutions >= 1
    assert r.solution_path is not None and len(r.solution_path) == 6
    print(f"  [1] empty 5x6 {''.join(pieces)}: solvable={r.solvable}, n_sol={r.num_solutions}, "
          f"nodes={r.nodes}, depth={r.solution_depth}, t={r.solve_time_ms:.1f}ms")

    # 2) Verify the solution tiles correctly
    test_board = [['.'] * 6 for _ in range(5)]
    for piece, ori_id, ar, ac in r.solution_path:
        cells = placement_cells(piece, ori_id, ar, ac)
        for cr, cc in cells:
            assert test_board[cr][cc] == '.'
            test_board[cr][cc] = piece
    assert all(c != '.' for row in test_board for c in row)
    print(f"  [2] solution tiles 5x6 board completely: ✓")

    # 3) Empty cells count mismatch → unsolvable cheap
    bad = [['L', '.', '.', '.', '.', '.']] + [['.'] * 6 for _ in range(4)]
    r = s.solve(bad, pieces)
    assert not r.solvable
    print(f"  [3] area mismatch: solvable={r.solvable}, n_sol={r.num_solutions}")

    # 4) is_viable + find_one_solution shortcuts
    assert s.is_viable(empty, pieces) is True
    sol = s.find_one_solution(empty, pieces)
    assert sol is not None and len(sol) == 6
    print(f"  [4] is_viable + find_one_solution OK")

    # 5) Legacy 5×4 LPWY still works (backward compat)
    s4 = PentominoSolver(board_h=5, board_w=4)
    r = s4.solve([['.'] * 4 for _ in range(5)], ['L', 'P', 'W', 'Y'])
    assert r.solvable and r.solution_path is not None and len(r.solution_path) == 4
    print(f"  [5] legacy 5x4 LPWY: solvable={r.solvable}, depth={r.solution_depth}")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    _smoke()
