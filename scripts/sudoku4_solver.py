"""Instrumented Sudoku 4×4 solver for SAVE data generation.

Returns SolverResult with the fields required by save_sibling_set_v1.2
schema: solvable, num_solutions, nodes, backtracks, solution_depth,
solve_time_ms, solution_path. Runs exhaustively even on unsolvable
states (so `nodes` is finite-comparable, not a sentinel).

Does NOT touch src/environments/sudoku*.py — fresh implementation.
4×4 is small enough that an exhaustive backtracking search is cheap.

Public surface:
    Sudoku4Solver().solve(grid)            -> SolverResult
    Sudoku4Solver().is_viable(grid)        -> bool
    Sudoku4Solver().find_one_solution(grid) -> Optional[List[Action]]
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


SOLVER_VERSION = "sudoku4_solver_v1"

# Action: (row, col, value), all 0-indexed for internal use.
# The dataset's action_struct is 1-indexed for display; the solver works in
# 0-indexed coordinates and the env wrapper converts.
Action = Tuple[int, int, int]


@dataclass
class SolverResult:
    solvable: bool
    num_solutions: int          # 0 if unsolvable; otherwise >= 1 (capped at solution_cap)
    nodes: int                  # total search nodes expanded (exhaustive)
    backtracks: int             # number of times search retreated
    solution_depth: Optional[int]  # length of one example solution path; None if unsolvable
    solve_time_ms: float
    solution_path: Optional[List[Action]] = None  # one example viable continuation
    solver_version: str = SOLVER_VERSION


def _box_index(r: int, c: int) -> int:
    """Return 0..3 for the 2×2 box containing cell (r, c)."""
    return (r // 2) * 2 + (c // 2)


def _legal_values(grid: List[List[int]], r: int, c: int) -> List[int]:
    """Values 1..4 that don't immediately violate row/col/box constraints."""
    if grid[r][c] != 0:
        return []
    used = set()
    for k in range(4):
        used.add(grid[r][k])
        used.add(grid[k][c])
    box_r0, box_c0 = (r // 2) * 2, (c // 2) * 2
    for dr in (0, 1):
        for dc in (0, 1):
            used.add(grid[box_r0 + dr][box_c0 + dc])
    return [v for v in (1, 2, 3, 4) if v not in used]


def _is_complete(grid: List[List[int]]) -> bool:
    return all(grid[r][c] != 0 for r in range(4) for c in range(4))


def _is_consistent(grid: List[List[int]]) -> bool:
    """Check no current placements violate any constraint. (Empty cells = 0 ignored.)"""
    for r in range(4):
        seen = set()
        for c in range(4):
            v = grid[r][c]
            if v != 0:
                if v in seen:
                    return False
                seen.add(v)
    for c in range(4):
        seen = set()
        for r in range(4):
            v = grid[r][c]
            if v != 0:
                if v in seen:
                    return False
                seen.add(v)
    for box_r0 in (0, 2):
        for box_c0 in (0, 2):
            seen = set()
            for dr in (0, 1):
                for dc in (0, 1):
                    v = grid[box_r0 + dr][box_c0 + dc]
                    if v != 0:
                        if v in seen:
                            return False
                        seen.add(v)
    return True


class Sudoku4Solver:
    """Sudoku 4×4 solver with full instrumentation.

    Uses MRV heuristic (pick empty cell with fewest legal values first).
    Counts every explored node and every backtrack. By default runs
    exhaustively up to a `solution_cap` to give a meaningful `num_solutions`
    (and `nodes` count) even for unsolvable states. The first solution
    found becomes the recorded `solution_path`.
    """

    def __init__(self, solution_cap: int = 8, node_cap: int = 200_000):
        # solution_cap=8 is enough to distinguish "unique" from "multiple"
        # for paper purposes; node_cap is a hard safety limit (4×4 should
        # never come close to it).
        self.solution_cap = solution_cap
        self.node_cap = node_cap

    def solve(self, grid_or_state) -> SolverResult:
        """Run exhaustive search, return SolverResult.

        Accepts either a 2D list (4×4 grid with 0 for empty) or a dict
        {"grid": ...} for spec compatibility.
        """
        if isinstance(grid_or_state, dict):
            grid = grid_or_state["grid"]
        else:
            grid = grid_or_state
        # Deep copy to avoid mutation
        g = [row[:] for row in grid]

        t0 = time.perf_counter()
        if not _is_consistent(g):
            return SolverResult(
                solvable=False, num_solutions=0, nodes=0, backtracks=0,
                solution_depth=None, solve_time_ms=(time.perf_counter() - t0) * 1000.0,
                solution_path=None,
            )

        nodes = [0]
        backtracks = [0]
        first_solution: List[Action] = []
        n_solutions = [0]

        # Track the trail of placements made during this search so we can
        # record the first complete solution as a path.
        trail: List[Action] = []

        def pick_cell() -> Optional[Tuple[int, int, List[int]]]:
            best = None
            best_options: List[int] = []
            for r in range(4):
                for c in range(4):
                    if g[r][c] != 0:
                        continue
                    opts = _legal_values(g, r, c)
                    if not opts:
                        return (r, c, [])  # forced dead end at this cell
                    if best is None or len(opts) < len(best_options):
                        best = (r, c)
                        best_options = opts
                        if len(opts) == 1:
                            break
                if best is not None and len(best_options) == 1:
                    break
            if best is None:
                return None  # no empty cells
            return (best[0], best[1], best_options)

        def search() -> bool:
            """Returns True iff we should keep searching (caps not exhausted)."""
            nodes[0] += 1
            if nodes[0] > self.node_cap:
                return False  # safety stop
            picked = pick_cell()
            if picked is None:
                # Complete grid → record solution
                n_solutions[0] += 1
                if n_solutions[0] == 1:
                    first_solution.extend(trail)
                if n_solutions[0] >= self.solution_cap:
                    return False
                return True
            r, c, opts = picked
            if not opts:
                # Dead end: backtrack
                backtracks[0] += 1
                return True
            for v in opts:
                g[r][c] = v
                trail.append((r, c, v))
                cont = search()
                trail.pop()
                g[r][c] = 0
                if not cont:
                    return False
            backtracks[0] += 1
            return True

        search()

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

    def is_viable(self, grid_or_state) -> bool:
        # Cheap path: stop after first solution (saves ~half the search).
        # This re-runs solve with solution_cap=1 for speed; SAVE data gen
        # uses solve() directly when it needs full stats.
        original = self.solution_cap
        try:
            self.solution_cap = 1
            return self.solve(grid_or_state).solvable
        finally:
            self.solution_cap = original

    def find_one_solution(self, grid_or_state) -> Optional[List[Action]]:
        original = self.solution_cap
        try:
            self.solution_cap = 1
            return self.solve(grid_or_state).solution_path
        finally:
            self.solution_cap = original


# --- Smoke test ---

def _smoke():
    s = Sudoku4Solver()

    # 1) Fully solved board: should report 1 solution, 0-depth path
    full = [
        [1, 2, 3, 4],
        [3, 4, 1, 2],
        [2, 1, 4, 3],
        [4, 3, 2, 1],
    ]
    r = s.solve(full)
    assert r.solvable and r.num_solutions == 1 and r.solution_depth == 0, r
    print(f"  [1] full board: solvable={r.solvable}, n_sol={r.num_solutions}, "
          f"nodes={r.nodes}, depth={r.solution_depth}, t={r.solve_time_ms:.2f}ms")

    # 2) Almost-empty board: many solutions
    one = [
        [1, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    r = s.solve(one)
    assert r.solvable and r.num_solutions >= 1 and r.solution_path is not None
    print(f"  [2] almost-empty: solvable={r.solvable}, n_sol={r.num_solutions}, "
          f"nodes={r.nodes}, depth={r.solution_depth}, "
          f"backtracks={r.backtracks}, t={r.solve_time_ms:.2f}ms")

    # 3) Inconsistent board: should return unsolvable, num_solutions=0
    bad = [
        [1, 1, 0, 0],  # two 1s in row 0 — illegal
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    r = s.solve(bad)
    assert not r.solvable and r.num_solutions == 0
    print(f"  [3] inconsistent: solvable={r.solvable}, n_sol={r.num_solutions}")

    # 4) Locally legal but globally doomed: needs construction.
    # Place 1 in (0,0), 2 in (0,1), 1 in (1,3) — row 0 and (1,3) constraint
    # conflict will only show up when we try to fill cells.
    # Simpler test: a state that has no legal completion.
    # Construct: column 0 has 1, 2, 3 — only 4 left for (3,0). But also
    # row 3 already has a 4 → impossible.
    doomed = [
        [1, 0, 0, 0],
        [2, 0, 0, 0],
        [3, 0, 0, 0],
        [0, 0, 0, 4],  # (3,0) must be 4 (col), but row 3 has 4 already
    ]
    r = s.solve(doomed)
    assert not r.solvable, r
    assert r.nodes >= 1, "Doomed state must explore at least one node before failing"
    print(f"  [4] doomed (locally consistent, globally unsolvable): "
          f"solvable={r.solvable}, nodes={r.nodes}, backtracks={r.backtracks}")

    # 5) Dict-form input
    r = s.solve({"grid": full})
    assert r.solvable
    print(f"  [5] dict-form input: solvable={r.solvable}")

    # 6) is_viable + find_one_solution shortcuts
    assert s.is_viable(one) is True
    assert s.is_viable(bad) is False
    sol = s.find_one_solution(one)
    assert sol is not None and len(sol) == 15  # 16 cells, 1 prefilled
    print(f"  [6] is_viable + find_one_solution OK; len(sol)={len(sol)}")

    print("All smoke tests passed.")


if __name__ == "__main__":
    _smoke()
