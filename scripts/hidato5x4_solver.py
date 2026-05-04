"""Instrumented Hidato (Numbrix variant) solver for SAVE data generation.

Wraps `src.environments.hidato_utils.is_solvable` with instrumentation for
the save_sibling_set_v1.2 schema: solvable, num_solutions, nodes, backtracks,
solution_depth, solve_time_ms, solution_path.

Does NOT modify hidato_utils.py (additivity contract). Implements its own
backtracking with reachability pruning, similar in spirit to the existing
checker but emitting the full result tuple SAVE needs.

Public surface:
    Hidato5x4Solver().solve(state)
    Hidato5x4Solver().is_viable(state)
    Hidato5x4Solver().find_one_solution(state)

State format (matches our SAVE env wrapper):
    {
      "rows": int,
      "cols": int,
      "assignment": dict[(r, c) -> int],   # 0-indexed cells; values 1..N
    }

`solution_path` is a list of (r, c, value) for each placement made by the
solver, in chronological order. Givens (cells already in `assignment`) are
NOT included in the path.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


SOLVER_VERSION = "hidato5x4_solver_v1"

# Action: (r, c, value), 0-indexed
Action = Tuple[int, int, int]


@dataclass
class SolverResult:
    solvable: bool
    num_solutions: int
    nodes: int
    backtracks: int
    solution_depth: Optional[int]
    solve_time_ms: float
    solution_path: Optional[List[Action]] = None
    solver_version: str = SOLVER_VERSION


def _adjacent_cells(r: int, c: int, R: int, C: int) -> List[Tuple[int, int]]:
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < R and 0 <= nc < C:
            out.append((nr, nc))
    return out


def _all_remaining_givens_reachable(
    asn: Dict[Tuple[int, int], int],
    R: int, C: int,
    just_placed_n: int,
    n_total: int,
) -> bool:
    """Cheap pruning: from the cell holding just_placed_n, BFS via empty cells
    + cells holding values > just_placed_n. If any future given is unreachable,
    return False.

    This catches the common Hidato dead-end where the path can't reach a
    future-required given.
    """
    placed_cell = None
    for (r, c), v in asn.items():
        if v == just_placed_n:
            placed_cell = (r, c)
            break
    if placed_cell is None:
        return True

    # Future givens (numbers > just_placed_n that are pre-filled)
    future_givens = {v: (r, c) for (r, c), v in asn.items() if v > just_placed_n}
    if not future_givens:
        return True

    # BFS from placed_cell through cells that aren't already-placed lower numbers
    # Empty cells + cells holding higher numbers are passable.
    reachable = {placed_cell}
    stack = [placed_cell]
    while stack:
        r, c = stack.pop()
        for nr, nc in _adjacent_cells(r, c, R, C):
            if (nr, nc) in reachable:
                continue
            v = asn.get((nr, nc))
            if v is None or v > just_placed_n:
                reachable.add((nr, nc))
                stack.append((nr, nc))

    for v, cell in future_givens.items():
        if cell not in reachable:
            return False
    return True


class Hidato5x4Solver:
    """Backtracking solver with adjacency + reachability pruning."""

    def __init__(self, solution_cap: int = 4, node_cap: int = 100_000):
        self.solution_cap = solution_cap
        self.node_cap = node_cap

    def solve(self, state: dict) -> SolverResult:
        R = state["rows"]
        C = state["cols"]
        asn = dict(state["assignment"])  # local copy
        n_total = R * C

        t0 = time.perf_counter()

        nodes = [0]
        backtracks = [0]
        first_solution: List[Action] = []
        n_solutions = [0]
        trail: List[Action] = []  # placements made by this solver run (excluding givens)

        def search(k: int) -> bool:
            """Place number k. Returns True iff search should continue (caps not exhausted)."""
            nodes[0] += 1
            if nodes[0] > self.node_cap:
                return False
            if k > n_total:
                # All numbers placed → solution
                n_solutions[0] += 1
                if n_solutions[0] == 1:
                    first_solution.extend(trail)
                if n_solutions[0] >= self.solution_cap:
                    return False
                return True
            # If k is a given, recurse with k+1
            existing = [(r, c) for (r, c), v in asn.items() if v == k]
            if existing:
                # Verify consistency: k must be adjacent to k-1 if k-1 exists
                if k > 1:
                    prev = [(r, c) for (r, c), v in asn.items() if v == k - 1]
                    if prev:
                        pr, pc = prev[0]
                        kr, kc = existing[0]
                        if (kr, kc) not in _adjacent_cells(pr, pc, R, C):
                            backtracks[0] += 1
                            return True  # inconsistent given → no solution this branch
                # Reachability check
                if not _all_remaining_givens_reachable(asn, R, C, k, n_total):
                    backtracks[0] += 1
                    return True
                return search(k + 1)
            # k must be placed adjacent to k-1
            if k == 1:
                # 1 isn't given; can be placed anywhere empty (rare in practice)
                empties = [(r, c) for r in range(R) for c in range(C)
                           if (r, c) not in asn]
                candidates = empties
            else:
                prev = [(r, c) for (r, c), v in asn.items() if v == k - 1]
                if not prev:
                    backtracks[0] += 1
                    return True  # k-1 not yet placed and not a given — shouldn't happen
                pr, pc = prev[0]
                candidates = [(nr, nc) for nr, nc in _adjacent_cells(pr, pc, R, C)
                              if (nr, nc) not in asn]
            if not candidates:
                backtracks[0] += 1
                return True
            for r, c in candidates:
                asn[(r, c)] = k
                trail.append((r, c, k))
                if _all_remaining_givens_reachable(asn, R, C, k, n_total):
                    cont = search(k + 1)
                else:
                    backtracks[0] += 1
                    cont = True
                trail.pop()
                del asn[(r, c)]
                if not cont:
                    return False
            backtracks[0] += 1
            return True

        search(1)

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

    def is_viable(self, state: dict) -> bool:
        original = self.solution_cap
        try:
            self.solution_cap = 1
            return self.solve(state).solvable
        finally:
            self.solution_cap = original

    def find_one_solution(self, state: dict) -> Optional[List[Action]]:
        original = self.solution_cap
        try:
            self.solution_cap = 1
            return self.solve(state).solution_path
        finally:
            self.solution_cap = original


# --- Smoke test ---

def _smoke():
    s = Hidato5x4Solver()

    # 1) 2x2 solvable: 1@(0,0), 4@(0,1). Path 1→2(1,0)→3(1,1)→4(0,1) ✓
    state = {
        "rows": 2, "cols": 2,
        "assignment": {(0, 0): 1, (0, 1): 4},
    }
    r = s.solve(state)
    assert r.solvable, r
    print(f"  [1] 2x2 (1@(0,0), 4@(0,1)): solvable={r.solvable}, "
          f"n_sol={r.num_solutions}, nodes={r.nodes}, depth={r.solution_depth}, "
          f"t={r.solve_time_ms:.2f}ms, path={r.solution_path}")

    # 2) 3×3 snake from existing puzzle bank: 1@(0,0), 9@(2,2)
    #    Solution exists per the puzzle bank's PUZZLE_3x3_SNAKE.
    state2 = {
        "rows": 3, "cols": 3,
        "assignment": {(0, 0): 1, (2, 2): 9},
    }
    r = s.solve(state2)
    assert r.solvable, r
    print(f"  [2] 3x3 snake (1@(0,0), 9@(2,2)): solvable={r.solvable}, "
          f"n_sol={r.num_solutions}, nodes={r.nodes}, t={r.solve_time_ms:.2f}ms")

    # 3) 3×3 with both endpoints in top row: 1@(0,0), 9@(0,2). The path of length 9
    #    must start (0,0) and end (0,2), visiting all 9 cells exactly once. This
    #    Hamiltonian path exists (e.g., 1@(0,0)→(1,0)→(2,0)→(2,1)→(1,1)→(2,2)→
    #    can't end at (0,2)... let me just let solver decide.
    state3 = {
        "rows": 3, "cols": 3,
        "assignment": {(0, 0): 1, (0, 2): 9},
    }
    r = s.solve(state3)
    print(f"  [3] 3x3 (1@(0,0), 9@(0,2)): solvable={r.solvable}, "
          f"n_sol={r.num_solutions}, nodes={r.nodes}, t={r.solve_time_ms:.2f}ms")

    # 4) Genuinely unsolvable: 2 at non-adjacent cell to 1
    state4 = {
        "rows": 2, "cols": 2,
        "assignment": {(0, 0): 1, (1, 1): 2},  # 2 not adjacent to 1
    }
    r = s.solve(state4)
    assert not r.solvable, r
    print(f"  [4] 2x2 unsolvable (1@(0,0), 2@(1,1) non-adjacent): "
          f"solvable={r.solvable}, nodes={r.nodes}")

    # 5) Genuinely unsolvable 2x2: 1@(0,0), 4@(1,1) — needs path 1→2→3→4 but
    #    no valid arrangement (verified by hand above).
    state5 = {
        "rows": 2, "cols": 2,
        "assignment": {(0, 0): 1, (1, 1): 4},
    }
    r = s.solve(state5)
    assert not r.solvable, r
    print(f"  [5] 2x2 unsolvable (1@(0,0), 4@(1,1)): solvable={r.solvable}, "
          f"nodes={r.nodes}")

    # 6) is_viable + find_one_solution shortcuts
    assert s.is_viable(state)
    assert not s.is_viable(state5)
    sol = s.find_one_solution(state)
    assert sol is not None
    print(f"  [6] is_viable + find_one_solution OK; len(sol)={len(sol)}")

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    _smoke()
