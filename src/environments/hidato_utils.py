"""Hidato (Numbrix variant) utilities — adjacency, solvability check,
and helpers.

A Hidato puzzle is an R×C grid where each cell either holds an integer in
[1, R·C] or is empty (denoted 0 / None). The constraint is that the integers
form a path: for each k in [1, R·C-1], the cells holding k and k+1 must be
*orthogonally adjacent* (share an edge).

The grid is *solvable* iff there's an assignment of the missing integers
such that the path constraint holds for every consecutive pair.

This module:
  - Provides adjacency primitives (4-connectivity).
  - Implements a solvability checker via backtracking with pruning.
  - Provides utility to verify a complete assignment satisfies the rules.
"""
from __future__ import annotations
from typing import Optional


# ----------------------------------------------------------------------------
# Adjacency
# ----------------------------------------------------------------------------

# 4-connected (orthogonal): up, down, left, right.
DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def adjacent_cells(r: int, c: int, rows: int, cols: int) -> list:
    """Return the in-bounds 4-neighbors of (r, c)."""
    return [(r + dr, c + dc) for dr, dc in DELTAS
            if 0 <= r + dr < rows and 0 <= c + dc < cols]


# ----------------------------------------------------------------------------
# Verify a complete assignment
# ----------------------------------------------------------------------------

def verify_solution(rows: int, cols: int, assignment: dict) -> tuple:
    """Check that `assignment` (a dict mapping (r, c) -> int) is a complete
    valid Hidato solution.

    Returns (ok: bool, reason: Optional[str]).
    """
    n_cells = rows * cols
    if len(assignment) != n_cells:
        return (False, f"assignment has {len(assignment)} cells, expected {n_cells}")
    # Check it's a permutation of 1..n_cells
    values = sorted(assignment.values())
    if values != list(range(1, n_cells + 1)):
        return (False, f"values are not a permutation of 1..{n_cells}: {values}")
    # Build inverse map: number -> position
    pos = {v: k for k, v in assignment.items()}
    # Check each consecutive pair is adjacent
    for k in range(1, n_cells):
        a = pos[k]
        b = pos[k + 1]
        if not _are_adjacent(a, b):
            return (False, f"{k} at {a} and {k+1} at {b} are not adjacent")
    return (True, None)


def _are_adjacent(a: tuple, b: tuple) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


# ----------------------------------------------------------------------------
# Solvability — can the partial state be extended to a valid solution?
# ----------------------------------------------------------------------------

def is_solvable(rows: int, cols: int, state: dict, *, max_backtrack: int = 100_000) -> tuple:
    """Determine whether the partial `state` (dict mapping (r, c) -> int, with
    only assigned cells present) can be extended to a complete Hidato
    solution. Returns (ok: bool, reason: Optional[str]).

    Uses backtracking from the highest already-placed number, advancing one
    step at a time to whichever empty adjacent cell can host the next number.
    Pruning:
      1. Adjacency: next number must go in a cell adjacent to the current.
      2. Connectivity check: after each placement, verify all remaining-but-
         pre-placed numbers (givens) are reachable in the right relative order
         via BFS over still-empty cells. If not, prune.

    Returns (False, reason) on doom; (True, None) on solvable.
    """
    n_cells = rows * cols
    if not state:
        # Empty puzzle — always solvable (just snake through the grid).
        # But typically we have at least one given; this shortcut isn't critical.
        return (True, None)

    # Validate assignment so far: distinct values, all in [1..n_cells]
    if len(set(state.values())) != len(state):
        return (False, "duplicate values in given assignment")
    for v in state.values():
        if not 1 <= v <= n_cells:
            return (False, f"value {v} out of range [1, {n_cells}]")
    # Check existing consecutive pairs are adjacent (for cells already placed)
    pos = {v: k for k, v in state.items()}
    for k in sorted(pos.keys()):
        if k + 1 in pos:
            if not _are_adjacent(pos[k], pos[k + 1]):
                return (False, f"given {k} at {pos[k]} and {k+1} at {pos[k+1]} not adjacent")

    counter = [0]

    def backtrack(asn: dict, k_target: int) -> bool:
        """Try to place numbers k_target, k_target+1, ..., n_cells.
        `asn` is the current assignment dict (with all placements so far).
        """
        counter[0] += 1
        if counter[0] > max_backtrack:
            return False
        if k_target > n_cells:
            return True
        # If k_target is already placed (was a given), advance.
        if k_target in {v for v in asn.values()}:
            return backtrack(asn, k_target + 1)
        # Otherwise: place k_target adjacent to where k_target-1 is.
        if k_target == 1:
            # Special case: 1 has no predecessor. Try every empty cell.
            cells_to_try = [(r, c) for r in range(rows) for c in range(cols)
                            if (r, c) not in asn]
        else:
            prev_pos = next((p for p, v in asn.items() if v == k_target - 1), None)
            if prev_pos is None:
                # Shouldn't happen — k_target-1 must already be placed if k_target == max_assigned + 1
                return False
            cells_to_try = [c for c in adjacent_cells(*prev_pos, rows, cols)
                            if c not in asn]

        for cell in cells_to_try:
            # Pre-check: would placing k_target here disconnect remaining givens?
            asn_new = dict(asn)
            asn_new[cell] = k_target
            if not _connectivity_check(asn_new, rows, cols, k_target, n_cells):
                continue
            if backtrack(asn_new, k_target + 1):
                return True
        return False

    # Find the smallest unplaced number to start backtracking from
    placed_values = sorted(state.values())
    k_start = 1
    while k_start in placed_values:
        k_start += 1
    if backtrack(dict(state), k_start):
        return (True, None)
    if counter[0] > max_backtrack:
        return (False, f"backtrack exceeded {max_backtrack} steps")
    return (False, "no valid completion")


def _connectivity_check(asn: dict, rows: int, cols: int, k_just_placed: int, n_cells: int) -> bool:
    """Cheap pruning: after placing k_just_placed, verify that any *future*
    given number (already in asn but with value > k_just_placed) is reachable
    from the current path tail via empty cells, in roughly the right number
    of steps.

    Specifically: for each remaining given k_g > k_just_placed, the empty-cell
    BFS distance from the current path's tail (or future placement region)
    to the cell holding k_g must be ≤ (k_g - k_just_placed).

    Approximation: just check that the cell containing k_g has at least one
    adjacent empty cell that's reachable from k_just_placed's cell via empty
    cells. Cheap to verify and prunes most dead-ends.
    """
    # Find current tail
    current_pos = next((p for p, v in asn.items() if v == k_just_placed), None)
    if current_pos is None:
        return True  # weird, shouldn't happen

    # All givens with value > k_just_placed
    future_givens = [(p, v) for p, v in asn.items() if v > k_just_placed]
    if not future_givens:
        # No future givens — just check there are enough empty cells
        # to fit the remaining numbers (k_just_placed+1 .. n_cells)
        n_empty = rows * cols - len(asn)
        n_remaining = n_cells - k_just_placed
        return n_empty == n_remaining

    # BFS over empty cells (and the path tail) to find reachable cells
    visited = {current_pos}
    queue = [current_pos]
    while queue:
        cell = queue.pop()
        for nb in adjacent_cells(*cell, rows, cols):
            if nb in visited:
                continue
            # nb is reachable if it's empty or if it's a future given
            if nb not in asn or asn[nb] > k_just_placed:
                visited.add(nb)
                queue.append(nb)

    # Each future given must be in `visited`
    for pos, _ in future_givens:
        if pos not in visited:
            return False
    return True


# ----------------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------------

def _smoke():
    print("Smoke test for Hidato utils:")

    # 2x2 grid, fully assigned: 1-2-3-4 path
    asn = {(0, 0): 1, (0, 1): 2, (1, 1): 3, (1, 0): 4}
    ok, reason = verify_solution(2, 2, asn)
    print(f"  Verify 2x2 valid path: {ok} (reason={reason})")

    # 2x2 grid, broken path
    asn = {(0, 0): 1, (1, 1): 2, (0, 1): 3, (1, 0): 4}
    ok, reason = verify_solution(2, 2, asn)
    print(f"  Verify 2x2 broken path: {ok} (reason={reason})  ← should be False")

    # 3x3 grid, partial state with given 1 at corner
    state = {(0, 0): 1, (2, 2): 9}
    ok, reason = is_solvable(3, 3, state)
    print(f"  Solvable 3x3 with givens (0,0)=1 (2,2)=9: {ok} (reason={reason})")

    # 3x3, given 5 at center, need 1..4 on one side and 6..9 on the other
    state = {(1, 1): 5}
    ok, reason = is_solvable(3, 3, state)
    print(f"  Solvable 3x3 with center=5: {ok} (reason={reason})")

    # 3x3, doomed: 1 and 9 placed at non-corners that disconnect the path
    state = {(0, 1): 1, (2, 1): 9}  # 1 at top-middle, 9 at bottom-middle
    ok, reason = is_solvable(3, 3, state)
    print(f"  Solvable 3x3 with (0,1)=1 (2,1)=9: {ok} (reason={reason})")


if __name__ == "__main__":
    _smoke()
