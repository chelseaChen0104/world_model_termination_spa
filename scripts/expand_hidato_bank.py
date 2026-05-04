"""Algorithmic Hidato puzzle bank expansion (NEAR-6 in future_steps).

Generates N solvable Hidato puzzles for SAVE data generation. The output is a
Python list of puzzle dicts in the same format as
`src/environments/hidato_puzzle_bank.py` PUZZLES, but written to a new file
to preserve additivity (does NOT modify the existing 8-puzzle bank).

Algorithm:
  1. For each puzzle to generate:
     a. Sample a random Hamiltonian path on the R×C grid.
        - Start at a random cell with value 1.
        - At each step, move to a random unvisited orthogonally-adjacent cell.
        - If stuck, backtrack (DFS).
     b. The path defines the full solution: cell (r, c) → number 1..N.
     c. Pick a subset of cells to expose as "givens".
        - Always expose 1 (start) and N (end).
        - Plus n_mid random middle-numbers as additional givens.
     d. Verify the resulting puzzle is solvable (sanity — should always pass
        since we generated it from a valid solution).

Output:
  data/hidato_bank_5x4_N200/bank.py — Python module with a PUZZLES list

Usage:
  python scripts/expand_hidato_bank.py --rows 5 --cols 4 --n-puzzles 200 \\
      --n-mid-givens 1 --output data/hidato_bank_5x4_N200/bank.py --seed 42

Per spec_hidato.md and future_steps.md NEAR-6, target is ~200 puzzles
to bring records-per-state ratio into the healthy 3-5 range at paper-final
scale (11K records / 200-puzzle bank ≈ 55 records per puzzle ≈ 4 records per
unique state assuming ~14 anchor states per puzzle).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.hidato5x4_solver import Hidato5x4Solver


def _adjacent_cells(r: int, c: int, R: int, C: int) -> List[Tuple[int, int]]:
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < R and 0 <= nc < C:
            out.append((nr, nc))
    return out


def random_hamiltonian_path(R: int, C: int, rng: random.Random,
                              max_attempts: int = 50) -> Optional[Dict[Tuple[int, int], int]]:
    """Sample a random Hamiltonian path on R×C grid; return cell→number mapping
    (1..R*C). Returns None if generation fails after max_attempts.

    Uses random-DFS with backtracking from a random start cell.
    """
    n_total = R * C

    for attempt in range(max_attempts):
        start_r = rng.randrange(R)
        start_c = rng.randrange(C)
        path: List[Tuple[int, int]] = [(start_r, start_c)]
        visited = {(start_r, start_c)}

        def dfs(r: int, c: int) -> bool:
            """Extend path from (r, c). Returns True if full Hamiltonian path completed."""
            if len(path) == n_total:
                return True
            neighbors = _adjacent_cells(r, c, R, C)
            rng.shuffle(neighbors)
            for nr, nc in neighbors:
                if (nr, nc) in visited:
                    continue
                # Warnsdorff-like heuristic: prefer cells with fewer unvisited neighbors
                # (simplified: just try in random order; small grids don't need full Warnsdorff)
                visited.add((nr, nc))
                path.append((nr, nc))
                if dfs(nr, nc):
                    return True
                path.pop()
                visited.remove((nr, nc))
            return False

        if dfs(start_r, start_c):
            # Convert path to assignment: path[i] gets number (i+1)
            return {cell: i + 1 for i, cell in enumerate(path)}

    return None


def select_givens(assignment: Dict[Tuple[int, int], int], n_mid_givens: int,
                   rng: random.Random) -> Dict[Tuple[int, int], int]:
    """Pick which cells to expose as givens. Always include 1 and N; add
    n_mid_givens random middle numbers."""
    n_total = max(assignment.values())
    given_values = {1, n_total}
    if n_mid_givens > 0:
        middle_values = list(range(2, n_total))
        rng.shuffle(middle_values)
        for v in middle_values[:n_mid_givens]:
            given_values.add(v)
    return {cell: v for cell, v in assignment.items() if v in given_values}


def generate_puzzle(R: int, C: int, n_mid_givens: int,
                     rng: random.Random,
                     solver: Hidato5x4Solver,
                     puzzle_id: str) -> Optional[dict]:
    """Generate one Hidato puzzle. Returns puzzle dict or None on failure."""
    assignment = random_hamiltonian_path(R, C, rng)
    if assignment is None:
        return None
    givens = select_givens(assignment, n_mid_givens, rng)

    # Verify the puzzle is solvable from the givens (sanity).
    state = {"rows": R, "cols": C, "assignment": dict(givens)}
    res = solver.solve(state)
    if not res.solvable:
        return None  # rare; should not happen since assignment was a valid path

    return {
        "id": puzzle_id,
        "rows": R,
        "cols": C,
        "givens": dict(givens),
        "solution": dict(assignment),
        "num_solutions": res.num_solutions,
        "n_givens": len(givens),
        "n_empty": R * C - len(givens),
    }


def emit_puzzle_bank_py(puzzles: List[dict], output_path: str):
    """Write puzzles as a .py module. Format mimics hidato_puzzle_bank.py
    so generate_save_data.py + sft_formatter import patterns work as-is."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    body = [
        '"""Algorithmically-generated Hidato puzzle bank.',
        "",
        "Generated by scripts/expand_hidato_bank.py.",
        f"Total: {len(puzzles)} puzzles.",
        "",
        "Schema matches src/environments/hidato_puzzle_bank.py:",
        '    {"id": str, "rows": int, "cols": int, '
        '"givens": dict[(r,c) -> int], "solution": dict}',
        '"""',
        "",
        "PUZZLES = [",
    ]
    for p in puzzles:
        body.append("    {")
        body.append(f'        "id": {p["id"]!r},')
        body.append(f'        "rows": {p["rows"]},')
        body.append(f'        "cols": {p["cols"]},')
        # givens
        body.append('        "givens": {')
        for (r, c), v in sorted(p["givens"].items()):
            body.append(f'            ({r}, {c}): {v},')
        body.append('        },')
        # solution
        body.append('        "solution": {')
        for (r, c), v in sorted(p["solution"].items()):
            body.append(f'            ({r}, {c}): {v},')
        body.append('        },')
        body.append('    },')
    body.append("]")
    body.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(body))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=int, default=5)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--n-puzzles", type=int, default=200)
    p.add_argument("--n-mid-givens", type=int, default=1,
                   help="Number of mid-sequence givens BEYOND 1 and N. "
                        "n_mid_givens=1 → 3 givens total per puzzle.")
    p.add_argument("--output", required=True,
                   help="Output .py file path")
    p.add_argument("--metadata-output", default=None,
                   help="Optional path to write metadata JSON")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    solver = Hidato5x4Solver(solution_cap=4, node_cap=200_000)

    print(f"=== Hidato bank expansion ===")
    print(f"  rows={args.rows} cols={args.cols} n_puzzles={args.n_puzzles}")
    print(f"  n_mid_givens={args.n_mid_givens} (3 givens total per puzzle)")
    print(f"  output: {args.output}")

    puzzles = []
    seen_solution_hashes = set()
    n_attempts = 0
    n_dup = 0
    n_unsolvable = 0
    n_failed_path = 0

    while len(puzzles) < args.n_puzzles and n_attempts < args.n_puzzles * 10:
        n_attempts += 1
        puzzle_id = f"{args.rows}x{args.cols}_gen_{len(puzzles):04d}"
        p = generate_puzzle(args.rows, args.cols, args.n_mid_givens, rng, solver, puzzle_id)
        if p is None:
            n_failed_path += 1
            continue

        # Dedup by solution path (canonical hash of sorted assignment)
        sol_key = tuple(sorted(p["solution"].items()))
        if sol_key in seen_solution_hashes:
            n_dup += 1
            continue
        seen_solution_hashes.add(sol_key)
        puzzles.append(p)

        if len(puzzles) % 20 == 0:
            print(f"  ...{len(puzzles)}/{args.n_puzzles} ({n_attempts} attempts, "
                  f"{n_dup} dups, {n_failed_path} path-fails)")

    print(f"\n=== Generation done ===")
    print(f"  generated: {len(puzzles)} unique puzzles in {n_attempts} attempts")
    print(f"  duplicates rejected: {n_dup}")
    print(f"  path generation failures: {n_failed_path}")

    # Distributional sanity
    if puzzles:
        from collections import Counter
        n_sol_dist = Counter(p["num_solutions"] for p in puzzles)
        n_givens_dist = Counter(p["n_givens"] for p in puzzles)
        print(f"\n  num_solutions distribution: {dict(n_sol_dist)}")
        print(f"  n_givens distribution: {dict(n_givens_dist)}")

    emit_puzzle_bank_py(puzzles, args.output)
    print(f"\n  Wrote {len(puzzles)} puzzles to {args.output}")

    if args.metadata_output:
        meta = {
            "n_puzzles": len(puzzles),
            "rows": args.rows,
            "cols": args.cols,
            "n_mid_givens": args.n_mid_givens,
            "n_attempts": n_attempts,
            "n_duplicates": n_dup,
            "seed": args.seed,
            "num_solutions_distribution": dict(n_sol_dist) if puzzles else {},
        }
        with open(args.metadata_output, "w") as f:
            json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
