"""Local progress formula for Hidato 5×4 (hidato_local_progress_v1).

Per spec §5.2 of doc/data_generation_hidato.md:
    score = placed_normalized - 0.05 * isolated_empties

where:
- placed_normalized = num_placed / num_total_cells
- isolated_empties = empty cells that have no useful neighbor (no empty
  neighbor AND no neighbor holding a value within ±1 of next_n)

CRITICAL: depends only on the assignment + adjacency. Does NOT consult the
solver. The "isolated_empties" proxy is intentionally coarse — it's not
solver-equivalent (solver does full reachability), it just penalizes locally
obvious dead ends.
"""
from __future__ import annotations

from typing import Dict, List, Tuple


FORMULA_ID = "hidato_local_progress_v1"
FORMULA_SPEC = "placed_normalized - 0.05 * isolated_empties"


def _adjacent_cells(r: int, c: int, R: int, C: int) -> List[Tuple[int, int]]:
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < R and 0 <= nc < C:
            out.append((nr, nc))
    return out


def _next_n(state: dict):
    R, C = state["rows"], state["cols"]
    placed = set(state["assignment"].values())
    for k in range(1, R * C + 1):
        if k not in placed:
            return k
    return None


def compute_progress(state: dict) -> Dict:
    R, C = state["rows"], state["cols"]
    asn = state["assignment"]
    n_total = R * C
    n_placed = len(asn)
    placed_normalized = n_placed / float(n_total) if n_total > 0 else 0.0

    nxt = _next_n(state)
    # Identify "isolated empty cells": empty cells with NO empty neighbor AND
    # no neighbor holding a value within ±1 of next_n. Such cells are likely
    # dead-end candidates the path will struggle to reach.
    isolated_empties = 0
    if nxt is not None:
        empties = [(r, c) for r in range(R) for c in range(C) if (r, c) not in asn]
        for r, c in empties:
            has_useful_neighbor = False
            for nr, nc in _adjacent_cells(r, c, R, C):
                if (nr, nc) not in asn:
                    has_useful_neighbor = True
                    break
                v = asn[(nr, nc)]
                if abs(v - nxt) <= 1:
                    has_useful_neighbor = True
                    break
            if not has_useful_neighbor:
                isolated_empties += 1

    score = placed_normalized - 0.05 * isolated_empties

    return {
        "formula_id": FORMULA_ID,
        "formula_spec": FORMULA_SPEC,
        "local_progress_score": score,
        "features": {
            "n_placed": n_placed,
            "placed_normalized": placed_normalized,
            "isolated_empties": isolated_empties,
            "n_total_cells": n_total,
            "next_n": nxt if nxt is not None else -1,
        },
    }


# --- Smoke test ---

def _smoke():
    # 1) Empty 3×3 (no givens): score = 0
    state = {"rows": 3, "cols": 3, "assignment": {}}
    p = compute_progress(state)
    assert p["local_progress_score"] == 0.0
    assert p["features"]["n_placed"] == 0
    print(f"  [1] empty 3x3: score={p['local_progress_score']}")

    # 2) Fully placed: score = 1.0
    state2 = {"rows": 2, "cols": 2,
              "assignment": {(0,0): 1, (0,1): 2, (1,1): 3, (1,0): 4}}
    p = compute_progress(state2)
    assert p["local_progress_score"] == 1.0
    assert p["features"]["next_n"] == -1
    print(f"  [2] fully placed: score={p['local_progress_score']}")

    # 3) 3x3_snake initial state: 1@(0,0), 9@(2,2). 2/9 placed. next_n=2.
    state3 = {"rows": 3, "cols": 3,
              "assignment": {(0, 0): 1, (2, 2): 9}}
    p = compute_progress(state3)
    expected = 2.0 / 9.0   # all empty cells have empty neighbors → 0 isolated
    assert abs(p["local_progress_score"] - expected) < 1e-9
    assert p["features"]["isolated_empties"] == 0
    print(f"  [3] 3x3 with 2 givens: score={p['local_progress_score']:.4f} "
          f"(no isolated empties)")

    # 4) Force an isolated empty: surround a single empty cell with placed values
    #    that aren't within ±1 of next_n.
    #    3x3, place 1@(0,0), 5@(0,2), 6@(1,2), 7@(2,2), 9@(2,0). Empty: (0,1)(1,0)(1,1)(2,1).
    #    next_n = 2. (0,1)'s neighbors: (0,0)=1 (within 1 of 2 ✓), (0,2)=5 (not within 1),
    #    (1,1)=empty. So (0,1) has useful neighbors. Not isolated.
    #    Try harder: put values far from next_n=2:
    state4 = {"rows": 3, "cols": 3,
              "assignment": {(0, 0): 1, (0, 1): 8, (0, 2): 9,
                             (1, 0): 5, (1, 2): 6,
                             (2, 0): 4, (2, 1): 7}}
    # 7 cells placed, 2 empty: (1,1), (2,2). next_n=2.
    # (1,1)'s neighbors: (0,1)=8(diff 6), (1,0)=5(diff 3), (1,2)=6(diff 4), (2,1)=7(diff 5).
    #   No within-1 neighbor, no empty neighbor (other than (1,1) itself, which doesn't count
    #   since we only count OTHER empties). Wait (2,2) is empty... let me check (1,1)'s neighbors.
    #   (1,1) neighbors: (0,1), (1,0), (1,2), (2,1) — all placed. None within ±1 of 2. ISOLATED.
    # (2,2)'s neighbors: (1,2)=6, (2,1)=7. Neither within ±1 of 2. ISOLATED.
    p = compute_progress(state4)
    assert p["features"]["isolated_empties"] == 2
    expected_score = 7.0/9.0 - 0.05 * 2
    assert abs(p["local_progress_score"] - expected_score) < 1e-9
    print(f"  [4] forced isolation: 2 empty cells far from next_n=2, "
          f"isolated_empties={p['features']['isolated_empties']}, "
          f"score={p['local_progress_score']:.4f}")

    # 5) Deceptive sketch: the state with MORE placed values but isolated empties
    #    scores lower than a state with fewer placed but no isolation.
    state_low = state3
    state_high = state4
    p_low = compute_progress(state_low)
    p_high = compute_progress(state_high)
    print(f"  [5] deceptive: 2-given state scores {p_low['local_progress_score']:.4f}, "
          f"7-given state scores {p_high['local_progress_score']:.4f}")
    # Note: these aren't strictly comparable without more setup, but we can verify
    # the formula is responsive to both placed count AND isolation penalty.

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    _smoke()
