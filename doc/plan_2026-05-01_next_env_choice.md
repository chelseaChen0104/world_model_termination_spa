# Next-Env Choice: 5×10 Pentomino vs Brand-New Game (2026-05-01)

The question: which env is *less* likely to exhibit the greedy Pass@1 collapse
documented in [eval_2026-05-01_pentomino_greedy_gap.md](eval_2026-05-01_pentomino_greedy_gap.md)?

## The greedy-collapse mechanism, distilled

For a K-step task:
```
P(greedy_solves) ≈ ∏_{k=1..K} p_argmax_correct(s_k)
```

This is small whenever:
1. **K is large** (errors compound), AND/OR
2. **p_argmax_correct is small** (action space sparse with valid moves), especially
   on early-trajectory steps where the env hasn't constrained the choice yet.

**Sudoku 4×4 doesn't collapse** because constraint propagation forces high
p_argmax_correct on most cells (often a unique valid number per cell).
**Pentomino-easy 5×4 collapses** because the first move has ~150-200 options
of which only ~10-20% are valid → p_argmax_correct ≈ 0.15 → over 4 steps,
greedy ≈ 0.05% ≈ 0.

## Option 1: 5×10 / 10-piece Pentomino (B-9)

| Property | 5×10 / 10-piece | 5×4 / 4-piece (B-8 baseline) |
|---|---|---|
| Total tilings | 4,664 | 20 |
| Trajectory length K | 10 | 4 |
| First-move action space | ~3,000 | ~200 |
| Approx valid first moves | ~500-1,500 | ~20-40 |
| Approx first-move p_correct | 0.17-0.50 | 0.10-0.20 |
| Estimated greedy Pass@1 (per-step ≈ 0.3) | 0.3^10 ≈ **0.0006%** | 0.15^4 ≈ 0.05% |
| Estimated greedy Pass@1 (per-step ≈ 0.5) | 0.5^10 ≈ 0.1% | 0.5^4 ≈ 6% |

**Key tension**: 5×10 has MORE solution paths but ALSO more trajectory steps.
**Compounding errors over 10 steps tends to dominate the gain from more
solutions.** Even with optimistic per-step argmax-correct of 0.5 (which would
require unprecedented learning concentration), greedy ~0.1%.

**Realistic outlook**: greedy likely improves from 0% to ~1-2%, not toward the
33%+ that Sudoku achieves.

**Cost**: minimal. Data generated, augmenter ready. ~1 day SFT + RL.

## Option 2: Brand-new game

The best candidates rank by greedy-friendliness:

### Kakuro (cross-sums) — strongest fit

- **Mechanic**: fill cells with digits 1-9 such that each row/column "run"
  sums to a given target. Different from Sudoku in cells filled, similar in
  constraint propagation.
- **Action space per cell**: 1-9, but constraints typically reduce to 1-3 valid
  digits per cell.
- **K (trajectory length)**: ~12-25 cells in a small Kakuro puzzle (5×5 or 6×6).
- **Per-step argmax-correct**: 0.7-0.95 (constraint propagation forces unique
  answers on most cells).
- **Estimated greedy Pass@1**: 0.85^15 ≈ 9% to 0.9^15 ≈ 21%. **Likely positive.**
- **Predictive gap**: yes — placing a wrong digit in a constrained cell can
  make a downstream cell impossible to fill.
- **Cost**: ~2 days for new env (oracle solver via constraint propagation +
  validator), data gen pipeline already env-agnostic.

### Sudoku 6×6 — incremental, fastest

- **Mechanic**: same as 4×4 but bigger.
- **Action space per cell**: 1-6.
- **K**: ~25-30 cells to fill.
- **Per-step argmax-correct**: similar to 4×4 (Sudoku is greedy-friendly).
- **Estimated greedy Pass@1**: similar to or slightly worse than 4×4
  (Run A on 4×4 hit 33%; 6×6 might hit ~15-25%).
- **Predictive gap**: yes (constraints can cascade).
- **Cost**: trivial — same env code, just `--grid-size 6`. Need new SFT data.
- **Caveat**: not really a "new game"; just a harder Sudoku.

### Hidato (number sequence path) — moderate fit

- **Mechanic**: place numbers 1..N² on an N×N grid such that consecutive
  numbers are adjacent.
- **Action space per step**: ≤8 (adjacent cells).
- **K**: N² (e.g., 16 for 4×4 grid; 36 for 6×6).
- **Per-step argmax-correct**: high mid-trajectory (often forced by adjacency
  constraints), but trajectories are *long*.
- **Estimated greedy Pass@1**: 0.85^16 ≈ 7% on 4×4. Could be acceptable.
- **Predictive gap**: yes — placing N at the wrong adjacent cell can isolate
  remaining cells.
- **Cost**: ~1.5 days new env (Hamiltonian-path checker for solvability).

### Lights Out / Rush Hour / Tower of Hanoi — alternative options

- **Lights Out**: turn lights off in min steps. Linear-algebra-solvable; greedy
  heuristics work. K ≤ N². Limited "doom" states (any state is reachable),
  but "doom" = exceeds budget. Predictive gap weak.
- **Rush Hour**: sliding cars. K = number of moves to free target car.
  Many states are doomed (no path). K typically 10-30 moves. Greedy works
  with right heuristics. Predictive gap strong.
- **Tower of Hanoi**: 3 disks = 7 moves; 4 disks = 15. Solution is fully
  algorithmic (greedy heuristic exists). No predictive gap (always solvable).
  Bad fit (no predictive gap).

## Recommendation matrix

| Goal | Best Option |
|---|---|
| **Lowest greedy collapse risk** | Kakuro |
| **Lowest effort + reasonable greedy** | Sudoku 6×6 |
| **Stick with Pentomino track** | 5×10 (with caveat: probably still collapses) |
| **Most-different env** | Hidato or Rush Hour |

## My recommendation

**Kakuro** as the new game.

Reasoning:
1. **Highest expected greedy Pass@1** of all candidates (constraint propagation
   forces argmax-correct, similar mechanism to Sudoku).
2. **Maintains the predictive gap property** (placing wrong digit can doom
   downstream cells).
3. **Cleaner cross-env transfer story** for the paper than "Sudoku 6×6"
   (different mechanic, same recipe).
4. **Action space is small** (1-9 digit options per cell, often pruned to 1-3
   by constraints) — easy to render, easy to evaluate solvability.

If implementation effort is the constraint, **Sudoku 6×6** is the fastest path
(reuses the existing Sudoku env). Almost as greedy-friendly.

**Avoid 5×10 Pentomino** for the greedy-collapse criterion specifically — even
though the data is ready, the env structure is the wrong shape for argmax-correct
to be high. The greedy gap will likely persist (just less severe than 5×4).

## What would the new env give us for the paper?

1. **A second cleanly-greedy env beyond Sudoku** — turns "the recipe works on
   Sudoku, partially on Pentomino" into "the recipe works on multiple
   constraint-satisfaction games (Sudoku, Kakuro), partially on tiling games
   (Pentomino) where the action-space structure isn't greedy-friendly."
2. **Validates v8 (single-token) anchor on a new env** — Sudoku worked, Pentomino
   needed v8.2. Kakuro would tell us if v8 is sufficient for constraint-propagation
   envs in general.
3. **Truncation experiment in a third setting** — does the gate save compute on
   Kakuro too?

## Cost summary (rough)

| Path | Code | Data | SFT | RL | Total |
|---|---|---|---|---|---|
| 5×10 Pentomino (B-9) | 0 hr | 0 hr (done) | ~1.5 hr | ~5 hr | **~6.5 hr** |
| Sudoku 6×6 | 0 hr | ~3 hr LLM-policy | ~2 hr | ~5 hr | **~10 hr** |
| Kakuro (new) | ~12 hr (env + DLX/CP solver) | ~3-4 hr | ~2 hr | ~5 hr | **~22 hr** |

If we want greedy to actually work, Kakuro is the right answer — but it's a
significant time investment. Sudoku 6×6 is the cheap middle option.
