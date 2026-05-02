# Spec — Kakuro Env (2026-05-01)

A constraint-satisfaction puzzle env added as a third game (after Sudoku and
Pentomino) to test whether the recipe transfers to a *greedy-friendly* env
that doesn't have the action-space sparsity that drove Pentomino's greedy
collapse. See [plan_2026-05-01_next_env_choice.md](plan_2026-05-01_next_env_choice.md)
for the rationale.

## Goal

Implement `KakuroEnv` extending `BaseTerminationEnv`, with:
- Reset from a puzzle bank
- Step (place a digit 1-9 in a chosen white cell)
- Solvability check (constraint propagation; doom if no valid digit assignment exists)
- Predictive gap (states that look fillable but are doomed)
- Render (text representation)
- Action enumeration

## Game rules

- An R×C grid where each cell is either **black** (with optional clues) or **white**.
- Black cells contain row-clue / column-clue numbers — sums for the run of
  white cells immediately to the right (row clue) or below (column clue).
- White cells must be filled with a digit 1-9.
- Each "run" (maximal sequence of consecutive white cells in a row or column)
  must:
  1. Sum to exactly the clue value
  2. Have all distinct digits

A puzzle is **solvable** if there exists a valid assignment of digits to all
white cells satisfying both constraints across every run. **Doom** = no
extension of the current partial assignment is solvable.

## Why Kakuro is greedy-friendly

Constraint propagation forces digits in many cells:
- A run of length 2 with clue 3 → digits must be {1, 2}
- A run with one cell and clue 4 → must be 4 (forced)
- Once any cell is filled, neighboring cells have reduced valid-digit sets

→ At many states, only 1-2 digits are valid for a given cell. argmax over the
viable digit set is much more likely to be correct than on Pentomino's
spread-thin first-move distribution.

## API (mirrors PolyominoEnv)

```python
class KakuroEnv(BaseTerminationEnv):
    def __init__(self, puzzle_bank: list, max_steps: Optional[int] = None):
        """
        puzzle_bank: list of dicts, each describing one puzzle:
          {
            "rows": int, "cols": int,
            "cells": list of (r, c, type, ...) where type ∈ {'black', 'white'},
                     black cells optionally carry "right_clue" and "down_clue"
                     for the runs starting at the cell to their right or below.
            "solution": dict mapping (r, c) -> digit for verification
          }
        """

    def reset(self, seed: Optional[int] = None) -> str:
        """Pick a puzzle from the bank deterministically by seed; return rendered str."""

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Action format: 'place {digit} at row {R} col {C}' (1-indexed)."""

    def render(self) -> str:
        """ASCII render of the partial board."""

    def check_solvability(self) -> Tuple[bool, Optional[str]]:
        """Constraint propagation + bounded backtracking."""

    def get_all_actions(self) -> list:
        """All valid 'place N at R C' strings — only digits not yet used in
        any run containing the cell, restricted to digits that don't violate
        sum constraints given current partial assignment."""
```

## Action format

Same convention as Sudoku/Polyomino:
```
<answer>place {digit} at row {R} col {C}</answer>
```

where digit is 1-9, (R, C) are 1-indexed cell coordinates that must be a
white cell.

## Tag set (mirrors Sudoku since the env is constraint-satisfaction)

```
<observation>...</observation>
<prediction>...</prediction>
<solvable>true|false</solvable>
<answer>place 5 at row 2 col 3</answer>
```

Use the existing `<solvable>` tag (not `<viability>`) since Kakuro is closer
in spirit to Sudoku's constraint-propagation style. Maybe.

(Decision pending: could also use `<viability>` for consistency with newer
envs. Implication: trainer regex already matches both, so the choice doesn't
affect code, only consistency in docs/data.)

## Solvability checker design

Constraint propagation:
1. For each run, compute the set of *valid digit subsets* that sum to its clue.
   Example: clue 6, length 3 → valid subsets {1,2,3}, but in different orders.
   For length 2, clue 6 → subsets {1,5}, {2,4} (i.e., {1,5} and {2,4}, NOT {3,3}).
2. For each white cell, intersect the per-run subsets that contain it.
3. If any cell has empty valid-digit set → unsolvable.
4. Otherwise: backtracking with MRV (most-constrained variable first).
   Bounded depth = total empty cells. With propagation, typically resolves quickly.

Example sum subsets table (precomputed once at startup):
```
sum 3, length 2: {1,2}                 (only one subset)
sum 4, length 2: {1,3}                 (only one)
sum 5, length 2: {1,4}, {2,3}          (two)
sum 6, length 2: {1,5}, {2,4}          (two; {3,3} invalid)
sum 7, length 2: {1,6}, {2,5}, {3,4}   (three)
...
```

This is a small lookup table — Kakuro players use it constantly.

## Puzzle bank: simple generation strategy

Hand-curate ~30-50 puzzles of varying difficulty (3×3 to 5×5 white-cell areas).
Each puzzle has:
- A grid layout
- Black-cell clues
- A known solution (for env validation)

Easier than building a puzzle generator from scratch. Quality > quantity.

For initial implementation, start with **5-10 puzzles** to validate the env,
then expand the bank as needed for SFT training.

## Data generation convention

Same pipeline as Sudoku/Polyomino:
- `LLMTrajectoryGenerator` does LLM-policy rollouts on KakuroEnv
- Filter out post-BP samples (we don't train on doom→doom transitions)
- Produce parquet files with prompt/response/extra_info columns

`SFTFormatter` needs a new variant: `kakuro_minimal`. Mostly a copy of
`sudoku_minimal` with kakuro-specific rendering details.

## Predictive gap test

Before committing to Kakuro for the recipe, verify the predictive gap exists:
- Generate 500 trajectories with random play
- Measure: fraction of states where partial assignment looks valid (no immediate
  rule violation) but is actually unsolvable (constraint propagation finds no
  extension). Target: ≥30% (matching Pentomino's 73% which was too high, but
  more than Sudoku's ~10%).

## Files to create

```
src/environments/
  kakuro.py                     -- KakuroEnv class (~400 LOC est)
  kakuro_utils.py               -- sum-subsets lookup, constraint propagation (~200 LOC)
  kakuro_puzzle_bank.py         -- 30-50 hand-curated puzzles (~200 LOC for data + generators)

src/data/sft_formatter.py       -- add `kakuro_minimal` variant (~50 LOC delta)

scripts/
  generate_kakuro.sh            -- LLM-policy data gen launcher (~50 LOC)
  p1_count_kakuro_solutions.py  -- generate / verify puzzle bank (~150 LOC)
```

## Effort estimate

| Phase | Effort |
|---|---|
| 1. Puzzle data model + ~10 hand-curated puzzles | ~2 hr |
| 2. Constraint-propagation solvability checker | ~2 hr |
| 3. KakuroEnv class (reset/step/render/get_all_actions) | ~3 hr |
| 4. Test env: ensure all 10 puzzles can be solved + doom states detected | ~1 hr |
| 5. SFTFormatter `kakuro_minimal` variant | ~1 hr |
| 6. LLM-policy data gen on autodl1/2 | ~3 hr GPU |
| 7. SFT training (B-K1 = first Kakuro SFT) | ~2 hr GPU |
| 8. Eval (logprob threshold sweep + sanity rollout) | ~30 min |
| 9. RL with v8 anchor | ~5 hr GPU |
| **Total** | **~9 hr local + ~10 hr GPU = ~19 hr** |

## Success criteria

- **B-K1 SFT**: AUC ≥ 0.95, **greedy Pass@1 ≥ 5%** (this is the key — do not
  expect 0% like Pentomino).
- **B-K1 RL with v8 anchor**: Pass@1 climbs above SFT level, calibration
  preserved (`solvable_acc` ≥ 0.95).

If Kakuro greedy Pass@1 lifts above 0% on B-K1 SFT, we have validated the
hypothesis that constraint propagation + small action space enables greedy
in this env class.

## Risks

1. **Puzzle bank quality**: hand-curated puzzles might not span enough variety
   for SFT to generalize. Mitigation: aim for 30+ puzzles spanning 3×3 / 4×4 /
   5×5 sizes, varying clue patterns.
2. **Solvability checker performance**: backtracking on Kakuro can be slow
   on bigger boards. Mitigation: 5×5 max for our experiments.
3. **Predictive gap might be too small**: Kakuro is *very* greedy-friendly,
   maybe to the point that the model can solve every state trivially via
   constraint propagation without RL. Need to verify there are non-trivial
   doom states to predict.

If predictive gap is <20%, this won't be a useful env for the recipe — fall
back to Sudoku 6×6 instead.

## Status (2026-05-01)

- [x] Spec written
- [ ] Puzzle data model + bank
- [ ] Solvability checker
- [ ] Env class
- [ ] Tests
- [ ] SFT formatter variant
- [ ] LLM-policy data gen
- [ ] SFT
- [ ] RL
