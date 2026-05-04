# Spec — Hidato Env

A path-fill puzzle env added as a third game (after Sudoku and Pentomino) to
test whether the recipe transfers to a *greedy-friendly* env where forced
moves dominate. Replaces the previously-considered Kakuro after a switch on
2026-05-01 — Hidato is structurally simpler and even more greedy-friendly.

See [plan_2026-05-01_next_env_choice.md](plan_2026-05-01_next_env_choice.md)
for the rationale of choosing a third constraint-propagation env.

> **Spec status (2026-05-03)**: env + SFT + RL all built and running.
> See "Status & Results" section at the bottom for the current numbers.

## Game rules

- An R×C grid (no holes for v1; full rectangle).
- Each cell either contains an integer in [1, R·C] or is empty (denoted 0).
- The cells with assigned numbers form (eventually) the entire sequence
  1, 2, 3, …, R·C, with the additional rule:
  **Consecutive numbers (k and k+1) must be in adjacent cells** — adjacent here
  means orthogonally adjacent (share an edge). This is the Numbrix variant;
  the classic Hidato uses 8-connectivity (also diagonal). We pick orthogonal
  to keep the action space and validity check small.
- A *puzzle* gives the model a partial state: some cells are pre-filled (the
  "givens") with their final numbers, others are empty.
- A *solution* is an assignment of the missing numbers such that the
  consecutive-adjacency rule holds for every k in 1..R·C-1.

## Why Hidato is greedy-friendly

At any state where number k has been placed at cell X, **the next number k+1
must go at one of the (up to 4) cells orthogonally adjacent to X that are
still empty**. In practice:
- Often only 1 or 2 of those adjacent cells are valid (others are blocked by
  given numbers from later in the sequence, or by walls).
- Solvability frequently requires a unique choice (placing k+1 in the wrong
  adjacent cell creates a downstream dead-end).

→ The argmax over valid next-cell choices is much more likely to be correct
than on Pentomino's spread-thin first-move distribution.

## API (mirrors PolyominoEnv)

```python
class HidatoEnv(BaseTerminationEnv):
    def __init__(self, puzzle_bank: list, max_steps: Optional[int] = None):
        """
        puzzle_bank: list of dicts, each describing one puzzle:
          {
            "id": str,
            "rows": int,
            "cols": int,
            "givens": dict[(r, c) -> int],   # pre-filled cells
            "solution": dict[(r, c) -> int], # full solution (for validation)
          }
        """

    def reset(self, seed: Optional[int] = None) -> str:
        """Pick a puzzle from the bank deterministically by seed; return rendered str."""

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Action format: 'place {N} at row {R} col {C}' (1-indexed).

        N must equal `max_assigned + 1` (the next number in the sequence).
        Cell (R, C) must be empty AND adjacent to the cell containing
        `max_assigned`. Otherwise action_is_valid=False.
        """

    def render(self) -> str:
        """ASCII grid render, with empty cells shown as '.'."""

    def check_solvability(self) -> Tuple[bool, Optional[str]]:
        """Backtracking search for a valid completion of the partial state."""

    def get_all_actions(self) -> list:
        """All cells that are valid next-step placements (empty + adjacent to k)."""
```

## Action format

```
<answer>place {N} at row {R} col {C}</answer>
```

where N is the sequential number to place (= max number already on the board + 1)
and (R, C) are 1-indexed cell coordinates.

The model could also emit just the cell (since N is implied), but for
consistency with Sudoku / Polyomino we keep the explicit number.

## Tag set

Same as Sudoku (`<solvable>`, NOT `<viability>`). Hidato is a constraint-
satisfaction puzzle; the existing Sudoku tag aligns well.

```
<observation>...</observation>
<prediction>...</prediction>
<solvable>true|false</solvable>
<answer>place 5 at row 2 col 3</answer>
```

## Solvability checker design

State: partial grid with some cells filled, max k already on the board.

Algorithm (backtracking with strong pruning):

```python
def is_solvable(state, k_target):
    if k_target > rows*cols:
        return True   # all numbers placed
    if k_target is already on the board (was a given):
        return is_solvable(state, k_target + 1)
    # else: place k_target adjacent to where k_target-1 lives
    cell_with_prev = locate(k_target - 1)
    for each empty cell c adjacent to cell_with_prev:
        # Pre-check: would placing k_target at c make some FUTURE given unreachable?
        # Cheap: check that all empty cells remain reachable from c (BFS connectivity)
        place k_target at c
        if all_remaining_givens_reachable(state):  # cheap propagation
            if is_solvable(state, k_target + 1):
                return True
        unplace
    return False
```

Pruning to keep the search bounded:
1. **Adjacency restriction**: only try cells adjacent to where k-1 is.
2. **Reachability check**: after each placement, BFS to confirm all remaining
   given cells are reachable in the right order. Constant-time per cell.
3. **Depth bound**: hard cap at e.g. 10K backtrack steps.

For 5×5 grids (25 cells) this is fast. For 6×6 it's still tractable.

## Predictive gap

Examples of "looks fine but doomed" states:
- After placing 1, 2, 3 in a sequence, the cell that should hold 4 is empty
  but adjacent to the cell with 3. However, if (4 → 5 → 6 → ...) needs to
  reach a given at the far corner, the choice of (4) may fork into a region
  that disconnects from the corner. Visually local (the 4 placement looks
  fine) but globally doomed.

This is the same flavor of predictive gap as Pentomino (locally valid,
globally unreachable), but the per-step argmax-correct probability is much
higher because adjacency forces the option set to be small.

## Puzzle bank

Hand-curated, lives in [src/environments/hidato_puzzle_bank.py](../src/environments/hidato_puzzle_bank.py).
**8 puzzles** spanning sizes 3×3 to 5×4:

| ID | Size | Cells | Givens | Empty (= rollout length) |
|---|---|---|---|---|
| 3x3_snake / 3x3_u / 3x3_spiral | 3×3 | 9 | 2 | 7 |
| 4x3_snake | 4×3 | 12 | 2 | 10 |
| 5x3_snake | 5×3 | 15 | 3 | 12 |
| 4x4_boustrophedon / 4x4_spiral | 4×4 | 16 | 2-3 | 13–14 |
| 5x4_snake | 5×4 | 20 | 3 | 17 |

> **Known limitation (2026-05-03):** 8 puzzles is too small. Eval cycles
> through them ~3.75× per 30-puzzle eval, and training data hits 98.5%
> exact-duplicate rate (only ~183 unique (prompt, response) pairs after
> augmentation × oversample). The 60% Pass@1 we see post-RL is partly
> memorization-driven; held-out generalization untested. **Expanding the
> bank to 50-200 puzzles via algorithmic generation is queued in
> [future_steps.md](future_steps.md) NEAR-6.** Visualization of three
> representative puzzles in [plots/hidato_examples.png](plots/hidato_examples.png).

## Files (built; no longer "to create")

| File | Role |
|---|---|
| [src/environments/hidato.py](../src/environments/hidato.py) | `HidatoEnv` class |
| [src/environments/hidato_utils.py](../src/environments/hidato_utils.py) | Solvability checker (backtracking + connectivity pruning) |
| [src/environments/hidato_puzzle_bank.py](../src/environments/hidato_puzzle_bank.py) | 8 hand-curated puzzles |
| [src/data/sft_formatter.py](../src/data/sft_formatter.py) `hidato_minimal` variant | SFT response formatter |
| [scripts/generate_hidato.sh](../scripts/generate_hidato.sh) | LLM-policy data-gen launcher |
| [src/data/hidato_solution_path_augmenter.py](../src/data/hidato_solution_path_augmenter.py) | Solution-path augmenter (needed for SFT to escape regime-1) |
| [scripts/combine_hidato_with_augmented.py](../scripts/combine_hidato_with_augmented.py) | LLM-policy + augmented data combiner |
| [scripts/run_hidato_sft.sh](../scripts/run_hidato_sft.sh) | SFT launcher |
| [scripts/run_hidato_rl_v8.sh](../scripts/run_hidato_rl_v8.sh) | RL launcher (v8 anchor) |
| [scripts/run_hidato_full_pipeline_no_leak.sh](../scripts/run_hidato_full_pipeline_no_leak.sh) | Strip leak + SFT + RL + eval |

## Risks (current state)

1. **Predictive gap is narrow but not too narrow.** ~67% of random rollouts
   end in doom; the predictive gap is real but adjacency forcing makes it
   easier than Pentomino. Confirmed via random-rollout test 2026-05-01.

2. **Trajectory length is the key cost driver.** A 5×4 puzzle with 3 givens
   = 17 placements. Per-step `p_correct = 0.95` gives `0.95^17 ≈ 0.42` —
   high enough to score 60% greedy Pass@1 with our current model.

3. **Bank is too small** (above) → memorization risk → unmitigated unless
   we expand. See NEAR-6.

## Status & Results (snapshot 2026-05-03)

- [x] Solvability checker (built; backtracking + reachability pruning)
- [x] Puzzle bank (8 puzzles)
- [x] HidatoEnv class
- [x] Tests
- [x] SFT formatter variant `hidato_minimal`
- [x] LLM-policy data gen (3000 trajectories, 9627 single-step samples)
- [x] Solution-path augmenter (80 unique augmented samples × 30 oversample = 2400)
- [x] B-H1 combined SFT dataset (12,027 train, 80% solvable / 20% doom)
- [x] **B-H1 SFT**: AUC = 1.000 (perfect logprob discrimination), eval_loss 0.0014
- [x] **B-H1 SFT Pass@1 with eval-pipeline fix**: 16.7% greedy (after `--prepend-current-state --single-turn-eval --max-response-tokens 512`); 0% without
- [x] **B-H1 RL with v8 anchor (leaked SFT)**: Pass@1 60% greedy at step 175+ — major lift; calibration preserved (solvable_acc 1.0, bp_recall 1.0)
- [ ] **Hidato no-leak SFT + RL**: queued to launch on autodl1 after current RL finishes (will use v8 anchor + `--action-quality-bonus 0.5`)
- [ ] **Held-out evaluation**: pending bank expansion (NEAR-6 in future_steps.md)

The SFT-stage greedy Pass@1 success criterion (≥ 5%) was achieved at 16.7%.
The RL-stage criterion (Pass@1 climbs above SFT, calibration preserved) was
achieved at 60% with `solvable_acc=1.0`. Both well past spec targets.

Caveat: the 60% number is on the same 8-puzzle bank as training. We don't
yet know how the model performs on held-out Hidato puzzles.
