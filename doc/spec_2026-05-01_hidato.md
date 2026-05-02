# Spec — Hidato Env (2026-05-01)

A path-fill puzzle env added as a third game (after Sudoku and Pentomino) to
test whether the recipe transfers to a *greedy-friendly* env where forced
moves dominate. Replaces the previously-considered Kakuro after a switch on
2026-05-01 — Hidato is structurally simpler and even more greedy-friendly.

See [plan_2026-05-01_next_env_choice.md](plan_2026-05-01_next_env_choice.md)
for the rationale of choosing a third constraint-propagation env.

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

## API (mirrors PolyominoEnv / KakuroEnv)

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

## Puzzle bank: simple curation strategy

Hand-design ~10-15 puzzles by:
1. Pick a Hamiltonian path through a small grid by hand (or random search +
   verification).
2. Strip 30-50% of the cells to make them empty.
3. Verify the partial puzzle has a unique solution (or at least one valid
   completion) using the solvability checker.

For 4×4 grids (16 cells, K=15 transitions) this is fast.

Initial bank: 5-10 puzzles spanning 3×3, 4×4, 5×5 sizes.

## SFT formatter variant

Add `hidato_minimal` to `SFTFormatter`. Mostly a copy of `sudoku_minimal`
with hidato-specific rendering details (empty cells as `.`, integer cells
as their digits).

## Predictive gap test

Same as Pentomino: generate ~500 random rollouts, measure the fraction of
states that look valid (no immediate adjacency violation) but are actually
doomed.

Target: between 30% (good) and 70% (too sparse). If too low, increase grid
size or strip more givens.

## Files to create

```
src/environments/
  hidato.py                     -- HidatoEnv class (~300 LOC est)
  hidato_utils.py               -- solvability checker, helpers (~150 LOC)
  hidato_puzzle_bank.py         -- 10-15 hand-curated puzzles (~150 LOC)

src/data/sft_formatter.py       -- add `hidato_minimal` variant (~30 LOC delta)

scripts/
  generate_hidato.sh            -- LLM-policy data gen launcher (~50 LOC)
```

## Effort estimate

| Phase | Effort |
|---|---|
| 1. Solvability checker (`hidato_utils.py`) | ~2 hr |
| 2. Puzzle bank with 5-10 hand-curated entries | ~1.5 hr |
| 3. HidatoEnv class | ~2.5 hr |
| 4. Tests | ~30 min |
| 5. SFTFormatter `hidato_minimal` variant | ~45 min |
| 6. LLM-policy data gen | ~3 hr GPU |
| 7. SFT training (B-H1 = first Hidato SFT) | ~2 hr GPU |
| 8. Eval (logprob + sanity rollout) | ~30 min |
| 9. RL with v8 anchor | ~5 hr GPU |
| **Total** | **~7 hr local + ~10 hr GPU = ~17 hr** |

## Success criteria

- **B-H1 SFT**: AUC ≥ 0.95, **greedy Pass@1 ≥ 5%** (non-zero unlike Pentomino).
- **B-H1 RL with v8 anchor**: Pass@1 climbs above SFT level, calibration
  preserved (`solvable_acc` ≥ 0.95).
- **Predictive gap test**: 30%+ of states "look valid but doom" under random
  play.

## Risks

1. **Predictive gap might be too narrow**: Hidato's adjacency forcing is so
   strong that the model could solve every state by just following the only
   valid adjacent cell. If that happens, there's no predictive gap to learn.
   Mitigation: pick puzzles where multiple adjacent cells are valid early
   in the sequence.

2. **Backtracking solver could be slow on bigger grids**: 6×6 = 36 cells
   might push the depth bound. Mitigation: stick to 4×4 / 5×5 for v1.

3. **Trajectory length is long**: a 5×5 puzzle with 5 givens needs the model
   to place 20 numbers — a 20-step trajectory. Greedy probability compounds.
   Even at p_correct = 0.85 per step, 0.85^20 ≈ 4%. Mitigation: keep grids
   small (4×4 = 16 steps) or include more givens (fewer empty cells = shorter
   trajectories).

## Status (2026-05-01)

- [x] Spec written
- [ ] Solvability checker
- [ ] Puzzle bank
- [ ] Env class
- [ ] Tests
- [ ] SFT formatter variant
- [ ] LLM-policy data gen
- [ ] SFT
- [ ] RL
