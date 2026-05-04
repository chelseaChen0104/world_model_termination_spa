# Spec — Sudoku Env

The original env for the "world model SFT for termination prediction" recipe
and the canonical case for the project's predictive-gap thesis. Sudoku
predates the per-env-spec convention adopted with Pentomino (2026-04-29) and
Hidato (2026-05-01); this file backfills the env-level spec for parity with
[spec_pentomino.md](spec_pentomino.md) and
[spec_hidato.md](spec_hidato.md). Project-level
scope/decisions still live in [spec_project.md](spec_project.md); the Sudoku
*architecture* lives in [../CLAUDE.md](../CLAUDE.md); the Sudoku *pipeline*
lives in [pipeline_design.md](pipeline_design.md).

## Game rules

- Standard Sudoku on a `grid_size × grid_size` board (project default: 9×9;
  current 4×4 sub-task uses a 4×4 board with 2×2 boxes).
- Each cell holds an integer in `[1, grid_size]` or is empty.
- A *solution* fills every empty cell so that every row, every column, and
  every box contains each integer in `[1, grid_size]` exactly once.
- A *puzzle* exposes a partial state — some cells pre-filled (the "givens")
  with a difficulty parameter controlling the fill rate (`easy` ≈ 60% filled
  for 9×9; for 4×4 ~50% filled).

## Why Sudoku is the canonical predictive-gap env

Sudoku's predictive gap is constraint-cascade: a placement that is locally
valid (no immediate row/col/box conflict) can eliminate the only candidate
for some other cell several steps downstream. The board still *looks*
valid; no visual marker reveals the doom. Detecting this requires
constraint-propagation reasoning across rows, columns, and boxes — exactly
the kind of structured world-model task that motivates the recipe.

- **No simple deadlock detector exists** (in contrast to Sokoban's pattern
  matcher) — you have to reason about constraints.
- **Random play hits a breaking point quickly** — ~98% of random
  trajectories on 9×9 hit a doom state within 1-3 moves; on 4×4 the rate is
  similarly high. This drives strong class imbalance toward unsolvable
  states.

## Predictive gap

Examples of "looks fine but doomed" states:
- After placing 5 at R3C4, all of row 3, column 4, and the (1,1) box still
  have valid options for the empty cells. But the candidate set for R5C2
  has just collapsed to ∅ — invisible to a single-cell visual scan, takes
  constraint propagation over the (R5, C2, box-3) trio to detect.
- Two empty cells in the same row both have candidate set `{4}` — visually
  fine until you compare them.

This is the original predictive-gap motivation in [spec_project.md](spec_project.md) §1.

## API

```python
class SudokuEnv(BaseTerminationEnv):
    def __init__(
        self,
        grid_size: int = 9,
        difficulty: str = "easy",  # "easy" | "medium" | "hard"
        max_steps: Optional[int] = None,
    ):
        ...

    def reset(self, seed: Optional[int] = None) -> str:
        """Generate a fresh puzzle deterministically by seed; return rendered str."""

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Action format: 'place {N} at row {R} col {C}' (1-indexed).

        N must be a valid digit for that cell (no immediate row/col/box conflict).
        Otherwise action_is_valid=False.
        """

    def render(self) -> str:
        """ASCII grid render, with empty cells shown as '.'.
        Pipes separate boxes; dashes separate rows of boxes."""

    def check_solvability(self) -> Tuple[bool, Optional[str]]:
        """Constraint propagation + bounded backtracking (~2ms/check)."""

    def get_all_actions(self) -> list:
        """All (cell, value) pairs that pass the local-conflict filter
        (uses get_valid_numbers())."""
```

Implementation: [src/environments/sudoku.py](../src/environments/sudoku.py).
Solvability checker: `SudokuSolvabilityChecker` in [src/environments/sudoku_utils.py](../src/environments/sudoku_utils.py).

## Action format

```
<answer>place {N} at row {R} col {C}</answer>
```

where N ∈ `[1, grid_size]` and (R, C) are 1-indexed cell coordinates.

## Tag set

```
<observation>...</observation>
<prediction>...</prediction>
<solvable>true|false</solvable>
<answer>place 5 at row 3 col 7</answer>
```

The `<solvable>` tag is the original termination signal — Sudoku and Hidato
both use it. Pentomino renamed to `<viability>` to better reflect a tiling
context (see [spec_pentomino.md](spec_pentomino.md) §4).

## Solvability checker design

`SudokuSolvabilityChecker` (in `sudoku_utils.py`) combines:

1. **Constraint propagation** — for each empty cell, intersect the allowed
   values from its row, column, and box. If any cell's candidate set is ∅,
   the state is unsolvable.
2. **Naked single propagation** — if a cell has exactly one candidate,
   commit it and recurse.
3. **Bounded backtracking** — when no naked singles remain, branch on the
   most-constrained cell. Hard cap at ~10K backtrack steps to keep
   per-state cost ≈ 2ms.

`is_solvable(state)` returns `(True, None)` for solvable states or
`(False, reason)` for unsolvable ones, where `reason` flags the failure
type ("contradiction", "depth_exceeded", etc.) — used downstream for
deadlock-type recall metrics.

## SFT formatter variants

Two variants in [src/data/sft_formatter.py](../src/data/sft_formatter.py):

- `sudoku_full` — the v3-era full tag set
  (`<observation>` + `<prediction>` + `<terminate_prob>` + `<steps_left>`
  + `<solvable>` + `<breaking_point>`). Used for early data
  (`data/sudoku_termination/`). Deprecated post-v4 pivot.
- `sudoku_minimal` — current minimal tag set per
  [spec_project.md](spec_project.md) §7.5 v4: `<observation>` + `<prediction>`
  + `<solvable>` + `<answer>` only.

## Datasets

- **`data/sudoku_termination/`** — random-play, full-tag (v3); 32k samples
  (25,649 train / 6,413 val); class distribution 6.6% solvable / 93.4%
  unsolvable. Deprecated.
- **`data/sudoku_4x4_llm_policy/`** — 4×4 LLM-policy data, minimal tag set;
  used for B-5 and downstream RL. See
  [runs_reference_2026-05-01.md](runs_reference_2026-05-01.md) for the
  current canonical dataset.

LLM-policy generation uses [src/data/llm_trajectory_generator.py](../src/data/llm_trajectory_generator.py)
with `--env sudoku --grid-size 4 --difficulty easy`.

## SFT runs

The Sudoku 4×4-easy track (B-5 family) is the headline SFT recipe and the
basis for all RL phases. See
[runs_reference_2026-05-01.md](runs_reference_2026-05-01.md) "SFT runs
(Sudoku)" for the full ledger; key checkpoint:

- **B-5** (`outputs/sft_sudoku_4x4_minimal_b5_spa_hparams/final/`):
  Qwen2.5-1.5B-Instruct, lr=1e-4, ep=5, bs=16 effective, max_length=1024.
  AUC=0.726, the first SFT run with real `<solvable>` discrimination.

## RL runs

See [runs_reference_2026-05-01.md](runs_reference_2026-05-01.md) "RL runs
(Sudoku)" for full detail. Key checkpoint chain:

1. **Run A / Phase 1** (`outputs/rl_b5_phase1_v6_1`) — v6.1 reward,
   500 steps, lifted Pass@1 6.67% → 33.33% but `solvable_acc` drifted
   0.62 → 0.51 (calibration regression).
2. **Phase 2** (`outputs/rl_b5_phase2_continue`) — continuation, basis for
   v8 anchor.
3. **Phase 3 v8 anchor** (`outputs/rl_b5_phase3_v8_anchor`) — v8 viability-tag
   KL anchor, restored calibration AND held Pass@1 (Pass@1 33% → 50%,
   `solvable_acc` ~0.80, `bp_recall` 1.0). Current canonical RL checkpoint.
4. **Phase 3 v8.2 dual anchor** (`outputs/rl_b5_phase3_v8_2_dual_anchor`) —
   in flight at time of writing (2026-05-01); tests whether dual-token
   anchor cures bimodal confidence in the truncation gate.

## Truncation experiment

The current canonical compute-saving experiment runs on the Sudoku v8 anchor
checkpoint and probes the conservative truncation gate. Full doc:
[eval_2026-05-01_truncation_full.md](eval_2026-05-01_truncation_full.md).
Headline: 22% rollout-token savings at the cost of −10pp eval Pass@1 (clean
re-eval).

## Files

```
src/environments/
  sudoku.py                       -- SudokuEnv class
  sudoku_utils.py                 -- puzzle gen + solvability checker
  base.py                         -- BaseTerminationEnv (shared)

src/data/
  sft_formatter.py                -- sudoku_minimal + sudoku_full variants
  llm_trajectory_generator.py     -- LLM-policy data gen (--env sudoku)
  trajectory_generator.py         -- random-play data gen (legacy)

scripts/
  run_sudoku_4x4_rl_v6_phase1.sh             -- v6.1 RL launcher
  run_sudoku_4x4_rl_v8_phase3.sh          -- v8 anchor launcher
  run_sudoku_4x4_rl_v8_2_dual_anchor.sh        -- v8.2 dual-anchor launcher
  run_truncation_tau_sweep.sh     -- truncation τ-sweep
  run_truncation_min_step_sweep.sh -- truncation min-step sweep
```

## Status

Sudoku is the longest-running track in the project. All env infrastructure,
SFT recipe, RL recipe, and truncation experiment are in place; remaining
work is the v8.2 dual-anchor RL eval and any τ-sweep follow-ups it enables.

For currently-active runs see
[runs_reference_2026-05-01.md](runs_reference_2026-05-01.md) and
[HANDOFF.md](HANDOFF.md).
