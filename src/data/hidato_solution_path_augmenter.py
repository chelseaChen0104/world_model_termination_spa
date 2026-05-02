"""Solution-path SFT data augmenter for Hidato.

Same motivation as `solution_path_augmenter.py` (Polyomino): LLM-policy data
gen on Hidato shows 0% success rate (Qwen2.5 produces parseable but
strategically bad actions, never reaching a complete solution). Without
positive-class samples in SFT, the model can't learn what valid sequences
look like → recipe stalls at "Pass@1 stochastic = 0", same as B-7 Pentomino.

This augmenter cures it deterministically:
  1. For each puzzle in the bank, walk the env through the puzzle's known
     solution in sequence (place 1 (or skip if given), then 2 (or skip), ...).
  2. Each placement becomes one (state, action, is_solvable=True) sample.
  3. Format via SFTFormatter so output is bit-for-bit compatible with the
     existing data pipeline.

For 8 puzzles in the bank with empty-cell counts {7, 7, 7, 14, 13, 10, 12, 17},
total = ~87 unique solution-path samples per pass. Oversample (--repeat) to
balance against the LLM-policy doom samples.

Output: every sample has `is_solvable=True`, `is_breaking_point=False`, with
trajectory positions covering 0..N-1 for each puzzle's empty-cell count.

Usage:
  python -m src.data.hidato_solution_path_augmenter \\
      --output-dir data/hidato_solution_paths
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

# Repo path for src.* imports (when invoked as a script)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.environments.hidato import HidatoEnv  # noqa: E402
from src.environments import hidato_puzzle_bank as bank  # noqa: E402
from src.data.trajectory_generator import (  # noqa: E402
    TrajectoryStep, TrajectoryMetadata,
)
from src.data.sft_formatter import SFTFormatter  # noqa: E402


def _bucket(delta: int) -> str:
    if delta == 0:
        return "immediate"
    if delta <= 3:
        return "near"
    if delta <= 7:
        return "medium"
    return "far"


def hidato_solution_to_trajectory(puzzle: dict) -> tuple:
    """Step a fresh HidatoEnv through `puzzle`'s known solution and emit a
    sequence of TrajectoryStep records (one per empty cell to fill).

    Returns (steps, metadata). All steps have `is_solvable=True` (we're on a
    valid solution path). Final step has done_label=1 and success=True.
    """
    env = HidatoEnv(puzzle_bank=[puzzle])
    state = env.reset(seed=0)
    steps: list = []
    n_cells = env.rows * env.cols
    n_empty_total = n_cells - len(puzzle["givens"])

    while True:
        next_n = env._next_required_number()
        if next_n is None or next_n > n_cells:
            break
        # Find the cell where next_n lives in the solution.
        target_cell = next(
            (pos for pos, v in puzzle["solution"].items() if v == next_n), None
        )
        if target_cell is None:
            raise RuntimeError(
                f"Puzzle {puzzle['id']} solution missing number {next_n}"
            )
        r, c = target_cell
        action_str = f"place {next_n} at row {r + 1} col {c + 1}"
        next_state, reward, done, info = env.step(action_str)
        if not info["action_is_valid"]:
            raise RuntimeError(
                f"Solution-path action {action_str} rejected by env on "
                f"puzzle {puzzle['id']}: {env.last_action_feedback}"
            )
        i_step = len(steps)
        steps_left = (n_empty_total - 1) - i_step  # 0 on final
        step = TrajectoryStep(
            state=state,
            action=i_step,                  # placeholder (we use action_name)
            action_name=info.get("action_name", action_str),
            next_state=next_state,
            reward=reward,
            step=i_step,
            done_label=1 if done else 0,
            steps_left=max(0, steps_left),
            steps_left_bucket=_bucket(max(0, steps_left)),
            is_solvable=True,                # by construction (on a valid path)
            is_breaking_point=False,
            deadlock_type=None,
            steps_since_break=None,
            success=info.get("success", False),
        )
        steps.append(step)
        state = next_state
        if done:
            break

    metadata = TrajectoryMetadata(
        total_steps=len(steps),
        success=True,
        has_breaking_point=False,
        breaking_point_step=None,
        steps_wasted=0,
        final_reward=1.0,
        termination_reason="success",
    )
    return steps, metadata


def build_sft_rows(trajectories, formatter: SFTFormatter) -> list:
    """Convert (steps, metadata) list to SFTFormatter-formatted rows."""
    df = formatter.create_sft_dataset(trajectories)
    if "extra_info" in df.columns:
        df["extra_info"] = df["extra_info"].apply(
            lambda v: v if isinstance(v, str) else json.dumps(v, default=str)
        )
    return df.to_dict("records")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="hidato_minimal", help="SFTFormatter variant")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--repeat", type=int, default=1,
                   help="Repeat each solution-path sample this many times "
                        "(simple oversample to balance against larger doom datasets).")
    p.add_argument("--val-frac", type=float, default=0.1,
                   help="Fraction of puzzles held out for val.")
    args = p.parse_args()

    print(f"=== hidato_solution_path_augmenter ===")
    print(f"  variant:    {args.variant}")
    print(f"  repeat:     {args.repeat}")
    print(f"  val_frac:   {args.val_frac}")
    print(f"  puzzle bank: {len(bank.PUZZLES)} puzzles")

    # Generate one trajectory per puzzle
    print(f"\nWalking each puzzle through its solution...")
    trajectories = []
    for puzzle in bank.PUZZLES:
        steps, meta = hidato_solution_to_trajectory(puzzle)
        trajectories.append((steps, meta))
        n_empty = puzzle["rows"] * puzzle["cols"] - len(puzzle["givens"])
        print(f"  {puzzle['id']:>22s}: {len(steps)} steps emitted (empty cells = {n_empty})")

    # Train/val split at puzzle granularity
    n_val = max(1, int(args.val_frac * len(trajectories)))
    val_trajs = trajectories[:n_val]
    train_trajs = trajectories[n_val:]
    print(f"  train puzzles: {len(train_trajs)}, val puzzles: {len(val_trajs)}")

    if args.repeat > 1:
        train_trajs = train_trajs * args.repeat
        val_trajs = val_trajs * args.repeat
        print(f"  after repeat × {args.repeat}: train={len(train_trajs)}, val={len(val_trajs)}")

    # Format via SFTFormatter
    print(f"\nFormatting via SFTFormatter(variant={args.variant!r})...")
    formatter = SFTFormatter(variant=args.variant)
    train_rows = build_sft_rows(train_trajs, formatter)
    val_rows = build_sft_rows(val_trajs, formatter)
    print(f"  train rows: {len(train_rows)}, val rows: {len(val_rows)}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "wm_train_solution_paths.parquet")
    val_path = os.path.join(args.output_dir, "wm_val_solution_paths.parquet")
    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(val_rows).to_parquet(val_path, index=False)
    print(f"\nWrote:")
    print(f"  {train_path}")
    print(f"  {val_path}")

    # Trajectory-position distribution
    from collections import Counter
    step_counts = Counter()
    for r in train_rows:
        info = r["extra_info"]
        if isinstance(info, str):
            info = json.loads(info)
        step_counts[info.get("step", -1)] += 1
    print(f"\nTrain trajectory-position distribution:")
    for k in sorted(step_counts):
        print(f"  step {k:>3d}: {step_counts[k]} samples")


if __name__ == "__main__":
    main()
