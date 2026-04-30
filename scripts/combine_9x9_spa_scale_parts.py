"""Combine 9x9 SPA-scale data parts (A from autodl1 + B from autodl2) into final dataset.

Run AFTER both parts have been synced to local (or to one cloud).

Steps:
  1. For each difficulty in {easy, medium, hard}: concat raw multi-turn parquets
     from `*_spa_scale_A/` and `*_spa_scale_B/` into `*_spa_scale/`.
  2. Reformat each combined difficulty to single-step minimal (calls
     scripts/reformat_to_minimal.py).
  3. Concat across difficulties into data/sudoku_llm_policy_minimal_spa_scale/.
  4. Apply length filter and post-BP filter.

Usage:
  python scripts/combine_9x9_spa_scale_parts.py [--root .]
"""
import argparse
import os
import subprocess
import sys
import pandas as pd

DIFFICULTIES = ["easy", "medium", "hard"]


def concat_parts(root, diff):
    """Concat A + B raw parquets for a single difficulty into combined dir."""
    out_dir = os.path.join(root, f"data/sudoku_llm_policy_{diff}_spa_scale")
    os.makedirs(out_dir, exist_ok=True)
    for split in ["wm_train.parquet", "wm_val.parquet"]:
        parts = []
        for suf in ["A", "B"]:
            p = os.path.join(root, f"data/sudoku_llm_policy_{diff}_spa_scale_{suf}/{split}")
            if os.path.exists(p):
                df = pd.read_parquet(p)
                parts.append(df)
                print(f"  {p}: {len(df)} rows")
        if not parts:
            print(f"  no parts for {diff}/{split} — skipping")
            continue
        out = pd.concat(parts, ignore_index=True)
        out_path = os.path.join(out_dir, split)
        out.to_parquet(out_path)
        print(f"  -> {out_path}: {len(out)} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Repo root")
    args = ap.parse_args()
    root = os.path.abspath(args.root)

    print("=== Step 1: concat A+B per difficulty ===")
    for diff in DIFFICULTIES:
        print(f"\n[{diff}]")
        concat_parts(root, diff)

    print("\n=== Step 2: reformat each difficulty to single-step minimal ===")
    for diff in DIFFICULTIES:
        src = f"data/sudoku_llm_policy_{diff}_spa_scale"
        dst = f"data/sudoku_llm_policy_{diff}_spa_scale_minimal"
        if not os.path.isdir(os.path.join(root, src)):
            print(f"  skip {diff}: {src} missing")
            continue
        print(f"  reformat {diff}: {src} -> {dst}")
        subprocess.run(
            [sys.executable, "scripts/reformat_to_minimal.py",
             "--input-dir", src, "--output-dir", dst],
            check=True, cwd=root,
        )

    print("\n=== Step 3: concat across difficulties ===")
    out_dir = os.path.join(root, "data/sudoku_llm_policy_minimal_spa_scale")
    os.makedirs(out_dir, exist_ok=True)
    for split in ["wm_train.parquet", "wm_val.parquet"]:
        parts = []
        for diff in DIFFICULTIES:
            p = os.path.join(root, f"data/sudoku_llm_policy_{diff}_spa_scale_minimal/{split}")
            if os.path.exists(p):
                df = pd.read_parquet(p)
                df["difficulty"] = diff
                parts.append(df)
                print(f"  {p}: {len(df)} rows")
        if not parts:
            print(f"  no parts for {split}")
            continue
        out = pd.concat(parts, ignore_index=True)
        out = out.sample(frac=1.0, random_state=42).reset_index(drop=True)
        out.to_parquet(os.path.join(out_dir, split))
        print(f"  -> {split}: {len(out)} rows")

    print("\n=== Step 4: filter long samples + post-BP ===")
    for script in ["filter_long_samples.py", "filter_post_bp.py"]:
        script_path = os.path.join(root, "scripts", script)
        if os.path.exists(script_path):
            subprocess.run(
                [sys.executable, script_path,
                 "--input-dir", "data/sudoku_llm_policy_minimal_spa_scale"],
                check=True, cwd=root,
            )
        else:
            print(f"  skip: {script} not found")

    print("\n=== Final sample counts ===")
    for fn in sorted(os.listdir(out_dir)):
        if fn.endswith(".parquet"):
            n = len(pd.read_parquet(os.path.join(out_dir, fn)))
            print(f"  {fn:48s} {n:6d} samples")


if __name__ == "__main__":
    main()
