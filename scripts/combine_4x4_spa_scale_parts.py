"""Combine 4x4 SPA-scale data parts (A from autodl1 + B from autodl2) into final dataset.

Run AFTER both parts are present locally (sync down from both clouds first).

Steps:
  1. Concat raw multi-turn parquets from `_spa_scale_A/` and `_spa_scale_B/` into `_spa_scale/`.
  2. Reformat combined raw to single-step minimal.
  3. Apply post-BP filter (4x4 has no length filter — short sequences).

Usage:
  python scripts/combine_4x4_spa_scale_parts.py [--root .]
"""
import argparse
import os
import subprocess
import sys
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Repo root")
    args = ap.parse_args()
    root = os.path.abspath(args.root)

    print("=== Step 1: concat A+B raw parquets ===")
    out_dir = os.path.join(root, "data/sudoku_4x4_llm_policy_spa_scale")
    os.makedirs(out_dir, exist_ok=True)
    for split in ["wm_train.parquet", "wm_val.parquet"]:
        parts = []
        for suf in ["A", "B"]:
            p = os.path.join(root, f"data/sudoku_4x4_llm_policy_spa_scale_{suf}/{split}")
            if os.path.exists(p):
                df = pd.read_parquet(p)
                parts.append(df)
                print(f"  {p}: {len(df)} rows")
        if not parts:
            print(f"  no parts for {split} — skipping")
            continue
        out = pd.concat(parts, ignore_index=True)
        out = out.sample(frac=1.0, random_state=42).reset_index(drop=True)
        out_path = os.path.join(out_dir, split)
        out.to_parquet(out_path)
        print(f"  -> {out_path}: {len(out)} rows")

    print("\n=== Step 2: reformat to single-step minimal ===")
    subprocess.run(
        [sys.executable, "scripts/reformat_to_minimal.py",
         "--input-dir", "data/sudoku_4x4_llm_policy_spa_scale",
         "--output-dir", "data/sudoku_4x4_llm_policy_minimal_spa_scale"],
        check=True, cwd=root,
    )

    print("\n=== Step 3: filter post-BP ===")
    subprocess.run(
        [sys.executable, "scripts/filter_post_bp.py",
         "--input-dir", "data/sudoku_4x4_llm_policy_minimal_spa_scale"],
        check=True, cwd=root,
    )

    print("\n=== Final sample counts ===")
    final_dir = os.path.join(root, "data/sudoku_4x4_llm_policy_minimal_spa_scale")
    for fn in sorted(os.listdir(final_dir)):
        if fn.endswith(".parquet"):
            n = len(pd.read_parquet(os.path.join(final_dir, fn)))
            print(f"  {fn:48s} {n:6d} samples")


if __name__ == "__main__":
    main()
