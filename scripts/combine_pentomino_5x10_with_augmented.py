"""Build the 5×10 augmented Pentomino SFT dataset.

Mirror of combine_pentomino_5x4_with_augmented.py but for the 5×10 / 10-piece
configuration. Targets the same ~30–50% augmented ratio that B-8 used.

Why subsample at the augmenter level (not here): with 4664 distinct tilings ×
10 pieces = 46,640 augmented samples and only 4,652 LLM-policy samples, using
the full augmented set would push the dataset to ~91% solvable. We cap
augmented at the augmenter level (--max-tilings 466) so it's roughly 1× the
LLM-policy size, then optionally repeat here for finer control.

Usage:
  python scripts/combine_pentomino_5x10_with_augmented.py \\
      --llm-dir data/pentomino_b9_llm_policy_minimal \\
      --aug-dir data/pentomino_5x10_solution_paths \\
      --output-dir data/pentomino_5x10_combined
"""
from __future__ import annotations
import argparse
import os
import json
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--llm-dir", default="data/pentomino_b9_llm_policy_minimal")
    p.add_argument("--aug-dir", default="data/pentomino_5x10_solution_paths")
    p.add_argument("--output-dir", default="data/pentomino_5x10_combined")
    p.add_argument("--aug-repeat", type=int, default=1,
                   help="How many times to repeat each augmented sample. "
                        "With --max-tilings 466 in the augmenter, repeat=1 "
                        "yields ~50%% augmented ratio.")
    p.add_argument("--use-no-post-bp", action="store_true", default=True)
    args = p.parse_args()

    suffix = "_no_post_bp" if args.use_no_post_bp else ""
    llm_train = pd.read_parquet(os.path.join(args.llm_dir, f"wm_train{suffix}.parquet"))
    llm_val = pd.read_parquet(os.path.join(args.llm_dir, f"wm_val{suffix}.parquet"))
    aug_train = pd.read_parquet(os.path.join(args.aug_dir, "wm_train_solution_paths.parquet"))
    aug_val = pd.read_parquet(os.path.join(args.aug_dir, "wm_val_solution_paths.parquet"))

    print("=== Combining 5×10 LLM-policy + augmented ===")
    print(f"  LLM  train: {len(llm_train):>6} samples")
    print(f"  LLM  val:   {len(llm_val):>6} samples")
    print(f"  Aug  train: {len(aug_train):>6} samples (× {args.aug_repeat} repeat)")
    print(f"  Aug  val:   {len(aug_val):>6} samples (× {args.aug_repeat} repeat)")

    aug_train_repeated = pd.concat([aug_train] * args.aug_repeat, ignore_index=True)
    aug_val_repeated = pd.concat([aug_val] * args.aug_repeat, ignore_index=True)
    combined_train = pd.concat([llm_train, aug_train_repeated], ignore_index=True)
    combined_val = pd.concat([llm_val, aug_val_repeated], ignore_index=True)
    combined_train = combined_train.sample(frac=1.0, random_state=42).reset_index(drop=True)
    combined_val = combined_val.sample(frac=1.0, random_state=42).reset_index(drop=True)

    os.makedirs(args.output_dir, exist_ok=True)
    for name, df in [("wm_train.parquet", combined_train),
                     ("wm_val.parquet", combined_val),
                     ("wm_train_no_post_bp.parquet", combined_train),
                     ("wm_val_no_post_bp.parquet", combined_val)]:
        df.to_parquet(os.path.join(args.output_dir, name), index=False)

    print(f"\n  Combined train: {len(combined_train):>6} samples")
    print(f"  Combined val:   {len(combined_val):>6} samples")
    print(f"  Augmented frac (train): {(len(aug_train_repeated)/len(combined_train))*100:.1f}%")

    def get_solv(row):
        info = row["extra_info"]
        if isinstance(info, str):
            info = json.loads(info)
        return info.get("is_solvable", None)
    combined_train["solv"] = combined_train.apply(get_solv, axis=1)
    n_solv = (combined_train["solv"] == True).sum()
    n_doom = (combined_train["solv"] == False).sum()
    print(f"\nCombined train class composition:")
    print(f"  is_solvable=True:  {n_solv} ({100*n_solv/len(combined_train):.1f}%)")
    print(f"  is_solvable=False: {n_doom} ({100*n_doom/len(combined_train):.1f}%)")


if __name__ == "__main__":
    main()
