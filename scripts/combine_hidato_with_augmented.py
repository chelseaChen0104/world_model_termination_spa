"""Build the B-H1 SFT dataset by concatenating Hidato LLM-policy data with
solution-path augmented samples.

Mirror of scripts/combine_pentomino_5x4_with_augmented.py, but for Hidato.

Why oversample: the augmented data has ~80 unique samples (one per
empty cell across the 8-puzzle bank). Without oversampling, augmented
samples are tiny relative to the LLM-policy doom corpus (~5000+ samples).
Oversampling by a meaningful factor (default 30×) brings augmented samples
to ~30-45% of the combined dataset, ensuring enough gradient passes on the
multi-step solution-path examples.
"""
from __future__ import annotations
import argparse
import os
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--llm-dir", default="data/hidato_llm_policy_minimal")
    p.add_argument("--aug-dir", default="data/hidato_solution_paths")
    p.add_argument("--output-dir", default="data/hidato_b_h1_combined")
    p.add_argument("--aug-repeat", type=int, default=30,
                   help="How many times to repeat each augmented sample. "
                        "Default 30 → ~30-45%% of combined dataset is augmented.")
    p.add_argument("--use-no-post-bp", action="store_true", default=True,
                   help="Use the *_no_post_bp.parquet variants of the LLM-policy data.")
    args = p.parse_args()

    suffix = "_no_post_bp" if args.use_no_post_bp else ""
    llm_train = pd.read_parquet(os.path.join(args.llm_dir, f"wm_train{suffix}.parquet"))
    llm_val = pd.read_parquet(os.path.join(args.llm_dir, f"wm_val{suffix}.parquet"))
    aug_train = pd.read_parquet(os.path.join(args.aug_dir, "wm_train_solution_paths.parquet"))
    aug_val = pd.read_parquet(os.path.join(args.aug_dir, "wm_val_solution_paths.parquet"))

    print("=== Combining Hidato LLM-policy + augmented → B-H1 ===")
    print(f"  LLM  train: {len(llm_train):>6} samples")
    print(f"  LLM  val:   {len(llm_val):>6} samples")
    print(f"  Aug  train: {len(aug_train):>6} samples (× {args.aug_repeat} repeat)")
    print(f"  Aug  val:   {len(aug_val):>6} samples (× {args.aug_repeat} repeat)")

    aug_train_repeated = pd.concat([aug_train] * args.aug_repeat, ignore_index=True)
    aug_val_repeated = pd.concat([aug_val] * args.aug_repeat, ignore_index=True)

    combined_train = pd.concat([llm_train, aug_train_repeated], ignore_index=True)
    combined_val = pd.concat([llm_val, aug_val_repeated], ignore_index=True)

    # Shuffle
    combined_train = combined_train.sample(frac=1.0, random_state=42).reset_index(drop=True)
    combined_val = combined_val.sample(frac=1.0, random_state=42).reset_index(drop=True)

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "wm_train.parquet")
    val_path = os.path.join(args.output_dir, "wm_val.parquet")
    train_path_nopb = os.path.join(args.output_dir, "wm_train_no_post_bp.parquet")
    val_path_nopb = os.path.join(args.output_dir, "wm_val_no_post_bp.parquet")

    combined_train.to_parquet(train_path, index=False)
    combined_val.to_parquet(val_path, index=False)
    combined_train.to_parquet(train_path_nopb, index=False)
    combined_val.to_parquet(val_path_nopb, index=False)

    print(f"\n  Combined train: {len(combined_train):>6} samples")
    print(f"  Combined val:   {len(combined_val):>6} samples")
    print(f"  Augmented frac (train): {(len(aug_train_repeated) / len(combined_train)) * 100:.1f}%")
    print(f"\nWrote:")
    for pth in (train_path, val_path, train_path_nopb, val_path_nopb):
        print(f"  {pth}")

    # Class composition (is_solvable distribution)
    import json
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
