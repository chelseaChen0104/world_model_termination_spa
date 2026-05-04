"""Build the B-8 SFT dataset by concatenating the existing B-7 data with the
solution-path augmented samples.

Why oversample: B-7 has 2964 train samples (~80% step-0); the augmenter produces
72 train samples uniformly across step 0-3. Without oversampling, augmented
samples are only ~2% of the dataset and the gradient signal for late-stage
states is very diluted. We oversample the augmented set so it's ~20% of the
combined dataset, ensuring the model gets enough passes on the multi-step
samples to learn coherent late-stage format.

Output: data/pentomino_b8_combined/ with the same parquet schema as B-7.
"""
from __future__ import annotations
import argparse
import os
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--b7-dir", default="data/pentomino_easy_llm_policy_minimal")
    p.add_argument("--aug-dir", default="data/pentomino_easy_solution_paths")
    p.add_argument("--output-dir", default="data/pentomino_b8_combined")
    p.add_argument("--aug-repeat", type=int, default=10,
                   help="How many times to repeat each augmented sample. "
                        "Default 10 → ~20% of combined dataset is augmented.")
    p.add_argument("--use-no-post-bp", action="store_true", default=True,
                   help="Use the *_no_post_bp.parquet variants of B-7.")
    args = p.parse_args()

    suffix = "_no_post_bp" if args.use_no_post_bp else ""
    b7_train = pd.read_parquet(os.path.join(args.b7_dir, f"wm_train{suffix}.parquet"))
    b7_val = pd.read_parquet(os.path.join(args.b7_dir, f"wm_val{suffix}.parquet"))
    aug_train = pd.read_parquet(os.path.join(args.aug_dir, "wm_train_solution_paths.parquet"))
    aug_val = pd.read_parquet(os.path.join(args.aug_dir, "wm_val_solution_paths.parquet"))

    print(f"=== Combining B-7 + augmented → B-8 ===")
    print(f"  B-7  train: {len(b7_train):>6} samples")
    print(f"  B-7  val:   {len(b7_val):>6} samples")
    print(f"  Aug  train: {len(aug_train):>6} samples (× {args.aug_repeat} repeat)")
    print(f"  Aug  val:   {len(aug_val):>6} samples (× {args.aug_repeat} repeat)")

    aug_train_repeated = pd.concat([aug_train] * args.aug_repeat, ignore_index=True)
    aug_val_repeated = pd.concat([aug_val] * args.aug_repeat, ignore_index=True)

    combined_train = pd.concat([b7_train, aug_train_repeated], ignore_index=True)
    combined_val = pd.concat([b7_val, aug_val_repeated], ignore_index=True)

    # Shuffle the combined train so augmented samples aren't all at the end
    combined_train = combined_train.sample(frac=1.0, random_state=42).reset_index(drop=True)
    combined_val = combined_val.sample(frac=1.0, random_state=42).reset_index(drop=True)

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "wm_train.parquet")
    val_path = os.path.join(args.output_dir, "wm_val.parquet")
    # Also write the no_post_bp variant under the standard name (since that's
    # what the SFT trainer expects from existing launchers)
    train_path_nopb = os.path.join(args.output_dir, "wm_train_no_post_bp.parquet")
    val_path_nopb = os.path.join(args.output_dir, "wm_val_no_post_bp.parquet")

    combined_train.to_parquet(train_path, index=False)
    combined_val.to_parquet(val_path, index=False)
    combined_train.to_parquet(train_path_nopb, index=False)
    combined_val.to_parquet(val_path_nopb, index=False)

    print(f"\n  Combined train: {len(combined_train):>6} samples")
    print(f"  Combined val:   {len(combined_val):>6} samples")
    print(f"  Augmented frac: {(len(aug_train_repeated) / len(combined_train)) * 100:.1f}%")
    print(f"\nWrote:")
    for pth in (train_path, val_path, train_path_nopb, val_path_nopb):
        print(f"  {pth}")

    # Step distribution in combined
    import json
    from collections import Counter
    def get_step(row):
        info = row["extra_info"]
        if isinstance(info, str):
            info = json.loads(info)
        return info.get("step", -1)
    combined_train["step"] = combined_train.apply(get_step, axis=1)
    step_counts = Counter(combined_train["step"])
    print(f"\nCombined train trajectory-position distribution:")
    total = len(combined_train)
    for k in sorted(step_counts):
        pct = 100 * step_counts[k] / total
        print(f"  step {k:>3d}: {step_counts[k]:>5d} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
