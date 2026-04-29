"""Oversample BP-class samples in a parquet file by replicating them N×.

Use case: class-weighted SFT diagnostic (Run B-3). Mathematically equivalent to
applying a 2× class weight to BP samples in the cross-entropy loss, but
implemented via dataset replication so the existing trainer needs no code change.

Usage:
    python scripts/oversample_bp.py \\
        --input data/sudoku_llm_policy_minimal/wm_train_filtered_no_post_bp.parquet \\
        --output data/sudoku_llm_policy_minimal/wm_train_b3_2x.parquet \\
        --bp-multiplier 2

This produces:
    --bp-multiplier 1 → no change (1531 pre-BP + 951 BP = 2482 total)
    --bp-multiplier 2 → BP samples doubled (1531 + 1902 = 3433 total)
    --bp-multiplier 5 → BP samples 5x (1531 + 4755 = 6286, BP-majority)
"""
import argparse
import json
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--bp-multiplier", type=int, default=2,
                   help="How many copies of each BP sample to keep (1 = no change)")
    args = p.parse_args()

    df = pd.read_parquet(args.input)
    print(f"reading: {args.input} ({len(df)} rows)")

    # Identify BP samples
    bp_idx = []
    pre_bp_idx = []
    other_idx = []
    for i in range(len(df)):
        info = df.iloc[i]["extra_info"]
        if isinstance(info, str):
            info = json.loads(info)
        sol = bool(info.get("is_solvable", False))
        bp = bool(info.get("is_breaking_point", False))
        if not sol and bp:
            bp_idx.append(i)
        elif sol and not bp:
            pre_bp_idx.append(i)
        else:
            other_idx.append(i)

    print(f"  pre-BP true: {len(pre_bp_idx)}")
    print(f"  BP false:    {len(bp_idx)}")
    print(f"  other:       {len(other_idx)}")

    # Keep all non-BP samples once + all BP samples × multiplier
    keep_idx = pre_bp_idx + other_idx + (bp_idx * args.bp_multiplier)
    out_df = df.iloc[keep_idx].reset_index(drop=True)

    # Shuffle so BP duplicates aren't all at the end
    out_df = out_df.sample(frac=1, random_state=42).reset_index(drop=True)
    out_df.to_parquet(args.output)

    n_total = len(out_df)
    n_bp = len(bp_idx) * args.bp_multiplier
    n_pre = len(pre_bp_idx)
    print(f"\noutput: {args.output} ({n_total} rows)")
    print(f"  pre-BP true: {n_pre} ({100*n_pre/n_total:.1f}%)")
    print(f"  BP false:    {n_bp} ({100*n_bp/n_total:.1f}%)")
    print(f"  effective BP class weight: {args.bp_multiplier}x")


if __name__ == "__main__":
    main()
