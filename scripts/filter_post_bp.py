"""Filter out post-BP filler samples (is_solvable=False AND is_breaking_point=False).

Keeps:
  - Pre-BP solvable: (is_solvable=True, is_breaking_point=False)
  - BP transitions:  (is_solvable=False, is_breaking_point=True)

Drops:
  - Post-BP filler:  (is_solvable=False, is_breaking_point=False)

Why: post-BP samples have label=false regardless of action, so they don't teach
action-conditional reasoning. They also dominate training (60% of data), pushing
the model toward "always false" predictions. Removing them rebalances classes
and forces the model to discriminate pre-BP from BP using grid features.

Usage:
    python scripts/filter_post_bp.py \\
        --input-dir data/sudoku_llm_policy_minimal \\
        --output-suffix _no_post_bp
"""
import argparse
import json
import os
import collections
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-suffix", default="_no_post_bp")
    args = p.parse_args()

    splits = ["wm_train_filtered.parquet", "wm_val_filtered.parquet",
              "wm_train.parquet", "wm_val.parquet"]
    for split in splits:
        in_p = os.path.join(args.input_dir, split)
        if not os.path.exists(in_p):
            continue
        out_p = in_p.replace(".parquet", f"{args.output_suffix}.parquet")
        df = pd.read_parquet(in_p)
        print(f"\n=== {split} ===")
        print(f"  reading {in_p}: {len(df)} rows")

        keep_idx = []
        cls_total = collections.Counter()
        cls_kept = collections.Counter()
        for i in range(len(df)):
            info = df.iloc[i]["extra_info"]
            if isinstance(info, str):
                info = json.loads(info)
            sol = bool(info.get("is_solvable", False))
            bp = bool(info.get("is_breaking_point", False))
            cls_total[(sol, bp)] += 1
            # Drop post-BP filler: solvable=False AND bp=False
            if not sol and not bp:
                continue
            keep_idx.append(i)
            cls_kept[(sol, bp)] += 1

        out_df = df.iloc[keep_idx].reset_index(drop=True)
        out_df.to_parquet(out_p)
        print(f"  kept {len(out_df)} / {len(df)} ({100*len(out_df)/len(df):.1f}%) → {out_p}")
        print(f"  class breakdown (kept / total):")
        for cls in sorted(cls_total):
            label = f"solvable={cls[0]}, bp={cls[1]}"
            print(f"    {label:30s} {cls_kept[cls]}/{cls_total[cls]}")
        if len(out_df) > 0:
            kept_sol = sum(1 for cls, n in cls_kept.items() if cls[0])
            print(f"  new class composition: "
                  f"{cls_kept[(True, False)]} pre-BP true ({100*cls_kept[(True, False)]/len(out_df):.1f}%), "
                  f"{cls_kept[(False, True)]} BP false ({100*cls_kept[(False, True)]/len(out_df):.1f}%)")


if __name__ == "__main__":
    main()
