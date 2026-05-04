"""Build SPA-baseline-style ablation datasets from an existing minimal-tag dataset.

Two ablations:
  --variant se_pred   keep <observation> + <prediction>, DROP <solvable>
                      (= SPA's full recipe — state estimation + transition
                       modeling, no termination tag.)
  --variant se_only   keep <observation> only, DROP <prediction> + <solvable>
                      (= SPA's State Estimation RL row.)

Both ablations:
  1. Update the system-message instructions to remove the corresponding lines.
  2. Strip the corresponding tags from each response.
  3. Re-save as a new parquet directory.

Usage:
  python scripts/strip_tags_from_parquet.py \
      --input  data/sudoku_4x4_llm_policy_minimal_spa_scale \
      --output data/sudoku_4x4_no_solvable \
      --variant se_pred

  python scripts/strip_tags_from_parquet.py \
      --input  data/sudoku_4x4_llm_policy_minimal_spa_scale \
      --output data/sudoku_4x4_se_only \
      --variant se_only

The output dataset has the same column structure as the input
(prompt / response / data_source / ability / reward_model / extra_info)
and is drop-in compatible with src/training/simple_sft_trainer.py.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil

import pandas as pd


SYSTEM_MSG_SE_PRED = (
    "You are solving a Sudoku puzzle. Fill in empty cells (shown as .) with "
    "numbers 1-9 so that each row, column, and 3x3 box contains each number "
    "exactly once.\n\n"
    "Grid format: Numbers separated by spaces, | separates 3x3 boxes, - "
    "separates rows of boxes.\n\n"
    "In your reasoning:\n"
    "1. Describe the current state in <observation>\n"
    "2. Predict the next state after your move in <prediction>\n\n"
    "Then provide your action in <answer> using format: place N at row R col C"
)

SYSTEM_MSG_SE_ONLY = (
    "You are solving a Sudoku puzzle. Fill in empty cells (shown as .) with "
    "numbers 1-9 so that each row, column, and 3x3 box contains each number "
    "exactly once.\n\n"
    "Grid format: Numbers separated by spaces, | separates 3x3 boxes, - "
    "separates rows of boxes.\n\n"
    "In your reasoning:\n"
    "1. Describe the current state in <observation>\n\n"
    "Then provide your action in <answer> using format: place N at row R col C"
)


_SOLVABLE_RE = re.compile(r"<solvable>.*?</solvable>\s*", re.DOTALL)
_PREDICTION_RE = re.compile(r"<prediction>.*?</prediction>\s*", re.DOTALL)


def transform_response(response: str, variant: str) -> str:
    if variant == "se_pred":
        out = _SOLVABLE_RE.sub("", response)
    elif variant == "se_only":
        out = _SOLVABLE_RE.sub("", response)
        out = _PREDICTION_RE.sub("", out)
    else:
        raise ValueError(f"unknown variant {variant!r}")
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out


def transform_prompt(prompt, variant: str):
    new_prompt = []
    for msg in prompt:
        if msg["role"] == "system":
            sys_msg = (
                SYSTEM_MSG_SE_PRED if variant == "se_pred" else SYSTEM_MSG_SE_ONLY
            )
            new_prompt.append({"role": "system", "content": sys_msg})
        else:
            new_prompt.append(dict(msg))
    return new_prompt


def transform_file(in_path: str, out_path: str, variant: str) -> None:
    df = pd.read_parquet(in_path)
    df["prompt"] = df["prompt"].apply(lambda p: transform_prompt(list(p), variant))
    df["response"] = df["response"].apply(lambda r: transform_response(str(r), variant))
    df.to_parquet(out_path, index=False)
    print(f"  {in_path} ({len(df)} rows) → {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input dataset directory")
    p.add_argument("--output", required=True, help="Output dataset directory")
    p.add_argument("--variant", required=True, choices=["se_pred", "se_only"])
    args = p.parse_args()

    if not os.path.isdir(args.input):
        raise FileNotFoundError(args.input)
    os.makedirs(args.output, exist_ok=True)

    files = ["wm_train.parquet", "wm_val.parquet",
             "wm_train_no_post_bp.parquet", "wm_val_no_post_bp.parquet"]
    print(f"=== strip_tags_from_parquet — variant={args.variant} ===")
    for f in files:
        ip = os.path.join(args.input, f)
        op = os.path.join(args.output, f)
        if os.path.isfile(ip):
            transform_file(ip, op, args.variant)
        else:
            print(f"  [skip] {ip} not found")

    print("\n--- sanity check first row of train ---")
    df = pd.read_parquet(os.path.join(args.output, "wm_train_no_post_bp.parquet"))
    r = df.iloc[0]
    print("system message after transform:")
    print(r["prompt"][0]["content"])
    print()
    print("response after transform:")
    print(r["response"])


if __name__ == "__main__":
    main()
