"""Reformat existing multi-turn Track B parquet to single-step minimal format.

Converts each multi-turn sample to a single-step (system + user_state) → response,
where the response keeps only <observation>, <prediction>, <solvable>, <answer>
(strips <terminate_prob>, <steps_left>, <breaking_point>).

This avoids regenerating trajectories on the GPU — same labels, same actions,
same predictions, just different prompt structure and shorter response.

Usage:
    python scripts/reformat_to_minimal.py \\
        --input-dir data/sudoku_llm_policy \\
        --output-dir data/sudoku_llm_policy_minimal
"""
import argparse
import os
import re
import pandas as pd

# Import the new minimal system prompt directly so this script and SFTFormatter
# stay in sync.
import sys
sys.path.insert(0, ".")
from src.data.sft_formatter import SFTFormatter


# --- Patterns to strip from the existing response strings ---
TAG_STRIP_PATTERNS = [
    re.compile(r'<terminate_prob>[^<]*</terminate_prob>\n?'),
    re.compile(r'<steps_left>[^<]*</steps_left>\n?'),
    re.compile(r'<breaking_point>[^<]*</breaking_point>\n?'),
]


def normalize_user_content(content: str) -> str:
    """Make the last-turn user message look like a single-turn step-0 message."""
    if content.startswith("Action executed. Current state:"):
        return "Current state:" + content[len("Action executed. Current state:"):]
    return content


def reformat_response(response: str) -> str:
    out = response
    for pat in TAG_STRIP_PATTERNS:
        out = pat.sub('', out)
    return out


def reformat_split(input_path: str, output_path: str, system_prompt: str) -> None:
    df = pd.read_parquet(input_path)
    print(f"  reading {input_path}: {len(df)} rows")

    new_rows = []
    for i in range(len(df)):
        row = df.iloc[i]
        msgs = row["prompt"]
        if hasattr(msgs, "tolist"):
            msgs = msgs.tolist()
        msgs = [{"role": m["role"], "content": m["content"]} for m in msgs]

        last_user = msgs[-1]
        new_user_content = normalize_user_content(last_user["content"])

        new_prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": new_user_content},
        ]
        new_response = reformat_response(row["response"])

        new_rows.append({
            "data_source": row["data_source"],
            "prompt": new_prompt,
            "response": new_response,
            "ability": row["ability"],
            "reward_model": row["reward_model"],
            "extra_info": row["extra_info"],
        })

    out_df = pd.DataFrame(new_rows)
    out_df.to_parquet(output_path)
    print(f"  wrote {output_path}: {len(out_df)} rows")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--variant", default="sudoku_minimal",
                   help="SFTFormatter variant whose system prompt to use")
    p.add_argument("--include-filtered", action="store_true",
                   help="Also reformat *_filtered.parquet variants if present")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    system_prompt = SFTFormatter.SYSTEM_PROMPTS[args.variant]
    print(f"using variant: {args.variant} (prompt {len(system_prompt)} chars)")

    splits = ["wm_train.parquet", "wm_val.parquet"]
    if args.include_filtered:
        splits += ["wm_train_filtered.parquet", "wm_val_filtered.parquet"]

    for split in splits:
        in_p = os.path.join(args.input_dir, split)
        if not os.path.exists(in_p):
            print(f"  skip {split}: not found")
            continue
        out_p = os.path.join(args.output_dir, split)
        print(f"\n=== {split} ===")
        reformat_split(in_p, out_p, system_prompt)

    print("\ndone")


if __name__ == "__main__":
    main()
