"""Filter SFT parquet files to drop samples whose prompt+response exceed a token budget.

Why: simple_sft_trainer.py's dataset_from_parquet() builds labels as
  labels = [-100] * prompt_len + full_tokens['input_ids'][prompt_len:]
when full_tokens is truncated to max_length but prompt_len exceeds it,
the slice is empty and the sample ends up with all -100 labels — which
produces NaN loss in eval and corrupts training.

This script reads wm_{train,val}.parquet from `--input-dir`, tokenizes
each sample's prompt+response, and writes only those that fit under
`--max-tokens` to `wm_{train,val}_filtered.parquet`.
"""
import argparse
import pandas as pd
from transformers import AutoTokenizer
import json
import collections


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True, help="Dir containing wm_train.parquet and wm_val.parquet")
    p.add_argument("--output-suffix", default="_filtered", help="Suffix appended to output filenames")
    p.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--max-tokens", type=int, default=4000,
                   help="Max prompt+response tokens; samples over this are dropped (give yourself "
                        "a safety margin below the trainer's max_length, e.g. 4000 for max_length=4096)")
    args = p.parse_args()

    print(f"Loading tokenizer: {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    for split in ["train", "val"]:
        in_path = f"{args.input_dir}/wm_{split}.parquet"
        out_path = f"{args.input_dir}/wm_{split}{args.output_suffix}.parquet"
        print(f"\n=== {split} ===\n  reading: {in_path}")
        df = pd.read_parquet(in_path)
        print(f"  rows: {len(df)}")

        keep = []
        cls_kept = collections.Counter()
        cls_total = collections.Counter()
        for i, row in df.iterrows():
            msgs = row["prompt"]
            if hasattr(msgs, "tolist"):
                msgs = msgs.tolist()
            prompt_text = ""
            for m in msgs:
                prompt_text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
            prompt_text += "<|im_start|>assistant\n"
            full = prompt_text + row["response"] + "<|im_end|>"
            n = len(tok(full, add_special_tokens=False)["input_ids"])

            info = row["extra_info"]
            if isinstance(info, str):
                info = json.loads(info)
            cls = (bool(info.get("is_solvable", False)), bool(info.get("is_breaking_point", False)))
            cls_total[cls] += 1
            if n <= args.max_tokens:
                keep.append(i)
                cls_kept[cls] += 1

        out = df.loc[keep].reset_index(drop=True)
        out.to_parquet(out_path)
        print(f"  kept: {len(out)}/{len(df)} ({100*len(out)/len(df):.1f}%) → {out_path}")

        print("  class distribution (kept / total):")
        for cls in sorted(cls_total):
            label = f"solvable={cls[0]}, bp={cls[1]}"
            tot = cls_total[cls]
            kept = cls_kept[cls]
            print(f"    {label:30s} {kept}/{tot} ({100*kept/max(tot,1):.1f}% retention)")


if __name__ == "__main__":
    main()
