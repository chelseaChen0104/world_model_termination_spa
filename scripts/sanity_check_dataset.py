"""Schema / format / content sanity checks for an SFT training dataset.

Designed to catch silent data-quality issues — missing tags, action/answer
mismatches, content leaks, duplicate samples, length anomalies.

Usage:
  python scripts/sanity_check_dataset.py --input data/sudoku_4x4_llm_policy_minimal_spa_scale --env sudoku
  python scripts/sanity_check_dataset.py --input data/pentomino_5x10_combined --env polyomino
  python scripts/sanity_check_dataset.py --input data/hidato_b_h1_combined --env hidato
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter

import pandas as pd

ENV_TAGS = {
    "sudoku":     {"viability_tag": "solvable",  "next_tag": "prediction"},
    "polyomino":  {"viability_tag": "viability", "next_tag": "next_state"},
    "hidato":     {"viability_tag": "solvable",  "next_tag": "prediction"},
}

ACTION_RE_BY_ENV = {
    "sudoku":    re.compile(r"place\s+(\d+)\s+at\s+row\s+(\d+)\s+col\s+(\d+)", re.IGNORECASE),
    "polyomino": re.compile(r"place\s+([A-Z])\s+ori=(\d+)\s+at\s+row\s+(\d+)\s+col\s+(\d+)", re.IGNORECASE),
    "hidato":    re.compile(r"place\s+(\d+)\s+at\s+row\s+(\d+)\s+col\s+(\d+)", re.IGNORECASE),
}

DOOM_LEAK_RE = re.compile(r"\s*[—\-–]\s*board now unsolvable", re.IGNORECASE)


def check_response_format(resp: str, viability_tag: str, next_tag: str) -> dict:
    """Return dict of presence-flags + parsed values."""
    out = {
        "has_think": "<think>" in resp and "</think>" in resp,
        "has_observation": "<observation>" in resp,
        f"has_{next_tag}": f"<{next_tag}>" in resp,
        f"has_{viability_tag}": f"<{viability_tag}>" in resp,
        "has_answer": "<answer>" in resp and "</answer>" in resp,
        "has_doom_leak": bool(DOOM_LEAK_RE.search(resp)),
    }
    m = re.search(rf"<{viability_tag}>\s*(true|false)\s*</{viability_tag}>", resp, re.IGNORECASE)
    out[f"{viability_tag}_value"] = m.group(1).lower() if m else None
    m = re.search(r"<answer>(.*?)</answer>", resp, re.DOTALL)
    out["answer_text"] = m.group(1).strip() if m else None
    return out


def parse_extra_info(s):
    if isinstance(s, str):
        return json.loads(s)
    return s if isinstance(s, dict) else {}


DEFAULT_THRESHOLDS = {
    "max_doom_leak_frac": 0.0,        # any doom leak is a fail
    "max_duplicate_frac": 0.50,       # >50% dups is a warn-fail
    "min_class_min_frac": 0.10,       # smallest class must be >=10%
    "min_unique_samples": 200,        # absolute floor on unique (prompt,response) pairs
    "min_action_parse_frac": 0.99,    # nearly all actions must parse
    "min_tag_consistency_frac": 0.99, # nearly all <viability> must match label
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--env", required=True, choices=list(ENV_TAGS.keys()))
    p.add_argument("--max-samples-shown", type=int, default=2)
    p.add_argument("--strict", action="store_true",
                   help="Exit with non-zero status if any threshold is violated.")
    p.add_argument("--max-duplicate-frac", type=float, default=DEFAULT_THRESHOLDS["max_duplicate_frac"])
    p.add_argument("--min-unique-samples", type=int, default=DEFAULT_THRESHOLDS["min_unique_samples"])
    args = p.parse_args()

    cfg = ENV_TAGS[args.env]
    via_tag = cfg["viability_tag"]
    next_tag = cfg["next_tag"]
    action_re = ACTION_RE_BY_ENV[args.env]

    print(f"=== sanity check: {args.input} (env={args.env}) ===\n")

    fname = "wm_train_no_post_bp.parquet"
    path = os.path.join(args.input, fname)
    if not os.path.isfile(path):
        path = os.path.join(args.input, "wm_train.parquet")
    df = pd.read_parquet(path)
    print(f"# 1. Schema")
    print(f"  source file:    {os.path.basename(path)}")
    print(f"  rows:           {len(df)}")
    print(f"  columns:        {list(df.columns)}")
    n_null_resp = df["response"].isna().sum()
    n_null_prompt = df["prompt"].isna().sum()
    print(f"  null response:  {n_null_resp}")
    print(f"  null prompt:    {n_null_prompt}\n")

    # Per-row format check
    print(f"# 2. Tag presence (response)")
    flags = df["response"].apply(lambda r: check_response_format(str(r), via_tag, next_tag))
    pf = pd.DataFrame(list(flags))
    for col in [c for c in pf.columns if c.startswith("has_") or c == f"{via_tag}_value"]:
        if col == f"{via_tag}_value":
            print(f"  {via_tag}_value distribution: {dict(Counter(pf[col]))}")
        else:
            n = pf[col].sum() if pf[col].dtype == bool else (pf[col] != False).sum()
            print(f"  {col}: {n}/{len(pf)} ({100*n/len(pf):.1f}%)")
    n_leak = pf["has_doom_leak"].sum()
    print(f"  has_doom_leak (env-render leak): {n_leak}/{len(pf)} ({100*n_leak/len(pf):.1f}%)")
    print()

    # Action-vs-answer consistency
    print(f"# 3. Action format")
    valid_action = pf["answer_text"].apply(lambda a: bool(action_re.match(a)) if a else False)
    n_valid = valid_action.sum()
    print(f"  parseable actions: {n_valid}/{len(pf)} ({100*n_valid/len(pf):.1f}%)")
    if n_valid < len(pf):
        bad = pf[~valid_action]["answer_text"].head(3)
        print(f"  unparseable examples (first 3): {list(bad)}")
    print()

    # Class composition
    df["solv"] = df["extra_info"].apply(lambda s: parse_extra_info(s).get("is_solvable"))
    df["step"] = df["extra_info"].apply(lambda s: parse_extra_info(s).get("step"))
    n_solv = (df["solv"] == True).sum()
    n_doom = (df["solv"] == False).sum()
    n_unknown = ((df["solv"] != True) & (df["solv"] != False)).sum()
    print(f"# 4. Class balance (extra_info.is_solvable)")
    print(f"  solvable=True:  {n_solv} ({100*n_solv/len(df):.1f}%)")
    print(f"  solvable=False: {n_doom} ({100*n_doom/len(df):.1f}%)")
    print(f"  unknown/None:   {n_unknown}\n")

    print(f"# 5. Step distribution (extra_info.step)")
    step_dist = df["step"].value_counts().sort_index()
    for s, c in step_dist.items():
        print(f"  step {s}: {c} ({100*c/len(df):.1f}%)")
    print()

    # Cross-check: <viability>true + extra_info.is_solvable=True consistency
    df["resp_via"] = pf[f"{via_tag}_value"]
    df["resp_via_bool"] = df["resp_via"].map({"true": True, "false": False})
    consistent = (df["resp_via_bool"] == df["solv"])
    n_consistent = consistent.sum()
    print(f"# 6. Tag↔label consistency (response <{via_tag}> matches extra_info.is_solvable)")
    print(f"  consistent: {n_consistent}/{len(df)} ({100*n_consistent/len(df):.1f}%)")
    inc = df[~consistent].head(2)
    if len(inc) > 0:
        print(f"  first inconsistent sample(s):")
        for i, row in inc.iterrows():
            print(f"    row {i}: resp_via={row['resp_via']}, extra_info_solv={row['solv']}")
    print()

    # Length stats
    print(f"# 7. Length stats")
    df["resp_len"] = df["response"].astype(str).apply(len)
    df["prompt_user_len"] = df["prompt"].apply(lambda p: len(str(p[1]["content"])))
    print(f"  response length (chars):       min={df['resp_len'].min()}  med={int(df['resp_len'].median())}  max={df['resp_len'].max()}  mean={int(df['resp_len'].mean())}")
    print(f"  user-msg length (chars):       min={df['prompt_user_len'].min()}  med={int(df['prompt_user_len'].median())}  max={df['prompt_user_len'].max()}\n")

    # Duplicate detection (prompt, response pairs)
    print(f"# 8. Duplicates")
    df["prompt_str"] = df["prompt"].apply(lambda p: str(p))
    n_dup = df.duplicated(subset=["prompt_str", "response"]).sum()
    print(f"  exact (prompt, response) duplicates: {n_dup}\n")

    # Show a sample
    print(f"# 9. Sample row (first solvable + first doom)")
    for kind, mask in [("solvable", df["solv"] == True), ("doom", df["solv"] == False)]:
        sub = df[mask]
        if len(sub) == 0:
            print(f"  no {kind} samples")
            continue
        row = sub.iloc[0]
        print(f"  --- {kind} ---")
        print(f"    prompt user-msg (last 200 chars): ...{str(row['prompt'][1]['content'])[-200:]}")
        print(f"    response (last 200 chars):        ...{str(row['response'])[-200:]}")
        print()

    # ── Threshold checks ─────────────────────────────────────────────
    print(f"# 10. Threshold checks")
    failures = []
    n = len(df)
    n_unique = n - n_dup
    leak_frac = n_leak / max(1, n)
    dup_frac = n_dup / max(1, n)
    parse_frac = n_valid / max(1, n)
    cons_frac = n_consistent / max(1, n)
    class_min_frac = min(n_solv, n_doom) / max(1, n)

    def check(name, ok, detail):
        marker = "✅" if ok else "❌"
        print(f"  {marker} {name}: {detail}")
        if not ok:
            failures.append(name)

    check("doom-leak fraction == 0",
          leak_frac == 0.0,
          f"{leak_frac*100:.1f}%")
    # Duplicate-fraction is INFORMATIONAL only. We intentionally oversample
    # augmented samples (--aug-repeat 30 for Hidato, 10 for Pentomino 5×4),
    # so high duplicate-fraction can be by design. The real diversity metric
    # is unique-sample count.
    print(f"  ℹ️  duplicate fraction (informational): {dup_frac*100:.1f}% ({n_dup}/{n})")
    print(f"     (intentional oversample is OK; only {chr(96)}min unique samples{chr(96)} is enforced)")
    check(f"unique samples >= {args.min_unique_samples}",
          n_unique >= args.min_unique_samples,
          f"{n_unique} unique")
    check(f"smallest class >= {DEFAULT_THRESHOLDS['min_class_min_frac']*100:.0f}%",
          class_min_frac >= DEFAULT_THRESHOLDS["min_class_min_frac"],
          f"smallest class is {class_min_frac*100:.1f}%")
    check(f"action parse rate >= {DEFAULT_THRESHOLDS['min_action_parse_frac']*100:.0f}%",
          parse_frac >= DEFAULT_THRESHOLDS["min_action_parse_frac"],
          f"{parse_frac*100:.1f}%")
    check(f"tag↔label consistency >= {DEFAULT_THRESHOLDS['min_tag_consistency_frac']*100:.0f}%",
          cons_frac >= DEFAULT_THRESHOLDS["min_tag_consistency_frac"],
          f"{cons_frac*100:.1f}%")

    print()
    if failures:
        print(f"=== ⚠️ {len(failures)} threshold failure(s): {', '.join(failures)} ===")
        if args.strict:
            import sys as _sys
            _sys.exit(1)
    else:
        print(f"=== ✅ all threshold checks passed ===")


if __name__ == "__main__":
    main()
