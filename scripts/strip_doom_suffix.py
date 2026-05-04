"""Strip the env-render doom-reason leak from SFT training data.

Background: PolyominoEnv and HidatoEnv's render() output appends a doom-reason
suffix to the `Last action:` line of doomed states, e.g.
  `Last action: placed L ori=4 at row 4 col 7 — board now unsolvable (island_size_mismatch_29)`

This text is then captured into both the user-msg `Last action:` line of the
SFT prompt AND the model's `<next_state>`/`<prediction>` body. The model
learns a trivial shortcut: emit the doom suffix → emit `<viability>=false`.
At greedy decode, the model hallucinates the suffix on valid actions, then
follows with `<viability>=false`, even when the action is actually fine.

This script removes the ` — board now unsolvable (...)` suffix from both
fields, leaving the plain `Last action: placed X at row Y col Z` form. The
`<viability>` / `<solvable>` ground-truth tag is preserved unchanged. The
model is then forced to predict viability from the actual board content,
not from a text shortcut.

Affects: Pentomino (5×4, 5×10) and Hidato datasets. Sudoku is already clean.

Usage:
  python scripts/strip_doom_suffix.py \\
      --input  data/pentomino_5x10_combined \\
      --output data/pentomino_5x10_combined_no_leak
"""
from __future__ import annotations

import argparse
import os
import re

import pandas as pd


# Match " — board now unsolvable (REASON)" or " — board now unsolvable" with
# optional inline whitespace. Em-dash and hyphen-minus both supported.
DOOM_RE = re.compile(r"\s*[—\-–]\s*board now unsolvable(?:\s*\([^\)]*\))?", re.IGNORECASE)


def strip_text(s):
    if not isinstance(s, str):
        return s
    return DOOM_RE.sub("", s)


def strip_prompt(prompt):
    """prompt is a list of {role, content} dicts (or numpy array of those)."""
    new = []
    for msg in prompt:
        new_msg = dict(msg)
        new_msg["content"] = strip_text(msg["content"])
        new.append(new_msg)
    return new


def transform_file(in_path: str, out_path: str) -> tuple:
    df = pd.read_parquet(in_path)
    n_response_hits = sum(bool(DOOM_RE.search(str(r))) for r in df["response"])
    n_prompt_hits = sum(
        any(DOOM_RE.search(str(m["content"])) for m in p)
        for p in df["prompt"]
    )
    df["response"] = df["response"].apply(strip_text)
    df["prompt"] = df["prompt"].apply(lambda p: strip_prompt(list(p)))
    df.to_parquet(out_path, index=False)
    # Re-verify no leak remains
    leftover_resp = sum(bool(DOOM_RE.search(str(r))) for r in df["response"])
    leftover_prompt = sum(
        any(DOOM_RE.search(str(m["content"])) for m in p)
        for p in df["prompt"]
    )
    return n_response_hits, n_prompt_hits, leftover_resp, leftover_prompt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input dataset directory")
    p.add_argument("--output", required=True, help="Output dataset directory")
    args = p.parse_args()

    if not os.path.isdir(args.input):
        raise FileNotFoundError(args.input)
    os.makedirs(args.output, exist_ok=True)

    candidates = ["wm_train.parquet", "wm_val.parquet",
                  "wm_train_no_post_bp.parquet", "wm_val_no_post_bp.parquet"]
    print(f"=== strip_doom_suffix: {args.input} → {args.output} ===")
    total_resp = 0
    total_prompt = 0
    for fname in candidates:
        ip = os.path.join(args.input, fname)
        if not os.path.isfile(ip):
            continue
        op = os.path.join(args.output, fname)
        n_r, n_p, lr, lp = transform_file(ip, op)
        print(f"  {fname}:")
        print(f"     response leak hits stripped: {n_r}  (leftover: {lr})")
        print(f"     prompt leak hits stripped:   {n_p}  (leftover: {lp})")
        total_resp += n_r
        total_prompt += n_p
    print(f"\n  TOTAL stripped: response={total_resp}, prompt={total_prompt}")


if __name__ == "__main__":
    main()
