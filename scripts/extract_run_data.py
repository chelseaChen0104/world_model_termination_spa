"""Unified extractor for plotting data across SFT and RL run output dirs.

Auto-detects whether an output dir is from an RL run (has rl_log.jsonl) or
SFT run (has runs/ TB events) and emits a normalized JSONL with per-step
metrics. Designed to make plotting/comparison trivial across run types.

Usage:
  python scripts/extract_run_data.py outputs/rl_b5_phase3_v8_anchor   # one RL run
  python scripts/extract_run_data.py outputs/sft_hidato_b_h1          # one SFT run
  python scripts/extract_run_data.py outputs/*                        # all runs

Output: <run_dir>/extracted_metrics.jsonl + summary printed to stdout.

Each JSONL line has the form:
  {"step": int, "phase": "train"|"eval", "metric_name": value, ...}

For RL, this is a re-emission of rl_log.jsonl (simpler format).
For SFT, this is a TB-events → JSONL conversion.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def is_rl_run(path):
    return os.path.isfile(os.path.join(path, "rl_log.jsonl"))


def is_sft_run(path):
    return os.path.isdir(os.path.join(path, "runs"))


def extract_rl(run_dir):
    """RL: just copy the rl_log.jsonl, summarize."""
    src = os.path.join(run_dir, "rl_log.jsonl")
    dst = os.path.join(run_dir, "extracted_metrics.jsonl")
    rows = []
    with open(src) as f:
        for line in f:
            rows.append(json.loads(line))
    with open(dst, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_train = sum(1 for r in rows if r.get("phase") != "eval")
    n_eval = sum(1 for r in rows if r.get("phase") == "eval")
    pass1s = [r["pass@1"] for r in rows if r.get("phase") == "eval" and "pass@1" in r]
    final_pass1 = pass1s[-1] if pass1s else None
    peak_pass1 = max(pass1s) if pass1s else None
    return {
        "run_type": "rl",
        "n_train_steps": n_train,
        "n_eval_points": n_eval,
        "final_pass1": final_pass1,
        "peak_pass1": peak_pass1,
        "extracted": dst,
    }


def extract_sft(run_dir):
    """SFT: parse the TensorBoard events, emit per-step JSONL."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {
            "run_type": "sft",
            "error": "tensorboard not installed; pip install tensorboard",
        }
    runs = os.path.join(run_dir, "runs")
    # Find the first events file (HF Trainer creates a subdirectory)
    events_files = []
    for root, _, files in os.walk(runs):
        for f in files:
            if f.startswith("events.out.tfevents"):
                events_files.append(os.path.join(root, f))
    if not events_files:
        return {"run_type": "sft", "error": "no TB events file"}
    ea = EventAccumulator(events_files[0])
    ea.Reload()
    tags = ea.Tags().get("scalars", [])

    # Collect per-step metrics
    per_step = {}
    for tag in tags:
        for ev in ea.Scalars(tag):
            per_step.setdefault(ev.step, {})[tag] = ev.value

    dst = os.path.join(run_dir, "extracted_metrics.jsonl")
    with open(dst, "w") as f:
        for step in sorted(per_step):
            row = {"step": step, **per_step[step]}
            f.write(json.dumps(row) + "\n")
    eval_losses = [per_step[s].get("eval/loss") for s in sorted(per_step) if "eval/loss" in per_step[s]]
    final_eval_loss = eval_losses[-1] if eval_losses else None
    return {
        "run_type": "sft",
        "n_steps": len(per_step),
        "tags": tags,
        "final_eval_loss": final_eval_loss,
        "extracted": dst,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="+")
    args = p.parse_args()

    for run_dir in args.run_dirs:
        run_dir = run_dir.rstrip("/")
        if not os.path.isdir(run_dir):
            continue
        name = os.path.basename(run_dir)
        if is_rl_run(run_dir):
            info = extract_rl(run_dir)
        elif is_sft_run(run_dir):
            info = extract_sft(run_dir)
        else:
            info = {"run_type": "unknown", "note": "no rl_log.jsonl and no runs/"}
        print(f"=== {name} ===")
        print(json.dumps(info, indent=2))
        print()


if __name__ == "__main__":
    main()
