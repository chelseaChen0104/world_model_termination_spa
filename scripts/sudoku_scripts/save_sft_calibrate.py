"""Calibration for trained SAVE f_phi.

Steps (paper §2.4 + §3 calibration paragraph):

  1. Run f_phi on val_natural_calibration to collect raw (ell_viab, label) pairs.
  2. Fit temperature scaling on val: T* = argmin_T NLL(σ(ell/T), label).
  3. Recompute Brier / ECE pre- and post-calibration.
  4. Sweep τ_keep on calibrated probs: smallest τ such that
       Pr[v=1 | p̂ ≥ τ] ≥ 1 − ε_keep, default ε_keep=0.05.
  5. Sweep τ_fb on calibrated probs: largest τ_fb < τ_keep such that
       Pr[v=0 | p̂ < τ_fb] ≥ 1 − ε_fb, default ε_fb=0.05.
  6. Save calibration.json with T*, τ_keep, τ_fb + diagnostic stats.

Usage:
    python scripts/sudoku_scripts/save_sft_calibrate.py \\
        --checkpoint outputs/save_sudoku4_f_phi/final \\
        --val data/sudoku4/sft/val_natural_calibration.sft.jsonl \\
        --output_json outputs/save_sudoku4_f_phi/calibration.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from save_sft_dataset import SaveSFTDataset  # noqa: E402
from save_sft_eval import collect_predictions, brier, ece_10bin, roc_auc  # noqa: E402


# -----------------------------------------------------------------------------
# Calibration math
# -----------------------------------------------------------------------------


def fit_temperature(ell: np.ndarray, y: np.ndarray) -> float:
    """Find T > 0 minimizing NLL of σ(ell/T) vs y, via line search."""
    # Coarse grid search then golden refine; small problem, no need for autograd
    grid = np.concatenate([np.linspace(0.05, 1.0, 50), np.linspace(1.0, 8.0, 50)])
    def nll(T):
        z = ell / T
        # log-sigmoid stable
        log_p = -np.logaddexp(0.0, -z)
        log_1mp = -np.logaddexp(0.0, z)
        return -float(np.mean(y * log_p + (1 - y) * log_1mp))
    losses = np.array([nll(T) for T in grid])
    T0 = grid[int(np.argmin(losses))]
    # Refine around T0
    lo, hi = max(0.01, T0 / 1.5), T0 * 1.5
    for _ in range(60):
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if nll(m1) < nll(m2):
            hi = m2
        else:
            lo = m1
    return float((lo + hi) / 2)


def find_tau_keep(probs: np.ndarray, y: np.ndarray, target_precision: float) -> Dict:
    """Smallest τ s.t. Pr[y=1 | p ≥ τ] ≥ target_precision.

    Sweeps over candidate thresholds = unique probs in ascending order.
    Returns the chosen τ + diagnostic stats. If no τ achieves the target,
    returns τ=1.0 (filter all out).
    """
    order = np.argsort(probs)
    p_sorted = probs[order]
    y_sorted = y[order]
    n = len(probs)
    # For each cutoff k (keep top n-k highest), precision = mean(y[k:])
    best_tau = 1.0  # fallback: keep nothing
    best_kept = 0
    for k in range(n):
        kept_y = y_sorted[k:]
        if len(kept_y) == 0:
            break
        prec = float(np.mean(kept_y))
        if prec >= target_precision:
            tau = float(p_sorted[k])
            return {
                "tau": tau,
                "precision_on_True": prec,
                "n_kept": int(len(kept_y)),
                "kept_fraction": float(len(kept_y) / n),
                "achieved_target": True,
            }
    return {
        "tau": best_tau,
        "precision_on_True": float("nan"),
        "n_kept": 0,
        "kept_fraction": 0.0,
        "achieved_target": False,
    }


def find_tau_fb(probs: np.ndarray, y: np.ndarray, target_precision: float, tau_keep: float) -> Dict:
    """Largest τ_fb < tau_keep s.t. Pr[y=0 | p < τ_fb] ≥ target_precision.

    Sweeps candidate thresholds in descending order to find the largest
    feasible τ_fb (i.e., terminate as often as possible while staying safe).
    """
    order = np.argsort(probs)[::-1]  # descending
    p_sorted = probs[order]
    y_sorted = y[order]
    n = len(probs)
    # For cutoff k (samples ranked 0..k-1 are >= threshold), below = ranks k..n-1
    # precision_on_False = mean(1 - y[k:])
    best = None
    # Iterate cutoffs: each unique boundary
    for k in range(n + 1):
        if k == 0:
            tau = float(p_sorted[0]) + 1e-9  # > all probs
        elif k == n:
            tau = float(p_sorted[-1])
        else:
            tau = float(p_sorted[k - 1])
        if tau >= tau_keep:
            continue
        below = y_sorted[k:]
        if len(below) == 0:
            continue
        prec_f = 1.0 - float(np.mean(below))
        if prec_f >= target_precision:
            cand = {
                "tau": tau,
                "precision_on_False": prec_f,
                "n_below": int(len(below)),
                "below_fraction": float(len(below) / n),
                "achieved_target": True,
            }
            if best is None or cand["tau"] > best["tau"]:
                best = cand
    if best is None:
        return {
            "tau": 0.0,
            "precision_on_False": float("nan"),
            "n_below": 0,
            "below_fraction": 0.0,
            "achieved_target": False,
        }
    return best


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--val", required=True, type=Path)
    ap.add_argument("--epsilon_keep", type=float, default=0.05,
                    help="Target false-keep rate (1 - precision_on_True target)")
    ap.add_argument("--epsilon_fb", type=float, default=0.05,
                    help="Target false-termination rate")
    ap.add_argument("--per_device_batch_sets", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--output_json", type=Path, default=None)
    args = ap.parse_args()

    print(f"[load] {args.checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, local_files_only=True,
    ).to("cuda")

    print(f"[load] {args.val}")
    ds = SaveSFTDataset(args.val, tokenizer, max_length=args.max_length)
    print(f"  n={len(ds)} samples")

    print("[predict]")
    preds = collect_predictions(model, tokenizer, ds, sets_per_batch=args.per_device_batch_sets)
    ell = np.array([p["ell_viab"] for p in preds], dtype=np.float64)
    y = np.array([int(p["next_viable"]) for p in preds], dtype=np.float64)
    raw_p = 1.0 / (1.0 + np.exp(-ell))

    # --- Pre-calibration metrics ---
    pre_auc = roc_auc(raw_p.tolist(), y.astype(int).tolist())
    pre_brier = brier(raw_p.tolist(), y.astype(int).tolist())
    pre_ece = ece_10bin(raw_p.tolist(), y.astype(int).tolist())

    # --- Fit temperature ---
    T = fit_temperature(ell, y)
    cal_p = 1.0 / (1.0 + np.exp(-ell / T))

    post_auc = roc_auc(cal_p.tolist(), y.astype(int).tolist())  # unchanged in theory
    post_brier = brier(cal_p.tolist(), y.astype(int).tolist())
    post_ece = ece_10bin(cal_p.tolist(), y.astype(int).tolist())

    # --- Threshold sweeps on CALIBRATED probs ---
    target_keep = 1.0 - args.epsilon_keep
    target_fb = 1.0 - args.epsilon_fb
    keep = find_tau_keep(cal_p, y, target_keep)
    fb = find_tau_fb(cal_p, y, target_fb, keep["tau"])

    out = {
        "n_val_samples": len(preds),
        "viable_label_rate": float(y.mean()),
        "epsilon_keep": args.epsilon_keep,
        "epsilon_fb": args.epsilon_fb,
        "temperature": T,
        "metrics_raw": {
            "auc": pre_auc, "brier": pre_brier, "ece_10bin": pre_ece,
        },
        "metrics_calibrated": {
            "auc": post_auc, "brier": post_brier, "ece_10bin": post_ece,
        },
        "tau_keep": keep,
        "tau_fb": fb,
    }
    print("\n=== CALIBRATION ===")
    print(json.dumps(out, indent=2))

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(out, indent=2))
        print(f"\n[save] {args.output_json}")


if __name__ == "__main__":
    main()
