"""Standalone eval for a trained SAVE f_phi checkpoint.

Computes:
  - Viability ROC-AUC (per-candidate v̂_φ vs oracle next_viable)
  - Same-state pairwise ranking accuracy (mixed sets only)
  - Deceptive benchmark accuracy (paper §3.4)
  - Brier score (viability)
  - ECE @ 10 bins (viability)
  - Transition exact-match accuracy (would require sampling generation; deferred)

Run on autodl2:
    /root/miniconda3/bin/python scripts/sudoku_scripts/save_sft_eval.py \\
        --checkpoint outputs/save_sudoku4_f_phi/final \\
        --eval data/sudoku4/sft/val_natural_calibration.sft.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from save_sft_dataset import SaveSFTDataset, SaveSFTCollator, GroupedSampler  # noqa: E402


# -----------------------------------------------------------------------------
# Inference: collect per-sample logits at viab/state slots
# -----------------------------------------------------------------------------


@torch.no_grad()
def collect_predictions(
    model,
    tokenizer,
    dataset: SaveSFTDataset,
    sets_per_batch: int = 8,
    device: str = "cuda",
) -> List[Dict]:
    """Return list of dicts with predicted viability prob + metadata for every sample."""
    collator = SaveSFTCollator(pad_token_id=tokenizer.pad_token_id)
    sampler = GroupedSampler(dataset, sets_per_batch=sets_per_batch, shuffle=False, seed=0)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collator)

    true_id = dataset.true_token_id
    false_id = dataset.false_token_id

    # Track samples by (sibling_set_id, candidate_id) to merge with deceptive_pairs later
    src_lookup = {(s["sibling_set_id"], s["candidate_id"]): s for s in dataset.samples}

    out = []
    model.eval()
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        viab_pos = batch["viab_value_pos"].to(device)
        state_pos = batch["state_value_pos"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attn)
        logits = outputs.logits  # [B, T, V]
        B, T, V = logits.shape

        pred_pos_viab = (viab_pos - 1).clamp(min=0, max=T - 1)
        pred_pos_state = (state_pos - 1).clamp(min=0, max=T - 1)
        bidx = torch.arange(B, device=device)
        ell_viab = logits[bidx, pred_pos_viab, true_id] - logits[bidx, pred_pos_viab, false_id]
        ell_state = logits[bidx, pred_pos_state, true_id] - logits[bidx, pred_pos_state, false_id]
        prob_viab = torch.sigmoid(ell_viab).float().cpu().numpy()
        prob_state = torch.sigmoid(ell_state).float().cpu().numpy()

        for i in range(B):
            sid = batch["sibling_set_ids"][i]
            cid = batch["candidate_ids"][i]
            src = src_lookup.get((sid, cid), {})
            out.append({
                "sibling_set_id": sid,
                "candidate_id": cid,
                "ell_viab": float(ell_viab[i].item()),
                "prob_viab": float(prob_viab[i]),
                "prob_state": float(prob_state[i]),
                "next_viable": bool(batch["next_viable"][i].item()),
                "state_viable": bool(batch["state_viable"][i].item()),
                "candidate_class": batch["candidate_classes"][i],
                "set_mixed": bool(batch["set_mixed"][i].item()),
                "deceptive_pair_memberships": src.get("deceptive_pair_memberships", []),
            })
    return out


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def roc_auc(scores: List[float], labels: List[int]) -> float:
    """Tie-corrected AUC via the rank-sum (Mann-Whitney U) formula."""
    n = len(scores)
    if n == 0:
        return float("nan")
    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Average rank for ties
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    sum_pos_ranks = sum(r for r, (_, y) in zip(ranks, pairs) if y == 1)
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def brier(probs: List[float], labels: List[int]) -> float:
    return float(np.mean([(p - y) ** 2 for p, y in zip(probs, labels)]))


def ece_10bin(probs: List[float], labels: List[int]) -> float:
    n = len(probs)
    bins = [[] for _ in range(10)]
    for p, y in zip(probs, labels):
        b = min(int(p * 10), 9)
        bins[b].append((p, y))
    e = 0.0
    for bucket in bins:
        if not bucket:
            continue
        avg_p = np.mean([p for p, _ in bucket])
        avg_y = np.mean([y for _, y in bucket])
        e += (len(bucket) / n) * abs(avg_p - avg_y)
    return float(e)


def same_state_pairwise_acc(preds: List[Dict]) -> Tuple[float, int]:
    """Within each mixed sibling set, count fraction of (viable, doomed) pairs
    where v̂_φ(viable) > v̂_φ(doomed)."""
    by_set: Dict[str, List[Dict]] = defaultdict(list)
    for p in preds:
        if p["set_mixed"]:
            by_set[p["sibling_set_id"]].append(p)
    correct = 0
    total = 0
    for sid, items in by_set.items():
        viable = [p for p in items if p["candidate_class"] == "valid_viable"]
        doomed = [p for p in items if p["candidate_class"] == "valid_doomed"]
        for v in viable:
            for d in doomed:
                total += 1
                if v["prob_viab"] > d["prob_viab"]:
                    correct += 1
    return (correct / total if total > 0 else float("nan")), total


def pr_auc(scores: List[float], labels: List[int]) -> float:
    """Average precision: ∫ Precision dRecall.

    Implements the trapezoidal AP commonly reported as PR-AUC. Positive class
    has label==1 (caller flips for doomed-class PR-AUC).
    """
    n = len(scores)
    if n == 0 or sum(labels) == 0:
        return float("nan")
    # Sort by score descending
    order = sorted(range(n), key=lambda i: -scores[i])
    tp = 0
    fp = 0
    n_pos = sum(labels)
    last_recall = 0.0
    ap = 0.0
    for idx in order:
        if labels[idx] == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        recall = tp / n_pos
        if recall > last_recall:
            ap += precision * (recall - last_recall)
            last_recall = recall
    return float(ap)


def deceptive_acc(preds: List[Dict]) -> Tuple[float, int]:
    """For each deceptive pair, did v̂_φ(a+) > v̂_φ(a-)?

    A candidate may belong to multiple pairs (deceptive_pair_memberships is a
    list); we register the prediction under EACH (pair_id, role) it covers.
    """
    by_pair: Dict[str, Dict[str, Dict]] = defaultdict(dict)
    for p in preds:
        for m in p.get("deceptive_pair_memberships", []):
            by_pair[m["pair_id"]][m["role"]] = p
    correct = 0
    total = 0
    for pid, pair in by_pair.items():
        if "a_plus" not in pair or "a_minus" not in pair:
            continue
        total += 1
        if pair["a_plus"]["prob_viab"] > pair["a_minus"]["prob_viab"]:
            correct += 1
    return (correct / total if total > 0 else float("nan")), total


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--eval", required=True, type=Path)
    ap.add_argument("--output_json", type=Path, default=None)
    ap.add_argument("--calibration_json", type=Path, default=None,
                    help="If given, apply temperature scaling and report Prec@τ_keep "
                         "alongside raw metrics (Q2 table format).")
    ap.add_argument("--per_device_batch_sets", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=512)
    args = ap.parse_args()

    print(f"[load] {args.checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, local_files_only=True,
    ).to("cuda")

    print(f"[load] eval data {args.eval}")
    ds = SaveSFTDataset(args.eval, tokenizer, max_length=args.max_length)
    print(f"  n={len(ds)} samples")

    print("[predict]")
    preds = collect_predictions(model, tokenizer, ds, sets_per_batch=args.per_device_batch_sets)
    print(f"  collected {len(preds)} predictions")

    # --- metrics ---
    scores_v = [p["prob_viab"] for p in preds]
    labels_v = [int(p["next_viable"]) for p in preds]
    auc_v = roc_auc(scores_v, labels_v)
    brier_v = brier(scores_v, labels_v)
    ece_v = ece_10bin(scores_v, labels_v)

    scores_s = [p["prob_state"] for p in preds]
    labels_s = [int(p["state_viable"]) for p in preds]
    auc_s = roc_auc(scores_s, labels_s)
    brier_s = brier(scores_s, labels_s)
    ece_s = ece_10bin(scores_s, labels_s)

    pw_acc, pw_n = same_state_pairwise_acc(preds)
    dec_acc, dec_n = deceptive_acc(preds)

    # PR-AUC on doomed class: positive class = doomed (label 0 → 1, score = 1 - p_viab)
    doomed_scores = [1.0 - s for s in scores_v]
    doomed_labels = [1 - y for y in labels_v]
    pr_auc_doomed = pr_auc(doomed_scores, doomed_labels)

    # Distribution sanity: minority class ratio for viability predictions (binarized at 0.5)
    bin_pred = [1 if p["prob_viab"] >= 0.5 else 0 for p in preds]
    minority_ratio = min(sum(bin_pred), len(bin_pred) - sum(bin_pred)) / max(len(bin_pred), 1)

    metrics = {
        "n_samples": len(preds),
        "viability": {
            "auc": auc_v,
            "pr_auc_doomed": pr_auc_doomed,
            "brier": brier_v,
            "ece_10bin": ece_v,
            "minority_pred_ratio": minority_ratio,
        },
        "state_viability": {
            "auc": auc_s,
            "brier": brier_s,
            "ece_10bin": ece_s,
        },
        "same_state_pairwise": {"acc": pw_acc, "n_pairs": pw_n},
        "deceptive_bench": {"acc": dec_acc, "n_pairs": dec_n},
    }

    # --- Optional: post-calibration metrics (paper Q2 row) ---
    if args.calibration_json is not None:
        calib = json.loads(args.calibration_json.read_text())
        T = float(calib["temperature"])
        tau_keep = float(calib["tau_keep"]["tau"])
        cal_scores = [1.0 / (1.0 + math.exp(-p["ell_viab"] / T)) for p in preds]
        cal_brier = brier(cal_scores, labels_v)
        cal_ece = ece_10bin(cal_scores, labels_v)
        # Prec@τ_keep: precision among kept (calibrated p ≥ τ_keep)
        kept_labels = [y for s, y in zip(cal_scores, labels_v) if s >= tau_keep]
        prec_at_keep = float(np.mean(kept_labels)) if kept_labels else float("nan")
        metrics["calibrated"] = {
            "temperature": T,
            "tau_keep": tau_keep,
            "brier": cal_brier,
            "ece_10bin": cal_ece,
            "prec_at_tau_keep": prec_at_keep,
            "n_kept": len(kept_labels),
            "kept_fraction": len(kept_labels) / len(preds),
        }

    print("\n=== EVAL METRICS ===")
    print(json.dumps(metrics, indent=2))

    # Q2 paper-table row (Table 2)
    if args.calibration_json is not None:
        c = metrics["calibrated"]
        print("\n=== Q2 PAPER ROW (Table 2) ===")
        print(f"{'Metric':<24}{'Value':>10}")
        print("-" * 34)
        print(f"{'ROC-AUC':<24}{auc_v:>10.4f}")
        print(f"{'PR-AUC (doomed)':<24}{pr_auc_doomed:>10.4f}")
        print(f"{'Same-state pairwise':<24}{pw_acc:>10.4f}")
        print(f"{'Brier (calibrated)':<24}{c['brier']:>10.4f}")
        print(f"{'ECE@10 (calibrated)':<24}{c['ece_10bin']:>10.4f}")
        print(f"{'Prec@τ_keep':<24}{c['prec_at_tau_keep']:>10.4f}"
              f"  (τ={c['tau_keep']:.3f}, kept={c['kept_fraction']:.1%})")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(metrics, indent=2))
        print(f"\n[save] {args.output_json}")


if __name__ == "__main__":
    main()
