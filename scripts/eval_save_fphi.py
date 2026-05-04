"""Evaluate a SAVE f_phi checkpoint on val + test SFT files.

Metrics (per doc/SAVE_handoff.md §5):
  - Viability ROC-AUC                (per-candidate v̂ vs oracle next_viable)
  - Brier score                       (mean squared error of v̂ vs label)
  - Calibration: hi-confidence rate   (predicted P > 0.9 → empirical viable rate)
  - Same-state pairwise ranking acc   (within mixed sibling sets)
  - Deceptive-pair accuracy           (using deceptive_pair_memberships in records)
  - Format compliance                 (greedy generation parses to all 3 tags)

Scoring approach: teacher-forced. We tokenize:
    prompt + "<next_state>\\n{oracle next_state}\\n</next_state>\\n<viability>"
Then a single forward pass; we extract P(viable) by softmaxing logits at the
last position over {true, false} token ids only.

Same approach for the <state_viable> tag (separate forward pass with the
viability portion teacher-forced too).

Usage:
  python scripts/eval_save_fphi.py \\
    --model /tmp/save_fphi_hidato5x4/final \\
    --val   data/hidato5x4/sft/val.sft.jsonl \\
    --test  data/hidato5x4/sft/test.sft.jsonl \\
    --output /tmp/save_fphi_hidato_eval.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_true_false_token_ids(tok) -> Tuple[int, int]:
    """Identify the first-token id for 'true' and 'false' as they would appear
    after '<viability>' (i.e., immediately after the closing-angle of the open
    tag, no leading space)."""
    # Try the most likely tokenizations.
    candidates_true = [
        tok("true", add_special_tokens=False).input_ids,
        tok(">true", add_special_tokens=False).input_ids,
    ]
    candidates_false = [
        tok("false", add_special_tokens=False).input_ids,
        tok(">false", add_special_tokens=False).input_ids,
    ]
    # We want the FIRST token of "true"/"false" when emitted right after ">".
    # Use the suffix difference between ">" and ">true":
    base = tok(">", add_special_tokens=False).input_ids
    after_true = tok(">true", add_special_tokens=False).input_ids
    after_false = tok(">false", add_special_tokens=False).input_ids
    if len(after_true) <= len(base) or len(after_false) <= len(base):
        # Fallback: standalone token ids
        return candidates_true[0][0], candidates_false[0][0]
    true_id = after_true[len(base)]
    false_id = after_false[len(base)]
    return true_id, false_id


def _build_prompt_for_viability(tok, messages: List[Dict], oracle_next_state_text: str) -> str:
    """Prompt + everything up to '<viability>' (open tag). Model's next token
    is the viability boolean."""
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return (prompt
            + "<next_state>\n"
            + oracle_next_state_text
            + "\n</next_state>\n<viability>")


def _build_prompt_for_state_viability(tok, messages: List[Dict],
                                        oracle_next_state_text: str,
                                        oracle_next_viable: bool) -> str:
    """Prompt + everything up to '<state_viable>' (open tag) — i.e. with both
    next_state AND viability teacher-forced to oracle values."""
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    nv = "true" if oracle_next_viable else "false"
    return (prompt
            + "<next_state>\n"
            + oracle_next_state_text
            + "\n</next_state>\n<viability>"
            + nv
            + "</viability>\n<state_viable>")


def _next_token_p_true(model, tok, full_prompt: str,
                        true_id: int, false_id: int) -> float:
    """Forward pass; return P(true | next-token ∈ {true, false})."""
    ids = tok(full_prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        out = model(ids)
    last_logits = out.logits[0, -1].float()
    lp_t = last_logits[true_id].item()
    lp_f = last_logits[false_id].item()
    # softmax over just these two
    m = max(lp_t, lp_f)
    e_t = math.exp(lp_t - m)
    e_f = math.exp(lp_f - m)
    return e_t / (e_t + e_f)


def _roc_auc(scores: List[float], labels: List[bool]) -> float:
    """Mann-Whitney style ROC-AUC, no sklearn dep. Returns nan if degenerate."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    n_correct = 0.0
    for sp in pos:
        for sn in neg:
            if sp > sn:
                n_correct += 1.0
            elif sp == sn:
                n_correct += 0.5
    return n_correct / (len(pos) * len(neg))


def _brier(scores: List[float], labels: List[bool]) -> float:
    if not scores:
        return float("nan")
    return sum((s - (1.0 if y else 0.0)) ** 2 for s, y in zip(scores, labels)) / len(scores)


def _ece(scores: List[float], labels: List[bool], n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    if not scores:
        return float("nan")
    bins = [[] for _ in range(n_bins)]
    for s, y in zip(scores, labels):
        b = min(int(s * n_bins), n_bins - 1)
        bins[b].append((s, y))
    total = 0.0
    n = len(scores)
    for bucket in bins:
        if not bucket:
            continue
        avg_s = sum(s for s, _ in bucket) / len(bucket)
        avg_y = sum((1.0 if y else 0.0) for _, y in bucket) / len(bucket)
        total += len(bucket) / n * abs(avg_s - avg_y)
    return total


def _evaluate_split(model, tok, records: List[Dict], split_name: str,
                     true_id: int, false_id: int) -> Dict:
    print(f"\n--- evaluating {split_name} ({len(records)} samples) ---")
    via_scores = []
    via_labels = []
    sviab_scores = []
    sviab_labels = []
    by_set: Dict[str, List[Tuple[bool, float]]] = defaultdict(list)
    deceptive_pair_scores: Dict[str, Dict[str, float]] = defaultdict(dict)

    for i, r in enumerate(records):
        oracle_next_state = r["response"].split("<next_state>\n", 1)[1].split("\n</next_state>", 1)[0]
        # Some safety: if our parse fails fall back to messages-derived prompt
        messages = r["messages"]

        # 1. Viability teacher-forced
        pf_via = _build_prompt_for_viability(tok, messages, oracle_next_state)
        p_via_true = _next_token_p_true(model, tok, pf_via, true_id, false_id)
        via_scores.append(p_via_true)
        via_labels.append(bool(r["next_viable"]))

        # 2. State-viable teacher-forced (oracle next_viable)
        pf_sviab = _build_prompt_for_state_viability(tok, messages, oracle_next_state,
                                                       bool(r["next_viable"]))
        p_sviab_true = _next_token_p_true(model, tok, pf_sviab, true_id, false_id)
        sviab_scores.append(p_sviab_true)
        sviab_labels.append(bool(r["state_viable"]))

        # Track per-sibling-set
        sib_id = r["sibling_set_id"]
        by_set[sib_id].append((bool(r["next_viable"]), p_via_true))

        # Track deceptive pairs
        for m in r.get("deceptive_pair_memberships", []) or []:
            deceptive_pair_scores[m["pair_id"]][m["role"]] = p_via_true

        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(records)}")

    auc = _roc_auc(via_scores, via_labels)
    brier = _brier(via_scores, via_labels)
    ece = _ece(via_scores, via_labels)
    sv_auc = _roc_auc(sviab_scores, sviab_labels)
    sv_brier = _brier(sviab_scores, sviab_labels)

    # Pairwise ranking accuracy within sibling sets that have both classes
    pw_correct = 0
    pw_total = 0
    for sid, items in by_set.items():
        viables = [s for y, s in items if y]
        doomeds = [s for y, s in items if not y]
        if not viables or not doomeds:
            continue
        for v in viables:
            for d in doomeds:
                pw_total += 1
                if v > d:
                    pw_correct += 1
                elif v == d:
                    pw_correct += 0.5
    pairwise_acc = (pw_correct / pw_total) if pw_total else float("nan")

    # Deceptive-pair accuracy
    dec_correct = 0
    dec_total = 0
    for pid, sc in deceptive_pair_scores.items():
        if "a_plus" not in sc or "a_minus" not in sc:
            continue
        dec_total += 1
        if sc["a_plus"] > sc["a_minus"]:
            dec_correct += 1
        elif sc["a_plus"] == sc["a_minus"]:
            dec_correct += 0.5
    deceptive_acc = (dec_correct / dec_total) if dec_total else float("nan")

    # Hi-confidence calibration
    hi_conf_idx = [i for i, s in enumerate(via_scores) if s >= 0.9]
    hi_emp_viable = (sum(1 for i in hi_conf_idx if via_labels[i]) / len(hi_conf_idx)
                       if hi_conf_idx else float("nan"))

    return {
        "split": split_name,
        "n_samples": len(records),
        "viability": {
            "auc": auc,
            "brier": brier,
            "ece_10bins": ece,
            "hi_confidence_n": len(hi_conf_idx),
            "hi_confidence_empirical_viable_rate": hi_emp_viable,
            "pos_rate": sum(via_labels) / len(via_labels),
        },
        "state_viable": {
            "auc": sv_auc,
            "brier": sv_brier,
            "pos_rate": sum(sviab_labels) / len(sviab_labels),
        },
        "pairwise_within_set": {
            "pairs_evaluated": pw_total,
            "accuracy": pairwise_acc,
        },
        "deceptive": {
            "pairs_evaluated": dec_total,
            "accuracy": deceptive_acc,
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--test", default=None)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    print(f"=== f_phi eval ===")
    print(f"  model: {args.model}")
    print(f"  val:   {args.val}")
    print(f"  test:  {args.test}")

    print("Loading model + tokenizer...")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).cuda().eval()

    true_id, false_id = _resolve_true_false_token_ids(tok)
    print(f"  resolved true_id={true_id} ({tok.decode([true_id])!r}), "
          f"false_id={false_id} ({tok.decode([false_id])!r})")

    def _load(path):
        out = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
        return out

    val_records = _load(args.val)
    val_results = _evaluate_split(model, tok, val_records, "val", true_id, false_id)

    results = {
        "model": args.model,
        "val": val_results,
    }
    if args.test:
        test_records = _load(args.test)
        test_results = _evaluate_split(model, tok, test_records, "test", true_id, false_id)
        results["test"] = test_results

    print("\n=== Summary ===")
    for split_name in ("val", "test"):
        if split_name not in results:
            continue
        r = results[split_name]
        print(f"\n[{split_name.upper()}] n={r['n_samples']}")
        v = r["viability"]
        print(f"  viability AUC:        {v['auc']:.3f}")
        print(f"  viability Brier:      {v['brier']:.4f}")
        print(f"  viability ECE@10:     {v['ece_10bins']:.4f}")
        print(f"  viability pos_rate:   {v['pos_rate']:.3f}")
        print(f"  hi-conf (>=0.9):      {v['hi_confidence_n']} samples; "
              f"empirical viable rate {v['hi_confidence_empirical_viable_rate']:.3f}")
        sv = r["state_viable"]
        print(f"  state_viable AUC:     {sv['auc']:.3f} (pos_rate {sv['pos_rate']:.3f})")
        pw = r["pairwise_within_set"]
        print(f"  pairwise (mixed sets): n={pw['pairs_evaluated']}, acc={pw['accuracy']:.3f}")
        dec = r["deceptive"]
        print(f"  deceptive pair acc:   n={dec['pairs_evaluated']}, acc={dec['accuracy']:.3f}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    main()
