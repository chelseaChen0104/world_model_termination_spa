"""
Evaluate RL-trained model vs SFT model for termination prediction.

Generates a BALANCED eval set from the live environment:
- 100 solvable states
- 100 unsolvable states
- 50 breaking point states (subset of unsolvable)

Reports:
1. Format compliance (valid XML output)
2. Solvability prediction accuracy + confusion matrix
3. Breaking point detection: accuracy, precision, recall, F1
4. Per-deadlock-type accuracy (corner, dead_square, freeze)

Usage:
    python evaluate_rl.py [--sft-path PATH] [--rl-path PATH] [--num-samples N]
"""

import torch
import re
import json
import argparse
import numpy as np
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.environments.sokoban import SokobanEnv
from src.environments.sudoku import SudokuEnv
from src.data.sft_formatter import SFTFormatter
from src.data.trajectory_generator import TrajectoryGenerator


def parse_predictions(text):
    """Parse model output for termination predictions (flexible format)."""
    result = {
        "terminate_prob": None,
        "steps_left": None,
        "solvable": None,
        "breaking_point": None,
        "answer": None,
    }

    match = re.search(r'<[Tt]erminate[_\s]?[Pp]rob(?:ability)?>\s*([\d.]+)', text)
    if match:
        try:
            result["terminate_prob"] = float(match.group(1))
        except ValueError:
            pass

    match = re.search(r'<[Ss]teps[_\s]?(?:[Ll]eft|[Rr]emaining)>\s*(\w+)', text)
    if match:
        result["steps_left"] = match.group(1).lower()

    # Accept <solvable> (sudoku) OR <viability> (polyomino, MKD) — same semantics, env-specific tag
    match = re.search(r'<[Ss]olvable>\s*(\w+)', text)
    if not match:
        match = re.search(r'<[Vv]iability>\s*(\w+)', text)
    if match:
        val = match.group(1).lower()
        result["solvable"] = val in ["true", "yes", "1"]

    match = re.search(r'<[Bb]reaking[_\s]?[Pp]oint>\s*(\w+)', text)
    if not match:
        match = re.search(r'<[Dd]eadlock[_\s]?(?:[Ii]dentification)?>\s*(\w+)', text)
    if match:
        val = match.group(1).lower()
        result["breaking_point"] = val in ["true", "yes", "1"]

    match = re.search(r'<[Aa]nswer>\s*(\w+)', text)
    if match:
        result["answer"] = match.group(1)

    return result


def generate_balanced_eval_set(
    env,
    system_prompt: str,
    n_solvable: int = 100,
    n_unsolvable: int = 100,
    max_steps: int = 100,
    seed_start: int = 10000,
):
    """Generate a balanced eval set from the live environment.

    Collects ~n_solvable solvable states and ~n_unsolvable unsolvable states.
    For Sokoban with 1 box, unsolvable states are exclusively breaking points
    (trajectory ends at deadlock). For environments with post-deadlock states,
    those are included too.

    Returns list of dicts: {state, is_solvable, is_breaking_point, deadlock_type, step_index}
    """
    generator = TrajectoryGenerator(env)

    solvable_samples = []
    unsolvable_samples = []  # Includes breaking points

    seed = seed_start
    max_attempts = (n_solvable + n_unsolvable) * 20

    for _ in range(max_attempts):
        if len(solvable_samples) >= n_solvable and len(unsolvable_samples) >= n_unsolvable:
            break

        seed += 1
        try:
            traj, meta = generator.generate_random_trajectory(max_steps=max_steps, seed=seed)
        except Exception:
            continue

        if not traj:
            continue

        for step in traj:
            sample = {
                "state": step.state,
                "is_solvable": step.is_solvable,
                "is_breaking_point": step.is_breaking_point,
                "deadlock_type": step.deadlock_type,
                "step_index": step.step,
            }

            if not step.is_solvable and len(unsolvable_samples) < n_unsolvable:
                unsolvable_samples.append(sample)
            elif step.is_solvable and len(solvable_samples) < n_solvable:
                solvable_samples.append(sample)

    n_bp = sum(1 for s in unsolvable_samples if s["is_breaking_point"])
    print(f"  Balanced eval set: {len(solvable_samples)} solvable, "
          f"{len(unsolvable_samples)} unsolvable ({n_bp} breaking points)")

    all_samples = solvable_samples + unsolvable_samples
    np.random.shuffle(all_samples)
    return all_samples


def evaluate_solvable_logprob(model, tokenizer, eval_samples, system_prompt, model_name="Model",
                              tag_name: str = "solvable"):
    """Extract per-sample P(<TAG>=true) and P(<TAG>=false) via teacher-forced
    forward pass — bypasses greedy/sampling and reveals the model's actual confidence
    distribution.

    Args:
        tag_name: Which XML tag to probe at. Defaults to "solvable" for Sudoku
            backward-compat. Use "viability" for Polyomino (per spec_pentomino.md §4).

    For each sample:
      1. Build prompt = chat_template([system, user_state])
      2. Append the response prefix up to and including the literal "<{tag_name}>" string
         (using the parquet's `response` field for samples loaded from parquet)
      3. Single forward pass; read logits at the very next token position
      4. Softmax → P(true), P(false)

    Then sweep thresholds τ and report precision/recall at each.
    """
    import numpy as np
    print(f"\n{'='*60}")
    print(f"Logprob-based eval: {model_name}")
    print(f"Samples: {len(eval_samples)}")
    print(f"{'='*60}")

    model.eval()

    # Find single-token IDs for "true" and "false"
    true_ids = tokenizer.encode("true", add_special_tokens=False)
    false_ids = tokenizer.encode("false", add_special_tokens=False)
    print(f"  tokenizer 'true'  → {true_ids} ({tokenizer.decode(true_ids)!r})")
    print(f"  tokenizer 'false' → {false_ids} ({tokenizer.decode(false_ids)!r})")
    if len(true_ids) != 1 or len(false_ids) != 1:
        print("  WARNING: 'true'/'false' tokenize to multiple tokens — using first only")
    true_id = true_ids[0]
    false_id = false_ids[0]

    results = []
    skipped = 0
    for i, sample in enumerate(eval_samples):
        if "prompt_messages" not in sample or not sample.get("response"):
            # Need a response template; live-env samples don't have one. Try to
            # use the sample's own response field if present (we add it below for parquet loads).
            skipped += 1
            continue

        messages = sample["prompt_messages"]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        response = sample["response"]
        target_tag = f"<{tag_name}>"
        idx = response.find(target_tag)
        if idx < 0:
            skipped += 1
            continue
        prefix_response = response[: idx + len(target_tag)]
        full_text = prompt_text + prefix_response

        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits[0, -1, :]
            probs = torch.softmax(logits, dim=-1)
            p_true = probs[true_id].item()
            p_false = probs[false_id].item()

        results.append({
            "p_true": p_true,
            "p_false": p_false,
            "gt_solvable": sample["is_solvable"],
            "is_breaking_point": sample.get("is_breaking_point", False),
            "step_index": sample.get("step_index", -1),
        })
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(eval_samples)}...")

    if skipped:
        print(f"  Skipped {skipped} samples (no response template)")
    print(f"\n  Total samples evaluated: {len(results)}")

    if not results:
        return {"per_sample": []}

    # P(true) distribution by class
    pos = [r["p_true"] for r in results if r["gt_solvable"]]
    neg = [r["p_true"] for r in results if not r["gt_solvable"]]
    print(f"\n  P(true) distribution by class:")
    if pos:
        print(f"    GT=true (n={len(pos)}):  mean={np.mean(pos):.3f}  median={np.median(pos):.3f}  std={np.std(pos):.3f}")
    if neg:
        print(f"    GT=false (n={len(neg)}): mean={np.mean(neg):.3f}  median={np.median(neg):.3f}  std={np.std(neg):.3f}")
    if pos and neg:
        print(f"    Separation (mean true − mean false): {np.mean(pos) - np.mean(neg):+.3f}")
        # P(false) for unsolvable states is the relevant signal for early termination
        p_false_on_false = [1 - r["p_true"] for r in results if not r["gt_solvable"]]
        print(f"    P(false) on actually-unsolvable: mean={np.mean(p_false_on_false):.3f} median={np.median(p_false_on_false):.3f}")

    # Threshold sweep — predict <TAG>=True if p_true > τ
    print(f"\n  Threshold sweep (predict <{tag_name}>=True if P(true) > τ):")
    print(f"  {'τ':>6} {'Acc':>7} {'Prec(T)':>9} {'Rec(T)':>8} {'Spec':>7} {'F1(T)':>7} {'Prec(F)':>9} {'Rec(F)':>8}")
    print(f"  {'-'*6} {'-'*7} {'-'*9} {'-'*8} {'-'*7} {'-'*7} {'-'*9} {'-'*8}")
    for tau in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        tp = fp = tn = fn = 0
        for r in results:
            pred = r["p_true"] > tau
            if r["gt_solvable"] and pred:        tp += 1
            elif r["gt_solvable"] and not pred:  fn += 1
            elif not r["gt_solvable"] and pred:  fp += 1
            else:                                 tn += 1
        n = tp + fp + tn + fn
        acc  = (tp + tn) / max(1, n)
        prT  = tp / max(1, tp + fp)
        rcT  = tp / max(1, tp + fn)
        spec = tn / max(1, tn + fp)
        f1T  = 2 * prT * rcT / max(1e-9, prT + rcT)
        prF  = tn / max(1, tn + fn)  # precision when predicting False
        rcF  = tn / max(1, tn + fp)  # = specificity, recall on False class
        print(f"  {tau:.2f}   {acc*100:6.1f} {prT*100:8.1f}% {rcT*100:7.1f}% {spec*100:6.1f}% {f1T*100:6.1f} {prF*100:8.1f}% {rcF*100:7.1f}%")

    # ROC AUC if sklearn is available
    try:
        from sklearn.metrics import roc_auc_score
        y = np.array([1 if r["gt_solvable"] else 0 for r in results])
        s = np.array([r["p_true"] for r in results])
        auc = roc_auc_score(y, s)
        print(f"\n  ROC AUC: {auc:.3f}")
    except Exception as e:
        print(f"\n  ROC AUC: skipped ({e})")

    return {"per_sample": results}


def load_balanced_from_parquet(parquet_path, n_per_class=100, seed=42):
    """Load eval samples directly from a training-format parquet, balanced by class.

    The parquet's `prompt` column already contains the multi-turn message list as
    seen during training (system + user/assistant pairs + final user state).
    This makes the eval distribution match the training distribution exactly.

    Returns list of dicts with keys:
      - prompt_messages: list of {role, content} messages (multi-turn)
      - is_solvable, is_breaking_point, deadlock_type, step_index
    """
    import pandas as pd
    df = pd.read_parquet(parquet_path)

    by_class = {("sol_T", "bp_F"): [], ("sol_F", "bp_T"): [], ("sol_F", "bp_F"): []}
    for i in range(len(df)):
        info = df.iloc[i]["extra_info"]
        if isinstance(info, str):
            info = json.loads(info)
        sol = bool(info.get("is_solvable", False))
        bp = bool(info.get("is_breaking_point", False))
        if sol and not bp:
            by_class[("sol_T", "bp_F")].append(i)
        elif not sol and bp:
            by_class[("sol_F", "bp_T")].append(i)
        elif not sol and not bp:
            by_class[("sol_F", "bp_F")].append(i)

    rng = np.random.default_rng(seed)
    selected = []
    for cls, idxs in by_class.items():
        n = min(n_per_class, len(idxs))
        chosen = rng.choice(idxs, size=n, replace=False)
        for i in chosen:
            row = df.iloc[int(i)]
            info = row["extra_info"]
            if isinstance(info, str):
                info = json.loads(info)
            msgs = row["prompt"]
            if hasattr(msgs, "tolist"):
                msgs = msgs.tolist()
            prompt_messages = [{"role": m["role"], "content": m["content"]} for m in msgs]
            selected.append({
                "prompt_messages": prompt_messages,
                "response": row["response"],  # used by --metric solvable-logprob
                "is_solvable": bool(info.get("is_solvable", False)),
                "is_breaking_point": bool(info.get("is_breaking_point", False)),
                "deadlock_type": info.get("deadlock_type"),
                "step_index": info.get("step", -1),
            })

    rng.shuffle(selected)
    n_sol = sum(1 for s in selected if s["is_solvable"])
    n_bp = sum(1 for s in selected if s["is_breaking_point"])
    n_unsol = len(selected) - n_sol
    print(f"  Loaded multi-turn eval from {parquet_path}: {len(selected)} samples "
          f"({n_sol} solvable, {n_unsol} unsolvable, {n_bp} BP)")
    return selected


def evaluate_model(model, tokenizer, eval_samples, system_prompt, model_name="Model",
                    temperature=0.0):
    """Evaluate a model on the balanced eval set.

    Each sample can be either:
      - single-turn:  has a "state" field; eval builds [system, user_state] prompt
      - multi-turn:   has a "prompt_messages" field; eval uses it as-is (matches training)
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Samples: {len(eval_samples)}")
    print(f"{'='*60}")

    model.eval()

    metrics = {
        "valid_format": 0,
        "has_solvable": 0,
        "has_bp": 0,
        "has_steps_left": 0,
        "has_terminate_prob": 0,
        "has_answer": 0,
        # Solvable confusion matrix
        "sol_tp": 0, "sol_fp": 0, "sol_fn": 0, "sol_tn": 0,
        # Breaking point confusion matrix
        "bp_tp": 0, "bp_fp": 0, "bp_fn": 0, "bp_tn": 0,
        # Per-deadlock-type accuracy
        "deadlock_type_correct": defaultdict(int),
        "deadlock_type_total": defaultdict(int),
        "total": 0,
    }

    for i, sample in enumerate(eval_samples):
        # Build prompt — multi-turn if sample has prompt_messages, else single-turn
        if "prompt_messages" in sample:
            messages = sample["prompt_messages"]
            max_input_len = 4096
            max_new_tokens = 600  # full response can be ~500 tokens (two grids + tags)
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Current state:\n{sample['state']}"},
            ]
            max_input_len = 1024  # bumped from 512 — system+state fits comfortably
            max_new_tokens = 600

        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_input_len).to(model.device)
        with torch.no_grad():
            do_sample = temperature > 0.0
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                pad_token_id=tokenizer.eos_token_id
            )

        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_len:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        preds = parse_predictions(response)
        metrics["total"] += 1

        # Format compliance
        if preds["solvable"] is not None:
            metrics["has_solvable"] += 1
        if preds["breaking_point"] is not None:
            metrics["has_bp"] += 1
        if preds["steps_left"] is not None:
            metrics["has_steps_left"] += 1
        if preds["terminate_prob"] is not None:
            metrics["has_terminate_prob"] += 1
        if preds["answer"] is not None:
            metrics["has_answer"] += 1
        if preds["solvable"] is not None or preds["breaking_point"] is not None:
            metrics["valid_format"] += 1

        gt_solvable = sample["is_solvable"]
        gt_bp = sample["is_breaking_point"]
        deadlock_type = sample.get("deadlock_type")

        # Solvable confusion matrix
        if preds["solvable"] is not None:
            pred_sol = preds["solvable"]
            if gt_solvable and pred_sol:
                metrics["sol_tp"] += 1
            elif gt_solvable and not pred_sol:
                metrics["sol_fn"] += 1
            elif not gt_solvable and pred_sol:
                metrics["sol_fp"] += 1
            else:
                metrics["sol_tn"] += 1

        # Breaking point confusion matrix
        if preds["breaking_point"] is not None:
            pred_bp = preds["breaking_point"]
            if gt_bp and pred_bp:
                metrics["bp_tp"] += 1
            elif gt_bp and not pred_bp:
                metrics["bp_fn"] += 1
            elif not gt_bp and pred_bp:
                metrics["bp_fp"] += 1
            else:
                metrics["bp_tn"] += 1

            # Per-deadlock-type accuracy (for breaking point samples)
            if gt_bp and deadlock_type:
                metrics["deadlock_type_total"][deadlock_type] += 1
                if pred_bp:
                    metrics["deadlock_type_correct"][deadlock_type] += 1

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(eval_samples)}...")

    # Compute derived metrics
    results = _compute_results(metrics)
    _print_results(results, metrics, model_name)

    return results, metrics


def _compute_results(metrics):
    """Compute derived metrics from raw counts."""
    total = metrics["total"]
    results = {}

    # Format compliance
    results["format_compliance"] = metrics["valid_format"] / total * 100 if total else 0
    results["has_solvable_rate"] = metrics["has_solvable"] / total * 100 if total else 0
    results["has_bp_rate"] = metrics["has_bp"] / total * 100 if total else 0
    results["has_answer_rate"] = metrics["has_answer"] / total * 100 if total else 0

    # Solvable metrics
    sol_total = metrics["sol_tp"] + metrics["sol_fp"] + metrics["sol_fn"] + metrics["sol_tn"]
    sol_correct = metrics["sol_tp"] + metrics["sol_tn"]
    results["solvable_accuracy"] = sol_correct / sol_total * 100 if sol_total else 0

    sol_prec_denom = metrics["sol_tp"] + metrics["sol_fp"]
    sol_rec_denom = metrics["sol_tp"] + metrics["sol_fn"]
    results["solvable_precision"] = metrics["sol_tp"] / sol_prec_denom * 100 if sol_prec_denom else 0
    results["solvable_recall"] = metrics["sol_tp"] / sol_rec_denom * 100 if sol_rec_denom else 0

    if results["solvable_precision"] + results["solvable_recall"] > 0:
        results["solvable_f1"] = 2 * results["solvable_precision"] * results["solvable_recall"] / (
            results["solvable_precision"] + results["solvable_recall"])
    else:
        results["solvable_f1"] = 0

    # Breaking point metrics
    bp_total = metrics["bp_tp"] + metrics["bp_fp"] + metrics["bp_fn"] + metrics["bp_tn"]
    bp_correct = metrics["bp_tp"] + metrics["bp_tn"]
    results["bp_accuracy"] = bp_correct / bp_total * 100 if bp_total else 0

    bp_prec_denom = metrics["bp_tp"] + metrics["bp_fp"]
    bp_rec_denom = metrics["bp_tp"] + metrics["bp_fn"]
    results["bp_precision"] = metrics["bp_tp"] / bp_prec_denom * 100 if bp_prec_denom else 0
    results["bp_recall"] = metrics["bp_tp"] / bp_rec_denom * 100 if bp_rec_denom else 0

    if results["bp_precision"] + results["bp_recall"] > 0:
        results["bp_f1"] = 2 * results["bp_precision"] * results["bp_recall"] / (
            results["bp_precision"] + results["bp_recall"])
    else:
        results["bp_f1"] = 0

    # Per-deadlock-type accuracy
    results["per_deadlock"] = {}
    for dtype in metrics["deadlock_type_total"]:
        t = metrics["deadlock_type_total"][dtype]
        c = metrics["deadlock_type_correct"].get(dtype, 0)
        results["per_deadlock"][dtype] = c / t * 100 if t else 0

    return results


def _print_results(results, metrics, model_name):
    """Print formatted evaluation results."""
    print(f"\n--- Results: {model_name} ---")
    total = metrics["total"]
    print(f"  Samples evaluated: {total}")

    print(f"\n  FORMAT COMPLIANCE:")
    print(f"    Valid format:     {results['format_compliance']:.1f}%")
    print(f"    Has <solvable>:   {results['has_solvable_rate']:.1f}%")
    print(f"    Has <breaking_point>: {results['has_bp_rate']:.1f}%")
    print(f"    Has <answer>:     {results['has_answer_rate']:.1f}%")

    print(f"\n  SOLVABLE PREDICTION:")
    print(f"    Accuracy:  {results['solvable_accuracy']:.1f}%")
    print(f"    Precision: {results['solvable_precision']:.1f}%")
    print(f"    Recall:    {results['solvable_recall']:.1f}%")
    print(f"    F1:        {results['solvable_f1']:.1f}%")
    print(f"    Confusion matrix:")
    print(f"                  Pred=True  Pred=False")
    print(f"      GT=True     {metrics['sol_tp']:>8d}   {metrics['sol_fn']:>8d}")
    print(f"      GT=False    {metrics['sol_fp']:>8d}   {metrics['sol_tn']:>8d}")

    print(f"\n  BREAKING POINT DETECTION:")
    print(f"    Accuracy:  {results['bp_accuracy']:.1f}%")
    print(f"    Precision: {results['bp_precision']:.1f}%")
    print(f"    Recall:    {results['bp_recall']:.1f}%  (key metric!)")
    print(f"    F1:        {results['bp_f1']:.1f}%")
    print(f"    Confusion matrix:")
    print(f"                  Pred=True  Pred=False")
    print(f"      GT=True     {metrics['bp_tp']:>8d}   {metrics['bp_fn']:>8d}")
    print(f"      GT=False    {metrics['bp_fp']:>8d}   {metrics['bp_tn']:>8d}")

    if results["per_deadlock"]:
        print(f"\n  PER-DEADLOCK-TYPE RECALL:")
        for dtype, acc in sorted(results["per_deadlock"].items()):
            t = metrics["deadlock_type_total"][dtype]
            c = metrics["deadlock_type_correct"].get(dtype, 0)
            print(f"    {dtype:<15s}: {acc:.1f}%  ({c}/{t})")


def print_sample_outputs(model, tokenizer, eval_samples, system_prompt, num_samples, model_name):
    """Print sample model outputs for manual inspection."""
    for i in range(min(num_samples, len(eval_samples))):
        sample = eval_samples[i]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Current state:\n{sample['state']}"},
        ]

        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_len:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        print(f"\n[{model_name} Sample {i+1}]")
        print(f"  GT: solvable={sample['is_solvable']}, "
              f"breaking_point={sample['is_breaking_point']}, "
              f"deadlock_type={sample.get('deadlock_type')}")
        print(f"  Generated ({len(response)} chars):")
        print(f"  {response[:400]}")
        print(f"  ---")


def rollout_one(model, tokenizer, env, puzzle_seed, system_prompt,
                max_steps=30, temperature=0.7, max_context_turns=10,
                max_new_tokens=512):
    """Play one full game from the puzzle at `puzzle_seed`. Returns dict with success/steps/reason.

    Mirrors training distribution: multi-turn with sliding window of last
    `max_context_turns` user/assistant pairs.

    `temperature=0` → greedy decode (deterministic, used for Pass@1).
    `temperature>0` → sampling (used for Pass@k>1).
    """
    state = env.reset(seed=puzzle_seed)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Current state:\n{state}"},
    ]

    for step in range(max_steps):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)

        with torch.no_grad():
            do_sample = temperature > 0
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
            )
        response_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()

        # Parse action — must be in <answer> tag or as a "place N at row R col C" pattern
        m = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if m:
            action_str = m.group(1).strip()
        else:
            m = re.search(r'place\s+\d+\s+at\s+row\s+\d+\s+col\s+\d+', response, re.IGNORECASE)
            action_str = m.group(0) if m else None

        if not action_str:
            return {"success": False, "steps": step, "reason": "parse_failure"}

        try:
            next_state, reward, done, info = env.step(action_str)
        except Exception as e:
            return {"success": False, "steps": step, "reason": f"env_error:{type(e).__name__}"}

        if not info.get('action_is_valid', True):
            return {"success": False, "steps": step + 1, "reason": "invalid_action"}

        if done:
            success = info.get('success', False) or (info.get('is_solvable', False) and info.get('puzzle_complete', False))
            # SudokuEnv: success when all cells filled correctly
            if not success and 'puzzle_complete' in info:
                success = bool(info['puzzle_complete'])
            return {"success": bool(success), "steps": step + 1, "reason": "done"}

        # Append turn and apply sliding window
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": f"Action executed. Current state:\n{next_state}"})
        if max_context_turns is not None and len(messages) > 2 * max_context_turns + 2:
            # Keep system + last (2*max_context_turns + 1) messages (last user state included)
            messages = [messages[0]] + messages[-(2 * max_context_turns + 1):]

        state = next_state

    return {"success": False, "steps": max_steps, "reason": "max_steps"}


def evaluate_pass_at_k(model, tokenizer, env, system_prompt,
                       n_puzzles=50, k_values=(1, 8),
                       max_rollout_steps=30, sampling_temperature=0.7,
                       max_context_turns=10, seed_start=20000,
                       model_name="Model"):
    """Compute Pass@1 (greedy) and Pass@K (sampled) over n_puzzles.

    Pass@1 = fraction of puzzles where the greedy rollout solves it.
    Pass@K = fraction where AT LEAST ONE of K sampled rollouts solves it.

    Returns dict {f"pass_at_{k}": fraction, "n_puzzles": n}.
    """
    print(f"\n{'='*60}")
    print(f"Pass@k Evaluation: {model_name}")
    print(f"  Puzzles: {n_puzzles} | k_values: {k_values}")
    print(f"  Greedy for Pass@1, sampling (temp={sampling_temperature}) for Pass@k>1")
    print(f"{'='*60}")

    model.eval()
    max_k = max(k_values)
    counts = {k: 0 for k in k_values}

    for puzzle_idx in range(n_puzzles):
        seed = seed_start + puzzle_idx
        # Pass@1 — greedy
        if 1 in k_values:
            r = rollout_one(model, tokenizer, env, seed, system_prompt,
                            max_steps=max_rollout_steps, temperature=0.0,
                            max_context_turns=max_context_turns)
            if r["success"]:
                counts[1] += 1

        # Pass@K — for K>1, do K sampled rollouts and check if any solved
        for k in k_values:
            if k == 1:
                continue
            any_solved = False
            for k_idx in range(k):
                r = rollout_one(model, tokenizer, env, seed, system_prompt,
                                max_steps=max_rollout_steps,
                                temperature=sampling_temperature,
                                max_context_turns=max_context_turns)
                if r["success"]:
                    any_solved = True
                    break  # Pass@K only needs ONE success
            if any_solved:
                counts[k] += 1

        if (puzzle_idx + 1) % 5 == 0 or puzzle_idx == n_puzzles - 1:
            partial = " | ".join(f"Pass@{k}={counts[k]}/{puzzle_idx+1} ({100*counts[k]/(puzzle_idx+1):.1f}%)" for k in k_values)
            print(f"  Puzzle {puzzle_idx+1}/{n_puzzles} | {partial}")

    results = {f"pass_at_{k}": counts[k] / n_puzzles for k in k_values}
    results["n_puzzles"] = n_puzzles

    print(f"\n  Final ({model_name}):")
    for k in k_values:
        print(f"    Pass@{k}: {counts[k]}/{n_puzzles} = {100*counts[k]/n_puzzles:.2f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate termination prediction models")
    parser.add_argument("--env", default="sudoku", choices=["sudoku", "sokoban"],
                        help="Environment to evaluate on")
    parser.add_argument("--metric", default="all",
                        choices=["termination", "pass-at-k", "solvable-logprob", "all"],
                        help="Which evaluation mode to run. 'solvable-logprob' = teacher-forced "
                             "logprob extraction at the <solvable> token, with threshold sweep. "
                             "Requires --eval-from-parquet (needs response template).")
    parser.add_argument("--sft-path", default="outputs/sft_sudoku_llm_policy",
                        help="Path to SFT model checkpoint")
    parser.add_argument("--rl-path", default="outputs/rl_sudoku",
                        help="Path to RL model checkpoint")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Base model for tokenizer (and as a baseline if no SFT/RL paths exist)")
    # Termination-mode args
    parser.add_argument("--n-solvable", type=int, default=100, help="(termination, single-turn) solvable eval samples")
    parser.add_argument("--n-unsolvable", type=int, default=100, help="(termination, single-turn) unsolvable eval samples")
    parser.add_argument("--sample-outputs", type=int, default=3, help="(termination) sample outputs to print")
    parser.add_argument("--eval-from-parquet", default=None,
                        help="Path to a training-format parquet (e.g. wm_val_filtered.parquet). "
                             "When set, eval samples are loaded from this file and use the multi-turn "
                             "prompts as-is, matching training distribution (overrides --n-solvable/--n-unsolvable).")
    parser.add_argument("--n-per-class", type=int, default=100,
                        help="(eval-from-parquet) max samples per class (solvable/BP/post-BP)")
    parser.add_argument("--eval-temperature", type=float, default=0.0,
                        help="(termination) sampling temperature; 0 = greedy (default), >0 = stochastic decode "
                             "(useful as a probe to check whether the model has internal class discrimination "
                             "buried by greedy's winner-takes-all decoding)")
    # Pass@k-mode args
    parser.add_argument("--n-puzzles", type=int, default=50, help="(pass-at-k) puzzles to evaluate")
    parser.add_argument("--k", default="1,8", help="(pass-at-k) comma-separated k values, e.g. 1,8")
    parser.add_argument("--grid-size", type=int, default=9, help="(sudoku) grid size — 9 (default) or 4 for SPA-replica setup")
    parser.add_argument("--difficulty", default="easy", choices=["easy", "medium", "hard"], help="(sudoku) puzzle difficulty")
    parser.add_argument("--max-rollout-steps", type=int, default=30, help="(pass-at-k) max steps per rollout")
    parser.add_argument("--rollout-temperature", type=float, default=0.7, help="(pass-at-k) sampling temperature for k>1")
    parser.add_argument("--max-context-turns", type=int, default=10, help="(pass-at-k) multi-turn sliding window")
    parser.add_argument("--tag-name", type=str, default="solvable",
                        help="XML tag to probe in solvable-logprob mode. 'solvable' (default, Sudoku) or 'viability' (Polyomino).")
    # Skip flags
    parser.add_argument("--skip-sft", action="store_true", help="Skip SFT evaluation")
    parser.add_argument("--skip-rl", action="store_true", help="Skip RL evaluation")
    parser.add_argument("--include-base", action="store_true", help="Also evaluate the base model (no fine-tuning)")
    args = parser.parse_args()
    args.k_values = tuple(int(x) for x in args.k.split(","))

    print("=" * 60)
    print(f"Balanced Evaluation: env={args.env} metric={args.metric}")
    print("=" * 60)

    np.random.seed(42)
    torch.manual_seed(42)

    # ── Env + formatter ──────────────────────────────────────────────
    if args.env == "sudoku":
        env = SudokuEnv(grid_size=args.grid_size, difficulty=args.difficulty, max_steps=args.max_rollout_steps)
        formatter = SFTFormatter(variant="sudoku_full")
    else:  # sokoban
        env = SokobanEnv(dim_room=(6, 6), num_boxes=1, max_steps=100)
        formatter = SFTFormatter(variant="full")

    # ── Termination-eval set (only built if needed) ──────────────────
    eval_samples = None
    if args.metric in ("termination", "solvable-logprob", "all"):
        if args.eval_from_parquet:
            print(f"\nLoading multi-turn eval samples from parquet: {args.eval_from_parquet}")
            eval_samples = load_balanced_from_parquet(
                args.eval_from_parquet,
                n_per_class=args.n_per_class,
            )
        else:
            print("\nGenerating balanced eval set from live environment...")
            eval_samples = generate_balanced_eval_set(
                env=env,
                system_prompt=formatter.system_prompt,
                n_solvable=args.n_solvable,
                n_unsolvable=args.n_unsolvable,
            )
        n_sol = sum(1 for s in eval_samples if s["is_solvable"])
        n_unsol = sum(1 for s in eval_samples if not s["is_solvable"])
        n_bp = sum(1 for s in eval_samples if s["is_breaking_point"])
        print(f"Eval set: {len(eval_samples)} total ({n_sol} solvable, {n_unsol} unsolvable, {n_bp} breaking points)")

    # ── Tokenizer ────────────────────────────────────────────────────
    print(f"\nLoading tokenizer from: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = {}

    def _run_one_model(label, path, results_key):
        """Load a checkpoint, run the requested metrics, free GPU. Returns results dict."""
        print(f"\nLoading {label} model from: {path}")
        try:
            mdl = AutoModelForCausalLM.from_pretrained(
                path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"Failed to load {label} model: {e}")
            return

        bucket = {}
        if args.metric in ("termination", "all"):
            term_results, term_metrics = evaluate_model(
                mdl, tokenizer, eval_samples, formatter.system_prompt,
                model_name=f"{label} ({path})",
                temperature=args.eval_temperature,
            )
            bucket["termination"] = term_results
            print(f"\n--- Sample {label} Outputs ({args.sample_outputs} examples) ---")
            print_sample_outputs(mdl, tokenizer, eval_samples, formatter.system_prompt,
                                 args.sample_outputs, label)

        if args.metric in ("solvable-logprob",):
            lp_results = evaluate_solvable_logprob(
                mdl, tokenizer, eval_samples, formatter.system_prompt,
                model_name=f"{label} ({path})",
                tag_name=args.tag_name,
            )
            bucket["solvable_logprob"] = lp_results

        if args.metric in ("pass-at-k", "all"):
            pak_results = evaluate_pass_at_k(
                mdl, tokenizer, env, formatter.system_prompt,
                n_puzzles=args.n_puzzles,
                k_values=args.k_values,
                max_rollout_steps=args.max_rollout_steps,
                sampling_temperature=args.rollout_temperature,
                max_context_turns=args.max_context_turns,
                model_name=f"{label} ({path})",
            )
            bucket["pass_at_k"] = pak_results

        all_results[results_key] = bucket
        del mdl
        torch.cuda.empty_cache()

    # ── Base model (optional) ────────────────────────────────────────
    if args.include_base:
        _run_one_model("BASE", args.base_model, "base")

    # ── SFT ──────────────────────────────────────────────────────────
    if not args.skip_sft:
        _run_one_model("SFT", args.sft_path, "sft")

    # ── RL ───────────────────────────────────────────────────────────
    if not args.skip_rl:
        _run_one_model("RL", args.rl_path, "rl")

    # ── Comparison table ─────────────────────────────────────────────
    model_keys = [k for k in ("base", "sft", "rl") if k in all_results]
    if len(model_keys) >= 2:
        print("\n" + "=" * 80)
        print("COMPARISON: " + " vs ".join(k.upper() for k in model_keys))
        print("=" * 80)

        # Termination metrics (if available)
        if any("termination" in all_results[k] for k in model_keys):
            print("\n[Termination metrics]")
            term_rows = [
                ("Format Compliance", "format_compliance"),
                ("Solvable Accuracy", "solvable_accuracy"),
                ("Solvable F1", "solvable_f1"),
                ("BP Accuracy", "bp_accuracy"),
                ("BP Precision", "bp_precision"),
                ("BP Recall (key!)", "bp_recall"),
                ("BP F1", "bp_f1"),
            ]
            header = f"{'Metric':<25} " + " ".join(f"{k.upper():>10}" for k in model_keys)
            print(header)
            print("-" * len(header))
            for label, key in term_rows:
                vals = [all_results[k].get("termination", {}).get(key, 0) for k in model_keys]
                row = f"{label:<25} " + " ".join(f"{v:>9.1f}%" for v in vals)
                print(row)

        # Pass@k metrics (if available)
        if any("pass_at_k" in all_results[k] for k in model_keys):
            print("\n[Pass@k metrics]")
            sample = next(all_results[k]["pass_at_k"] for k in model_keys if "pass_at_k" in all_results[k])
            k_keys = sorted(int(k.split("_")[-1]) for k in sample if k.startswith("pass_at_"))
            header = f"{'Metric':<25} " + " ".join(f"{k.upper():>10}" for k in model_keys)
            print(header)
            print("-" * len(header))
            for k_val in k_keys:
                row_label = f"Pass@{k_val}"
                vals = [all_results[k].get("pass_at_k", {}).get(f"pass_at_{k_val}", 0) for k in model_keys]
                row = f"{row_label:<25} " + " ".join(f"{100*v:>9.2f}%" for v in vals)
                print(row)
            # n_puzzles for context
            n_p = sample.get("n_puzzles", "?")
            print(f"\n  (over n_puzzles={n_p})")

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
