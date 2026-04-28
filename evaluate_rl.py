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

    match = re.search(r'<[Ss]olvable>\s*(\w+)', text)
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


def evaluate_model(model, tokenizer, eval_samples, system_prompt, model_name="Model"):
    """Evaluate a model on the balanced eval set."""
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
        # Build prompt
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate termination prediction models")
    parser.add_argument("--sft-path", default="outputs/sft_termination/checkpoint-5730",
                        help="Path to SFT model checkpoint")
    parser.add_argument("--rl-path", default="outputs/rl_termination/step_1000",
                        help="Path to RL model checkpoint")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Base model for tokenizer")
    parser.add_argument("--n-solvable", type=int, default=100, help="Number of solvable eval samples")
    parser.add_argument("--n-unsolvable", type=int, default=100, help="Number of unsolvable eval samples")
    parser.add_argument("--sample-outputs", type=int, default=3, help="Number of sample outputs to print")
    parser.add_argument("--skip-sft", action="store_true", help="Skip SFT evaluation")
    parser.add_argument("--skip-rl", action="store_true", help="Skip RL evaluation")
    args = parser.parse_args()

    print("=" * 60)
    print("Balanced Evaluation for Termination Prediction")
    print("=" * 60)

    np.random.seed(42)
    torch.manual_seed(42)

    # Generate balanced eval set from live environment
    print("\nGenerating balanced eval set from live environment...")
    env = SokobanEnv(dim_room=(6, 6), num_boxes=1, max_steps=100)
    formatter = SFTFormatter(variant="full")

    eval_samples = generate_balanced_eval_set(
        env=env,
        system_prompt=formatter.system_prompt,
        n_solvable=args.n_solvable,
        n_unsolvable=args.n_unsolvable,
    )

    # Distribution summary
    n_sol = sum(1 for s in eval_samples if s["is_solvable"])
    n_unsol = sum(1 for s in eval_samples if not s["is_solvable"])
    n_bp = sum(1 for s in eval_samples if s["is_breaking_point"])
    print(f"Eval set: {len(eval_samples)} total ({n_sol} solvable, {n_unsol} unsolvable, {n_bp} breaking points)")

    # Load tokenizer
    print(f"\nLoading tokenizer from: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_results = {}

    # Evaluate SFT model
    if not args.skip_sft:
        print(f"\nLoading SFT model from: {args.sft_path}")
        try:
            sft_model = AutoModelForCausalLM.from_pretrained(
                args.sft_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            sft_results, sft_metrics = evaluate_model(
                sft_model, tokenizer, eval_samples, formatter.system_prompt,
                model_name=f"SFT ({args.sft_path})"
            )
            all_results["sft"] = sft_results

            print(f"\n--- Sample SFT Outputs ({args.sample_outputs} examples) ---")
            print_sample_outputs(sft_model, tokenizer, eval_samples, formatter.system_prompt,
                                args.sample_outputs, "SFT")

            del sft_model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Failed to load SFT model: {e}")

    # Evaluate RL model
    if not args.skip_rl:
        print(f"\nLoading RL model from: {args.rl_path}")
        try:
            rl_model = AutoModelForCausalLM.from_pretrained(
                args.rl_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
            rl_results, rl_metrics = evaluate_model(
                rl_model, tokenizer, eval_samples, formatter.system_prompt,
                model_name=f"RL ({args.rl_path})"
            )
            all_results["rl"] = rl_results

            print(f"\n--- Sample RL Outputs ({args.sample_outputs} examples) ---")
            print_sample_outputs(rl_model, tokenizer, eval_samples, formatter.system_prompt,
                                args.sample_outputs, "RL")

            del rl_model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Failed to load RL model: {e}")

    # Comparison table
    if len(all_results) >= 2:
        print("\n" + "=" * 70)
        print("COMPARISON: SFT vs RL")
        print("=" * 70)

        comparison_metrics = [
            ("Format Compliance", "format_compliance"),
            ("Solvable Accuracy", "solvable_accuracy"),
            ("Solvable F1", "solvable_f1"),
            ("BP Accuracy", "bp_accuracy"),
            ("BP Precision", "bp_precision"),
            ("BP Recall (key!)", "bp_recall"),
            ("BP F1", "bp_f1"),
        ]

        sft_r = all_results.get("sft", {})
        rl_r = all_results.get("rl", {})

        print(f"\n{'Metric':<25} {'SFT':>10} {'RL':>10} {'Delta':>10}")
        print("-" * 57)

        for label, key in comparison_metrics:
            sft_val = sft_r.get(key, 0)
            rl_val = rl_r.get(key, 0)
            delta = rl_val - sft_val
            delta_str = f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}"
            print(f"{label:<25} {sft_val:>9.1f}% {rl_val:>9.1f}% {delta_str:>9}%")

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
