"""Test Hidato's predictive-gap property — run random rollouts, measure
fraction of states that are "looks valid but actually doomed" (the SFT signal).

For each step in a random rollout:
- The action is "place next number in some valid adjacent cell"
- After placing, we check is_solvable (from env)
- A "predictive gap" event = action_was_valid (didn't violate adjacency) AND
  resulting state is is_solvable=False (the model couldn't have known just
  from local rules — needs to think globally)

Output: doom rate, average rollout length, distribution of where doom happens
in the trajectory.
"""
from __future__ import annotations
import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environments.hidato import HidatoEnv


def run_random_rollout(env: HidatoEnv, seed: int) -> dict:
    """Run one rollout with random action selection. Returns trajectory stats."""
    obs = env.reset(seed=seed)
    rng = random.Random(seed * 7919 + 1)
    n_steps = 0
    doom_step = None
    pre_doom_assignment_size = None
    while True:
        actions = env.get_all_actions()
        if not actions:
            # No valid actions — game ends here
            break
        action = rng.choice(actions)
        obs, reward, done, info = env.step(action)
        n_steps += 1
        if not info["is_solvable"] and doom_step is None:
            doom_step = n_steps
            pre_doom_assignment_size = len(env.assignment) - 1  # before this action
        if done:
            break
    return {
        "puzzle_id": env.puzzle["id"],
        "n_steps": n_steps,
        "n_filled_at_end": len(env.assignment),
        "n_total": env.rows * env.cols,
        "success": info.get("success", False),
        "doom_step": doom_step,
        "pre_doom_assignment_size": pre_doom_assignment_size,
        "is_doomed_at_end": not info.get("is_solvable", True),
    }


def main():
    n_rollouts = 200
    print(f"Hidato predictive-gap test ({n_rollouts} random rollouts):\n")
    env = HidatoEnv()
    stats = [run_random_rollout(env, seed=i) for i in range(n_rollouts)]

    n_success = sum(1 for s in stats if s["success"])
    n_doom = sum(1 for s in stats if s["is_doomed_at_end"])
    n_truncated = n_rollouts - n_success - n_doom

    avg_len = sum(s["n_steps"] for s in stats) / n_rollouts
    avg_doom_step = sum(s["doom_step"] for s in stats if s["doom_step"]) / max(1, n_doom)
    avg_n_filled = sum(s["n_filled_at_end"] for s in stats) / n_rollouts

    print(f"  Rollouts:         {n_rollouts}")
    print(f"  Success rate:     {n_success}/{n_rollouts} ({100*n_success/n_rollouts:.1f}%)")
    print(f"  Doom rate:        {n_doom}/{n_rollouts} ({100*n_doom/n_rollouts:.1f}%)")
    print(f"  Truncated rate:   {n_truncated}/{n_rollouts} ({100*n_truncated/n_rollouts:.1f}%)")
    print(f"  Avg rollout len:  {avg_len:.2f} steps")
    print(f"  Avg doom step:    {avg_doom_step:.2f} (over the {n_doom} doom rollouts)")
    print(f"  Avg n_filled at end: {avg_n_filled:.2f}")

    # Per-puzzle breakdown
    print("\n  Per-puzzle breakdown:")
    by_puzzle = {}
    for s in stats:
        by_puzzle.setdefault(s["puzzle_id"], []).append(s)
    for pid, items in sorted(by_puzzle.items()):
        n = len(items)
        succ = sum(1 for s in items if s["success"])
        doom = sum(1 for s in items if s["is_doomed_at_end"])
        avg_l = sum(s["n_steps"] for s in items) / n
        print(f"    {pid:>22s}: n={n}, success={succ}, doom={doom}, avg_len={avg_l:.1f}")

    # Verdict
    doom_rate = n_doom / n_rollouts
    print(f"\n  Verdict: doom rate = {100*doom_rate:.1f}%")
    if 30 <= 100*doom_rate <= 80:
        print("    ✓ Healthy predictive gap — model has plenty to learn")
    elif 100*doom_rate < 30:
        print("    ⚠ Low doom rate — might be too easy / no predictive gap")
    else:
        print("    ⚠ Very high doom rate — most random play dooms; check if env too restrictive")


if __name__ == "__main__":
    main()
