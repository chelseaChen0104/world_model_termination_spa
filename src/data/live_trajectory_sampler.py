"""
Live Trajectory Sampler for RL Training

Generates fresh, balanced training batches by playing trajectories in a live
environment and selecting steps with controlled class balance.
"""

import random
import numpy as np
import torch
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from src.data.trajectory_generator import TrajectoryGenerator, TrajectoryStep, TrajectoryMetadata
from src.environments.base import BaseTerminationEnv


@dataclass
class RLSample:
    """A single training sample for RL with ground truth labels."""
    state: str                    # Current grid state (observation)
    next_state: str               # Next grid state
    action_name: str              # Action taken
    is_solvable: bool             # Ground truth: is state solvable?
    is_breaking_point: bool       # Ground truth: did this action create deadlock?
    deadlock_type: Optional[str]  # Type of deadlock (if any)
    steps_left: int               # Steps until trajectory ends
    steps_left_bucket: str        # Categorical bucket
    step_index: int               # Step number in trajectory


class LiveTrajectorySampler:
    """Generates balanced RL training batches from live environment play.

    Instead of sampling from a static dataset, this class:
    1. Plays full trajectories using random actions in a live env
    2. Records all steps with ground-truth annotations
    3. Samples balanced batches (~50% solvable, ~50% unsolvable)
    4. Focuses sampling around breaking points
    5. Refreshes the trajectory pool periodically
    """

    def __init__(
        self,
        env: BaseTerminationEnv,
        system_prompt: str,
        tokenizer,
        batch_size: int = 16,
        solvable_ratio: float = 0.5,
        breaking_point_focus: float = 0.3,
        max_steps: int = 100,
        window_around_bp: int = 3,
        pool_size: int = 200,
        refresh_frequency: int = 50,
    ):
        self.env = env
        self.system_prompt = system_prompt
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.solvable_ratio = solvable_ratio
        self.breaking_point_focus = breaking_point_focus
        self.max_steps = max_steps
        self.window_around_bp = window_around_bp
        self.pool_size = pool_size
        self.refresh_frequency = refresh_frequency

        self.generator = TrajectoryGenerator(env)
        self._trajectory_pool: List[Tuple[List[TrajectoryStep], TrajectoryMetadata]] = []
        self._solvable_steps: List[RLSample] = []
        self._unsolvable_steps: List[RLSample] = []
        self._breaking_point_steps: List[RLSample] = []
        self._sample_count = 0
        self._seed_counter = 0

        # Fill initial pool
        self._fill_pool()

    def _fill_pool(self):
        """Generate trajectories to fill the pool and index steps by class."""
        self._solvable_steps.clear()
        self._unsolvable_steps.clear()
        self._breaking_point_steps.clear()

        attempts = 0
        max_attempts = self.pool_size * 10

        while len(self._trajectory_pool) < self.pool_size and attempts < max_attempts:
            attempts += 1
            self._seed_counter += 1
            try:
                traj, meta = self.generator.generate_random_trajectory(
                    max_steps=self.max_steps,
                    seed=self._seed_counter,
                )
            except Exception:
                continue

            if not traj:
                continue

            self._trajectory_pool.append((traj, meta))

            # Index steps by class
            for step in traj:
                sample = RLSample(
                    state=step.state,
                    next_state=step.next_state,
                    action_name=step.action_name,
                    is_solvable=step.is_solvable,
                    is_breaking_point=step.is_breaking_point,
                    deadlock_type=step.deadlock_type,
                    steps_left=step.steps_left,
                    steps_left_bucket=step.steps_left_bucket,
                    step_index=step.step,
                )

                if step.is_breaking_point:
                    self._breaking_point_steps.append(sample)
                    self._unsolvable_steps.append(sample)
                elif not step.is_solvable:
                    self._unsolvable_steps.append(sample)
                else:
                    self._solvable_steps.append(sample)

        stats = (
            f"Pool filled: {len(self._trajectory_pool)} trajectories, "
            f"{len(self._solvable_steps)} solvable, "
            f"{len(self._unsolvable_steps)} unsolvable "
            f"({len(self._breaking_point_steps)} breaking points)"
        )
        print(f"  [LiveSampler] {stats}")

    def _maybe_refresh_pool(self):
        """Refresh pool periodically to prevent staleness."""
        if self._sample_count > 0 and self._sample_count % self.refresh_frequency == 0:
            self._trajectory_pool.clear()
            self._fill_pool()

    def sample_batch(self) -> Dict:
        """Generate a balanced batch of training samples.

        Returns:
            Dict with keys:
            - input_ids: (batch_size, max_seq_len) tokenized prompts
            - attention_mask: (batch_size, max_seq_len)
            - extra_infos: list of JSON strings with ground truth
            - samples: list of RLSample objects (for debugging)
        """
        self._maybe_refresh_pool()
        self._sample_count += 1

        # Calculate how many of each class to sample
        n_solvable = int(self.batch_size * self.solvable_ratio)
        n_unsolvable = self.batch_size - n_solvable

        # Within unsolvable: prioritize breaking points
        n_bp = min(
            int(n_unsolvable * self.breaking_point_focus),
            len(self._breaking_point_steps),
        )
        n_other_unsolvable = n_unsolvable - n_bp

        # Sample with replacement if needed
        samples = []

        if self._solvable_steps:
            sol_indices = np.random.choice(len(self._solvable_steps), size=n_solvable, replace=True)
            samples.extend([self._solvable_steps[i] for i in sol_indices])

        if self._breaking_point_steps and n_bp > 0:
            bp_indices = np.random.choice(len(self._breaking_point_steps), size=n_bp, replace=True)
            samples.extend([self._breaking_point_steps[i] for i in bp_indices])

        if self._unsolvable_steps and n_other_unsolvable > 0:
            unsol_indices = np.random.choice(len(self._unsolvable_steps), size=n_other_unsolvable, replace=True)
            samples.extend([self._unsolvable_steps[i] for i in unsol_indices])

        # Handle edge case: not enough unsolvable samples
        while len(samples) < self.batch_size and self._solvable_steps:
            idx = np.random.randint(len(self._solvable_steps))
            samples.append(self._solvable_steps[idx])

        # Shuffle to avoid all-solvable-then-all-unsolvable ordering
        random.shuffle(samples)

        # Format as tokenized prompts
        return self._format_batch(samples)

    def _format_batch(self, samples: List[RLSample]) -> Dict:
        """Convert samples to tokenized batch."""
        all_input_ids = []
        all_attention_masks = []
        extra_infos = []

        for sample in samples:
            # Build conversation
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Current state:\n{sample.state}"},
            ]

            # Tokenize using chat template
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            encoded = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            all_input_ids.append(encoded["input_ids"].squeeze(0))
            all_attention_masks.append(encoded["attention_mask"].squeeze(0))

            # Ground truth as JSON
            extra_info = json.dumps({
                "is_solvable": sample.is_solvable,
                "is_breaking_point": sample.is_breaking_point,
                "deadlock_type": sample.deadlock_type,
                "steps_left": sample.steps_left,
                "steps_left_bucket": sample.steps_left_bucket,
                "step": sample.step_index,
            })
            extra_infos.append(extra_info)

        # Pad to same length (left padding for generation)
        max_len = max(ids.size(0) for ids in all_input_ids)
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        padded_input_ids = []
        padded_attention_masks = []
        for ids, mask in zip(all_input_ids, all_attention_masks):
            pad_len = max_len - ids.size(0)
            padded_input_ids.append(
                torch.cat([torch.full((pad_len,), pad_token_id, dtype=ids.dtype), ids])
            )
            padded_attention_masks.append(
                torch.cat([torch.zeros(pad_len, dtype=mask.dtype), mask])
            )

        return {
            "input_ids": torch.stack(padded_input_ids),
            "attention_mask": torch.stack(padded_attention_masks),
            "extra_infos": extra_infos,
            "samples": samples,
        }

    def get_stats(self) -> Dict:
        """Return statistics about the current pool."""
        return {
            "pool_trajectories": len(self._trajectory_pool),
            "solvable_steps": len(self._solvable_steps),
            "unsolvable_steps": len(self._unsolvable_steps),
            "breaking_point_steps": len(self._breaking_point_steps),
            "total_samples_generated": self._sample_count,
        }


if __name__ == "__main__":
    from src.environments.sokoban import SokobanEnv
    from src.data.sft_formatter import SFTFormatter

    env = SokobanEnv(dim_room=(6, 6), num_boxes=1, max_steps=50)
    formatter = SFTFormatter(variant="full")

    # Quick test without real tokenizer
    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 0
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return str(messages)
        def __call__(self, text, return_tensors="pt", truncation=True, max_length=512):
            ids = torch.randint(0, 100, (1, 50))
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    sampler = LiveTrajectorySampler(
        env=env,
        system_prompt=formatter.system_prompt,
        tokenizer=FakeTokenizer(),
        batch_size=16,
        pool_size=50,
    )

    print("\nStats:", sampler.get_stats())

    batch = sampler.sample_batch()
    print(f"\nBatch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch extra_infos count: {len(batch['extra_infos'])}")

    # Check class balance
    n_solvable = sum(1 for s in batch["samples"] if s.is_solvable)
    n_unsolvable = sum(1 for s in batch["samples"] if not s.is_solvable)
    n_bp = sum(1 for s in batch["samples"] if s.is_breaking_point)
    print(f"\nClass balance: {n_solvable} solvable, {n_unsolvable} unsolvable ({n_bp} breaking points)")
