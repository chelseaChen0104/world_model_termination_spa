"""
Trajectory Generation with Full Annotations

This module generates trajectories with termination and breaking point annotations
for training termination-aware world models.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import json
import random


@dataclass
class TrajectoryStep:
    """Single step in a trajectory with all annotations."""
    state: str  # Current state observation
    action: int  # Action taken
    action_name: str  # Human-readable action name
    next_state: str  # Resulting state
    reward: float
    step: int

    # Termination annotations
    done_label: int  # 1 if this is the last step, else 0
    steps_left: int  # Number of steps remaining
    steps_left_bucket: str  # Categorical: immediate/near/medium/far

    # Breaking point annotations
    is_solvable: bool  # Can the puzzle still be solved?
    is_breaking_point: bool  # Did this action create a deadlock?
    deadlock_type: Optional[str]  # Type of deadlock if any
    steps_since_break: Optional[int]  # Steps since breaking point (if applicable)

    # Additional info
    success: bool  # Did the episode end successfully?

    # LLM raw response (for multi-turn SFT context)
    # Populated by LLMTrajectoryGenerator; None for random-play trajectories
    llm_raw_response: Optional[str] = None


@dataclass
class TrajectoryMetadata:
    """Trajectory-level metadata."""
    total_steps: int
    success: bool
    has_breaking_point: bool
    breaking_point_step: Optional[int]
    steps_wasted: int  # Steps taken after breaking point
    final_reward: float
    termination_reason: str  # 'success', 'deadlock', 'timeout'


def bucket_steps(delta: int) -> str:
    """Convert raw steps to categorical bucket."""
    if delta == 1:
        return "immediate"
    elif delta <= 3:
        return "near"
    elif delta <= 7:
        return "medium"
    else:
        return "far"


class TrajectoryGenerator:
    """
    Generates trajectories with full termination and breaking point annotations.

    This generator can work with:
    1. A policy model (for RL-based generation)
    2. Random actions (for exploration)
    3. Expert demonstrations (from BFS solver)
    """

    def __init__(self, env, detector=None):
        """
        Initialize the trajectory generator.

        Args:
            env: Sokoban environment instance
            detector: Deadlock detector (optional, will use env's detector if available)
        """
        self.env = env
        self.detector = detector or getattr(env, 'detector', None)

    def generate_random_trajectory(self, max_steps: int = 100,
                                   seed: Optional[int] = None) -> Tuple[List[TrajectoryStep], TrajectoryMetadata]:
        """
        Generate a trajectory using random actions.

        Args:
            max_steps: Maximum steps per episode
            seed: Random seed for reproducibility

        Returns:
            Tuple of (list of trajectory steps, trajectory metadata)
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        trajectory = []
        state = self.env.reset(seed=seed)
        done = False
        t = 0

        prev_solvable = True
        breaking_point_step = None

        while not done and t < max_steps:
            # Select random action
            all_actions = self.env.get_all_actions() if hasattr(self.env, 'get_all_actions') else list(self.env.ACTION_LOOKUP.keys())
            if not all_actions:
                break  # No valid actions available
            action = random.choice(all_actions)
            # Get human-readable action name
            if hasattr(self.env, 'ACTION_LOOKUP') and action in self.env.ACTION_LOOKUP:
                action_name = self.env.ACTION_LOOKUP[action]
            else:
                action_name = str(action)

            # Take action
            next_state, reward, done, info = self.env.step(action)

            # Get solvability info
            is_solvable = info.get('is_solvable', True)
            deadlock_type = info.get('deadlock_type', None)

            # Detect breaking point
            is_breaking = prev_solvable and not is_solvable
            if is_breaking and breaking_point_step is None:
                breaking_point_step = t

            step = TrajectoryStep(
                state=state,
                action=action,
                action_name=action_name,
                next_state=next_state,
                reward=reward,
                step=t,
                done_label=0,  # Will be updated in post-processing
                steps_left=0,  # Will be updated in post-processing
                steps_left_bucket='',  # Will be updated in post-processing
                is_solvable=is_solvable,
                is_breaking_point=is_breaking,
                deadlock_type=deadlock_type,
                steps_since_break=None,  # Will be updated in post-processing
                success=info.get('success', False)
            )

            trajectory.append(step)

            prev_solvable = is_solvable
            state = next_state
            t += 1

        # Post-hoc annotation of termination signals
        T = len(trajectory)
        B = breaking_point_step

        for i, step in enumerate(trajectory):
            # Termination signals
            step.done_label = 1 if i == T - 1 else 0
            step.steps_left = T - i
            step.steps_left_bucket = bucket_steps(T - i)

            # Breaking point signals
            if B is not None and i >= B:
                step.steps_since_break = i - B

        # Determine termination reason
        if trajectory and trajectory[-1].success:
            termination_reason = 'success'
        elif B is not None:
            termination_reason = 'deadlock'
        else:
            termination_reason = 'timeout'

        metadata = TrajectoryMetadata(
            total_steps=T,
            success=trajectory[-1].success if trajectory else False,
            has_breaking_point=B is not None,
            breaking_point_step=B,
            steps_wasted=(T - 1 - B) if B is not None else 0,
            final_reward=trajectory[-1].reward if trajectory else 0,
            termination_reason=termination_reason
        )

        return trajectory, metadata

    def generate_batch(self, num_trajectories: int, max_steps: int = 100,
                       seed: Optional[int] = None) -> List[Tuple[List[TrajectoryStep], TrajectoryMetadata]]:
        """
        Generate a batch of trajectories.

        Args:
            num_trajectories: Number of trajectories to generate
            max_steps: Maximum steps per episode
            seed: Base random seed

        Returns:
            List of (trajectory, metadata) tuples
        """
        trajectories = []

        for i in range(num_trajectories):
            traj_seed = seed + i if seed is not None else None
            traj, meta = self.generate_random_trajectory(max_steps, traj_seed)
            trajectories.append((traj, meta))

        return trajectories

    def generate_balanced_dataset(self, target_size: int = 1280,
                                   success_ratio: float = 0.4,
                                   failure_ratio: float = 0.4,
                                   timeout_ratio: float = 0.2,
                                   max_steps: int = 100,
                                   seed: Optional[int] = None) -> List[Tuple[List[TrajectoryStep], TrajectoryMetadata]]:
        """
        Generate a balanced dataset with specified outcome ratios.

        Args:
            target_size: Total number of trajectories to generate
            success_ratio: Ratio of successful trajectories
            failure_ratio: Ratio of failure (deadlock) trajectories
            timeout_ratio: Ratio of timeout trajectories
            max_steps: Maximum steps per episode
            seed: Random seed

        Returns:
            List of balanced (trajectory, metadata) tuples
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        target_success = int(target_size * success_ratio)
        target_failure = int(target_size * failure_ratio)
        target_timeout = target_size - target_success - target_failure

        success_trajs = []
        failure_trajs = []
        timeout_trajs = []

        attempt = 0
        max_attempts = target_size * 10  # Prevent infinite loop

        while (len(success_trajs) < target_success or
               len(failure_trajs) < target_failure or
               len(timeout_trajs) < target_timeout) and attempt < max_attempts:

            traj, meta = self.generate_random_trajectory(max_steps, seed=attempt)

            if meta.success and len(success_trajs) < target_success:
                success_trajs.append((traj, meta))
            elif meta.has_breaking_point and len(failure_trajs) < target_failure:
                failure_trajs.append((traj, meta))
            elif not meta.success and not meta.has_breaking_point and len(timeout_trajs) < target_timeout:
                timeout_trajs.append((traj, meta))

            attempt += 1

        # Combine and shuffle
        all_trajs = success_trajs + failure_trajs + timeout_trajs
        random.shuffle(all_trajs)

        print(f"Generated balanced dataset:")
        print(f"  - Success: {len(success_trajs)} (target: {target_success})")
        print(f"  - Failure: {len(failure_trajs)} (target: {target_failure})")
        print(f"  - Timeout: {len(timeout_trajs)} (target: {target_timeout})")

        return all_trajs


def trajectory_to_dict(trajectory: List[TrajectoryStep],
                       metadata: TrajectoryMetadata) -> Dict:
    """Convert a trajectory to a dictionary for serialization."""
    steps = []
    for step in trajectory:
        steps.append({
            'state': step.state,
            'action': step.action,
            'action_name': step.action_name,
            'next_state': step.next_state,
            'reward': step.reward,
            'step': step.step,
            'done_label': step.done_label,
            'steps_left': step.steps_left,
            'steps_left_bucket': step.steps_left_bucket,
            'is_solvable': step.is_solvable,
            'is_breaking_point': step.is_breaking_point,
            'deadlock_type': step.deadlock_type,
            'steps_since_break': step.steps_since_break,
            'success': step.success,
            'llm_raw_response': step.llm_raw_response,
        })

    return {
        'steps': steps,
        'metadata': {
            'total_steps': metadata.total_steps,
            'success': metadata.success,
            'has_breaking_point': metadata.has_breaking_point,
            'breaking_point_step': metadata.breaking_point_step,
            'steps_wasted': metadata.steps_wasted,
            'final_reward': metadata.final_reward,
            'termination_reason': metadata.termination_reason,
        }
    }


def save_trajectories(trajectories: List[Tuple[List[TrajectoryStep], TrajectoryMetadata]],
                      output_path: str):
    """Save trajectories to a JSON file."""
    data = [trajectory_to_dict(traj, meta) for traj, meta in trajectories]
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(trajectories)} trajectories to {output_path}")


if __name__ == '__main__':
    # Test trajectory generation
    from environments.sokoban import SokobanEnv

    env = SokobanEnv(dim_room=(6, 6), num_boxes=1)
    generator = TrajectoryGenerator(env)

    # Generate a single trajectory
    trajectory, metadata = generator.generate_random_trajectory(max_steps=50, seed=42)

    print(f"Generated trajectory with {len(trajectory)} steps")
    print(f"Metadata: {metadata}")

    # Show first few steps
    for step in trajectory[:5]:
        print(f"\nStep {step.step}:")
        print(f"  Action: {step.action_name}")
        print(f"  Solvable: {step.is_solvable}")
        print(f"  Steps left: {step.steps_left} ({step.steps_left_bucket})")
