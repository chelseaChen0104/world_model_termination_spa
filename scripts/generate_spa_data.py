#!/usr/bin/env python3
"""
SPA Data Generation Script

This script generates training data for the SPA (Self-Play Agent) world model.
It creates trajectories with full annotations for termination and breaking point detection.

Usage:
    python scripts/generate_spa_data.py --num-trajectories 1000 --output-dir data/spa_sokoban
"""

import argparse
import os
import sys
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from environments.sokoban import SokobanEnv
from data.trajectory_generator import TrajectoryGenerator, save_trajectories
from data.sft_formatter import SFTFormatter


def parse_args():
    parser = argparse.ArgumentParser(description='Generate SPA training data')

    # Environment settings
    parser.add_argument('--env', type=str, default='sokoban',
                        choices=['sokoban', 'frozenlake', 'sudoku'],
                        help='Environment type')
    parser.add_argument('--dim-x', type=int, default=6,
                        help='Grid width')
    parser.add_argument('--dim-y', type=int, default=6,
                        help='Grid height')
    parser.add_argument('--num-boxes', type=int, default=1,
                        help='Number of boxes (Sokoban)')
    parser.add_argument('--max-steps', type=int, default=100,
                        help='Maximum steps per episode')

    # Sudoku settings
    parser.add_argument('--grid-size', type=int, default=9,
                        help='Sudoku grid size (4, 9, or 16)')
    parser.add_argument('--difficulty', type=str, default='easy',
                        choices=['easy', 'medium', 'hard'],
                        help='Sudoku difficulty')

    # Generation settings
    parser.add_argument('--num-trajectories', type=int, default=1280,
                        help='Total number of trajectories to generate')
    parser.add_argument('--success-ratio', type=float, default=0.4,
                        help='Target ratio of successful trajectories')
    parser.add_argument('--failure-ratio', type=float, default=0.4,
                        help='Target ratio of failure (deadlock) trajectories')
    parser.add_argument('--timeout-ratio', type=float, default=0.2,
                        help='Target ratio of timeout trajectories')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Format settings
    parser.add_argument('--variant', type=str, default='full',
                        choices=['baseline', 'termination', 'full', 'sudoku_full'],
                        help='Output format variant')
    parser.add_argument('--include-coordinates', action='store_true', default=True,
                        help='Include coordinate descriptions')

    # Output settings
    parser.add_argument('--output-dir', type=str, default='data/spa_sokoban',
                        help='Output directory')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split ratio')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("SPA Data Generation")
    print("=" * 60)
    print(f"Environment: {args.env}")
    print(f"Grid size: {args.dim_x}x{args.dim_y}")
    print(f"Num boxes: {args.num_boxes}")
    print(f"Max steps: {args.max_steps}")
    print(f"Num trajectories: {args.num_trajectories}")
    print(f"Format variant: {args.variant}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)

    # Create environment
    print("\n[1/4] Creating environment...")
    if args.env == 'sokoban':
        env = SokobanEnv(
            dim_room=(args.dim_y, args.dim_x),
            num_boxes=args.num_boxes,
            max_steps=args.max_steps
        )
        if args.variant == 'full' or args.variant not in ['sudoku_full']:
            pass  # Default variant is fine for Sokoban
    elif args.env == 'sudoku':
        from environments.sudoku import SudokuEnv
        env = SudokuEnv(
            grid_size=args.grid_size,
            difficulty=args.difficulty,
            max_steps=args.max_steps,
        )
        if args.variant == 'full':
            args.variant = 'sudoku_full'  # Auto-select Sudoku variant
        print(f"Sudoku: {args.grid_size}x{args.grid_size}, difficulty={args.difficulty}")
    else:
        raise NotImplementedError(f"Environment {args.env} not yet implemented")

    # Create trajectory generator
    print("[2/4] Generating trajectories...")
    generator = TrajectoryGenerator(env)

    # Generate balanced dataset
    trajectories = generator.generate_balanced_dataset(
        target_size=args.num_trajectories,
        success_ratio=args.success_ratio,
        failure_ratio=args.failure_ratio,
        timeout_ratio=args.timeout_ratio,
        max_steps=args.max_steps,
        seed=args.seed
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save raw trajectories
    print("[3/4] Saving raw trajectories...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = os.path.join(args.output_dir, f'raw_trajectories_{timestamp}.json')
    save_trajectories(trajectories, raw_path)

    # Format for SFT
    print("[4/4] Formatting for SFT training...")
    formatter = SFTFormatter(
        variant=args.variant,
        include_coordinates=args.include_coordinates
    )

    df = formatter.create_sft_dataset(trajectories)
    formatter.save_dataset(df, args.output_dir, split_ratio=args.val_split)

    # Print statistics
    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)

    # Count statistics
    success_count = sum(1 for _, meta in trajectories if meta.success)
    failure_count = sum(1 for _, meta in trajectories if meta.has_breaking_point)
    timeout_count = len(trajectories) - success_count - failure_count

    print(f"\nTrajectory Statistics:")
    print(f"  Total trajectories: {len(trajectories)}")
    print(f"  Success: {success_count} ({100*success_count/len(trajectories):.1f}%)")
    print(f"  Failure (deadlock): {failure_count} ({100*failure_count/len(trajectories):.1f}%)")
    print(f"  Timeout: {timeout_count} ({100*timeout_count/len(trajectories):.1f}%)")

    # Average trajectory length
    avg_len = sum(meta.total_steps for _, meta in trajectories) / len(trajectories)
    print(f"\n  Average trajectory length: {avg_len:.1f} steps")

    # Breaking point statistics
    bp_trajs = [(t, m) for t, m in trajectories if m.has_breaking_point]
    if bp_trajs:
        avg_bp_step = sum(m.breaking_point_step for _, m in bp_trajs) / len(bp_trajs)
        avg_wasted = sum(m.steps_wasted for _, m in bp_trajs) / len(bp_trajs)
        print(f"\n  Breaking point trajectories: {len(bp_trajs)}")
        print(f"  Average breaking point step: {avg_bp_step:.1f}")
        print(f"  Average steps wasted after breaking point: {avg_wasted:.1f}")

    print(f"\nOutput saved to: {args.output_dir}")
    print(f"  - raw_trajectories_{timestamp}.json")
    print(f"  - wm_train.parquet")
    print(f"  - wm_val.parquet")
    print(f"  - wm_train.csv")
    print(f"  - wm_val.csv")


if __name__ == '__main__':
    main()
