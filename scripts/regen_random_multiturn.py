"""Regenerate sudoku_multiturn random-play SFT data (CPU-only)."""
import sys, time
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, ".")

from src.environments.sudoku import SudokuEnv
from src.data.trajectory_generator import TrajectoryGenerator
from src.data.sft_formatter import SFTFormatter

t0 = time.time()
env = SudokuEnv(grid_size=9, difficulty="easy", max_steps=30)
gen = TrajectoryGenerator(env)
trajectories = gen.generate_balanced_dataset(
    target_size=1280,
    success_ratio=0.0,
    failure_ratio=1.0,
    timeout_ratio=0.0,
    max_steps=30,
    seed=42,
)
print(f"Trajectories generated: {len(trajectories)} in {time.time()-t0:.1f}s")

fmt = SFTFormatter(variant="sudoku_full")
df = fmt.create_sft_dataset(trajectories, multi_turn=True, max_context_turns=10)
fmt.save_dataset(df, "data/sudoku_multiturn", split_ratio=0.2)
print(f"DONE total={time.time()-t0:.1f}s rows={len(df)}")
