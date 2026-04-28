# Training modules for termination prediction
#
# SFT Training: Supervised fine-tuning on trajectory data
# RL Training: Reinforcement learning with termination-based rewards
#
# Note: The full trainers (sft_trainer.py, rl_trainer.py) require RAGEN/verl infrastructure.
# For standalone training, use simple_sft_trainer.py instead.

# Lazy imports to avoid importing verl/ragen when not needed
def get_sft_trainer():
    """Get the FSDP SFT Trainer (requires verl/ragen)."""
    from .sft_trainer import FSDPSFTTrainer
    return FSDPSFTTrainer

def get_simple_sft_trainer():
    """Get the simple SFT Trainer (standalone, uses HuggingFace Trainer)."""
    from .simple_sft_trainer import main as train_sft
    return train_sft

def get_rl_trainer():
    """Get the RL Trainer (requires verl/ragen)."""
    from .rl_trainer import TerminationRLTrainer
    return TerminationRLTrainer

__all__ = [
    "get_sft_trainer",
    "get_simple_sft_trainer",
    "get_rl_trainer",
]
