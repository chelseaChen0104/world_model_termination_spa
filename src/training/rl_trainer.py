# RL Trainer for Termination Prediction
# Adapted from RAGEN's agent_trainer.py for termination prediction with world models
#
# This trainer implements PPO/GRPO for training LLMs to predict termination signals
# in the context of world model learning.
#
# Updated for new verl API (2025)

"""
FSDP PPO Trainer with Ray-based single controller for Termination Prediction.
Adapted from RAGEN implementation.
"""

import os
import torch
import numpy as np
from collections import defaultdict
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from copy import deepcopy
from typing import Optional, Dict, Any, List

# Updated imports for new verl API
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
# New API: use extract_reward instead of compute_reward
from verl.trainer.ppo.reward import extract_reward, get_custom_reward_fn
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.torch_functional import masked_mean

# Import ray_trainer components
from verl.trainer.ppo.ray_trainer import apply_kl_penalty

try:
    from tensordict import TensorDict
except ImportError:
    TensorDict = None

import re


def parse_termination_predictions(text: str) -> Dict[str, Any]:
    """
    Parse XML-formatted model output to extract termination predictions.

    Handles various format variations:
    - With or without closing tags
    - With or without spaces
    - Various tag styles (case insensitive, underscores vs camelCase)

    Returns:
        Dict with parsed values or None for missing fields
    """
    result = {
        "terminate_prob": None,
        "steps_left": None,
        "solvable": None,
        "breaking_point": None,
        "answer": None,
    }

    # Parse terminate_prob - multiple tag name variations (case insensitive)
    # Handles: terminate_prob, Terminate_Probability, etc.
    match = re.search(r'<[Tt]erminate[_\s]?[Pp]rob(?:ability)?>\s*([\d.]+)', text)
    if match:
        try:
            result["terminate_prob"] = float(match.group(1))
        except ValueError:
            pass

    # Parse steps_left - multiple variations (case insensitive)
    # Handles: steps_left, Steps_Remaining, etc.
    match = re.search(r'<[Ss]teps[_\s]?(?:[Ll]eft|[Rr]emaining)>\s*(\w+)', text)
    if match:
        result["steps_left"] = match.group(1).lower()

    # Parse solvable - flexible format (case insensitive)
    match = re.search(r'<[Ss]olvable>\s*(\w+)', text)
    if match:
        val = match.group(1).lower()
        result["solvable"] = val in ["true", "yes", "1"]

    # Parse breaking_point - multiple variations (case insensitive)
    # Handles: breaking_point, Breaking_Point, Deadlock_identification, etc.
    match = re.search(r'<[Bb]reaking[_\s]?[Pp]oint>\s*(\w+)', text)
    if not match:
        match = re.search(r'<[Dd]eadlock[_\s]?(?:[Ii]dentification)?>\s*(\w+)', text)
    if match:
        val = match.group(1).lower()
        result["breaking_point"] = val in ["true", "yes", "1"]

    # Parse answer - flexible format
    match = re.search(r'<[Aa]nswer>\s*(\w+)', text)
    if match:
        result["answer"] = match.group(1)

    return result


def parse_ground_truth_from_extra_info(extra_info_str: str) -> Dict[str, Any]:
    """Parse ground truth from extra_info field in dataset."""
    result = {
        "is_solvable": None,
        "is_breaking_point": None,
        "step": None,
    }

    try:
        import json
        info = json.loads(extra_info_str.replace("'", '"'))
        result["is_solvable"] = info.get("is_solvable")
        result["is_breaking_point"] = info.get("is_breaking_point")
        result["step"] = info.get("step")
    except:
        pass

    return result


def compute_response_mask(data: DataProto) -> torch.Tensor:
    """Compute response mask from attention mask and input length."""
    attention_mask = data.batch["attention_mask"]
    input_lens = data.batch.get("input_lens", None)

    if input_lens is not None:
        batch_size, seq_len = attention_mask.shape
        response_mask = torch.zeros_like(attention_mask)
        for i in range(batch_size):
            response_mask[i, input_lens[i]:] = attention_mask[i, input_lens[i]:]
        return response_mask
    else:
        # Fallback: assume all non-padding tokens are response
        return attention_mask


def compute_termination_reward(
    batch: DataProto,
    config: Dict[str, Any],
) -> torch.Tensor:
    """
    Compute reward for termination prediction.

    Rewards are based on:
    1. Correct solvability prediction
    2. Correct breaking point detection
    3. Accurate steps_left estimation

    Args:
        batch: DataProto containing the model outputs
        config: Configuration dictionary

    Returns:
        reward_tensor: Tensor of shape (batch_size, seq_len)
    """
    batch_size = batch.batch["input_ids"].shape[0]
    seq_len = batch.batch["input_ids"].shape[1]
    reward_tensor = torch.zeros(batch_size, seq_len, device=batch.batch["input_ids"].device)

    # Get ground truth labels from non_tensor_batch
    is_solvable_gt = batch.non_tensor_batch.get("is_solvable_gt", None)
    is_breaking_point_gt = batch.non_tensor_batch.get("is_breaking_point_gt", None)

    # Get model predictions (parsed from response)
    solvable_pred = batch.non_tensor_batch.get("solvable_pred", None)
    breaking_point_pred = batch.non_tensor_batch.get("breaking_point_pred", None)
    steps_left_pred = batch.non_tensor_batch.get("steps_left_pred", None)

    # Reward weights from config
    reward_config = config.get("reward", {})
    solvable_weight = reward_config.get("solvable_reward_weight", 1.0) if hasattr(reward_config, 'get') else 1.0
    breaking_point_weight = reward_config.get("breaking_point_bonus", 1.5) if hasattr(reward_config, 'get') else 1.5
    format_penalty = -0.5  # Penalty for not producing valid predictions

    total_rewards = 0.0
    valid_samples = 0

    for i in range(batch_size):
        reward = 0.0

        # Check if prediction was parsed successfully
        has_valid_pred = solvable_pred is not None and solvable_pred[i] is not None

        if not has_valid_pred:
            # Penalty for invalid/unparseable output
            reward = format_penalty
        else:
            valid_samples += 1

            # Solvability prediction reward
            if is_solvable_gt is not None and is_solvable_gt[i] is not None:
                gt_solvable = bool(is_solvable_gt[i])
                pred_solvable = bool(solvable_pred[i])

                if gt_solvable == pred_solvable:
                    reward += solvable_weight
                else:
                    reward -= solvable_weight * 0.5  # Penalty for wrong prediction

            # Breaking point detection reward (high value - this is key for efficiency)
            if is_breaking_point_gt is not None and is_breaking_point_gt[i] is not None:
                gt_bp = bool(is_breaking_point_gt[i])
                pred_bp = bool(breaking_point_pred[i]) if breaking_point_pred is not None and breaking_point_pred[i] is not None else False

                if gt_bp and pred_bp:
                    # Correctly identified breaking point - big bonus!
                    reward += breaking_point_weight
                elif gt_bp and not pred_bp:
                    # Missed a breaking point - penalty
                    reward -= breaking_point_weight * 0.5
                elif not gt_bp and pred_bp:
                    # False positive - small penalty
                    reward -= breaking_point_weight * 0.25
                # True negative (correctly said no breaking point) - no change

        total_rewards += reward

        # Assign reward to the last token of the response
        response_mask = batch.batch.get("response_mask", None)
        if response_mask is not None and response_mask[i].any():
            last_token_idx = response_mask[i].nonzero()[-1].item()
            reward_tensor[i, last_token_idx] = reward
        else:
            reward_tensor[i, -1] = reward

    # Log reward statistics periodically
    if batch_size > 0:
        avg_reward = total_rewards / batch_size
        valid_ratio = valid_samples / batch_size if batch_size > 0 else 0
        # Print debug info occasionally
        if np.random.random() < 0.1:  # 10% of the time
            print(f"  [Reward Debug] avg={avg_reward:.3f}, valid_preds={valid_ratio:.1%}")

    return reward_tensor


def compute_termination_reward_v2(
    batch: DataProto,
    config: Dict[str, Any],
) -> torch.Tensor:
    """
    Improved reward function for termination prediction RL.

    Key changes vs v1:
    - Format compliance gives POSITIVE reward (prevents forgetting XML format)
    - Breaking point TP/FN have strong asymmetric weights (catches missed deadlocks)
    - steps_left NOT rewarded (trajectory-dependent, noisy — SFT only)
    - Solvable prediction uses moderate symmetric weights

    Reward table:
        Format: +0.1 per required tag found (max +0.5 for all 5 tags)
        Solvable correct:   +0.5
        Solvable wrong:     -0.25
        BP TP (caught!):    +3.0
        BP FN (missed!):    -2.0
        BP FP (false alarm): -0.5
        BP TN:               0.0
    """
    batch_size = batch.batch["input_ids"].shape[0]
    seq_len = batch.batch["input_ids"].shape[1]
    reward_tensor = torch.zeros(batch_size, seq_len, device=batch.batch["input_ids"].device)

    # Ground truth
    is_solvable_gt = batch.non_tensor_batch.get("is_solvable_gt", None)
    is_breaking_point_gt = batch.non_tensor_batch.get("is_breaking_point_gt", None)

    # Predictions
    solvable_pred = batch.non_tensor_batch.get("solvable_pred", None)
    breaking_point_pred = batch.non_tensor_batch.get("breaking_point_pred", None)

    # Optional: raw generated text for format checking
    generated_texts = batch.non_tensor_batch.get("generated_texts", None)

    # Reward weights from config (with v2 defaults)
    reward_config = config.get("reward", {}) if hasattr(config, 'get') else {}
    solvable_correct_r = reward_config.get("solvable_correct", 0.5) if hasattr(reward_config, 'get') else 0.5
    solvable_wrong_r = reward_config.get("solvable_wrong", -0.25) if hasattr(reward_config, 'get') else -0.25
    bp_tp_r = reward_config.get("bp_tp", 3.0) if hasattr(reward_config, 'get') else 3.0
    bp_fn_r = reward_config.get("bp_fn", -2.0) if hasattr(reward_config, 'get') else -2.0
    bp_fp_r = reward_config.get("bp_fp", -0.5) if hasattr(reward_config, 'get') else -0.5
    format_tag_r = reward_config.get("format_tag_reward", 0.1) if hasattr(reward_config, 'get') else 0.1

    # Required XML tags for format compliance
    required_tags = ["<solvable>", "<breaking_point>", "<steps_left>", "<terminate_prob>", "<answer>"]

    total_rewards = 0.0
    valid_samples = 0
    bp_tp = 0
    bp_fn = 0
    bp_fp = 0
    bp_tn = 0
    sol_correct = 0
    sol_wrong = 0

    for i in range(batch_size):
        reward = 0.0

        # 1. Format compliance: reward for each XML tag present
        if generated_texts is not None and i < len(generated_texts) and generated_texts[i]:
            text_lower = generated_texts[i].lower()
            for tag in required_tags:
                if tag in text_lower:
                    reward += format_tag_r

        # Check if prediction was parsed successfully
        has_solvable = solvable_pred is not None and i < len(solvable_pred) and solvable_pred[i] is not None
        has_bp = breaking_point_pred is not None and i < len(breaking_point_pred) and breaking_point_pred[i] is not None

        if has_solvable or has_bp:
            valid_samples += 1

        # 2. Solvability prediction reward
        if has_solvable and is_solvable_gt is not None and i < len(is_solvable_gt) and is_solvable_gt[i] is not None:
            gt_solvable = bool(is_solvable_gt[i])
            pred_solvable = bool(solvable_pred[i])

            if gt_solvable == pred_solvable:
                reward += solvable_correct_r
                sol_correct += 1
            else:
                reward += solvable_wrong_r
                sol_wrong += 1

        # 3. Breaking point detection — strongly asymmetric
        if is_breaking_point_gt is not None and i < len(is_breaking_point_gt) and is_breaking_point_gt[i] is not None:
            gt_bp = bool(is_breaking_point_gt[i])
            pred_bp = bool(breaking_point_pred[i]) if has_bp else False

            if gt_bp and pred_bp:
                reward += bp_tp_r
                bp_tp += 1
            elif gt_bp and not pred_bp:
                reward += bp_fn_r
                bp_fn += 1
            elif not gt_bp and pred_bp:
                reward += bp_fp_r
                bp_fp += 1
            else:
                bp_tn += 1

        total_rewards += reward

        # Assign reward to the last token of the response
        response_mask = batch.batch.get("response_mask", None)
        if response_mask is not None and response_mask[i].any():
            last_token_idx = response_mask[i].nonzero()[-1].item()
            reward_tensor[i, last_token_idx] = reward
        else:
            reward_tensor[i, -1] = reward

    # Log reward statistics periodically
    if batch_size > 0 and np.random.random() < 0.1:
        avg_reward = total_rewards / batch_size
        valid_ratio = valid_samples / batch_size
        print(f"  [Reward_v2] avg={avg_reward:.3f}, valid={valid_ratio:.0%}, "
              f"sol={sol_correct}ok/{sol_wrong}bad, "
              f"bp: TP={bp_tp} FN={bp_fn} FP={bp_fp} TN={bp_tn}")

    return reward_tensor


def _get_steps_bucket(steps: int) -> str:
    """Convert steps to bucket label."""
    if isinstance(steps, str):
        return steps  # Already a bucket
    if steps <= 3:
        return "immediate"
    elif steps <= 10:
        return "near"
    elif steps <= 25:
        return "medium"
    else:
        return "far"


def _bucket_to_int(bucket: str) -> int:
    """Convert bucket to integer for comparison."""
    bucket_map = {"immediate": 0, "near": 1, "medium": 2, "far": 3}
    return bucket_map.get(bucket, -1)


def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    episode_ids: Optional[np.ndarray] = None,
):
    """
    Compute advantage for GRPO with episode-level deduplication support.

    Adapted from RAGEN for termination prediction.
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]

        seen_pairs = set()
        for i in range(bsz):
            if episode_ids is not None:
                pair = (index[i], episode_ids[i])
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0, device=scores.device)
                id2std[idx] = torch.tensor(1.0, device=scores.device)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")

        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_advantage(
    data: DataProto,
    adv_estimator: str,
    gamma: float = 1.0,
    lam: float = 1.0,
    norm_adv_by_std_in_grpo: bool = True,
):
    """Compute advantages using specified estimator."""
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)

    if adv_estimator == "gae" or adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns

    elif adv_estimator == "grpo" or adv_estimator == AdvantageEstimator.GRPO:
        episode_ids = data.non_tensor_batch.get("episode_ids", None)
        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            episode_ids=episode_ids,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError(f"Advantage estimator {adv_estimator} not supported")

    return data


class TerminationRLTrainer:
    """
    RL Trainer for Termination Prediction.

    This trainer fine-tunes an LLM (after SFT) to predict:
    1. Whether an episode is done (done_label)
    2. How many steps are left (steps_left_bucket)
    3. Whether the state is solvable (is_solvable)
    4. Deadlock detection (deadlock_type)

    Uses GRPO or PPO for advantage estimation.
    """

    def __init__(
        self,
        config,
        tokenizer,
        model,
        reward_fn=None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.model = model
        self.reward_fn = reward_fn or compute_termination_reward

        self.global_steps = 0
        self.total_training_steps = config.trainer.total_training_steps

    def train(self):
        """Main training loop."""
        print("=" * 60)
        print("Termination Prediction - RL Training")
        print("=" * 60)
        print(f"Total training steps: {self.total_training_steps}")
        print(f"Advantage estimator: {self.config.algorithm.adv_estimator}")
        print("=" * 60)

        # Initialize wandb if configured
        if hasattr(self.config.trainer, 'logger') and "wandb" in self.config.trainer.logger:
            try:
                import wandb
                wandb.init(
                    project=self.config.trainer.get('project_name', 'termination-rl'),
                    name=self.config.trainer.get('experiment_name', 'rl_grpo'),
                    config=dict(self.config),
                )
                print("Wandb initialized successfully")
            except Exception as e:
                print(f"Warning: Failed to initialize wandb: {e}")
                print("Continuing without wandb logging...")

        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="RL Training"
        )

        for step in range(self.total_training_steps):
            metrics = {}

            # 1. Generate rollouts
            batch = self._generate_rollouts()

            # 2. Compute rewards
            reward_tensor = self.reward_fn(batch, self.config)
            batch.batch["token_level_rewards"] = reward_tensor

            # 3. Compute advantages
            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
            )

            # 4. Update policy
            actor_metrics = self._update_actor(batch)
            metrics.update(actor_metrics)

            # 5. Logging
            self._log_metrics(metrics)

            progress_bar.update(1)
            self.global_steps += 1

            # 6. Checkpointing
            if self.global_steps % self.config.trainer.save_freq == 0:
                self._save_checkpoint()

        progress_bar.close()
        print("RL Training Complete!")

    def _generate_rollouts(self) -> DataProto:
        """Generate rollouts using the current policy."""
        raise NotImplementedError("Subclasses must implement _generate_rollouts")

    def _update_actor(self, batch: DataProto) -> Dict[str, float]:
        """Update the actor using PPO/GRPO."""
        raise NotImplementedError("Subclasses must implement _update_actor")

    def _log_metrics(self, metrics: Dict[str, float]):
        """Log training metrics."""
        # Print to console
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        if self.global_steps % 10 == 0:  # Print every 10 steps
            print(f"Step {self.global_steps}: {metrics_str}")

        # Log to wandb if initialized
        if hasattr(self.config.trainer, 'logger') and "wandb" in self.config.trainer.logger:
            import wandb
            if wandb.run is not None:
                wandb.log(metrics, step=self.global_steps)

    def _save_checkpoint(self):
        """Save model checkpoint."""
        ckpt_dir = os.path.join(
            self.config.trainer.default_local_dir,
            f"step_{self.global_steps}"
        )
        os.makedirs(ckpt_dir, exist_ok=True)

        # Save model
        self.model.save_pretrained(ckpt_dir)
        self.tokenizer.save_pretrained(ckpt_dir)

        print(f"Saved checkpoint to {ckpt_dir}")


class SimpleTerminationRLTrainer(TerminationRLTrainer):
    """
    Simple single-GPU RL Trainer without Ray/distributed training.

    This is a simplified version that can run on a single GPU for testing.
    For full distributed training, use DistributedTerminationRLTrainer.
    """

    def __init__(
        self,
        config,
        tokenizer,
        model,
        train_dataset,
        reward_fn=None,
        collate_fn=None,
    ):
        super().__init__(config, tokenizer, model, reward_fn)
        self.train_dataset = train_dataset

        # Use smaller batch size to avoid OOM (micro_batch_size_per_gpu)
        batch_size = config.data.get('micro_batch_size_per_gpu', 4)
        print(f"Using batch size: {batch_size}")

        self.dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        self.dataloader_iter = iter(self.dataloader)

        # Optimizer - get lr from actor_rollout_ref.actor.optim
        lr = config.actor_rollout_ref.actor.optim.lr
        weight_decay = config.actor_rollout_ref.actor.optim.get('weight_decay', 0.01)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        print(f"Optimizer: AdamW, lr={lr}, weight_decay={weight_decay}")

    def _generate_rollouts(self) -> DataProto:
        """Generate rollouts by sampling from dataset and running model."""
        try:
            batch = next(self.dataloader_iter)
        except StopIteration:
            self.dataloader_iter = iter(self.dataloader)
            batch = next(self.dataloader_iter)

        # Move to device
        device = next(self.model.parameters()).device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        extra_infos = batch.get("extra_infos", [None] * input_ids.shape[0])

        # Generate responses
        rollout_cfg = self.config.actor_rollout_ref.rollout
        max_new_tokens = rollout_cfg.max_new_tokens
        temperature = rollout_cfg.temperature

        # Debug generation params
        if self.global_steps == 0:
            print(f"[Debug] Generation params: max_new_tokens={max_new_tokens}, temperature={temperature}")
            print(f"[Debug] Input shape: {input_ids.shape}")

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode generated text to parse predictions
        batch_size = outputs.shape[0]
        input_len = input_ids.shape[1]
        output_len = outputs.shape[1]

        # Debug shapes
        if self.global_steps % 20 == 0:
            print(f"\n[Debug] input_len={input_len}, output_len={output_len}, new_tokens={output_len - input_len}")

        # Extract only the generated part (after the prompt)
        if output_len > input_len:
            generated_ids = outputs[:, input_len:]
            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        else:
            # No new tokens generated - use full output and decode
            print(f"[Warning] No new tokens generated! output_len={output_len}, input_len={input_len}")
            generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Parse predictions from generated text
        solvable_preds = []
        breaking_point_preds = []
        steps_left_preds = []
        terminate_prob_preds = []

        # Debug: print first generated text occasionally
        if self.global_steps % 20 == 0 and len(generated_texts) > 0:
            print(f"[Debug] Step {self.global_steps} - Sample generated text (first 500 chars):")
            sample_text = generated_texts[0][:500] if generated_texts[0] else "(empty)"
            print(sample_text)
            print("---")

        for text in generated_texts:
            preds = parse_termination_predictions(text)
            solvable_preds.append(preds.get("solvable"))
            breaking_point_preds.append(preds.get("breaking_point"))
            steps_left_preds.append(preds.get("steps_left"))
            terminate_prob_preds.append(preds.get("terminate_prob"))

        # Parse ground truth from extra_info
        solvable_gt = []
        breaking_point_gt = []

        for info_str in extra_infos:
            if info_str:
                gt = parse_ground_truth_from_extra_info(info_str)
                solvable_gt.append(gt.get("is_solvable"))
                breaking_point_gt.append(gt.get("is_breaking_point"))
            else:
                solvable_gt.append(None)
                breaking_point_gt.append(None)

        # Create DataProto
        data = DataProto()
        data.batch = {
            "input_ids": outputs,
            "attention_mask": torch.ones_like(outputs),
            "input_lens": torch.tensor([input_len] * batch_size),
        }
        data.non_tensor_batch = {
            "uid": np.arange(batch_size),
            # Ground truth
            "is_solvable_gt": np.array(solvable_gt, dtype=object),
            "is_breaking_point_gt": np.array(breaking_point_gt, dtype=object),
            # Predictions
            "solvable_pred": np.array(solvable_preds, dtype=object),
            "breaking_point_pred": np.array(breaking_point_preds, dtype=object),
            "steps_left_pred": np.array(steps_left_preds, dtype=object),
            "terminate_prob_pred": np.array(terminate_prob_preds, dtype=object),
            # Raw text for format reward (needed by compute_termination_reward_v2)
            "generated_texts": np.array(generated_texts, dtype=object),
        }

        return data

    def _update_actor(self, batch: DataProto) -> Dict[str, float]:
        """Update actor using simple policy gradient."""
        self.optimizer.zero_grad()

        input_ids = batch.batch["input_ids"]
        attention_mask = batch.batch["attention_mask"]
        advantages = batch.batch["advantages"]
        response_mask = batch.batch["response_mask"]

        # Forward pass
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits

        # Compute log probs
        log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)

        # Policy gradient loss
        pg_loss = -masked_mean(token_log_probs * advantages[:, 1:], response_mask[:, 1:])

        pg_loss.backward()

        # Gradient clipping
        max_grad_norm = self.config.trainer.get('max_grad_norm', 1.0)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_grad_norm
        )

        self.optimizer.step()

        return {
            "policy_loss": pg_loss.item(),
            "mean_advantage": advantages.mean().item(),
        }


class LiveEnvTerminationRLTrainer(TerminationRLTrainer):
    """
    RL Trainer that generates fresh balanced batches from a live environment.

    Instead of sampling from a static parquet dataset, this trainer:
    1. Plays trajectories in a live env (Sokoban, Sudoku, etc.)
    2. Samples balanced batches (~50% solvable, ~50% unsolvable)
    3. Focuses on breaking points for better deadlock detection
    4. Uses compute_termination_reward_v2 by default

    This addresses class imbalance (99.2% solvable) in static datasets.
    """

    def __init__(
        self,
        config,
        tokenizer,
        model,
        env,
        system_prompt: str,
        reward_fn=None,
        solvable_ratio: float = 0.5,
        breaking_point_focus: float = 0.3,
        pool_size: int = 200,
        refresh_frequency: int = 50,
    ):
        super().__init__(config, tokenizer, model, reward_fn or compute_termination_reward_v2)

        from src.data.live_trajectory_sampler import LiveTrajectorySampler

        batch_size = config.data.get('micro_batch_size_per_gpu', 4)
        self.batch_size = batch_size

        # Initialize live sampler
        self.sampler = LiveTrajectorySampler(
            env=env,
            system_prompt=system_prompt,
            tokenizer=tokenizer,
            batch_size=batch_size,
            solvable_ratio=solvable_ratio,
            breaking_point_focus=breaking_point_focus,
            pool_size=pool_size,
            refresh_frequency=refresh_frequency,
        )

        # Optimizer
        lr = config.actor_rollout_ref.actor.optim.lr
        weight_decay = config.actor_rollout_ref.actor.optim.get('weight_decay', 0.01)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        print(f"[LiveEnvTrainer] batch_size={batch_size}, lr={lr}, "
              f"solvable_ratio={solvable_ratio}, bp_focus={breaking_point_focus}")

    def _generate_rollouts(self) -> DataProto:
        """Generate rollouts from live environment with balanced sampling."""
        # Get balanced batch from live sampler
        sampler_batch = self.sampler.sample_batch()
        samples = sampler_batch["samples"]

        device = next(self.model.parameters()).device
        input_ids = sampler_batch["input_ids"].to(device)
        attention_mask = sampler_batch["attention_mask"].to(device)

        # Generate responses
        rollout_cfg = self.config.actor_rollout_ref.rollout
        max_new_tokens = rollout_cfg.max_new_tokens
        temperature = rollout_cfg.temperature

        if self.global_steps == 0:
            stats = self.sampler.get_stats()
            print(f"[LiveEnvTrainer] Pool: {stats}")
            print(f"[LiveEnvTrainer] Generation: max_new_tokens={max_new_tokens}, temp={temperature}")

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode generated text to parse predictions
        batch_size = outputs.shape[0]
        input_len = input_ids.shape[1]
        output_len = outputs.shape[1]

        if output_len > input_len:
            generated_ids = outputs[:, input_len:]
            generated_texts = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        else:
            print(f"[Warning] No new tokens generated! output_len={output_len}, input_len={input_len}")
            generated_texts = [""] * batch_size

        # Parse predictions from generated text
        solvable_preds = []
        breaking_point_preds = []

        if self.global_steps % 20 == 0 and generated_texts:
            sample_text = generated_texts[0][:500] if generated_texts[0] else "(empty)"
            print(f"[LiveEnvTrainer] Step {self.global_steps} sample output:\n{sample_text}\n---")

        for text in generated_texts:
            preds = parse_termination_predictions(text)
            solvable_preds.append(preds.get("solvable"))
            breaking_point_preds.append(preds.get("breaking_point"))

        # Ground truth from live sampler samples
        solvable_gt = [s.is_solvable for s in samples]
        breaking_point_gt = [s.is_breaking_point for s in samples]

        # Log class balance periodically
        if self.global_steps % 20 == 0:
            n_sol = sum(1 for s in solvable_gt if s)
            n_unsol = sum(1 for s in solvable_gt if not s)
            n_bp = sum(1 for s in breaking_point_gt if s)
            print(f"[LiveEnvTrainer] Batch balance: {n_sol} solvable, {n_unsol} unsolvable, {n_bp} breaking points")

        # Create DataProto
        data = DataProto()
        data.batch = {
            "input_ids": outputs,
            "attention_mask": torch.ones_like(outputs),
            "input_lens": torch.tensor([input_len] * batch_size),
        }
        data.non_tensor_batch = {
            "uid": np.arange(batch_size),
            # Ground truth
            "is_solvable_gt": np.array(solvable_gt, dtype=object),
            "is_breaking_point_gt": np.array(breaking_point_gt, dtype=object),
            # Predictions
            "solvable_pred": np.array(solvable_preds, dtype=object),
            "breaking_point_pred": np.array(breaking_point_preds, dtype=object),
            # Raw text for format reward
            "generated_texts": np.array(generated_texts, dtype=object),
        }

        return data

    def _update_actor(self, batch: DataProto) -> Dict[str, float]:
        """Update actor using policy gradient (same as SimpleTerminationRLTrainer)."""
        self.optimizer.zero_grad()

        input_ids = batch.batch["input_ids"]
        attention_mask = batch.batch["attention_mask"]
        advantages = batch.batch["advantages"]
        response_mask = batch.batch["response_mask"]

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        logits = outputs.logits

        log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)

        pg_loss = -masked_mean(token_log_probs * advantages[:, 1:], response_mask[:, 1:])
        pg_loss.backward()

        max_grad_norm = self.config.trainer.get('max_grad_norm', 1.0)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
        self.optimizer.step()

        return {
            "policy_loss": pg_loss.item(),
            "mean_advantage": advantages.mean().item(),
        }


# Placeholder for the full distributed trainer
class DistributedTerminationRLTrainer(TerminationRLTrainer):
    """
    Distributed RL Trainer using Ray for multi-GPU training.

    This is a placeholder - full implementation requires:
    1. Ray worker setup
    2. FSDP/DeepSpeed integration
    3. vLLM for fast rollout generation
    """
    pass


def main():
    """Main entry point for RL training."""
    import hydra
    from omegaconf import DictConfig

    @hydra.main(config_path="config", config_name="rl_termination", version_base=None)
    def run(config: DictConfig):
        print("=" * 60)
        print("Termination Prediction - RL Training")
        print("=" * 60)

        # Load tokenizer and model
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import os

        # Convert relative path to absolute path for local checkpoints
        model_path = config.model.partial_pretrain
        if not model_path.startswith('/') and not '/' in model_path.split('/')[0]:
            # Looks like a local relative path, convert to absolute
            model_path = os.path.abspath(model_path)
            print(f"Using local model path: {model_path}")

        # Get base model name for tokenizer (SFT checkpoint may have incompatible tokenizer config)
        base_model = getattr(config.model, 'base_model', 'Qwen/Qwen2.5-0.5B-Instruct')
        print(f"Loading tokenizer from base model: {base_model}")

        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True,
        )

        print(f"Loading model from: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Load dataset
        import pandas as pd

        # Handle both train_file and train_files config options
        if hasattr(config.data, 'train_file'):
            train_files = config.data.train_file
        else:
            train_files = config.data.train_files

        # Ensure train_files is a list
        if isinstance(train_files, str):
            train_files = [train_files]
        else:
            train_files = list(train_files)

        # Load and concatenate all training files
        dfs = []
        for f in train_files:
            print(f"Loading: {f}")
            dfs.append(pd.read_parquet(f))
        train_df = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(train_df)} training samples from {len(train_files)} file(s)")

        # Print dataset columns for debugging
        print(f"Dataset columns: {train_df.columns.tolist()}")
        print(f"First row sample: {train_df.iloc[0].to_dict()}")

        # Set tokenizer padding
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'  # For generation

        # Use smaller max_length to save memory (512 is enough for Sokoban prompts)
        max_seq_length = min(config.data.get('max_length', 512), 512)
        print(f"Using max sequence length: {max_seq_length}")

        # Create simple dataset with ground truth
        class SimpleDataset(Dataset):
            def __init__(self, df, tokenizer, max_length=512):
                self.df = df.reset_index(drop=True)
                self.tokenizer = tokenizer
                self.max_length = max_length

                # Convert prompts to proper chat format
                self.prompts = []
                for prompt_data in df['prompt']:
                    # Handle array of message dicts
                    if isinstance(prompt_data, (list, np.ndarray)):
                        messages = list(prompt_data)
                        # Apply chat template
                        formatted = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True
                        )
                        self.prompts.append(formatted)
                    else:
                        # Already a string
                        self.prompts.append(str(prompt_data))

                # Pre-extract extra_info for ground truth
                self.extra_infos = df['extra_info'].astype(str).tolist() if 'extra_info' in df.columns else [None] * len(df)

                # Debug: print first prompt
                print(f"[Debug] First formatted prompt (first 300 chars):\n{self.prompts[0][:300]}...")

            def __len__(self):
                return len(self.prompts)

            def __getitem__(self, idx):
                prompt = self.prompts[idx]
                # Tokenize - return as list for collate_fn
                encoding = self.tokenizer(
                    prompt,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                return {
                    "input_ids": encoding["input_ids"].squeeze(0),
                    "attention_mask": encoding["attention_mask"].squeeze(0),
                    "extra_info": self.extra_infos[idx],
                }

        # Custom collate function for dynamic padding
        def collate_fn(batch):
            # Find max length in this batch
            max_len = max(item["input_ids"].size(0) for item in batch)

            input_ids_list = []
            attention_mask_list = []
            extra_infos = []

            for item in batch:
                seq_len = item["input_ids"].size(0)
                pad_len = max_len - seq_len

                # Left padding
                if pad_len > 0:
                    input_ids = torch.cat([
                        torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long),
                        item["input_ids"]
                    ])
                    attention_mask = torch.cat([
                        torch.zeros(pad_len, dtype=torch.long),
                        item["attention_mask"]
                    ])
                else:
                    input_ids = item["input_ids"]
                    attention_mask = item["attention_mask"]

                input_ids_list.append(input_ids)
                attention_mask_list.append(attention_mask)
                extra_infos.append(item.get("extra_info", None))

            return {
                "input_ids": torch.stack(input_ids_list),
                "attention_mask": torch.stack(attention_mask_list),
                "extra_infos": extra_infos,  # List of strings
            }

        train_dataset = SimpleDataset(train_df, tokenizer, max_length=max_seq_length)

        # Create trainer
        trainer = SimpleTerminationRLTrainer(
            config=config,
            tokenizer=tokenizer,
            model=model,
            train_dataset=train_dataset,
            collate_fn=collate_fn,
        )

        # Train
        trainer.train()

    run()


def main_live_env():
    """Entry point for live-environment RL training."""
    import hydra
    from omegaconf import DictConfig

    @hydra.main(config_path="config", config_name="rl_termination", version_base=None)
    def run(config: DictConfig):
        print("=" * 60)
        print("Termination Prediction - Live Env RL Training")
        print("=" * 60)

        from transformers import AutoTokenizer, AutoModelForCausalLM
        from src.environments.sokoban import SokobanEnv
        from src.data.sft_formatter import SFTFormatter

        # Load tokenizer
        base_model = getattr(config.model, 'base_model', 'Qwen/Qwen2.5-0.5B-Instruct')
        print(f"Loading tokenizer from: {base_model}")
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

        # Load model
        model_path = config.model.partial_pretrain
        if not model_path.startswith('/') and '/' not in model_path.split('/')[0]:
            model_path = os.path.abspath(model_path)
        print(f"Loading model from: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Create environment
        env_config = config.get('environment', {})
        env_type = env_config.get('type', 'sokoban')
        if env_type == 'sokoban':
            dim_room = tuple(env_config.get('dim_room', [6, 6]))
            num_boxes = env_config.get('num_boxes', 1)
            max_steps = env_config.get('max_steps', 100)
            env = SokobanEnv(dim_room=dim_room, num_boxes=num_boxes, max_steps=max_steps)
            print(f"Environment: Sokoban {dim_room}, {num_boxes} boxes, max_steps={max_steps}")
            formatter = SFTFormatter(variant="full")
        elif env_type == 'sudoku':
            from src.environments.sudoku import SudokuEnv
            grid_size = env_config.get('grid_size', 9)
            difficulty = env_config.get('difficulty', 'easy')
            max_steps = env_config.get('max_steps', 81)
            env = SudokuEnv(grid_size=grid_size, difficulty=difficulty, max_steps=max_steps)
            print(f"Environment: Sudoku {grid_size}x{grid_size}, {difficulty}, max_steps={max_steps}")
            formatter = SFTFormatter(variant="sudoku_full")
        else:
            raise ValueError(f"Unknown environment type: {env_type}")

        # System prompt
        system_prompt = formatter.system_prompt

        # Sampling config
        sampling_config = config.get('sampling', {})
        solvable_ratio = sampling_config.get('solvable_ratio', 0.5)
        bp_focus = sampling_config.get('breaking_point_focus', 0.3)
        pool_size = sampling_config.get('trajectory_pool_size', 200)
        refresh_freq = sampling_config.get('refresh_frequency', 50)

        # Create trainer
        trainer = LiveEnvTerminationRLTrainer(
            config=config,
            tokenizer=tokenizer,
            model=model,
            env=env,
            system_prompt=system_prompt,
            solvable_ratio=solvable_ratio,
            breaking_point_focus=bp_focus,
            pool_size=pool_size,
            refresh_frequency=refresh_freq,
        )

        # Train
        trainer.train()

    run()


if __name__ == "__main__":
    main()
