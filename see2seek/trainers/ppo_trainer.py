"""
ppo_trainer.py — Proximal Policy Optimisation trainer for recurrent policies.

Implements the full training loop:
    1. Collect `num_steps` steps from `num_envs` parallel environments.
    2. Encode observations and goals with frozen encoders (DINOv2 + CLIP).
       DINOv2 is called ONCE per frame via get_all_embeddings(), returning
       both the CLS token and the patch-token grid — see dino_encoder.py.
    3. Compute GAE returns and advantages.
    4. Run `num_epochs` epochs of mini-batch PPO updates.
    5. Log to W&B and save checkpoints.

The PPO loss has three terms:
    L = L_policy + value_loss_coef * L_value - entropy_coef * H[π]

    L_policy = -E[ min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t) ]
               where r_t = π(a_t|s_t) / π_old(a_t|s_t)

    L_value  = MSE( V(s_t) , R_t )

    H[π]     = -E[ π log π ]   (encourages exploration)

Usage:
    trainer = PPOTrainer(cfg)
    trainer.train()
"""

from __future__ import annotations

import logging
import os
import time
from collections import deque
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from see2seek.utils.config import Config
from see2seek.models.encoders.dino_encoder import DINOv2Encoder
from see2seek.models.encoders.clip_encoder import CLIPGoalEncoder
from see2seek.envs.vec_env import make_vec_envs
from see2seek.agents.gru_policy import build_policy
from see2seek.buffers.rollout_buffer import RolloutBuffer

logger = logging.getLogger(__name__)


class PPOTrainer:
    """
    Full PPO training loop with recurrent GRU policy.

    Args:
        cfg:      Config dataclass (from configs/config.py).
        resume:   Path to a checkpoint to resume from, or None.
    """

    def __init__(self, cfg: Config, resume: Optional[str] = None) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # ---- Logging ----
        self._setup_logging()

        # ---- Encoders (frozen, shared across all updates) ----
        logger.info("Building encoders ...")
        self.obs_encoder  = DINOv2Encoder(
            device=cfg.device,
            normalize=cfg.encoder.obs_normalize,
        )
        self.goal_encoder = CLIPGoalEncoder(
            device=cfg.device,
            normalize=cfg.encoder.goal_normalize,
        )

        # ---- Policy (trainable — includes SpatialCompressionHead + CLS proj) ----
        logger.info("Building GRU Actor-Critic policy ...")
        self.policy = build_policy(cfg, cfg.device)

        # ---- Optimiser ----
        self.optimiser = torch.optim.Adam(
            self.policy.parameters(),
            lr=cfg.ppo.lr,
            eps=cfg.ppo.eps,
        )

        # ---- Environments ----
        logger.info(f"Launching {cfg.env.num_envs} parallel environments ...")
        self.vec_env = make_vec_envs(cfg)

        # ---- Rollout buffer ----
        # storage_device/store_dtype are optional memory knobs — see
        # rollout_buffer.py docstring. Defaulting storage to the same
        # compute device here; if you hit OOM at your num_steps/num_envs,
        # pass storage_device=torch.device("cpu") instead.
        self.buffer = RolloutBuffer(
            num_steps=cfg.ppo.num_steps,
            num_envs=cfg.env.num_envs,
            patch_dim=self.obs_encoder.patch_dim,
            num_patches=self.obs_encoder.num_patches,
            cls_dim=self.obs_encoder.embed_dim,
            goal_dim=cfg.encoder.goal_embed_dim,
            hidden_size=cfg.policy.hidden_size,
            num_actions=cfg.env.num_actions,
            device=self.device,
            pointgoal_dim=cfg.encoder.pointgoal_input_dim,
            storage_device=getattr(cfg.ppo, "buffer_storage_device", None),
            store_dtype=getattr(cfg.ppo, "buffer_store_dtype", torch.float16),
        )

        # ---- Training state ----
        self._total_steps = 0
        self._num_updates = 0
        self._start_time  = time.time()

        # ---- Episode-level metric tracking (rolling window over last 100 episodes) ----
        self._recent_rewards   = deque(maxlen=100)
        self._recent_successes = deque(maxlen=100)
        self._recent_spls      = deque(maxlen=100)
        self._running_reward   = torch.zeros(cfg.env.num_envs, device=self.device)

        # Resume from checkpoint if provided
        if resume is not None:
            self._load_checkpoint(resume)

        logger.info("PPOTrainer initialised and ready.")

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _setup_logging(self) -> None:
        """Initialise W&B if enabled."""
        if self.cfg.logging.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=self.cfg.logging.wandb_project,
                    entity=self.cfg.logging.wandb_entity,
                    name=self.cfg.logging.run_name,
                    config={
                        "env":     vars(self.cfg.env),
                        "encoder": vars(self.cfg.encoder),
                        "policy":  vars(self.cfg.policy),
                        "ppo":     vars(self.cfg.ppo),
                    },
                )
                self._wandb = wandb
                logger.info("W&B initialised")
            except ImportError:
                logger.warning("wandb not installed — skipping W&B logging")
                self._wandb = None
        else:
            self._wandb = None

    def _setup_checkpoint_dir(self) -> None:
        os.makedirs(self.cfg.logging.checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full PPO training loop until total_num_steps is reached."""
        cfg = self.cfg
        self._setup_checkpoint_dir()

        logger.info("=== Starting training ===")
        logger.info(f"  Total env steps : {cfg.ppo.total_num_steps:,}")
        logger.info(f"  Num envs        : {cfg.env.num_envs}")
        logger.info(f"  Steps per rollout: {cfg.ppo.num_steps}")
        logger.info(f"  Policy input dim : {self.policy.policy_input_dim}")

        # Reset all environments and get initial observations
        obs_dict = self.vec_env.reset_all()
        # obs_dict["rgb"]  : (N, 3, H, W) on CPU
        # obs_dict["goal"] : (N, 512)     on CPU

        # Initial hidden state (all zeros at training start)
        hidden = self.policy.get_initial_hidden(cfg.env.num_envs, self.device)
        masks  = torch.ones(cfg.env.num_envs, 1, device=self.device)
        prev_actions = torch.full(
            (cfg.env.num_envs,), cfg.env.num_actions, dtype=torch.long, device=self.device
        )   # num_actions index = "no previous action" padding

        steps_since_reset = torch.zeros(cfg.env.num_envs, device=self.device)
        while self._total_steps < cfg.ppo.total_num_steps:
            # ---- Phase 1: Collect rollout ----
            self.policy.eval()    # eval for rollout (no dropout)
            collect_start = time.time()

            for _ in range(cfg.ppo.num_steps):
                with torch.no_grad():
                    # 1a. Encode observation (DINOv2 — frozen). Single
                    # backbone forward pass returns BOTH CLS and patch
                    # tokens — do not call forward()/get_patch_embeddings()
                    # separately here, that would run the ViT twice.
                    rgb = obs_dict["rgb"].to(self.device)                    # (N, 3, H, W)
                    cls_embed, patch_embed = self.obs_encoder.get_all_embeddings(rgb)
                    # cls_embed:   (N, 768)
                    # patch_embed: (N, 256, 768)

                    # 1b. Encode goal (CLIP — frozen; cached)
                    goal_embed = self._get_goal_embeddings(obs_dict)  # (N, 512)

                    can_stop = steps_since_reset >= cfg.env.min_steps_before_stop

                    # 1c. PointGoal sensor
                    pointgoal = obs_dict["pointgoal"].to(self.device)  # (N, 3)

                    # 1d. Policy forward
                    dist, value, hidden_next = self.policy.act(
                        patch_embed, cls_embed, goal_embed, prev_actions,
                        hidden, masks, pointgoal=pointgoal, can_stop=can_stop,
                    )
                    actions   = dist.sample()                      # (N,)
                    log_probs = dist.log_prob(actions)             # (N,)

                # 1d. Step environments
                obs_dict, rewards, dones, infos = self.vec_env.step(actions)
                rewards = rewards.to(self.device)

                steps_since_reset = steps_since_reset + 1
                steps_since_reset = torch.where(
                    dones.to(self.device), torch.zeros_like(steps_since_reset), steps_since_reset
                )

                # --- episode-level bookkeeping ---
                self._running_reward += rewards
                for env_idx in range(cfg.env.num_envs):
                    if dones[env_idx]:
                        self._recent_rewards.append(self._running_reward[env_idx].item())
                        info = infos[env_idx]
                        if "success" in info:
                            self._recent_successes.append(float(info["success"]))
                        if "spl" in info:
                            self._recent_spls.append(info["spl"])
                        self._running_reward[env_idx] = 0.0

                # 1e. Build masks for NEXT step (0 if this step was terminal)
                new_masks = (~dones).float().unsqueeze(1).to(self.device)  # (N, 1)

                # 1f. Insert into buffer (raw patch/CLS tokens — the
                # trainable SpatialCompressionHead / CLS proj re-run on
                # these during evaluate_actions()).
                self.buffer.insert(
                    patch_embed = patch_embed,
                    cls_embed   = cls_embed,
                    goal_embed  = goal_embed,
                    action      = actions,
                    prev_action = prev_actions,
                    reward      = rewards,
                    mask        = new_masks.squeeze(1),
                    value       = value.squeeze(-1),
                    log_prob    = log_probs,
                    hidden      = hidden,
                    can_stop    = can_stop,
                    pointgoal   = pointgoal,
                )

                # 1g. Update recurrent state
                hidden       = hidden_next * new_masks.unsqueeze(0)
                prev_actions = actions
                masks        = new_masks
                self._total_steps += cfg.env.num_envs

            # ---- Per-action reward diagnostic (gate to avoid log spam) ----
            if self._num_updates % cfg.ppo.log_interval == 0:
                self._log_per_action_rewards()

            # ---- Phase 2: Compute returns ----
            with torch.no_grad():
                rgb = obs_dict["rgb"].to(self.device)
                cls_embed, patch_embed = self.obs_encoder.get_all_embeddings(rgb)
                goal_embed = self._get_goal_embeddings(obs_dict)
                pointgoal = obs_dict["pointgoal"].to(self.device)
                _, last_value, _ = self.policy.act(
                    patch_embed, cls_embed, goal_embed, prev_actions, hidden, masks,
                    pointgoal=pointgoal,
                )

                # ---- Branch-norm diagnostic (gated to avoid log spam) ----
                # Reuses the observation just encoded above — no extra
                # DINOv2/CLIP forward passes. Confirms directly whether the
                # spatial branch's scale is comparable to the goal branch's
                # (see SpatialCompressionHead's LayerNorm), rather than
                # inferring it indirectly from entropy/value_loss trends.
                if self._num_updates % cfg.ppo.log_interval == 0:
                    self._log_branch_norms(patch_embed, cls_embed, goal_embed)

            self.buffer.compute_returns(
                last_value  = last_value.squeeze(-1),
                gamma       = cfg.ppo.gamma,
                gae_lambda  = cfg.ppo.gae_lambda,
            )

            # ---- Phase 3: PPO update ----
            self.policy.train()
            update_metrics = self._ppo_update()
            self._num_updates += 1

            # ---- Phase 4: Carry state forward ----
            self.buffer.after_update(hidden, masks.squeeze(1))

            # ---- Logging ----
            if self._num_updates % cfg.ppo.log_interval == 0:
                self._log(update_metrics, collect_start)

            # ---- Checkpointing ----
            if self._total_steps % cfg.ppo.checkpoint_interval < (
                cfg.ppo.num_steps * cfg.env.num_envs
            ):
                self._save_checkpoint()

        # Final checkpoint
        self._save_checkpoint(final=True)
        self.vec_env.close()
        logger.info("=== Training complete ===")

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _ppo_update(self) -> dict:
        """
        Run num_epochs epochs of PPO updates over the current rollout buffer.

        Returns:
            Dict of mean losses for logging.
        """
        cfg = self.cfg
        metrics = {"policy_loss": 0., "value_loss": 0., "entropy": 0., "total_loss": 0.}
        num_batches = 0

        for _ in range(cfg.ppo.num_epochs):
            for batch in self.buffer.recurrent_mini_batches(cfg.ppo.num_mini_batches):

                # Re-evaluate actions under current policy. This re-runs the
                # trainable SpatialCompressionHead + CLS projection on the
                # RAW patch/CLS tokens stored in the buffer, so gradients
                # flow into them here (they were run under no_grad() during
                # rollout collection).
                log_probs, values, entropy = self.policy.evaluate_actions(
                    patch_embeds = batch.patch_embeds,
                    cls_embed    = batch.cls_embeds,
                    goal_embed   = batch.goal_embeds,
                    prev_actions = batch.prev_actions,
                    hidden       = batch.hidden_states,
                    masks        = batch.masks,
                    actions      = batch.actions,
                    pointgoal    = batch.pointgoals,
                    can_stop     = batch.can_stop,
                )

                # PPO clipped policy loss
                ratio = torch.exp(log_probs - batch.old_log_probs)
                surr1 = ratio * batch.advantages
                surr2 = torch.clamp(ratio, 1 - cfg.ppo.clip_param, 1 + cfg.ppo.clip_param) * batch.advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (MSE)
                value_loss = F.mse_loss(values.squeeze(-1), batch.returns)

                # Total loss
                loss = (
                    policy_loss
                    + cfg.ppo.value_loss_coef * value_loss
                    - cfg.ppo.entropy_coef    * entropy
                )

                # Gradient step
                self.optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.ppo.max_grad_norm)
                self.optimiser.step()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"]  += value_loss.item()
                metrics["entropy"]     += entropy.item()
                metrics["total_loss"]  += loss.item()
                num_batches += 1

        # Average over all batches
        for k in metrics:
            metrics[k] /= max(num_batches, 1)

        return metrics

    def _log_branch_norms(
        self,
        patch_embed: torch.Tensor,
        cls_embed: torch.Tensor,
        goal_embed: torch.Tensor,
    ) -> None:
        """Log mean L2 norm of each fusion branch (spatial / cls / goal) as
        they enter the concat, so scale parity between branches (e.g. after
        adding LayerNorm to SpatialCompressionHead) can be verified directly
        instead of inferred from downstream loss curves."""
        norms = self.policy.branch_norms(patch_embed, cls_embed, goal_embed)
        parts = "  ".join(f"{name}={val:.3f}" for name, val in norms.items())
        logger.info(f"    branch_norms — {parts}")

        if self._wandb is not None:
            self._wandb.log({
                f"branch_norms/{name}": val for name, val in norms.items()
            })

    def _log_per_action_rewards(self) -> None:
        """Log mean reward per action type from the just-collected rollout buffer.
        Call this AFTER the collection loop, BEFORE buffer.after_update() clears state."""
        action_names = {0: "MoveAhead", 1: "RotateLeft", 2: "RotateRight", 3: "Stop"}
        T = self.buffer.num_steps
        for a_id, a_name in action_names.items():
            mask = (self.buffer.actions[:T] == a_id)
            count = mask.sum().item()
            if count > 0:
                mean_r = self.buffer.rewards[:T][mask].mean().item()
                logger.info(f"    action={a_name:12s} count={count:6d} mean_reward={mean_r:+.4f}")

    # ------------------------------------------------------------------
    # Goal embedding helper
    # ------------------------------------------------------------------

    def _get_goal_embeddings(self, obs_dict: dict) -> torch.Tensor:
        """
        Fetches the pre-calculated 512-d goal embedding directly from the environment.
        """
        return obs_dict["goal"].to(self.device)

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def _save_checkpoint(self, final: bool = False) -> None:
        """Save policy weights, optimiser state, and training metadata."""
        suffix = "final" if final else f"{self._total_steps:012d}"
        path = os.path.join(
            self.cfg.logging.checkpoint_dir,
            f"checkpoint_{suffix}.pth"
        )
        torch.save({
            "policy_state_dict":    self.policy.state_dict(),
            "optimiser_state_dict": self.optimiser.state_dict(),
            "total_steps":          self._total_steps,
            "num_updates":          self._num_updates,
            "cfg":                  self.cfg,
        }, path)
        logger.info(f"Checkpoint saved: {path}")

    def _load_checkpoint(self, path: str) -> None:
        """Load a previously saved checkpoint."""
        logger.info(f"Resuming from checkpoint: {path}")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy_state_dict"])
        self.optimiser.load_state_dict(ckpt["optimiser_state_dict"])
        self._total_steps = ckpt.get("total_steps", 0)
        self._num_updates = ckpt.get("num_updates", 0)
        logger.info(f"Resumed at step {self._total_steps:,}")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, metrics: dict, collect_start: float) -> None:
        """Log training metrics to console and W&B."""
        elapsed    = time.time() - self._start_time
        steps_ps   = self._total_steps / max(elapsed, 1)
        fps        = (self.cfg.ppo.num_steps * self.cfg.env.num_envs) / (time.time() - collect_start)

        mean_reward  = sum(self._recent_rewards)   / len(self._recent_rewards)   if self._recent_rewards   else 0.0
        success_rate = sum(self._recent_successes) / len(self._recent_successes) if self._recent_successes else 0.0
        mean_spl     = sum(self._recent_spls)      / len(self._recent_spls)      if self._recent_spls      else 0.0

        logger.info(
            f"[{self._total_steps:>10,} steps | update {self._num_updates:>5}] "
            f"policy_loss={metrics['policy_loss']:.4f}  "
            f"value_loss={metrics['value_loss']:.4f}  "
            f"entropy={metrics['entropy']:.4f}  "
            f"reward={mean_reward:.3f}  "
            f"SR={success_rate:.3f}  "
            f"SPL={mean_spl:.3f}  "
            f"fps={fps:.0f}"
        )

        if self._wandb is not None:
            self._wandb.log({
                "train/policy_loss": metrics["policy_loss"],
                "train/value_loss":  metrics["value_loss"],
                "train/entropy":     metrics["entropy"],
                "train/total_loss":  metrics["total_loss"],
                "train/fps":         fps,
                "train/total_steps": self._total_steps,
                "train/mean_episode_reward": mean_reward,
                "train/success_rate": success_rate,
                "train/spl": mean_spl,
            })