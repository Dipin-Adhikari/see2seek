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
import math
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
        self.obs_encoder_type = getattr(cfg.encoder, "obs_encoder_type", "dino")

        # ---- Logging ----
        self._setup_logging()

        # ---- Encoders (frozen, shared across all updates) ----
        logger.info(f"Building encoders (obs_encoder={self.obs_encoder_type}) ...")
        if self.obs_encoder_type == "dino":
            self.obs_encoder = DINOv2Encoder(
                device=cfg.device,
                normalize=cfg.encoder.obs_normalize,
            )
            buf_patch_dim = self.obs_encoder.patch_dim
            buf_num_patches = self.obs_encoder.num_patches
            buf_cls_dim = self.obs_encoder.embed_dim
        else:
            self.obs_encoder = None  # CLIP handles obs encoding
            buf_patch_dim = 0
            buf_num_patches = 0
            buf_cls_dim = cfg.encoder.goal_embed_dim  # 512

        self.goal_encoder = CLIPGoalEncoder(
            device=cfg.device,
            normalize=cfg.encoder.goal_normalize,
        )

        # ---- Policy (trainable) ----
        logger.info("Building GRU Actor-Critic policy ...")
        self.policy = build_policy(cfg, cfg.device)

        # ---- Optimiser ----
        self.optimiser = torch.optim.Adam(
            self.policy.parameters(),
            lr=cfg.ppo.lr,
            eps=cfg.ppo.eps,
        )
        total_updates = cfg.ppo.total_num_steps // (cfg.ppo.num_steps * cfg.env.num_envs)
        self.lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimiser, start_factor=1.0, end_factor=0.1, total_iters=total_updates
        )

        # ---- Environments ----
        logger.info(f"Launching {cfg.env.num_envs} parallel environments ...")
        self.vec_env = make_vec_envs(cfg)

        # ---- Rollout buffer ----
        self.buffer = RolloutBuffer(
            num_steps=cfg.ppo.num_steps,
            num_envs=cfg.env.num_envs,
            patch_dim=buf_patch_dim,
            num_patches=buf_num_patches,
            cls_dim=buf_cls_dim,
            goal_dim=cfg.encoder.goal_embed_dim,
            hidden_size=cfg.policy.hidden_size,
            num_actions=cfg.env.num_actions,
            device=self.device,
            pointgoal_dim=cfg.encoder.pointgoal_input_dim,
            num_recurrent_layers=getattr(cfg.policy, "num_recurrent_layers", 2),
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

        # ---- Per-GPS-condition metrics (split by pointgoal on/off) ----
        self._recent_rewards_gps_on    = deque(maxlen=100)
        self._recent_successes_gps_on  = deque(maxlen=100)
        self._recent_spls_gps_on       = deque(maxlen=100)
        self._recent_rewards_gps_off   = deque(maxlen=100)
        self._recent_successes_gps_off = deque(maxlen=100)
        self._recent_spls_gps_off      = deque(maxlen=100)

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

        # Episodic memory buffers (per-env, persisted across rollouts)
        memory_size = getattr(cfg.encoder, "memory_size", 64)
        cls_dim_for_mem = (
            cfg.encoder.dino_cls_dim if self.obs_encoder_type == "dino"
            else cfg.encoder.goal_embed_dim
        )
        memory_buffer = torch.zeros(
            cfg.env.num_envs, memory_size, cls_dim_for_mem, device=self.device
        )
        memory_pose_buffer = torch.zeros(
            cfg.env.num_envs, memory_size, 4, device=self.device
        )
        memory_mask = torch.zeros(
            cfg.env.num_envs, memory_size, device=self.device, dtype=torch.bool
        )

        # Dead-reckoned pose: [x, y, cos_theta, sin_theta] per env
        # Accumulated from discrete actions (MoveAhead=0.25m, Rotate=30deg)
        agent_poses = torch.zeros(cfg.env.num_envs, 4, device=self.device)
        agent_poses[:, 2] = 1.0  # cos(0) = 1
        # agent_poses[:, 3] = 0.0  # sin(0) = 0 (already zero)

        # Per-episode pointgoal dropout: decide once per episode whether GPS is available
        pg_episode_mask = (torch.rand(cfg.env.num_envs, 1, device=self.device)
                           > cfg.encoder.pointgoal_dropout).float()

        steps_since_reset = torch.zeros(cfg.env.num_envs, device=self.device)
        while self._total_steps < cfg.ppo.total_num_steps:
            # ---- Curriculum: update effective max_steps ----
            if cfg.env.curriculum_enabled:
                progress = min(self._total_steps / cfg.env.curriculum_ramp_steps, 1.0)
                curr_max_steps = int(
                    cfg.env.curriculum_start_max_steps
                    + progress * (cfg.env.curriculum_end_max_steps - cfg.env.curriculum_start_max_steps)
                )
                self.vec_env.set_max_steps(curr_max_steps)

            # ---- Exploration bonus decay ----
            if cfg.env.exploration_decay_steps > 0 and cfg.env.exploration_bonus > 0:
                decay_progress = min(self._total_steps / cfg.env.exploration_decay_steps, 1.0)
                curr_bonus = cfg.env.exploration_bonus * (1.0 - decay_progress)
                self.vec_env.set_exploration_bonus(curr_bonus)

            # ---- Phase 1: Collect rollout ----
            self.policy.eval()    # eval for rollout (no dropout)
            collect_start = time.time()

            for _ in range(cfg.ppo.num_steps):
                with torch.no_grad():
                    rgb = obs_dict["rgb"].to(self.device)  # (N, 3, H, W)

                    # 1a. Encode observation (frozen backbone)
                    if self.obs_encoder_type == "dino":
                        cls_embed, patch_embed = self.obs_encoder.get_all_embeddings(rgb)
                    else:
                        cls_embed = self.goal_encoder.get_obs_embedding(rgb)  # (N, 512)
                        patch_embed = None

                    # 1b. Encode goal (CLIP — frozen; cached)
                    goal_embed = self._get_goal_embeddings(obs_dict)  # (N, 512)

                    can_stop = steps_since_reset >= cfg.env.min_steps_before_stop

                    # 1c. PointGoal sensor (per-episode dropout for zero-shot transfer)
                    pointgoal = obs_dict["pointgoal"].to(self.device)  # (N, 3)
                    pointgoal = pointgoal * pg_episode_mask
                    pg_dropped = (pg_episode_mask.squeeze(1) == 0)

                    # 1d. Policy forward (with position-augmented episodic memory)
                    dist, value, hidden_next, memory_buffer, memory_pose_buffer, memory_mask = self.policy.act(
                        patch_embed, cls_embed, goal_embed, prev_actions,
                        hidden, masks, pointgoal=pointgoal, can_stop=can_stop,
                        memory_buffer=memory_buffer, memory_pose_buffer=memory_pose_buffer,
                        memory_mask=memory_mask, poses=agent_poses,
                    )
                    actions   = dist.sample()                      # (N,)
                    log_probs = dist.log_prob(actions)             # (N,)

                # Bug 2 fix: snapshot pre-step pose (what the policy actually saw)
                pose_for_buffer = agent_poses.clone()

                # 1d. Step environments
                obs_dict, rewards, dones, infos = self.vec_env.step(actions)
                rewards = rewards.to(self.device)

                # Dead-reckon pose update (vectorized)
                # actions: 0=MoveAhead, 1=RotateLeft, 2=RotateRight, 3=Stop
                # agent_poses: (N, 4) = [x, y, cos_theta, sin_theta]
                dones_dev = dones.to(self.device)
                acts_dev = actions

                # Reset pose for done envs
                agent_poses[dones_dev] = 0.0
                agent_poses[dones_dev, 2] = 1.0

                # Bug 0 fix: only update pose when MoveAhead actually succeeded
                move_success = torch.tensor(
                    [infos[i].get("move_success", True) for i in range(cfg.env.num_envs)],
                    dtype=torch.bool, device=self.device,
                )
                is_move = (acts_dev == 0) & ~dones_dev & move_success
                if is_move.any():
                    agent_poses[is_move, 0] += cfg.env.move_magnitude * agent_poses[is_move, 3]
                    agent_poses[is_move, 1] += cfg.env.move_magnitude * agent_poses[is_move, 2]

                # Rotation: apply 2D rotation matrix to (cos_theta, sin_theta)
                cos_r = math.cos(math.radians(cfg.env.rotate_degrees))
                sin_r = math.sin(math.radians(cfg.env.rotate_degrees))

                is_left = (acts_dev == 1) & ~dones_dev
                if is_left.any():
                    c = agent_poses[is_left, 2].clone()
                    s = agent_poses[is_left, 3].clone()
                    agent_poses[is_left, 2] = c * cos_r - s * sin_r
                    agent_poses[is_left, 3] = s * cos_r + c * sin_r

                is_right = (acts_dev == 2) & ~dones_dev
                if is_right.any():
                    c = agent_poses[is_right, 2].clone()
                    s = agent_poses[is_right, 3].clone()
                    agent_poses[is_right, 2] = c * cos_r + s * sin_r
                    agent_poses[is_right, 3] = s * cos_r - c * sin_r

                steps_since_reset = steps_since_reset + 1
                steps_since_reset = torch.where(
                    dones.to(self.device), torch.zeros_like(steps_since_reset), steps_since_reset
                )

                # --- episode-level bookkeeping ---
                self._running_reward += rewards
                for env_idx in range(cfg.env.num_envs):
                    if dones[env_idx]:
                        ep_reward = self._running_reward[env_idx].item()
                        self._recent_rewards.append(ep_reward)
                        info = infos[env_idx]
                        ep_success = float(info["success"]) if "success" in info else None
                        ep_spl = info.get("spl")
                        if ep_success is not None:
                            self._recent_successes.append(ep_success)
                        if ep_spl is not None:
                            self._recent_spls.append(ep_spl)

                        # Split metrics by GPS condition (mask was set at episode start)
                        gps_on = pg_episode_mask[env_idx].item() > 0.5
                        if gps_on:
                            self._recent_rewards_gps_on.append(ep_reward)
                            if ep_success is not None:
                                self._recent_successes_gps_on.append(ep_success)
                            if ep_spl is not None:
                                self._recent_spls_gps_on.append(ep_spl)
                        else:
                            self._recent_rewards_gps_off.append(ep_reward)
                            if ep_success is not None:
                                self._recent_successes_gps_off.append(ep_success)
                            if ep_spl is not None:
                                self._recent_spls_gps_off.append(ep_spl)

                        self._running_reward[env_idx] = 0.0

                # Re-roll per-episode pointgoal dropout for newly started episodes
                if dones_dev.any():
                    new_rolls = (torch.rand(dones_dev.sum().item(), 1, device=self.device)
                                 > cfg.encoder.pointgoal_dropout).float()
                    pg_episode_mask[dones_dev] = new_rolls

                # 1e. Build masks for NEXT step (0 if this step was terminal)
                new_masks = (~dones).float().unsqueeze(1).to(self.device)  # (N, 1)

                # 1f. Insert into buffer
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
                    pointgoal_dropped = pg_dropped,
                    pose        = pose_for_buffer,
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
                if self.obs_encoder_type == "dino":
                    cls_embed, patch_embed = self.obs_encoder.get_all_embeddings(rgb)
                else:
                    cls_embed = self.goal_encoder.get_obs_embedding(rgb)
                    patch_embed = None
                goal_embed = self._get_goal_embeddings(obs_dict)
                pointgoal = obs_dict["pointgoal"].to(self.device)
                _, last_value, _, memory_buffer, memory_pose_buffer, memory_mask = self.policy.act(
                    patch_embed, cls_embed, goal_embed, prev_actions, hidden, masks,
                    pointgoal=pointgoal,
                    memory_buffer=memory_buffer, memory_pose_buffer=memory_pose_buffer,
                    memory_mask=memory_mask, poses=agent_poses,
                )

                if self._num_updates % cfg.ppo.log_interval == 0:
                    self._log_branch_norms(
                        patch_embed, cls_embed, goal_embed,
                        pointgoal=pointgoal, poses=agent_poses,
                        memory_buffer=memory_buffer,
                        memory_pose_buffer=memory_pose_buffer,
                        memory_mask=memory_mask,
                    )

            self.buffer.compute_returns(
                last_value  = last_value.squeeze(-1),
                gamma       = cfg.ppo.gamma,
                gae_lambda  = cfg.ppo.gae_lambda,
            )

            # ---- Phase 3: PPO update ----
            self.policy.train()
            update_metrics = self._ppo_update()
            self.lr_scheduler.step()
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
                log_probs, values, entropy_per_sample = self.policy.evaluate_actions(
                    patch_embeds = batch.patch_embeds,
                    cls_embed    = batch.cls_embeds,
                    goal_embed   = batch.goal_embeds,
                    prev_actions = batch.prev_actions,
                    hidden       = batch.hidden_states,
                    masks        = batch.masks,
                    actions      = batch.actions,
                    pointgoal    = batch.pointgoals,
                    can_stop     = batch.can_stop,
                    poses        = batch.poses,
                )

                # PPO clipped policy loss
                ratio = torch.exp(log_probs - batch.old_log_probs)
                surr1 = ratio * batch.advantages
                surr2 = torch.clamp(ratio, 1 - cfg.ppo.clip_param, 1 + cfg.ppo.clip_param) * batch.advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (MSE)
                value_loss = F.mse_loss(values.squeeze(-1), batch.returns)

                # Per-sample entropy weighting: higher entropy bonus for GPS-off steps
                entropy_coefs = torch.where(
                    batch.pointgoal_dropped,
                    cfg.ppo.entropy_coef_gps_off,
                    cfg.ppo.entropy_coef_gps_on,
                )
                weighted_entropy = (entropy_coefs * entropy_per_sample).mean()
                entropy = entropy_per_sample.mean()

                # Total loss
                loss = (
                    policy_loss
                    + cfg.ppo.value_loss_coef * value_loss
                    - weighted_entropy
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
        pointgoal=None,
        poses=None,
        memory_buffer=None,
        memory_pose_buffer=None,
        memory_mask=None,
    ) -> None:
        """Log mean L2 norm of each fusion branch (spatial / cls / goal /
        pointgoal / egopose / memory) as they enter the concat, so scale
        parity between branches can be verified directly."""
        norms = self.policy.branch_norms(
            patch_embed, cls_embed, goal_embed,
            pointgoal=pointgoal, poses=poses,
            memory_buffer=memory_buffer,
            memory_pose_buffer=memory_pose_buffer,
            memory_mask=memory_mask,
        )
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
            "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
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
        if "lr_scheduler_state_dict" in ckpt:
            self.lr_scheduler.load_state_dict(ckpt["lr_scheduler_state_dict"])
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

        # GPS-ON metrics
        sr_gps_on  = sum(self._recent_successes_gps_on)  / len(self._recent_successes_gps_on)  if self._recent_successes_gps_on  else 0.0
        spl_gps_on = sum(self._recent_spls_gps_on)       / len(self._recent_spls_gps_on)       if self._recent_spls_gps_on       else 0.0
        rw_gps_on  = sum(self._recent_rewards_gps_on)    / len(self._recent_rewards_gps_on)    if self._recent_rewards_gps_on    else 0.0
        n_gps_on   = len(self._recent_successes_gps_on)

        # GPS-OFF metrics
        sr_gps_off  = sum(self._recent_successes_gps_off) / len(self._recent_successes_gps_off) if self._recent_successes_gps_off else 0.0
        spl_gps_off = sum(self._recent_spls_gps_off)      / len(self._recent_spls_gps_off)      if self._recent_spls_gps_off      else 0.0
        rw_gps_off  = sum(self._recent_rewards_gps_off)   / len(self._recent_rewards_gps_off)   if self._recent_rewards_gps_off   else 0.0
        n_gps_off   = len(self._recent_successes_gps_off)

        # Curriculum max_steps (if enabled)
        if self.cfg.env.curriculum_enabled:
            progress = min(self._total_steps / self.cfg.env.curriculum_ramp_steps, 1.0)
            curr_max_steps = int(
                self.cfg.env.curriculum_start_max_steps
                + progress * (self.cfg.env.curriculum_end_max_steps - self.cfg.env.curriculum_start_max_steps)
            )
        else:
            curr_max_steps = self.cfg.env.max_steps

        logger.info(
            f"[{self._total_steps:>10,} steps | update {self._num_updates:>5}] "
            f"policy_loss={metrics['policy_loss']:.4f}  "
            f"value_loss={metrics['value_loss']:.4f}  "
            f"entropy={metrics['entropy']:.4f}  "
            f"reward={mean_reward:.3f}  "
            f"SR={success_rate:.3f}  "
            f"SPL={mean_spl:.3f}  "
            f"fps={fps:.0f}  "
            f"max_steps={curr_max_steps}"
        )
        logger.info(
            f"    GPS-ON  (n={n_gps_on:>3}): SR={sr_gps_on:.3f}  SPL={spl_gps_on:.3f}  reward={rw_gps_on:.3f}  |  "
            f"GPS-OFF (n={n_gps_off:>3}): SR={sr_gps_off:.3f}  SPL={spl_gps_off:.3f}  reward={rw_gps_off:.3f}"
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
                "train/sr_gps_on":      sr_gps_on,
                "train/spl_gps_on":     spl_gps_on,
                "train/reward_gps_on":  rw_gps_on,
                "train/sr_gps_off":     sr_gps_off,
                "train/spl_gps_off":    spl_gps_off,
                "train/reward_gps_off": rw_gps_off,
            })