"""
Critical design note for recurrent PPO:
    Vanilla PPO samples random mini-batches from the replay buffer.
    With a GRU, this is WRONG because the hidden state at step t depends
    on all previous steps in the same episode.

    Correct approach:
        1. Collect full rollouts of `num_steps` steps per env.
        2. During mini-batch creation, split each env's rollout into
           contiguous CHUNKS (e.g. 32-step sequences).
        3. Pass the STORED hidden state from the start of each chunk as
           the initial hidden state for that chunk's GRU forward pass.

    This ensures the GRU sees the correct temporal context during updates.

Critical design note for the spatial-branch encoder change:
    The SpatialCompressionHead (and the CLS projection) in gru_policy.py are
    TRAINABLE — unlike the frozen DINOv2/CLIP backbones. That means we can
    NOT store the pre-compressed 1568-dim spatial vector here and expect
    gradients to flow into the CNN during PPO updates; the compression has
    to be re-run (with grad) inside evaluate_actions() every epoch. So this
    buffer stores the RAW DINOv2 outputs instead:
        patch_embeds : (T, N, 256, 768)   <- was obs_embeds (T, N, 768)
        cls_embeds   : (T, N, 768)        <- new
    This is a large memory jump versus the old CLS-only buffer. At
    T=128, N=16: patch_embeds alone is
        128 * 16 * 256 * 768 * 4 bytes (fp32)  ≈ 1.6 GB
    Two knobs are provided to manage this:
        - `store_dtype`: defaults to torch.float16 for patch_embeds (halves
          the footprint; DINOv2 features tolerate fp16 storage fine since
          they're frozen features, not gradients-in-flight). Cast back to
          the compute dtype right before feeding the trainable CNN.
        - `storage_device`: if GPU headroom is tight, pass "cpu" here and
          keep everything else (returns/advantages/hidden states, which are
          small) on `device`. Mini-batches move their slice to `device`
          on the fly in _collect_chunk_batch(); only one chunk_len-sized
          slice touches the GPU transfer bus at a time, not the whole buffer.

Buffer layout:
    For each of the num_steps steps and num_envs environments, we store:
        patch_embeds    : (num_steps, num_envs, 256, 768)
        cls_embeds      : (num_steps, num_envs, 768)
        goal_embeds     : (num_steps, num_envs, 512)
        actions         : (num_steps, num_envs)
        prev_actions    : (num_steps, num_envs)
        rewards         : (num_steps, num_envs)
        masks           : (num_steps+1, num_envs)   ← +1 for next-step mask
        values          : (num_steps+1, num_envs)   ← +1 for bootstrap
        log_probs       : (num_steps, num_envs)
        hidden_states   : (num_steps+1, num_envs, hidden_size)

Usage:
    buf = RolloutBuffer(num_steps=128, num_envs=16, ...)
    buf.insert(patch_embed, cls_embed, goal_embed, action, prev_action,
               reward, mask, value, log_prob, hidden, can_stop)
    buf.compute_returns(last_value, gamma, gae_lambda)
    for batch in buf.recurrent_mini_batches(num_mini_batches=2, chunk_len=32):
        ...   # update policy
    buf.reset()
"""

from __future__ import annotations

import logging
from typing import Generator, NamedTuple, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mini-batch named tuple
# ---------------------------------------------------------------------------

class RecurrentBatch(NamedTuple):
    """A single mini-batch fed to policy.evaluate_actions()."""
    patch_embeds:   torch.Tensor    # (chunk_len * num_chunks, 256, 768)
    cls_embeds:     torch.Tensor    # (chunk_len * num_chunks, 768)
    goal_embeds:    torch.Tensor    # (chunk_len * num_chunks, 512)
    prev_actions:   torch.Tensor    # (chunk_len * num_chunks,)
    hidden_states:  torch.Tensor    # (1, num_chunks, hidden_size)
    masks:          torch.Tensor    # (chunk_len * num_chunks, 1)
    actions:        torch.Tensor    # (chunk_len * num_chunks,)
    old_log_probs:  torch.Tensor    # (chunk_len * num_chunks,)
    returns:        torch.Tensor    # (chunk_len * num_chunks,)
    advantages:     torch.Tensor    # (chunk_len * num_chunks,)
    can_stop:       torch.Tensor    # (chunk_len * num_chunks,)
    pointgoals:     torch.Tensor    # (chunk_len * num_chunks, 3)


# ---------------------------------------------------------------------------
# Buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """
    Stores rollout data for recurrent PPO updates.

    Args:
        num_steps:      Steps per rollout per environment.
        num_envs:       Number of parallel environments.
        patch_dim:      Per-patch DINOv2 token width (768).
        num_patches:    Number of DINOv2 patch tokens (256 for 16x16 grid).
        cls_dim:        DINOv2 CLS token width (768).
        goal_dim:       Dimension of goal embedding (512 for CLIP ViT-B/32).
        hidden_size:    GRU hidden state size.
        num_actions:    Size of discrete action space.
        device:         Torch device used for compute (returns/advantages/
                         hidden states always live here; large per-step
                         tensors optionally live on `storage_device` instead).
        storage_device: Where to allocate the large per-step tensors
                         (patch_embeds, cls_embeds, goal_embeds). Defaults
                         to `device`. Pass torch.device("cpu") if GPU VRAM
                         is tight — see module docstring for the size math.
        store_dtype:    Dtype for patch_embeds/cls_embeds/goal_embeds.
                         Defaults to torch.float16 to roughly halve the
                         dominant patch_embeds footprint; these are frozen
                         DINOv2/CLIP features, not values with gradients
                         attached, so fp16 storage is safe here.
    """

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        patch_dim: int,
        num_patches: int,
        cls_dim: int,
        goal_dim: int,
        hidden_size: int,
        num_actions: int,
        device: torch.device,
        pointgoal_dim: int = 3,
        storage_device: Optional[torch.device] = None,
        store_dtype: torch.dtype = torch.float16,
    ) -> None:
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.patch_dim = patch_dim
        self.num_patches = num_patches
        self.cls_dim = cls_dim
        self.goal_dim = goal_dim
        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.device = device
        self.pointgoal_dim = pointgoal_dim
        self.storage_device = storage_device if storage_device is not None else device
        self.store_dtype = store_dtype

        self._step = 0   # current insertion pointer

        self._allocate()

    # ------------------------------------------------------------------
    # Buffer allocation
    # ------------------------------------------------------------------

    def _allocate(self) -> None:
        """Allocate all buffer tensors."""
        T, N = self.num_steps, self.num_envs
        P, D_patch, D_cls, D_goal, D_h = (
            self.num_patches, self.patch_dim, self.cls_dim,
            self.goal_dim, self.hidden_size,
        )
        sd, sdt = self.storage_device, self.store_dtype

        # Raw frozen-encoder outputs (large — see module docstring on
        # store_dtype/storage_device for memory management).
        self.patch_embeds = torch.zeros(T, N, P, D_patch, device=sd, dtype=sdt)
        self.cls_embeds   = torch.zeros(T, N, D_cls,       device=sd, dtype=sdt)
        self.goal_embeds  = torch.zeros(T, N, D_goal,       device=sd, dtype=sdt)

        # PointGoal sensor (small — always on compute device, fp32)
        self.pointgoals = torch.zeros(T, N, self.pointgoal_dim, device=self.device)

        # Actions
        self.actions      = torch.zeros(T, N, dtype=torch.long, device=self.device)
        self.prev_actions = torch.zeros(T, N, dtype=torch.long, device=self.device)

        self.can_stop = torch.ones(T, N, dtype=torch.bool, device=self.device)

        # Rewards and masks
        # masks[t] = 0 if step t starts a new episode, else 1
        self.rewards = torch.zeros(T,   N, device=self.device)
        self.masks   = torch.ones(T+1, N, device=self.device)

        # Value function and log probabilities
        self.values    = torch.zeros(T+1, N, device=self.device)
        self.log_probs = torch.zeros(T,   N, device=self.device)

        # GRU hidden states (stored at EVERY step for recurrent mini-batches)
        # Shape: (T+1, 1, N, D_h) — 1 is GRU num_layers
        self.hidden_states = torch.zeros(T+1, 1, N, D_h, device=self.device)

        # Computed after rollout ends
        self.returns    = torch.zeros(T, N, device=self.device)
        self.advantages = torch.zeros(T, N, device=self.device)

    # ------------------------------------------------------------------
    # Insertion
    # ------------------------------------------------------------------

    def insert(
        self,
        patch_embed:  torch.Tensor,
        cls_embed:    torch.Tensor,
        goal_embed:   torch.Tensor,
        action:       torch.Tensor,
        prev_action:  torch.Tensor,
        reward:       torch.Tensor,
        mask:         torch.Tensor,
        value:        torch.Tensor,
        log_prob:     torch.Tensor,
        hidden:       torch.Tensor,
        can_stop:     torch.Tensor,
        pointgoal:    torch.Tensor = None,
    ) -> None:
        """
        Insert one time-step of data from all environments.

        Args:
            patch_embed: (N, num_patches, patch_dim) — raw DINOv2 patch tokens.
            cls_embed:   (N, cls_dim)                — raw DINOv2 CLS token.
            goal_embed:  (N, goal_dim)                — CLIP goal embedding.
            action:      (N,) long
            prev_action: (N,) long
            reward:      (N,)
            mask:        (N,) — 0.0 at episode start, 1.0 otherwise
            value:       (N,) or (N,1)
            log_prob:    (N,)
            hidden:      (1, N, hidden_size) — hidden state BEFORE this step
            can_stop:    (N,) bool
            pointgoal:   (N, 3) — [geodesic_dist, cos(angle), sin(angle)]
        """
        t = self._step

        self.patch_embeds[t] = patch_embed.detach().to(
            device=self.storage_device, dtype=self.store_dtype
        )
        self.cls_embeds[t] = cls_embed.detach().to(
            device=self.storage_device, dtype=self.store_dtype
        )
        self.goal_embeds[t] = goal_embed.detach().to(
            device=self.storage_device, dtype=self.store_dtype
        )
        self.actions[t]          = action
        self.prev_actions[t]     = prev_action
        self.rewards[t]          = reward
        self.masks[t+1]          = mask
        self.values[t]           = value.view(self.num_envs).detach()
        self.log_probs[t]        = log_prob.detach()
        self.hidden_states[t+1]  = hidden.detach()   # hidden AFTER this step
        self.can_stop[t]         = can_stop
        if pointgoal is not None:
            self.pointgoals[t] = pointgoal.detach().to(device=self.device)

        self._step += 1

    def after_update(self, last_hidden: torch.Tensor, last_mask: torch.Tensor) -> None:
        """
        Call this AFTER compute_returns() + mini-batch updates.

        Carries the last hidden state and mask forward to the next rollout
        so the GRU context is not lost between rollout segments.

        Args:
            last_hidden: (1, N, hidden_size) — final hidden state of rollout.
            last_mask:   (N,) — mask at the last step.
        """
        self.hidden_states[0] = last_hidden.detach()
        self.masks[0]         = last_mask.detach()
        self._step = 0

    # ------------------------------------------------------------------
    # Return / advantage computation
    # ------------------------------------------------------------------

    def compute_returns(
        self,
        last_value: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """
        Compute GAE advantages and discounted returns.

        Implements Generalised Advantage Estimation (Schulman et al., 2016):
            δ_t  = r_t + γ * V(s_{t+1}) * mask_{t+1} - V(s_t)
            A_t  = Σ_{l=0}^{∞} (γλ)^l * δ_{t+l}

        Args:
            last_value: (N,) — critic estimate at the final step (bootstrap).
            gamma:      Discount factor (default 0.99).
            gae_lambda: GAE lambda (default 0.95).
        """
        self.values[self.num_steps] = last_value.view(self.num_envs).detach()

        gae = torch.zeros(self.num_envs, device=self.device)

        for t in reversed(range(self.num_steps)):
            # TD error
            delta = (
                self.rewards[t]
                + gamma * self.values[t+1] * self.masks[t+1]
                - self.values[t]
            )
            # GAE (recursive formula, with episode reset via mask)
            gae = delta + gamma * gae_lambda * self.masks[t+1] * gae

            self.advantages[t] = gae
            self.returns[t]    = gae + self.values[t]

        # Normalise advantages for training stability
        adv_flat = self.advantages[:self.num_steps].reshape(-1)
        self.advantages[:self.num_steps] = (
            (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        ).reshape(self.num_steps, self.num_envs)

    # ------------------------------------------------------------------
    # Mini-batch generator
    # ------------------------------------------------------------------

    def recurrent_mini_batches(
        self,
        num_mini_batches: int,
        chunk_len: int = 32,
    ) -> Generator[RecurrentBatch, None, None]:
        """
        Yield mini-batches suitable for recurrent PPO updates.

        Each mini-batch consists of multiple fixed-length chunks drawn
        from the rollout. The hidden state at the start of each chunk is
        the stored value from the buffer (so the GRU sees correct context).

        Args:
            num_mini_batches: How many mini-batches to split rollout into.
            chunk_len:        Length of each contiguous sequence chunk.

        Yields:
            RecurrentBatch namedtuples, with patch/cls/goal embeds cast
            back to float32 and moved to `self.device` (a no-op if
            storage_device == device already).
        """
        T, N = self.num_steps, self.num_envs

        # Split each env's T-step rollout into (T // chunk_len) chunks
        num_chunks_per_env = T // chunk_len
        total_chunks = num_chunks_per_env * N

        # Shuffle chunk ordering for stochasticity
        chunk_indices = torch.randperm(total_chunks, device=self.device)

        chunks_per_batch = total_chunks // num_mini_batches

        for start in range(0, total_chunks, chunks_per_batch):
            batch_idx = chunk_indices[start : start + chunks_per_batch]
            yield self._collect_chunk_batch(batch_idx, chunk_len, num_chunks_per_env)

    def _collect_chunk_batch(
        self,
        chunk_indices: torch.Tensor,
        chunk_len: int,
        num_chunks_per_env: int,
    ) -> RecurrentBatch:
        """
        Collect data for a set of chunk indices into a RecurrentBatch.

        Each chunk index maps to a (env_id, chunk_start_step) pair.
        """
        num_chunks = len(chunk_indices)

        # env_ids/start_steps are used to index storage_device tensors, so
        # keep them on storage_device to avoid a device-mismatch index error
        # when storage_device != device (e.g. patch_embeds kept on CPU).
        chunk_indices_storage = chunk_indices.to(self.storage_device)
        env_ids_storage   = chunk_indices_storage // num_chunks_per_env
        chunk_ids_storage = chunk_indices_storage  % num_chunks_per_env
        start_steps_storage = chunk_ids_storage * chunk_len

        # Same indices, but on compute device, for the small tensors that
        # already live there (actions, masks, returns, hidden states, ...).
        env_ids     = chunk_indices // num_chunks_per_env
        chunk_ids   = chunk_indices  % num_chunks_per_env
        start_steps = chunk_ids * chunk_len

        patch_list, cls_list, goal_list, prev_act_list, mask_list = [], [], [], [], []
        act_list, lp_list, ret_list, adv_list = [], [], [], []
        can_stop_list = []
        pointgoal_list = []

        for local_t in range(chunk_len):
            t_idx_storage = (start_steps_storage + local_t).clamp(max=self.num_steps - 1)
            t_idx = (start_steps + local_t).clamp(max=self.num_steps - 1)

            patch_list.append(self.patch_embeds[t_idx_storage, env_ids_storage])
            cls_list.append(self.cls_embeds[t_idx_storage, env_ids_storage])
            goal_list.append(self.goal_embeds[t_idx_storage, env_ids_storage])

            prev_act_list.append(self.prev_actions[t_idx, env_ids])
            mask_list.append(self.masks[t_idx, env_ids].unsqueeze(-1))
            act_list.append(self.actions[t_idx, env_ids])
            lp_list.append(self.log_probs[t_idx, env_ids])
            ret_list.append(self.returns[t_idx, env_ids])
            adv_list.append(self.advantages[t_idx, env_ids])
            can_stop_list.append(self.can_stop[t_idx, env_ids])
            pointgoal_list.append(self.pointgoals[t_idx, env_ids])

        # Stack along time dimension: (chunk_len, num_chunks, ...) → flatten time
        def _flat(lst):
            return torch.stack(lst, dim=0).reshape(-1, *lst[0].shape[1:])

        # Large tensors: cast to float32 and move to compute device here —
        # only this chunk-sized slice crosses the storage->compute boundary,
        # not the whole buffer.
        patch_batch = _flat(patch_list).to(device=self.device, dtype=torch.float32)
        cls_batch   = _flat(cls_list).to(device=self.device, dtype=torch.float32)
        goal_batch  = _flat(goal_list).to(device=self.device, dtype=torch.float32)

        # Hidden state at the start of each chunk: shape (1, num_chunks, hidden_size)
        # hidden_states[t] is the hidden BEFORE step t
        h_idx = start_steps
        hidden_batch = self.hidden_states[
            h_idx, 0, env_ids, :
        ].unsqueeze(0)   # (1, num_chunks, D_h)

        return RecurrentBatch(
            patch_embeds  = patch_batch,
            cls_embeds    = cls_batch,
            goal_embeds   = goal_batch,
            prev_actions  = _flat(prev_act_list),
            hidden_states = hidden_batch,
            masks         = _flat(mask_list),
            actions       = _flat(act_list),
            old_log_probs = _flat(lp_list),
            returns       = _flat(ret_list),
            advantages    = _flat(adv_list),
            can_stop      = _flat(can_stop_list),
            pointgoals    = _flat(pointgoal_list),
        )

    # ------------------------------------------------------------------
    # State inspection
    # ------------------------------------------------------------------

    def is_full(self) -> bool:
        return self._step >= self.num_steps

    def reset(self) -> None:
        self._step = 0

    @property
    def current_hidden(self) -> torch.Tensor:
        """The hidden state at the very start of this rollout."""
        return self.hidden_states[0]

    @property
    def current_mask(self) -> torch.Tensor:
        """The mask at the very start of this rollout."""
        return self.masks[0]