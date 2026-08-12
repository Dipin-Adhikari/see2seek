"""
gru_policy.py — GRU-based Actor-Critic policy for embodied navigation.

Architecture (mirrors ZSON's policy, but with our encoder dims):

    Input per step:
        obs_embed        : (B, 768)   ← DINOv2 CLS token
        goal_embed       : (B, 512)   ← CLIP   CLS token (pre-cached)
        prev_action_embed: (B, 32)    ← learned embedding of last discrete action

    Concatenation: (B, 1312)

    GRU (single layer, hidden_size=512):
        input : (seq_len, B, 1312)
        output: (seq_len, B, 512)

    Actor head  → logits  (B, num_actions)
    Critic head → value   (B, 1)

Design notes:
    - Hidden state (h_t) is the agent's episodic memory and MUST be
      correctly propagated across steps and reset at episode boundaries.
    - The policy is in train() mode during rollout collection (so dropout
      layers would be active, though we have none by default).
    - Recurrent PPO requires chunked mini-batch updates; the RolloutBuffer
      handles storing and replaying hidden states correctly.

Usage:
    policy = GRUActorCritic(
        policy_input_dim=1312,
        hidden_size=512,
        num_actions=4,
    )
    # Single step:
    dist, value, h_next = policy.act(obs_embed, goal_embed, prev_action, h_prev, masks)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper layers
# ---------------------------------------------------------------------------

class LinearNormAct(nn.Sequential):
    """Linear → LayerNorm → ELU block used in Actor/Critic heads."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.ELU(inplace=True),
        )


# ---------------------------------------------------------------------------
# Main policy
# ---------------------------------------------------------------------------

class GRUActorCritic(nn.Module):
    """
    Single-layer GRU Actor-Critic policy for discrete-action navigation.
 
    frozen encoders (DINOv2 + CLIP). The GRU provides temporal context.

    Args:
        policy_input_dim: Dimension of the concatenated input vector.
                          Default 1312 = 768 + 512 + 32.
        hidden_size:      GRU hidden state size. Default 512.
        num_actions:      Size of the discrete action space. Default 4.
        num_actions_embed:Dimension of the learned prev-action embedding. Default 32.
        actor_hidden_dim: Intermediate linear dim in actor head. Default 256.
        critic_hidden_dim:Intermediate linear dim in critic head. Default 256.
    """

    def __init__(
        self,
        policy_input_dim: int = 1312,
        hidden_size: int = 512,
        num_actions: int = 4,
        num_action_embed: int = 32,
        actor_hidden_dim: int = 256,
        critic_hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_actions = num_actions

        # ---- Previous-action embedding ----
        # Embedding for the no-op / padding action (action index num_actions)
        # is also allocated so index (num_actions) can be used for step 0.
        self.prev_action_embed = nn.Embedding(
            num_embeddings=num_actions + 1,    # +1 for start-of-episode padding
            embedding_dim=num_action_embed,
        )

        # ---- GRU ----
        self.gru = nn.GRU(
            input_size=policy_input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=False,   # (seq_len, batch, features)
        )

        # ---- Actor head ----
        self.actor = nn.Sequential(
            LinearNormAct(hidden_size, actor_hidden_dim),
            nn.Linear(actor_hidden_dim, num_actions),
        )

        # ---- Critic head ----
        self.critic = nn.Sequential(
            LinearNormAct(hidden_size, critic_hidden_dim),
            nn.Linear(critic_hidden_dim, 1),
        )

        # Weight initialisation
        self._init_weights()

        logger.info(
            f"GRUActorCritic — input={policy_input_dim}, "
            f"hidden={hidden_size}, actions={num_actions}"
        )

    # ------------------------------------------------------------------
    # Weight initialisation (orthogonal is standard for RL)
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        for module in [self.actor, self.critic]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=0.01)
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Core forward (used internally)
    # ------------------------------------------------------------------

    def forward(
        self,
        obs_embed: torch.Tensor,
        goal_embed: torch.Tensor,
        prev_actions: torch.Tensor,
        hidden: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass through the GRU policy.

        Args:
            obs_embed:    (N, 768)  — DINOv2 observation embedding.
            goal_embed:   (N, 512)  — CLIP goal embedding.
            prev_actions: (N,)  long — index of the action taken last step.
                          Use index `num_actions` for the very first step.
            hidden:       (1, N, hidden_size) — GRU hidden state.
            masks:        (N, 1) float — 0.0 at episode boundaries (resets h),
                          1.0 otherwise. This gates the hidden state correctly.

        Returns:
            logits:     (N, num_actions) — raw action scores (pre-softmax).
            value:      (N, 1)          — state-value estimate.
            hidden_out: (1, N, hidden_size) — updated GRU hidden state.

        Note on masks:
            masks zeros out h_t at episode boundaries so the GRU starts fresh
            for new episodes without requiring an explicit reset() call. This is
            critical for correctness in vectorised environments.
        """
        total_B    = obs_embed.shape[0]      # N during rollout, chunk_len*num_chunks during update
        num_chunks = hidden.shape[1]         # hidden: (1, num_chunks, hidden_size)
        chunk_len  = total_B // num_chunks   # 1 during rollout, >1 during PPO update

        # 1. Embed previous action
        prev_act_embed = self.prev_action_embed(prev_actions)   # (total_B, 32)

        # 2. Concatenate all inputs: (total_B, 1312)
        x = torch.cat([obs_embed, goal_embed, prev_act_embed], dim=-1)

        # 3. Reshape to time-major: (chunk_len, num_chunks, features)
        x = x.view(chunk_len, num_chunks, -1)
        masks_seq = masks.view(chunk_len, num_chunks, 1)

        # 4. Step through the chunk manually, resetting hidden state at
        #    episode boundaries (mask == 0) at each timestep.
        h = hidden
        outputs = []
        for t in range(chunk_len):
            h = h * masks_seq[t].unsqueeze(0)              # (1, num_chunks, hidden_size)
            out_t, h = self.gru(x[t].unsqueeze(0), h)       # x[t]: (num_chunks, 1312)
            outputs.append(out_t.squeeze(0))

        gru_out = torch.stack(outputs, dim=0).reshape(total_B, self.hidden_size)
        hidden_out = h

        # 5. Actor and Critic heads
        logits = self.actor(gru_out)      # (total_B, num_actions)
        value  = self.critic(gru_out)     # (total_B, 1)

        return logits, value, hidden_out

    # ------------------------------------------------------------------
    # Convenience wrappers used by the PPO trainer
    # ------------------------------------------------------------------

    def act(
        self,
        obs_embed: torch.Tensor,
        goal_embed: torch.Tensor,
        prev_actions: torch.Tensor,
        hidden: torch.Tensor,
        masks: torch.Tensor,
        can_stop: Optional[torch.Tensor] = None,   # ADD: (N,) bool, True = Stop allowed
    ) -> Tuple[Categorical, torch.Tensor, torch.Tensor]:
        logits, value, hidden_out = self.forward(
            obs_embed, goal_embed, prev_actions, hidden, masks
        )
        if can_stop is not None:
            stop_idx = self.num_actions - 1   # Stop is action index 3 (last)
            logits = logits.clone()
            logits[~can_stop, stop_idx] = float("-inf")
        dist = Categorical(logits=logits)
        return dist, value, hidden_out

    def evaluate_actions(
        self,
        obs_embed: torch.Tensor,
        goal_embed: torch.Tensor,
        prev_actions: torch.Tensor,
        hidden: torch.Tensor,
        masks: torch.Tensor,
        actions: torch.Tensor,
        can_stop: Optional[torch.Tensor] = None,   # ADD
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value, _ = self.forward(
            obs_embed, goal_embed, prev_actions, hidden, masks
        )
        if can_stop is not None:
            stop_idx = self.num_actions - 1
            logits = logits.clone()
            logits[~can_stop, stop_idx] = float("-inf")
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy().mean()
        return log_probs, value, entropy

    def get_initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return a zeroed GRU hidden state for episode start."""
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_policy(cfg, device: str = "cuda") -> GRUActorCritic:
    """
    Construct a GRUActorCritic from a Config object.

    Args:
        cfg:    configs.config.Config instance.
        device: Device string.

    Returns:
        GRUActorCritic on the specified device.
    """
    policy = GRUActorCritic(
        policy_input_dim=cfg.encoder.policy_input_dim,
        hidden_size=cfg.policy.hidden_size,
        num_actions=cfg.env.num_actions,
        num_action_embed=cfg.encoder.action_embed_dim,
        actor_hidden_dim=cfg.policy.actor_hidden_dim,
        critic_hidden_dim=cfg.policy.critic_hidden_dim,
    )
    return policy.to(torch.device(device))