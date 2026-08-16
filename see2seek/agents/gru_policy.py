"""
gru_policy.py — GRU-based Actor-Critic policy for embodied navigation.

Architecture (ZSON/EmbCLIP-inspired spatial fusion):

    Frozen backbones (outside this module, see dino_encoder.py / clip_encoder.py):
        DINOv2 ViT-B/14  -> cls_embed    (B, 768)
                          -> patch_embeds (B, 256, 768)   16x16 grid
        CLIP   ViT-B/32  -> goal_embed   (B, 512)

    Four independent branches (no cross-conditioning between them):

    1. Spatial branch (trainable, SpatialCompressionHead):
           patch_embeds (B, 256, 768)
           -> reshape (B, 16, 16, 768) -> permute (B, 768, 16, 16)
           -> 2-layer CNN (EmbCLIP-style compression) -> (B, 32, 7, 7)
           -> flatten -> (B, 1568)
       This is the only branch carrying "where things are" spatial info.
       Unlike EmbCLIP, no goal-tiling happens here — goal fusion is deferred
       to the flat concatenation below (ZSON-style).

    2. CLS branch (trainable, small linear projection):
           cls_embed (B, 768) -> Linear/LayerNorm/ELU -> (B, 64)
       Light global scene context ("this is a bedroom"). Gated by `use_cls`
       so it can be ablated to check whether it earns its 64 dims.

    3. Goal branch (no projection — fed raw):
           goal_embed (B, 512) passed through unchanged.

    4. Previous-action branch:
           learned embedding, (B, 32).

    Fusion (flat concat, single fusion point):
        1568 (spatial) + 64 (CLS-proj, if use_cls) + 512 (goal, raw) + 32 (prev action)
        = 2176-dim (or 2112-dim if use_cls=False)

    Recurrent policy: 1-layer GRU, hidden_size=512, input_size=policy_input_dim.
    (2-layer GRU is a deliberately separate, later ablation — see project
    notes; keep this validation run isolated to the encoder change alone.)

Design notes:
    - Hidden state (h_t) is the agent's episodic memory and MUST be
      correctly propagated across steps and reset at episode boundaries.
    - The SpatialCompressionHead and CLS projection are TRAINABLE — unlike
      the frozen DINOv2/CLIP backbones, gradients flow through them during
      PPO updates. This means the RolloutBuffer must store raw patch tokens
      (and raw CLS embeddings), not pre-compressed vectors, so the compression
      head can be re-run (with gradients) during evaluate_actions() in every
      PPO epoch. See rollout_buffer.py for the buffer-side implications.
    - Recurrent PPO requires chunked mini-batch updates; the RolloutBuffer
      handles storing and replaying hidden states correctly.

Usage:
    policy = GRUActorCritic(
        hidden_size=512,
        num_actions=4,
        use_cls=True,
    )
    dist, value, h_next = policy.act(
        patch_embeds, cls_embed, goal_embed, prev_action, h_prev, masks
    )
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
    """Linear → LayerNorm → ELU block used in Actor/Critic heads and CLS proj."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.ELU(inplace=True),
        )


class SpatialCompressionHead(nn.Module):
    """
    Trainable 2-layer CNN (EmbCLIP-style) that compresses DINOv2's flat
    patch-token sequence into a compact spatial feature vector.

    Input:  patch_tokens (B, 256, 768) — DINOv2's flat row-major patch
            sequence (index = row * 16 + col) over a 16x16 grid.
    Output: (B, 1568) — flattened 32x7x7 compressed grid.

    Reshape correctness (this matters — a naive reshape scrambles space):
        patch_tokens.view(B, 16, 16, 768)   # spatial layout preserved:
                                             # row-major (256,) -> (16,16)
                                             # is a safe, order-preserving
                                             # reshape since PyTorch's
                                             # .view() is also row-major.
        .permute(0, 3, 1, 2)                # (B, 16, 16, 768) -> (B, 768, 16, 16)
                                             # NCHW for Conv2d.
        Do NOT do patch_tokens.reshape(B, 768, 16, 16) directly — that
        reads the wrong axis as the spatial one and produces garbage
        spatial locations despite looking shape-correct.

    Conv dims (16x16 -> 7x7, verified):
        Conv2d(768, 128, kernel_size=3, stride=2, padding=0):
            out = floor((16 - 3) / 2) + 1 = 7           -> (B, 128, 7, 7)
        Conv2d(128, 32,  kernel_size=3, stride=1, padding=1):
            out = 7 (padding=1 preserves spatial size)  -> (B, 32, 7, 7)
        flatten -> (B, 1568)
    """

    def __init__(self, in_channels: int = 768, grid_size: int = 16) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    @property
    def output_dim(self) -> int:
        return 32 * 7 * 7  # 1568

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: (B, num_patches, C) where num_patches == grid_size**2.
        Returns:
            (B, output_dim) flattened spatial feature vector.
        """
        B, num_patches, C = patch_tokens.shape
        g = self.grid_size
        if num_patches != g * g:
            raise ValueError(
                f"SpatialCompressionHead expected {g * g} patches "
                f"(grid_size={g}), got {num_patches}"
            )
        x = patch_tokens.view(B, g, g, C).permute(0, 3, 1, 2).contiguous()  # (B, C, g, g)
        x = self.conv(x)                    # (B, 32, 7, 7)
        x = x.flatten(start_dim=1)          # (B, 1568)
        return x


# ---------------------------------------------------------------------------
# Main policy
# ---------------------------------------------------------------------------

class GRUActorCritic(nn.Module):
    """
    Single-layer GRU Actor-Critic policy for discrete-action navigation.

    Consumes frozen DINOv2 (CLS + patch tokens) and CLIP goal embeddings.
    The spatial-compression CNN and CLS projection are trainable; the GRU
    provides temporal context.

    Args:
        dino_patch_dim:    Per-patch DINOv2 token width. Default 768.
        dino_grid_size:    Side length of the DINOv2 patch grid. Default 16.
        dino_cls_dim:      DINOv2 CLS token width. Default 768.
        goal_embed_dim:    CLIP goal embedding width (fed raw, unprojected). Default 512.
        cls_proj_dim:      CLS projection output width. Default 64.
        use_cls:           If False, the CLS branch is dropped entirely from
                            the fused vector (ablation switch). Default True.
        hidden_size:       GRU hidden state size. Default 512.
        num_actions:       Size of the discrete action space. Default 4.
        num_action_embed:  Dimension of the learned prev-action embedding. Default 32.
        actor_hidden_dim:  Intermediate linear dim in actor head. Default 256.
        critic_hidden_dim: Intermediate linear dim in critic head. Default 256.

    Note on policy_input_dim:
        Computed automatically from the above as:
            spatial_head.output_dim (1568)
            + (cls_proj_dim if use_cls else 0)
            + goal_embed_dim
            + num_action_embed
        Default: 1568 + 64 + 512 + 32 = 2176.
    """

    def __init__(
        self,
        dino_patch_dim: int = 768,
        dino_grid_size: int = 16,
        dino_cls_dim: int = 768,
        goal_embed_dim: int = 512,
        cls_proj_dim: int = 64,
        use_cls: bool = True,
        hidden_size: int = 512,
        num_actions: int = 4,
        num_action_embed: int = 32,
        actor_hidden_dim: int = 256,
        critic_hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_actions = num_actions
        self.use_cls = use_cls
        self.goal_embed_dim = goal_embed_dim
        self.num_action_embed = num_action_embed

        # ---- Spatial branch (trainable) ----
        self.spatial_head = SpatialCompressionHead(
            in_channels=dino_patch_dim, grid_size=dino_grid_size
        )

        # ---- CLS branch (trainable, ablatable) ----
        self.cls_proj_dim = cls_proj_dim
        self.cls_proj = LinearNormAct(dino_cls_dim, cls_proj_dim)

        # ---- Previous-action embedding ----
        self.prev_action_embed = nn.Embedding(
            num_embeddings=num_actions + 1,    # +1 for start-of-episode padding
            embedding_dim=num_action_embed,
        )

        # ---- Fused input dim ----
        self.policy_input_dim = (
            self.spatial_head.output_dim
            + (cls_proj_dim if use_cls else 0)
            + goal_embed_dim
            + num_action_embed
        )

        # ---- GRU ----
        self.gru = nn.GRU(
            input_size=self.policy_input_dim,
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
            f"GRUActorCritic — policy_input_dim={self.policy_input_dim} "
            f"(spatial={self.spatial_head.output_dim}, "
            f"cls={cls_proj_dim if use_cls else 0}, "
            f"goal={goal_embed_dim}, prev_action={num_action_embed}), "
            f"hidden={hidden_size}, actions={num_actions}, use_cls={use_cls}"
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

        for m in self.spatial_head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.cls_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Fusion helper
    # ------------------------------------------------------------------

    def _fuse_inputs(
        self,
        patch_embeds: torch.Tensor,
        cls_embed: torch.Tensor,
        goal_embed: torch.Tensor,
        prev_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build the flat fused input vector for a batch of timesteps.

        Args:
            patch_embeds: (B, 256, 768) — DINOv2 patch tokens.
            cls_embed:    (B, 768)      — DINOv2 CLS token.
            goal_embed:   (B, 512)      — CLIP goal embedding (raw, unprojected).
            prev_actions: (B,) long     — index of the action taken last step.

        Returns:
            (B, policy_input_dim)
        """
        spatial_feat = self.spatial_head(patch_embeds)          # (B, 1568)
        prev_act_embed = self.prev_action_embed(prev_actions)   # (B, 32)

        parts = [spatial_feat]
        if self.use_cls:
            parts.append(self.cls_proj(cls_embed))              # (B, 64)
        parts.append(goal_embed)                                # (B, 512), raw
        parts.append(prev_act_embed)                             # (B, 32)

        return torch.cat(parts, dim=-1)

    # ------------------------------------------------------------------
    # Core forward (used internally)
    # ------------------------------------------------------------------

    def forward(
        self,
        patch_embeds: torch.Tensor,
        cls_embed: torch.Tensor,
        goal_embed: torch.Tensor,
        prev_actions: torch.Tensor,
        hidden: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass through the spatial-compression head, CLS
        projection, and GRU policy.

        Args:
            patch_embeds: (N, 256, 768) — DINOv2 patch tokens.
            cls_embed:    (N, 768)      — DINOv2 CLS token.
            goal_embed:   (N, 512)      — CLIP goal embedding.
            prev_actions: (N,)  long — index of the action taken last step.
                          Use index `num_actions` for the very first step.
            hidden:       (1, N, hidden_size) — GRU hidden state.
            masks:        (N, 1) float — 0.0 at episode boundaries (resets h),
                          1.0 otherwise.

        Returns:
            logits:     (N, num_actions) — raw action scores (pre-softmax).
            value:      (N, 1)          — state-value estimate.
            hidden_out: (1, N, hidden_size) — updated GRU hidden state.
        """
        total_B    = patch_embeds.shape[0]   # N during rollout, chunk_len*num_chunks during update
        num_chunks = hidden.shape[1]         # hidden: (1, num_chunks, hidden_size)
        chunk_len  = total_B // num_chunks   # 1 during rollout, >1 during PPO update

        # 1. Fuse all branches: (total_B, policy_input_dim)
        x = self._fuse_inputs(patch_embeds, cls_embed, goal_embed, prev_actions)

        # 2. Reshape to time-major: (chunk_len, num_chunks, features)
        x = x.view(chunk_len, num_chunks, -1)
        masks_seq = masks.view(chunk_len, num_chunks, 1)

        # 3. Step through the chunk manually, resetting hidden state at
        #    episode boundaries (mask == 0) at each timestep.
        h = hidden
        outputs = []
        for t in range(chunk_len):
            h = h * masks_seq[t].unsqueeze(0)              # (1, num_chunks, hidden_size)
            out_t, h = self.gru(x[t].unsqueeze(0), h)       # x[t]: (num_chunks, policy_input_dim)
            outputs.append(out_t.squeeze(0))

        gru_out = torch.stack(outputs, dim=0).reshape(total_B, self.hidden_size)
        hidden_out = h

        # 4. Actor and Critic heads
        logits = self.actor(gru_out)      # (total_B, num_actions)
        value  = self.critic(gru_out)     # (total_B, 1)

        return logits, value, hidden_out

    # ------------------------------------------------------------------
    # Convenience wrappers used by the PPO trainer
    # ------------------------------------------------------------------

    def act(
        self,
        patch_embeds: torch.Tensor,
        cls_embed: torch.Tensor,
        goal_embed: torch.Tensor,
        prev_actions: torch.Tensor,
        hidden: torch.Tensor,
        masks: torch.Tensor,
        can_stop: Optional[torch.Tensor] = None,   # (N,) bool, True = Stop allowed
    ) -> Tuple[Categorical, torch.Tensor, torch.Tensor]:
        logits, value, hidden_out = self.forward(
            patch_embeds, cls_embed, goal_embed, prev_actions, hidden, masks
        )
        if can_stop is not None:
            stop_idx = self.num_actions - 1   # Stop is action index 3 (last)
            logits = logits.clone()
            logits[~can_stop, stop_idx] = float("-inf")
        dist = Categorical(logits=logits)
        return dist, value, hidden_out

    def evaluate_actions(
        self,
        patch_embeds: torch.Tensor,
        cls_embed: torch.Tensor,
        goal_embed: torch.Tensor,
        prev_actions: torch.Tensor,
        hidden: torch.Tensor,
        masks: torch.Tensor,
        actions: torch.Tensor,
        can_stop: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value, _ = self.forward(
            patch_embeds, cls_embed, goal_embed, prev_actions, hidden, masks
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

    Expects the following (new) fields on cfg.encoder:
        dino_patch_dim   (default 768)
        dino_grid_size   (default 16)
        dino_cls_dim     (default 768)
        goal_embed_dim   (default 512)
        cls_proj_dim     (default 64)
        use_cls          (default True)

    Args:
        cfg:    configs.config.Config instance.
        device: Device string.

    Returns:
        GRUActorCritic on the specified device.
    """
    enc = cfg.encoder
    policy = GRUActorCritic(
        dino_patch_dim=getattr(enc, "dino_patch_dim", 768),
        dino_grid_size=getattr(enc, "dino_grid_size", 16),
        dino_cls_dim=getattr(enc, "dino_cls_dim", 768),
        goal_embed_dim=getattr(enc, "goal_embed_dim", 512),
        cls_proj_dim=getattr(enc, "cls_proj_dim", 64),
        use_cls=getattr(enc, "use_cls", True),
        hidden_size=cfg.policy.hidden_size,
        num_actions=cfg.env.num_actions,
        num_action_embed=getattr(enc, "action_embed_dim", 32),
        actor_hidden_dim=cfg.policy.actor_hidden_dim,
        critic_hidden_dim=cfg.policy.critic_hidden_dim,
    )
    return policy.to(torch.device(device))