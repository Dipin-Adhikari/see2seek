"""
gru_policy.py — GRU-based Actor-Critic policy for embodied navigation.

Architecture (spatial fusion + episodic memory):

    Frozen backbones (outside this module):
        DINOv2 ViT-B/14  -> cls_embed    (B, 768)
                          -> patch_embeds (B, 256, 768)   16x16 grid
        CLIP   ViT-B/32  -> goal_embed   (B, 512)

    Trainable branches:

    1. Spatial branch (SpatialCompressionHead):
           patch_embeds (B, 256, 768) -> CNN -> (B, 1568)

    2. CLS branch (linear projection):
           cls_embed (B, 768) -> Linear/LN/ELU -> (B, 64)

    3. Goal branch (trainable projection):
           goal_embed (B, 512) -> Linear/LN/ELU -> (B, 512)

    4. Episodic memory (cross-attention over past CLS tokens):
           query=current_cls, KV=past_cls_buffer -> (B, 128)

    5. Previous-action branch: embedding -> (B, 32)

    6. PointGoal branch: [d,cos,sin] -> Linear/ReLU -> (B, 32)

    7. Ego-pose branch: [x,y,cos_theta,sin_theta] -> Linear/ReLU -> (B, 32)
       Dead-reckoned position relative to episode start. Gives the agent
       explicit spatial awareness for loop detection and room escape.

    Fusion (flat concat):
        1568 + 64 + 512 + 128 + 32 + 32 + 32 = 2368-dim

    Recurrent policy: 2-layer GRU, hidden_size=512.

    The episodic memory operates within the recurrent loop: at each
    timestep, the current CLS token queries a buffer of past CLS tokens
    via single-head cross-attention. The buffer is built up step-by-step
    (no BPTT through time — stored tokens are detached). This gives the
    agent a "have I seen this before?" signal for loop detection.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper layers
# ---------------------------------------------------------------------------

class LinearNormAct(nn.Sequential):
    """Linear -> LayerNorm -> ELU block."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(
            nn.Linear(in_features, out_features),
            nn.LayerNorm(out_features),
            nn.ELU(inplace=True),
        )


class SpatialCompressionHead(nn.Module):
    """
    Trainable 2-layer CNN that compresses DINOv2 patch tokens into a
    compact spatial feature vector.

    Input:  (B, 256, 768) -> Output: (B, 1568)
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
        self.norm = nn.LayerNorm(32 * 7 * 7)

    @property
    def output_dim(self) -> int:
        return 32 * 7 * 7  # 1568

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        B, num_patches, C = patch_tokens.shape
        g = self.grid_size
        x = patch_tokens.view(B, g, g, C).permute(0, 3, 1, 2).contiguous()
        x = self.conv(x)
        x = x.flatten(start_dim=1)
        x = self.norm(x)
        x = F.normalize(x, p=2, dim=-1)
        return x


class EpisodicMemory(nn.Module):
    """
    Position-augmented episodic memory via cross-attention.

    Stores (CLS_token, dead_reckoned_pose) pairs in a circular buffer.
    Query: current CLS token + current pose (projected)
    Keys/Values: past CLS tokens + past poses (projected)
    Output: 128-dim memory context vector

    Dead-reckoned pose is (x, y, cos_theta, sin_theta) — 4 values,
    accumulated from discrete actions. This gives the agent spatial
    awareness of WHERE it has been, enabling implicit loop detection
    and frontier-seeking behavior without depth sensors.

    The buffer contents are detached (no BPTT through time). Only the
    Q/K/V projections, pose embedding, and output projection are trainable.
    """

    POSE_DIM = 4  # (x, y, cos_theta, sin_theta)

    def __init__(
        self,
        cls_dim: int = 768,
        memory_proj_dim: int = 128,
        num_heads: int = 1,
        pose_embed_dim: int = 32,
    ) -> None:
        super().__init__()
        self.cls_dim = cls_dim
        self.memory_proj_dim = memory_proj_dim
        self.head_dim = memory_proj_dim // num_heads
        self.num_heads = num_heads
        self.pose_embed_dim = pose_embed_dim

        # Pose embedding: raw 4-dim -> pose_embed_dim
        self.pose_proj = nn.Sequential(
            nn.Linear(self.POSE_DIM, pose_embed_dim),
            nn.ReLU(inplace=True),
        )

        kv_input_dim = cls_dim + pose_embed_dim
        q_input_dim = cls_dim + pose_embed_dim

        self.q_proj = nn.Linear(q_input_dim, memory_proj_dim)
        self.k_proj = nn.Linear(kv_input_dim, memory_proj_dim)
        self.v_proj = nn.Linear(kv_input_dim, memory_proj_dim)
        self.out_proj = nn.Linear(memory_proj_dim, memory_proj_dim)
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        query_cls: torch.Tensor,
        query_pose: torch.Tensor,
        memory_cls: torch.Tensor,
        memory_poses: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query_cls:    (B, cls_dim) — current CLS token.
            query_pose:   (B, 4) — current dead-reckoned pose.
            memory_cls:   (B, M, cls_dim) — past CLS tokens in buffer.
            memory_poses: (B, M, 4) — past poses in buffer.
            memory_mask:  (B, M) bool — True where buffer slot is valid.

        Returns:
            (B, memory_proj_dim) — memory readout vector.
        """
        B, M, _ = memory_cls.shape

        # Embed poses
        q_pose_feat = self.pose_proj(query_pose)               # (B, pose_embed_dim)
        m_pose_feat = self.pose_proj(memory_poses.reshape(B * M, -1)).view(B, M, -1)

        # Concat cls + pose for Q, K, V
        q_input = torch.cat([query_cls, q_pose_feat], dim=-1)  # (B, cls+pose)
        kv_input = torch.cat([memory_cls, m_pose_feat], dim=-1)  # (B, M, cls+pose)

        q = self.q_proj(q_input).unsqueeze(1)      # (B, 1, proj_dim)
        k = self.k_proj(kv_input)                  # (B, M, proj_dim)
        v = self.v_proj(kv_input)                  # (B, M, proj_dim)

        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (B, 1, M)

        if memory_mask is not None:
            attn = attn.masked_fill(~memory_mask.unsqueeze(1), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        out = torch.bmm(attn, v).squeeze(1)        # (B, proj_dim)
        out = self.out_proj(out)
        return out


# ---------------------------------------------------------------------------
# Main policy
# ---------------------------------------------------------------------------

class GRUActorCritic(nn.Module):
    """
    2-layer GRU Actor-Critic with episodic memory and goal projection.
    """

    def __init__(
        self,
        obs_encoder_type: str = "dino",
        dino_patch_dim: int = 768,
        dino_grid_size: int = 16,
        dino_cls_dim: int = 768,
        clip_obs_dim: int = 512,
        clip_obs_proj_dim: int = 512,
        goal_embed_dim: int = 512,
        goal_proj_dim: int = 512,
        cls_proj_dim: int = 64,
        use_cls: bool = True,
        hidden_size: int = 512,
        num_recurrent_layers: int = 2,
        num_actions: int = 4,
        num_action_embed: int = 32,
        pointgoal_input_dim: int = 3,
        pointgoal_embed_dim: int = 32,
        egopose_input_dim: int = 4,
        egopose_embed_dim: int = 32,
        use_egopose: bool = True,
        memory_size: int = 64,
        memory_proj_dim: int = 128,
        actor_hidden_dim: int = 256,
        critic_hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        self.obs_encoder_type = obs_encoder_type
        self.hidden_size = hidden_size
        self.num_recurrent_layers = num_recurrent_layers
        self.num_actions = num_actions
        self.use_cls = use_cls
        self.goal_embed_dim = goal_embed_dim
        self.goal_proj_dim = goal_proj_dim
        self.num_action_embed = num_action_embed
        self.pointgoal_embed_dim = pointgoal_embed_dim
        self.egopose_embed_dim = egopose_embed_dim
        self.use_egopose = use_egopose
        self.memory_size = memory_size
        self.memory_proj_dim = memory_proj_dim
        self.dino_cls_dim = dino_cls_dim

        if obs_encoder_type == "dino":
            self.spatial_head = SpatialCompressionHead(
                in_channels=dino_patch_dim, grid_size=dino_grid_size
            )
            self.cls_proj_dim = cls_proj_dim
            self.cls_proj = LinearNormAct(dino_cls_dim, cls_proj_dim)

            obs_dim = (
                self.spatial_head.output_dim
                + (cls_proj_dim if use_cls else 0)
            )
        elif obs_encoder_type == "clip":
            self.clip_obs_proj_dim = clip_obs_proj_dim
            self.obs_proj = nn.Sequential(
                LinearNormAct(clip_obs_dim, clip_obs_proj_dim),
            )
            obs_dim = clip_obs_proj_dim
        else:
            raise ValueError(f"Unknown obs_encoder_type: {obs_encoder_type}")

        # ---- Goal projection (trainable) ----
        self.goal_proj = LinearNormAct(goal_embed_dim, goal_proj_dim)

        # ---- Episodic memory ----
        self.episodic_memory = EpisodicMemory(
            cls_dim=dino_cls_dim if obs_encoder_type == "dino" else clip_obs_dim,
            memory_proj_dim=memory_proj_dim,
        )

        # ---- Previous-action embedding ----
        self.prev_action_embed = nn.Embedding(
            num_embeddings=num_actions + 1,
            embedding_dim=num_action_embed,
        )

        # ---- PointGoal embedding ----
        self.pointgoal_proj = nn.Sequential(
            nn.Linear(pointgoal_input_dim, pointgoal_embed_dim),
            nn.ReLU(inplace=True),
        )

        # ---- Ego-pose embedding (dead-reckoned position relative to start) ----
        if use_egopose:
            self.egopose_proj = nn.Sequential(
                nn.Linear(egopose_input_dim, egopose_embed_dim),
                nn.ReLU(inplace=True),
            )

        # ---- Fused input dim (without memory — memory added per-step in loop) ----
        self._base_input_dim = (
            obs_dim
            + goal_proj_dim
            + num_action_embed
            + pointgoal_embed_dim
            + (egopose_embed_dim if use_egopose else 0)
        )
        self.policy_input_dim = self._base_input_dim + memory_proj_dim

        # ---- 2-layer GRU ----
        self.gru = nn.GRU(
            input_size=self.policy_input_dim,
            hidden_size=hidden_size,
            num_layers=num_recurrent_layers,
            batch_first=False,
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

        self._init_weights()

        egopose_str = f", egopose={egopose_embed_dim}" if use_egopose else ""
        logger.info(
            f"GRUActorCritic [{obs_encoder_type}] — "
            f"policy_input_dim={self.policy_input_dim} "
            f"(obs={obs_dim}, goal_proj={goal_proj_dim}, "
            f"memory={memory_proj_dim}, "
            f"prev_action={num_action_embed}, pointgoal={pointgoal_embed_dim}{egopose_str}), "
            f"GRU={num_recurrent_layers}x{hidden_size}, actions={num_actions}"
        )

    # ------------------------------------------------------------------
    # Weight initialisation
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

        if self.obs_encoder_type == "dino":
            for m in self.spatial_head.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.orthogonal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            for m in self.cls_proj.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight)
                    nn.init.zeros_(m.bias)
        elif self.obs_encoder_type == "clip":
            for m in self.obs_proj.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight)
                    nn.init.zeros_(m.bias)

        for m in self.goal_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)

        for m in self.pointgoal_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)

        if self.use_egopose:
            for m in self.egopose_proj.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight)
                    nn.init.zeros_(m.bias)

        for m in self.episodic_memory.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Base fusion (everything except memory — computed for all steps at once)
    # ------------------------------------------------------------------

    def _fuse_base_inputs(
        self,
        patch_embeds: torch.Tensor,
        cls_embed: torch.Tensor,
        goal_embed: torch.Tensor,
        prev_actions: torch.Tensor,
        pointgoal: torch.Tensor,
        poses: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Build the base fused input (without memory) for all timesteps.

        Returns:
            (B, base_input_dim)
        """
        prev_act_embed = self.prev_action_embed(prev_actions)
        pointgoal_feat = self.pointgoal_proj(pointgoal)
        pointgoal_feat = F.normalize(pointgoal_feat, p=2, dim=-1)
        goal_feat = self.goal_proj(goal_embed)
        goal_feat = F.normalize(goal_feat, p=2, dim=-1)

        if self.obs_encoder_type == "dino":
            spatial_feat = self.spatial_head(patch_embeds)
            parts = [spatial_feat]
            if self.use_cls:
                cls_feat = self.cls_proj(cls_embed)
                cls_feat = F.normalize(cls_feat, p=2, dim=-1)
                parts.append(cls_feat)
        else:
            obs_feat = self.obs_proj(cls_embed)
            obs_feat = F.normalize(obs_feat, p=2, dim=-1)
            parts = [obs_feat]

        parts.append(goal_feat)
        parts.append(prev_act_embed)
        parts.append(pointgoal_feat)

        # Add ego-pose as direct input (dead-reckoned position relative to start)
        if self.use_egopose and poses is not None:
            egopose_feat = self.egopose_proj(poses)
            egopose_feat = F.normalize(egopose_feat, p=2, dim=-1)
            parts.append(egopose_feat)

        return torch.cat(parts, dim=-1)

    # ------------------------------------------------------------------
    # Core forward with episodic memory in recurrent loop
    # ------------------------------------------------------------------

    def forward(
        self,
        patch_embeds: torch.Tensor,
        cls_embed: torch.Tensor,
        goal_embed: torch.Tensor,
        prev_actions: torch.Tensor,
        hidden: torch.Tensor,
        masks: torch.Tensor,
        pointgoal: torch.Tensor = None,
        memory_buffer: Optional[torch.Tensor] = None,
        memory_pose_buffer: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        poses: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass with position-augmented episodic memory.

        Args:
            patch_embeds: (total_B, 256, 768)
            cls_embed:    (total_B, 768)
            goal_embed:   (total_B, 512)
            prev_actions: (total_B,) long
            hidden:       (num_layers, num_chunks, hidden_size)
            masks:        (total_B, 1)
            pointgoal:    (total_B, 3)
            memory_buffer:      (num_chunks, memory_size, cls_dim)
            memory_pose_buffer: (num_chunks, memory_size, 4)
            memory_mask:        (num_chunks, memory_size) bool
            poses:        (total_B, 4) — dead-reckoned [x, y, cos_theta, sin_theta]

        Returns:
            logits, value, hidden_out, memory_buf_out, memory_pose_buf_out, memory_mask_out
        """
        total_B = cls_embed.shape[0]
        num_chunks = hidden.shape[1]
        chunk_len = total_B // num_chunks
        cls_dim = cls_embed.shape[-1]

        # 1. Compute base features for all steps at once
        base_feat = self._fuse_base_inputs(
            patch_embeds, cls_embed, goal_embed, prev_actions, pointgoal, poses
        )
        base_feat = base_feat.view(chunk_len, num_chunks, -1)

        # Reshape CLS and poses for per-step memory operations
        cls_seq = cls_embed.view(chunk_len, num_chunks, cls_dim)
        masks_seq = masks.view(chunk_len, num_chunks, 1)

        if poses is not None:
            pose_seq = poses.view(chunk_len, num_chunks, EpisodicMemory.POSE_DIM)
        else:
            pose_seq = torch.zeros(
                chunk_len, num_chunks, EpisodicMemory.POSE_DIM,
                device=cls_embed.device, dtype=cls_embed.dtype,
            )

        # 2. Initialize memory buffers if not provided
        if memory_buffer is None:
            memory_buffer = torch.zeros(
                num_chunks, self.memory_size, cls_dim,
                device=cls_embed.device, dtype=cls_embed.dtype,
            )
            memory_pose_buffer = torch.zeros(
                num_chunks, self.memory_size, EpisodicMemory.POSE_DIM,
                device=cls_embed.device, dtype=cls_embed.dtype,
            )
            memory_mask = torch.zeros(
                num_chunks, self.memory_size,
                device=cls_embed.device, dtype=torch.bool,
            )
        elif memory_pose_buffer is None:
            memory_pose_buffer = torch.zeros(
                num_chunks, self.memory_size, EpisodicMemory.POSE_DIM,
                device=cls_embed.device, dtype=cls_embed.dtype,
            )

        # Track write position in circular buffer
        write_pos = memory_mask.sum(dim=1).clamp(max=self.memory_size - 1).long()

        # 3. Step through time with memory
        h = hidden
        outputs = []
        for t in range(chunk_len):
            # Reset hidden state at episode boundaries
            h = h * masks_seq[t].unsqueeze(0).expand_as(h)

            # Reset memory buffer for envs that just started a new episode
            episode_start = (masks_seq[t].squeeze(-1) == 0)
            if episode_start.any():
                memory_buffer[episode_start] = 0
                memory_pose_buffer[episode_start] = 0
                memory_mask[episode_start] = False
                write_pos[episode_start] = 0

            # Compute memory readout via cross-attention
            has_memory = memory_mask.any(dim=1)
            memory_context = torch.zeros(
                num_chunks, self.memory_proj_dim,
                device=cls_embed.device, dtype=cls_embed.dtype,
            )
            if has_memory.any():
                mem_out = self.episodic_memory(
                    cls_seq[t][has_memory],
                    pose_seq[t][has_memory],
                    memory_buffer[has_memory],
                    memory_pose_buffer[has_memory],
                    memory_mask[has_memory],
                )
                memory_context[has_memory] = F.normalize(mem_out, p=2, dim=-1)

            # Write current CLS + pose to buffer (detached)
            idx_cls = write_pos.unsqueeze(1).unsqueeze(2).expand(-1, 1, cls_dim)
            memory_buffer.scatter_(1, idx_cls, cls_seq[t].detach().unsqueeze(1))

            idx_pose = write_pos.unsqueeze(1).unsqueeze(2).expand(-1, 1, EpisodicMemory.POSE_DIM)
            memory_pose_buffer.scatter_(1, idx_pose, pose_seq[t].detach().unsqueeze(1))

            memory_mask.scatter_(1, write_pos.unsqueeze(1), True)
            write_pos = (write_pos + 1) % self.memory_size

            # Concatenate base features + memory context
            x_t = torch.cat([base_feat[t], memory_context], dim=-1)

            # GRU step
            out_t, h = self.gru(x_t.unsqueeze(0), h)
            outputs.append(out_t.squeeze(0))

        gru_out = torch.stack(outputs, dim=0).reshape(total_B, self.hidden_size)
        hidden_out = h

        # 4. Actor and Critic heads
        logits = self.actor(gru_out)
        value = self.critic(gru_out)

        return logits, value, hidden_out, memory_buffer, memory_pose_buffer, memory_mask

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def act(
        self,
        patch_embeds: torch.Tensor,
        cls_embed: torch.Tensor,
        goal_embed: torch.Tensor,
        prev_actions: torch.Tensor,
        hidden: torch.Tensor,
        masks: torch.Tensor,
        pointgoal: torch.Tensor = None,
        can_stop: Optional[torch.Tensor] = None,
        memory_buffer: Optional[torch.Tensor] = None,
        memory_pose_buffer: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        poses: Optional[torch.Tensor] = None,
    ) -> Tuple[Categorical, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: (dist, value, hidden_out, memory_buffer_out, memory_pose_buffer_out, memory_mask_out)
        """
        logits, value, hidden_out, mem_buf, mem_pose_buf, mem_mask = self.forward(
            patch_embeds, cls_embed, goal_embed, prev_actions, hidden, masks,
            pointgoal, memory_buffer, memory_pose_buffer, memory_mask, poses,
        )
        if can_stop is not None:
            stop_idx = self.num_actions - 1
            logits = logits.clone()
            logits[~can_stop, stop_idx] = float("-inf")
        dist = Categorical(logits=logits)
        return dist, value, hidden_out, mem_buf, mem_pose_buf, mem_mask

    def evaluate_actions(
        self,
        patch_embeds: torch.Tensor,
        cls_embed: torch.Tensor,
        goal_embed: torch.Tensor,
        prev_actions: torch.Tensor,
        hidden: torch.Tensor,
        masks: torch.Tensor,
        actions: torch.Tensor,
        pointgoal: torch.Tensor = None,
        can_stop: Optional[torch.Tensor] = None,
        memory_buffer: Optional[torch.Tensor] = None,
        memory_pose_buffer: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        poses: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-evaluate actions for PPO update. Memory is built from scratch
        within each chunk (first steps have limited history, which is
        acceptable since the GRU hidden state carries longer context).

        Returns:
            log_probs: (total_B,)
            value:     (total_B, 1)
            entropy:   (total_B,) — per-sample entropy (NOT mean)
        """
        logits, value, _, _, _, _ = self.forward(
            patch_embeds, cls_embed, goal_embed, prev_actions, hidden, masks,
            pointgoal, memory_buffer, memory_pose_buffer, memory_mask, poses,
        )
        if can_stop is not None:
            stop_idx = self.num_actions - 1
            logits = logits.clone()
            logits[~can_stop, stop_idx] = float("-inf")
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, value, entropy

    def get_initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return zeroed GRU hidden state for episode start."""
        return torch.zeros(
            self.num_recurrent_layers, batch_size, self.hidden_size, device=device
        )

    @torch.no_grad()
    def branch_norms(
        self,
        patch_embeds: torch.Tensor,
        cls_embed: torch.Tensor,
        goal_embed: torch.Tensor,
        pointgoal: Optional[torch.Tensor] = None,
        poses: Optional[torch.Tensor] = None,
        memory_buffer: Optional[torch.Tensor] = None,
        memory_pose_buffer: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Diagnostic: mean per-sample L2 norm of each fusion branch."""
        norms = {}

        goal_feat = F.normalize(self.goal_proj(goal_embed), p=2, dim=-1)
        norms["goal"] = goal_feat.norm(dim=-1).mean().item()

        if self.obs_encoder_type == "dino":
            spatial_feat = self.spatial_head(patch_embeds)
            norms["spatial"] = spatial_feat.norm(dim=-1).mean().item()
            if self.use_cls:
                cls_feat = F.normalize(self.cls_proj(cls_embed), p=2, dim=-1)
                norms["cls"] = cls_feat.norm(dim=-1).mean().item()
        else:
            obs_feat = F.normalize(self.obs_proj(cls_embed), p=2, dim=-1)
            norms["obs"] = obs_feat.norm(dim=-1).mean().item()

        if pointgoal is not None:
            pg_feat = self.pointgoal_proj(pointgoal)
            norms["pointgoal"] = pg_feat.norm(dim=-1).mean().item()

        if self.use_egopose and poses is not None:
            ego_feat = self.egopose_proj(poses)
            norms["egopose"] = ego_feat.norm(dim=-1).mean().item()

        if memory_buffer is not None and memory_mask is not None:
            has_memory = memory_mask.any(dim=1)
            if has_memory.any() and poses is not None:
                mem_out = self.episodic_memory(
                    cls_embed[has_memory],
                    poses[has_memory],
                    memory_buffer[has_memory],
                    memory_pose_buffer[has_memory],
                    memory_mask[has_memory],
                )
                mem_out = F.normalize(mem_out, p=2, dim=-1)
                norms["memory"] = mem_out.norm(dim=-1).mean().item()

        return norms


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_policy(cfg, device: str = "cuda") -> GRUActorCritic:
    """Construct a GRUActorCritic from a Config object."""
    enc = cfg.encoder
    pol = cfg.policy
    policy = GRUActorCritic(
        obs_encoder_type=getattr(enc, "obs_encoder_type", "dino"),
        dino_patch_dim=getattr(enc, "dino_patch_dim", 768),
        dino_grid_size=getattr(enc, "dino_grid_size", 16),
        dino_cls_dim=getattr(enc, "dino_cls_dim", 768),
        clip_obs_dim=getattr(enc, "goal_embed_dim", 512),
        clip_obs_proj_dim=getattr(enc, "clip_obs_proj_dim", 512),
        goal_embed_dim=getattr(enc, "goal_embed_dim", 512),
        goal_proj_dim=getattr(enc, "goal_proj_dim", 512),
        cls_proj_dim=getattr(enc, "cls_proj_dim", 64),
        use_cls=getattr(enc, "use_cls", True),
        hidden_size=pol.hidden_size,
        num_recurrent_layers=getattr(pol, "num_recurrent_layers", 2),
        num_actions=cfg.env.num_actions,
        num_action_embed=getattr(enc, "action_embed_dim", 32),
        pointgoal_input_dim=getattr(enc, "pointgoal_input_dim", 3),
        pointgoal_embed_dim=getattr(enc, "pointgoal_embed_dim", 32),
        egopose_input_dim=getattr(enc, "egopose_input_dim", 4),
        egopose_embed_dim=getattr(enc, "egopose_embed_dim", 32),
        use_egopose=getattr(enc, "use_egopose", True),
        memory_size=getattr(enc, "memory_size", 64),
        memory_proj_dim=getattr(enc, "memory_proj_dim", 128),
        actor_hidden_dim=pol.actor_hidden_dim,
        critic_hidden_dim=pol.critic_hidden_dim,
    )
    return policy.to(torch.device(device))
