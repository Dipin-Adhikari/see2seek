"""
fusion.py — Merges visual, goal, and action embeddings into a single joint representation.

Design rationale:
    Following the ZSON architecture, the agent needs a joint representation of:
        1. What it currently sees (DINOv2 visual embedding)
        2. What it is looking for (CLIP goal embedding)
        3. What it just did (Previous action embedding)
    
    This module converts the discrete previous action into a dense 32-dim vector,
    and then concatenates it with the 512-dim visual and 512-dim goal embeddings.
    The resulting 1056-dim feature vector is ready to be fed into the GRU memory.

    By isolating this logic here, we can easily experiment with more advanced 
    fusion mechanisms (e.g., FiLM, Cross-Attention) in the future without 
    breaking your Actor-Critic architecture.

Usage:
    fusion_module = MultimodalFusion(obs_dim=512, goal_dim=512, action_dim=32, num_actions=6)
    joint_features = fusion_module(obs_embed, goal_embed, prev_actions)  # (B, 1056)
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class MultimodalFusion(nn.Module):
    def __init__(self, obs_dim: int =512, goal_dim: int = 512, action_dim: int = 32,
                 num_actions: int = 6) -> None:
        
        """
        Fuses observation, goal, and previous action into a single feature vector.
        
        Args:
            obs_dim: Dimension of the visual observation embedding (DINOv2 = 512).
            goal_dim: Dimension of the goal embedding (CLIP = 512).
            action_dim: Target embedding dimension for the previous action (default: 32).
            num_actions: Number of discrete navigation actions in the environment.
                        Note: We internally add +1 to this number to account for a 
                        special "start of episode" dummy action (usually represented as -1 or 0
                        depending on your env wrapper).
        """

        super().__init__()
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.action_dim = action_dim
        self.num_actions = num_actions

        self.output_dim = self.obs_dim + self.goal_dim + self.action_dim

        # Embedding layer for previous actions. 
        # We use num_actions + 1 to allocate a specific embedding for the 
        # "Start of Episode" state where there is no legitimate previous action.
        self.action_embedder = nn.Embedding(
            num_embeddings=self.num_actions + 1,
            embedding_dim = self.action_dim,
        )

        logger.info(
            f"MultimodalFusion initialized — Output dim: {self.output_dim} "
            f"(Obs: {obs_dim} | Goal: {goal_dim} | Action: {action_dim})"
        )

    def forward(self, obs_embed: torch.Tensor, goal_embed: torch.Tensor,
                prev_actions: torch.Tensor) -> torch.Tensor:
        
        if obs_embed.dim() != 2 or obs_embed.size(1) != self.obs_dim:
            raise ValueError(f"Expected obs_embed shape (B, {self.obs_dim}), got {tuple(obs_embed.shape)}")
        
        if goal_embed.dim() != 2 or goal_embed.size(1) != self.goal_dim:
            raise ValueError(f"Expected goal_embed shape (B, {self.goal_dim}), got {tuple(goal_embed.shape)}")
        
        if prev_actions.dim() == 2 and prev_actions.size(1) == 1:
            prev_actions = prev_actions.squeeze(1)
        elif prev_actions.dim() != 1:
            raise ValueError(f"Expected prev_actions shape (B,) or (B, 1), got {tuple(prev_actions.shape)}")
        
        prev_actions = prev_actions.long()

        action_embed = self.action_embedder(prev_actions)

        fused_features = torch.cat([obs_embed, goal_embed, action_embed], dim=1)
        return fused_features # (B, 1056)
    

    def __repr__(self) -> str:
        return (
            f"MultimodalFusion(obs_dim={self.obs_dim}, goal_dim={self.goal_dim}, "
            f"action_dim={self.action_dim}, output_dim={self.output_dim})"
        )