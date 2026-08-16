"""
dino_encoder.py — Frozen DINOv2 ViT-B/14 observation encoder.

Motivation (from the paper):
    DINOv2's multi-crop self-distillation produces object-coherent, linearly
    separable patch features (~21-point gap over I-JEPA on STL-10 mean-pool
    linear probe). This makes it a strong frozen backbone for navigation, where
    the agent must identify semantically distinct regions across viewpoints.

Design decisions:
    - Model: dinov2_vitb14  (ViT-B with 14x14 patch size)
    - Outputs (single forward pass, see get_all_embeddings):
        * CLS token   -> shape (B, 768)             — global scene summary
        * Patch tokens-> shape (B, 256, 768)         — 16x16 spatial grid
                         (224 / 14 = 16 patches per side)
    - Weights: always frozen (no gradient flow)
    - Normalisation: ImageNet mean/std applied before forward pass
    - Loaded via torch.hub to avoid heavy dependency chains

IMPORTANT — avoid double forward passes:
    forward() and get_patch_embeddings() are kept for backward compatibility,
    but each independently calls self._backbone.forward_features(rgb). If you
    need BOTH the CLS token and patch tokens for a given frame (which the
    spatial-branch policy now does, every single rollout step), call
    get_all_embeddings() instead — it runs the ViT forward pass exactly once
    and returns both outputs. At 16 parallel envs stepping every frame,
    calling forward() + get_patch_embeddings() back-to-back silently doubles
    your DINOv2 compute for no reason.

Usage:
    encoder = DINOv2Encoder(device="cuda")
    cls_embed, patch_embeds = encoder.get_all_embeddings(rgb_tensor)
    # rgb_tensor:   (B, 3, 224, 224)
    # cls_embed:    (B, 768)
    # patch_embeds: (B, 256, 768)
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchvision import transforms

logger = logging.getLogger(__name__)


# ImageNet normalisation constants (DINOv2 was pre-trained with these)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


class DINOv2Encoder(nn.Module):
    """
    Frozen DINOv2 ViT-B/14 encoder that maps RGB images to 768-dim CLS
    embeddings and 256x768 patch-token grids.

    The encoder is always in eval() mode and gradients are disabled. It is
    designed to be instantiated once and shared across all parallel environments.

    Args:
        device:     Torch device string, e.g. "cuda" or "cpu".
        normalize:  If True (default), apply ImageNet normalisation before
                    forwarding through DINOv2. Set False only if your data
                    pipeline already normalises.
        hub_source: Where torch.hub loads the model from. Default "facebookresearch/dinov2".

    Attributes:
        embed_dim (int):  CLS token dimension = 768.
        patch_dim (int):  Per-patch token dimension = 768 (same width as CLS).
        num_patches (int): Number of patch tokens for a 224x224 input = 256 (16x16).
        grid_size (int):  Side length of the patch grid = 16.
    """

    embed_dim: int = 768
    patch_dim: int = 768
    num_patches: int = 256
    grid_size: int = 16

    def __init__(
        self,
        device: str = "cuda",
        normalize: bool = True,
        hub_source: str = "facebookresearch/dinov2",
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self._normalize = normalize

        logger.info("Loading DINOv2 ViT-B/14 from torch.hub ...")

        import os
        local_repo = os.path.expanduser(
            "~/.cache/torch/hub/facebookresearch_dinov2_main"
        )
        if os.path.isdir(local_repo):
            # Repo already cloned — skip the GitHub network round-trip entirely
            self._backbone = torch.hub.load(
                local_repo,
                "dinov2_vitb14",
                source="local",
                pretrained=True,
                verbose=False,
            )
        else:
            self._backbone = torch.hub.load(
                hub_source,
                "dinov2_vitb14",
                pretrained=True,
                verbose=False,
            )

        # Freeze all parameters — this encoder is never fine-tuned
        self._freeze()

        # Move to target device
        self._backbone.to(self.device)

        # Pre-build normalisation transform (applied inside forward)
        if normalize:
            self._norm = transforms.Normalize(
                mean=_IMAGENET_MEAN,
                std=_IMAGENET_STD,
            )
        else:
            self._norm = nn.Identity()

        logger.info(
            f"DINOv2Encoder ready — embed_dim={self.embed_dim}, "
            f"num_patches={self.num_patches}, device={self.device}, frozen=True"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _freeze(self) -> None:
        """Disable gradients and lock batch-norm stats."""
        for param in self._backbone.parameters():
            param.requires_grad = False
        self._backbone.eval()

    def _shape_guard(self, rgb: torch.Tensor) -> None:
        if rgb.dim() != 4 or rgb.shape[1] != 3:
            raise ValueError(
                f"Expected rgb shape (B, 3, H, W), got {tuple(rgb.shape)}"
            )

    # ------------------------------------------------------------------
    # Forward — CLS only (kept for backward compatibility)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of RGB images into 768-dim DINOv2 CLS embeddings.

        NOTE: if you also need patch tokens for the same frame, use
        get_all_embeddings() instead of calling this and
        get_patch_embeddings() separately — that runs the ViT twice.

        Args:
            rgb: Float tensor of shape (B, 3, H, W), pixel values in [0, 1].
                 H and W must be divisible by 14 (patch size); use 224x224.

        Returns:
            Tensor of shape (B, 768) — the CLS token from the last ViT block.
        """
        self._shape_guard(rgb)
        rgb = self._norm(rgb.to(self.device))
        features = self._backbone.forward_features(rgb)
        cls_token: torch.Tensor = features["x_norm_clstoken"]  # (B, 768)
        return cls_token

    @torch.no_grad()
    def get_patch_embeddings(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Extract dense spatial patch tokens from a batch of RGB images.

        NOTE: if you also need the CLS token for the same frame, use
        get_all_embeddings() instead — this runs the ViT twice otherwise.

        Args:
            rgb: Float tensor of shape (B, 3, H, W), pixel values in [0, 1].
                 H and W must be divisible by 14.

        Returns:
            Tensor of shape (B, 256, 768) — flat row-major patch sequence
            over a 16x16 grid (index = row * 16 + col).
        """
        self._shape_guard(rgb)
        rgb = self._norm(rgb.to(self.device))
        features = self._backbone.forward_features(rgb)
        patch_tokens: torch.Tensor = features["x_norm_patchtokens"]
        return patch_tokens

    # ------------------------------------------------------------------
    # Forward — CLS + patches in a single backbone pass (preferred)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_all_embeddings(
        self, rgb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the DINOv2 backbone ONCE and return both the CLS token and the
        patch-token grid. Use this everywhere both are needed (rollout
        collection, PPO re-evaluation) instead of forward() +
        get_patch_embeddings(), which would duplicate the forward pass.

        Args:
            rgb: Float tensor of shape (B, 3, H, W), pixel values in [0, 1].

        Returns:
            cls_token:    (B, 768)
            patch_tokens: (B, 256, 768) — flat row-major sequence over a
                          16x16 grid. To recover spatial layout as a CNN
                          feature map: patch_tokens.view(B, 16, 16, 768)
                          .permute(0, 3, 1, 2) -> (B, 768, 16, 16).
                          Do NOT reshape directly to (B, 768, 16, 16) —
                          that scrambles spatial position across channels.
        """
        self._shape_guard(rgb)
        rgb = self._norm(rgb.to(self.device))
        features = self._backbone.forward_features(rgb)
        cls_token: torch.Tensor = features["x_norm_clstoken"]       # (B, 768)
        patch_tokens: torch.Tensor = features["x_norm_patchtokens"]  # (B, 256, 768)
        return cls_token, patch_tokens

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def train(self, mode: bool = True) -> "DINOv2Encoder":
        """Override train() to keep backbone always in eval mode."""
        # The outer nn.Module can be in train mode (for PPO policy wrapper),
        # but the backbone itself must stay eval to keep BN stats frozen.
        super().train(mode)
        self._backbone.eval()
        return self

    def __repr__(self) -> str:
        return (
            f"DINOv2Encoder(model=dinov2_vitb14, embed_dim={self.embed_dim}, "
            f"num_patches={self.num_patches}, frozen=True, device={self.device})"
        )