"""
dino_encoder.py — Frozen DINOv2 ViT-B/14 observation encoder.

Motivation (from the paper):
    DINOv2's multi-crop self-distillation produces object-coherent, linearly
    separable patch features (~21-point gap over I-JEPA on STL-10 mean-pool
    linear probe). This makes it a strong frozen backbone for navigation, where
    the agent must identify semantically distinct regions across viewpoints.

Design decisions:
    - Model: dinov2_vitb14  (ViT-B with 14x14 patch size)
    - Output: CLS token  → shape (B, 512)
    - Weights: always frozen (no gradient flow)
    - Normalisation: ImageNet mean/std applied before forward pass
    - Loaded via torch.hub to avoid heavy dependency chains

Usage:
    encoder = DINOv2Encoder(device="cuda")
    obs_embed = encoder(rgb_tensor)   # (B, 3, 224, 224) → (B, 512)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from torchvision import transforms

logger = logging.getLogger(__name__)


# ImageNet normalisation constants (DINOv2 was pre-trained with these)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


class DINOv2Encoder(nn.Module):
    """
    Frozen DINOv2 ViT-B/14 encoder that maps RGB images to 512-dim embeddings.

    The encoder is always in eval() mode and gradients are disabled. It is
    designed to be instantiated once and shared across all parallel environments.

    Args:
        device:     Torch device string, e.g. "cuda" or "cpu".
        normalize:  If True (default), apply ImageNet normalisation before
                    forwarding through DINOv2. Set False only if your data
                    pipeline already normalises.
        hub_source: Where torch.hub loads the model from. Default "facebookresearch/dinov2".

    Attributes:
        embed_dim (int): Output embedding dimension = 512.
    """

    embed_dim: int = 512

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
            f"device={self.device}, frozen=True"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _freeze(self) -> None:
        """Disable gradients and lock batch-norm stats."""
        for param in self._backbone.parameters():
            param.requires_grad = False
        self._backbone.eval()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of RGB images into 512-dim DINOv2 CLS embeddings.

        Args:
            rgb: Float tensor of shape (B, 3, H, W), pixel values in [0, 1].
                 H and W must be divisible by 14 (patch size); use 224x224.

        Returns:
            Tensor of shape (B, 512) — the CLS token from the last ViT block.

        Notes:
            - This method is always called with torch.no_grad() context.
            - Normalisation is applied here, so the raw [0,1] tensor is expected.
        """
        # Shape guard
        if rgb.dim() != 4 or rgb.shape[1] != 3:
            raise ValueError(
                f"Expected rgb shape (B, 3, H, W), got {tuple(rgb.shape)}"
            )

        rgb = rgb.to(self.device)

        # Apply ImageNet normalisation
        rgb = self._norm(rgb)

        # DINOv2 forward_features returns a dict; 'x_norm_clstoken' is CLS
        features = self._backbone.forward_features(rgb)
        cls_token: torch.Tensor = features["x_norm_clstoken"]  # (B, 512)

        return cls_token
    
    
    @torch.no_grad()
    def get_patch_embeddings(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Extract dense spatial patch tokens from a batch of RGB images.

        Args:
            rgb: Float tensor of shape (B, 3, H, W), pixel values in [0, 1].
                 H and W must be divisible by 14.

        Returns:
            Tensor of shape (B, Num_Patches, 768) where Num_Patches = (H/14) * (W/14).
        """
        if rgb.dim() != 4 or rgb.shape[1] != 3:
            raise ValueError(
                f"Expected rgb shape (B, 3, H, W), got {tuple(rgb.shape)}"
            )

        rgb = rgb.to(self.device)

        # Leverage the encoder's internal ImageNet normalization
        rgb = self._norm(rgb)

        # Extract the raw patch features instead of the CLS token
        features = self._backbone.forward_features(rgb)
        patch_tokens: torch.Tensor = features["x_norm_patchtokens"]

        return patch_tokens
    

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
            f"frozen=True, device={self.device})"
        )
