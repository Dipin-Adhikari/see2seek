"""
clip_encoder.py — Frozen CLIP ViT-B/32 goal encoder.

Design rationale:
    ZSON's zero-shot transfer from ImageNav to ObjectNav relies on the CLIP
    shared embedding space: image goals and text category names ("find a chair")
    are mapped to nearby points in the same latent space.

    Replacing this encoder with DINOv2 would break that alignment because DINOv2
    has no joint language-vision training. Therefore:
        - Observation encoder: DINOv2 (our contribution)
        - Goal encoder:        CLIP   (kept from ZSON)

    This encoder handles BOTH image goals (ImageNav) and text goals (ObjectNav
    zero-shot). The shared embedding space is what enables zero-shot transfer.

    For efficiency, goal embeddings are pre-cached at the start of training
    using tools/cache_goal_embeddings.py; this encoder is then called only
    during eval or for on-the-fly goals.

Usage:
    encoder = CLIPGoalEncoder(device="cuda")
    img_embed  = encoder.encode_image(pil_image)   # (1, 512)
    text_embed = encoder.encode_text("a chair")    # (1, 512)
"""

from __future__ import annotations

import logging
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


class CLIPGoalEncoder(nn.Module):
    """
    Frozen CLIP ViT-B/32 encoder for encoding image and text navigation goals.

    Internally uses open_clip (the community fork of OpenAI CLIP) which has
    the same architecture and weights but is easier to install.

    Args:
        model_name:  CLIP model variant. Default "ViT-B-32" (open_clip naming).
        pretrained:  Checkpoint name for open_clip. Default "openai" (original weights).
        device:      Torch device string.
        normalize:   L2-normalise embeddings (recommended for cosine similarity).

    Attributes:
        embed_dim (int): Output embedding dimension = 512.
    """

    embed_dim: int = 512

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "cuda",
        normalize: bool = True,
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self._normalize = normalize

        try:
            import open_clip
        except ImportError as e:
            raise ImportError(
                "open_clip_torch is required. Install with: pip install open_clip_torch"
            ) from e

        logger.info(f"Loading CLIP {model_name} ({pretrained}) via open_clip ...")

        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device,
        )
        self._tokenizer = open_clip.get_tokenizer(model_name)

        # Freeze everything
        self._freeze()

        logger.info(
            f"CLIPGoalEncoder ready — embed_dim={self.embed_dim}, "
            f"device={self.device}, frozen=True"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _freeze(self) -> None:
        for param in self._model.parameters():
            param.requires_grad = False
        self._model.eval()

    # ------------------------------------------------------------------
    # Public encoding methods
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_image(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        """
        Encode PIL image(s) into CLIP image embeddings.

        Args:
            images: A single PIL.Image or a list of PIL.Images.

        Returns:
            Tensor of shape (N, 512), optionally L2-normalised.
        """
        if isinstance(images, Image.Image):
            images = [images]

        # Apply CLIP's own preprocessing (resize, crop, normalise)
        batch = torch.stack([self._preprocess(img) for img in images]).to(self.device)

        embeddings = self._model.encode_image(batch)  # (N, 512)

        if self._normalize:
            embeddings = F.normalize(embeddings, dim=-1)

        return embeddings

    @torch.no_grad()
    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode natural-language goal string(s) into CLIP text embeddings.

        This is used during ObjectNav zero-shot evaluation where the goal is
        a category name like "chair" or "television".

        Can also be used for Nepali-language goals (e.g. "कुर्सी") since CLIP's
        multilingual text encoder generalises to non-English scripts.

        Args:
            texts: A single string or a list of strings.

        Returns:
            Tensor of shape (N, 512), optionally L2-normalised.
        """
        if isinstance(texts, str):
            texts = [texts]

        tokens = self._tokenizer(texts).to(self.device)
        embeddings = self._model.encode_text(tokens)  # (N, 512)

        if self._normalize:
            embeddings = F.normalize(embeddings, dim=-1)

        return embeddings

    @torch.no_grad()
    def encode_tensor_image(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Encode a pre-loaded RGB tensor (already preprocessed) into CLIP embeddings.

        Use this during training when the goal image tensor is already on GPU
        (avoids PIL round-trip). The tensor must already be CLIP-preprocessed
        (i.e. 224x224, ImageNet-normalised).

        Args:
            rgb: Tensor of shape (B, 3, 224, 224).

        Returns:
            Tensor of shape (B, 512).
        """
        rgb = rgb.to(self.device)
        embeddings = self._model.encode_image(rgb)

        if self._normalize:
            embeddings = F.normalize(embeddings, dim=-1)

        return embeddings

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def train(self, mode: bool = True) -> "CLIPGoalEncoder":
        super().train(mode)
        self._model.eval()   # backbone stays eval
        return self

    def __repr__(self) -> str:
        return (
            f"CLIPGoalEncoder(model=ViT-B/32, embed_dim={self.embed_dim}, "
            f"frozen=True, device={self.device})"
        )
