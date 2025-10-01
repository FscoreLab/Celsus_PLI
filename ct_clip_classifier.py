"""
CT-CLIP Classifier for Inference
Simple classification head for CT-CLIP fine-tuning and inference.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn


class SimpleCTCLIPClassifier(nn.Module):
    """Simple classification head for CT-CLIP fine-tuning."""

    def __init__(
        self,
        ct_clip_model,
        num_classes: int = 15,
        dropout: float = 0.3,
        latent_norm: str = "none",
        reinit_head: bool = False,
        head_init_std: float = 0.01,
    ):
        super().__init__()
        self.ct_clip = ct_clip_model
        self.num_classes = num_classes

        self.latent_norm = (latent_norm or "none").lower()
        if self.latent_norm == "layernorm":
            self.pre_head_norm = nn.LayerNorm(512)
        else:
            self.pre_head_norm = nn.Identity()

        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, num_classes))

        self.tokenizer = None

    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> Dict:
        """Forward pass."""
        device = images.device

        # Get image embeddings from CT-CLIP
        # Create dummy text tokens (not used for classification)
        dummy_text = [""] * images.size(0)

        text_tokens = self.tokenizer(
            dummy_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512
        ).to(device)

        _, image_latents, _ = self.ct_clip(text_tokens, images, device=device, return_latents=True)

        if self.latent_norm == "l2":
            image_latents = torch.nn.functional.normalize(image_latents, dim=1)
        else:
            image_latents = self.pre_head_norm(image_latents)

        logits = self.classifier(image_latents)

        result = {"logits": logits}

        return result
