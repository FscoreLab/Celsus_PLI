"""3D classification network built on MedNeXt blocks."""
from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn

from mednext.mednext_blocks import MedNeXtBlock, MedNeXtDownBlock


class MedNeXt3DClassifier(nn.Module):
    """A MedNeXt-based encoder tailored for volumetric classification.

    The network repeatedly applies MedNeXt blocks followed by down-sampling
    blocks, producing a compact latent representation that is pooled and fed
    to a linear classifier. The default configuration is sized for volumes of
    ``320 x 320 x 224`` but supports other inputs with spatial sizes that are
    divisible by ``2 ** (len(depth_config) - 1)``.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_channels: int = 32,
        depth_config: Sequence[int] | int = (2, 2, 2, 2, 2),
        exp_r: Sequence[int] | int = 4,
        kernel_size: int = 3,
        norm_type: str = "group",
        grn: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if isinstance(depth_config, int):
            depth_config = [depth_config] * 4
        else:
            depth_config = list(depth_config)

        if isinstance(exp_r, int):
            exp_r = [exp_r] * len(depth_config)
        else:
            exp_r = list(exp_r)

        if len(exp_r) != len(depth_config):
            raise ValueError(
                "exp_r must be an int or a sequence with the same length as depth_config"
            )

        self.stem = nn.Conv3d(in_channels, base_channels, kernel_size=3, stride=2, padding=0)

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_ch = base_channels
        for stage_idx, num_blocks in enumerate(depth_config):
            blocks: List[nn.Module] = []
            for _ in range(num_blocks):
                blocks.append(
                    MedNeXtBlock(
                        in_channels=in_ch,
                        out_channels=in_ch,
                        exp_r=exp_r[stage_idx],
                        kernel_size=kernel_size,
                        do_res=True,
                        norm_type=norm_type,
                        dim="3d",
                        grn=grn,
                    )
                )
            self.stages.append(nn.Sequential(*blocks))

            is_last_stage = stage_idx == len(depth_config) - 1
            if not is_last_stage:
                out_ch = in_ch * 2
                self.downsamples.append(
                    MedNeXtDownBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        exp_r=exp_r[stage_idx + 1],
                        kernel_size=kernel_size,
                        do_res=True,
                        norm_type=norm_type,
                        dim="3d",
                        grn=grn,
                    )
                )
                in_ch = out_ch

        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = nn.Flatten(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_ch, num_classes),
        )

        self.num_features = in_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage_idx, stage in enumerate(self.stages):
            x = stage(x)
            if stage_idx < len(self.downsamples):
                x = self.downsamples[stage_idx](x)
        x = self.global_pool(x)
        x = self.flatten(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters in ``model``."""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        model = MedNeXt3DClassifier(in_channels=1, num_classes=16).to(device)
        print(f"Trainable parameters: {count_parameters(model):,}")
        dummy_input = torch.randn(1, 1, 320, 320, 224, device=device)
        output = model(dummy_input)
        print(f"Output shape: {tuple(output.shape)}")
        print(f"Trainable parameters: {count_parameters(model):,}")
