"""3D U-Net with FiLM conditioning for scalar parameter injection."""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseModel, register_model


class ConvBlock3D(nn.Module):
    """Two conv3d layers with GroupNorm and GELU."""

    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(groups, out_ch), out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(groups, out_ch), out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: x = gamma * x + beta."""

    def __init__(self, scalar_dim: int, n_features: int, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(scalar_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_features * 2),
        )

    def forward(self, x: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        params = self.mlp(scalars)  # (B, 2*C)
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + 1.0  # residual
        beta = beta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return gamma * x + beta


@register_model
class UNet3D(BaseModel):
    """3D U-Net encoder-decoder with skip connections and FiLM conditioning.

    FiLM is applied at the bottleneck to inject scalar parameters.
    """

    name = "unet3d"

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 5,
        scalar_dim: int = 5,
        features: tuple[int, ...] = (32, 64, 128, 256),
        **kwargs,
    ):
        super().__init__()
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool3d(2)
        self.decoder_blocks = nn.ModuleList()
        self.upconvs = nn.ModuleList()

        # Encoder
        prev_ch = in_channels
        for feat in features:
            self.encoder_blocks.append(ConvBlock3D(prev_ch, feat))
            prev_ch = feat

        # Bottleneck
        self.bottleneck = ConvBlock3D(features[-1], features[-1] * 2)

        # FiLM conditioning at bottleneck
        self.film = FiLMLayer(scalar_dim, features[-1] * 2)

        # Decoder
        prev_ch = features[-1] * 2
        for feat in reversed(features):
            self.upconvs.append(nn.ConvTranspose3d(prev_ch, feat, kernel_size=2, stride=2))
            self.decoder_blocks.append(ConvBlock3D(feat * 2, feat))  # *2 for skip
            prev_ch = feat

        # Output
        self.out_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, input_dict: dict) -> dict:
        x = input_dict["fields"]  # (B, C_in, D, H, W)
        scalars = input_dict.get("scalar_params")  # (B, S) or None

        # Encoder path
        skips = []
        for enc in self.encoder_blocks:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # FiLM conditioning
        if scalars is not None:
            x = self.film(x, scalars)

        # Decoder path
        for upconv, dec, skip in zip(self.upconvs, self.decoder_blocks, reversed(skips)):
            x = upconv(x)
            # Handle size mismatch from non-power-of-2 dims (128,128,32)
            if x.shape != skip.shape:
                x = nn.functional.pad(
                    x,
                    [0, skip.shape[4] - x.shape[4],
                     0, skip.shape[3] - x.shape[3],
                     0, skip.shape[2] - x.shape[2]],
                )
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return {"fields": self.out_conv(x)}
