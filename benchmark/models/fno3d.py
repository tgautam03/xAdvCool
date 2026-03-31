"""FNO-3D model for field prediction.

Uses the neuraloperator library if available, otherwise provides a
self-contained implementation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseModel, register_model


class SpectralConv3d(nn.Module):
    """3D Fourier layer: applies learnable weights in frequency space."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int, modes3: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat))

    def _compl_mul3d(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # (B, in, m1, m2, m3), (in, out, m1, m2, m3) -> (B, out, m1, m2, m3)
        return torch.einsum("bixyz,ioxyz->boxyz", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # FFT
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        m1, m2, m3 = self.modes1, self.modes2, self.modes3
        out_ft = torch.zeros(B, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)

        out_ft[:, :, :m1, :m2, :m3] = self._compl_mul3d(x_ft[:, :, :m1, :m2, :m3], self.weights1)
        out_ft[:, :, -m1:, :m2, :m3] = self._compl_mul3d(x_ft[:, :, -m1:, :m2, :m3], self.weights2)
        out_ft[:, :, :m1, -m2:, :m3] = self._compl_mul3d(x_ft[:, :, :m1, -m2:, :m3], self.weights3)
        out_ft[:, :, -m1:, -m2:, :m3] = self._compl_mul3d(x_ft[:, :, -m1:, -m2:, :m3], self.weights4)

        return torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))


class FNOBlock3d(nn.Module):
    """Single FNO layer: spectral conv + linear bypass + activation."""

    def __init__(self, width: int, modes1: int, modes2: int, modes3: int):
        super().__init__()
        self.spectral = SpectralConv3d(width, width, modes1, modes2, modes3)
        self.linear = nn.Conv3d(width, width, 1)
        self.norm = nn.GroupNorm(min(8, width), width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.spectral(x) + self.linear(x)))


@register_model
class FNO3D(BaseModel):
    """Fourier Neural Operator for 3D field-to-field prediction.

    Scalar parameters are broadcast-concatenated as extra input channels.
    """

    name = "fno3d"

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 5,
        scalar_dim: int = 5,
        width: int = 32,
        modes: tuple[int, ...] = (12, 12, 8),
        n_layers: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.scalar_dim = scalar_dim
        total_in = in_channels + scalar_dim  # scalars broadcast as channels

        # Lifting
        self.lift = nn.Conv3d(total_in, width, 1)

        # Fourier layers
        self.layers = nn.ModuleList([
            FNOBlock3d(width, modes[0], modes[1], modes[2])
            for _ in range(n_layers)
        ])

        # Projection
        self.proj = nn.Sequential(
            nn.Conv3d(width, width * 2, 1),
            nn.GELU(),
            nn.Conv3d(width * 2, out_channels, 1),
        )

    def forward(self, input_dict: dict) -> dict:
        x = input_dict["fields"]  # (B, C_in, D, H, W)
        scalars = input_dict.get("scalar_params")  # (B, S)

        # Broadcast scalars as spatial channels
        if scalars is not None:
            B, _, D, H, W = x.shape
            s = scalars.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, S, 1, 1, 1)
            s = s.expand(-1, -1, D, H, W)  # (B, S, D, H, W)
            x = torch.cat([x, s], dim=1)  # (B, C_in + S, D, H, W)

        x = self.lift(x)
        for layer in self.layers:
            x = layer(x)
        x = self.proj(x)

        return {"fields": x}
