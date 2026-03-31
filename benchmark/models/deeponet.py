"""DeepONet: branch net (CNN on input fields) + trunk net (MLP on coordinates)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseModel, register_model


class BranchNet3D(nn.Module):
    """CNN encoder that maps 3D input fields to a latent vector."""

    def __init__(self, in_channels: int, scalar_dim: int, latent_dim: int, out_channels: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(in_channels, 32, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv3d(64, 128, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv3d(128, 128, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 + scalar_dim, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim * out_channels),
        )
        self.latent_dim = latent_dim
        self.out_channels = out_channels

    def forward(self, fields: torch.Tensor, scalars: torch.Tensor | None) -> torch.Tensor:
        """Returns (B, out_channels, latent_dim)."""
        x = self.cnn(fields).squeeze(-1).squeeze(-1).squeeze(-1)  # (B, 128)
        if scalars is not None:
            x = torch.cat([x, scalars], dim=1)
        x = self.fc(x)  # (B, latent_dim * out_channels)
        return x.view(-1, self.out_channels, self.latent_dim)  # (B, C_out, P)


class TrunkNet(nn.Module):
    """MLP that maps (x, y, z) coordinates to latent features."""

    def __init__(self, latent_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords: (N, 3) → (N, P)."""
        return self.mlp(coords)


@register_model
class DeepONet3D(BaseModel):
    """DeepONet for 3D field prediction.

    Output at each grid point = dot(branch_output, trunk_output) + bias.
    The trunk net is evaluated on a fixed grid, so we precompute coordinates.
    """

    name = "deeponet"

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 5,
        scalar_dim: int = 5,
        latent_dim: int = 128,
        grid_shape: tuple[int, ...] = (128, 128, 32),
        **kwargs,
    ):
        super().__init__()
        self.grid_shape = grid_shape
        self.out_channels = out_channels

        self.branch = BranchNet3D(in_channels, scalar_dim, latent_dim, out_channels)
        self.trunk = TrunkNet(latent_dim)
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # Precompute normalized grid coordinates [0, 1]
        D, H, W = grid_shape
        zz, yy, xx = torch.meshgrid(
            torch.linspace(0, 1, D),
            torch.linspace(0, 1, H),
            torch.linspace(0, 1, W),
            indexing="ij",
        )
        coords = torch.stack([zz, yy, xx], dim=-1).reshape(-1, 3)  # (D*H*W, 3)
        self.register_buffer("coords", coords)

    def forward(self, input_dict: dict) -> dict:
        fields = input_dict["fields"]  # (B, C_in, D, H, W)
        scalars = input_dict.get("scalar_params")

        B = fields.shape[0]
        D, H, W = self.grid_shape
        N = D * H * W

        # Branch: (B, C_out, P)
        branch_out = self.branch(fields, scalars)

        # Trunk: (N, P) — same for all samples in batch
        trunk_out = self.trunk(self.coords)  # (N, P)

        # Dot product: (B, C_out, N) = (B, C_out, P) @ (P, N)
        output = torch.bmm(branch_out, trunk_out.T.unsqueeze(0).expand(B, -1, -1))

        # Add bias and reshape
        output = output + self.bias.view(1, -1, 1)
        output = output.view(B, self.out_channels, D, H, W)

        return {"fields": output}
