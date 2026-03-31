"""Trivial baselines: MeanBaseline and MLPBaseline."""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseModel, register_model


@register_model
class MeanBaseline(BaseModel):
    """Returns a learned constant field (initialized to zero, trained to match mean)."""

    name = "mean_baseline"

    def __init__(self, out_channels: int = 5, grid_shape: tuple = (128, 128, 32), **kwargs):
        super().__init__()
        self.mean_field = nn.Parameter(torch.zeros(1, out_channels, *grid_shape))

    def forward(self, input_dict: dict) -> dict:
        B = input_dict["fields"].shape[0]
        return {"fields": self.mean_field.expand(B, -1, -1, -1, -1)}


@register_model
class MLPBaseline(BaseModel):
    """Global-average-pool input fields + scalar params → MLP → output.

    For field prediction: broadcasts MLP output back to volume.
    For scalar prediction: MLP directly outputs scalars.
    """

    name = "mlp_baseline"

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 5,
        scalar_dim: int = 5,
        hidden_dim: int = 256,
        task: str = "field_prediction",
        grid_shape: tuple = (128, 128, 32),
        n_scalar_targets: int = 11,
        **kwargs,
    ):
        super().__init__()
        self.task = task
        self.grid_shape = grid_shape

        feat_dim = in_channels + scalar_dim

        if task == "field_prediction":
            self.mlp = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_channels),
            )
            self.out_channels = out_channels
        else:
            self.mlp = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, n_scalar_targets),
            )

    def forward(self, input_dict: dict) -> dict:
        fields = input_dict["fields"]  # (B, C_in, D, H, W)
        scalars = input_dict.get("scalar_params")  # (B, S) or None

        # Global average pool over spatial dims
        pooled = fields.mean(dim=(2, 3, 4))  # (B, C_in)
        if scalars is not None:
            pooled = torch.cat([pooled, scalars], dim=1)  # (B, C_in + S)

        out = self.mlp(pooled)  # (B, C_out) or (B, N_scalars)

        if self.task == "field_prediction":
            # Broadcast to volume
            out = out.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (B, C_out, 1, 1, 1)
            out = out.expand(-1, -1, *self.grid_shape)  # (B, C_out, D, H, W)
            return {"fields": out}
        else:
            return {"scalars": out}
