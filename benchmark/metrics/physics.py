"""Physics-consistency metrics for predicted fields."""

from __future__ import annotations

import torch


def mass_conservation_violation(
    velocity: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean absolute divergence of velocity in fluid cells.

    Computes div(u) = du_x/dx + du_y/dy + du_z/dz using central finite differences.

    Args:
        velocity: (B, 3, D, H, W) predicted velocity field.
        mask: (B, 1, D, H, W) domain mask (0 = fluid).
    Returns:
        (B,) mean |div(u)| over fluid cells per sample.
    """
    ux = velocity[:, 0]  # (B, D, H, W)
    uy = velocity[:, 1]
    uz = velocity[:, 2]

    # Central differences (interior points only)
    dux_dx = (ux[:, 2:, 1:-1, 1:-1] - ux[:, :-2, 1:-1, 1:-1]) / 2.0
    duy_dy = (uy[:, 1:-1, 2:, 1:-1] - uy[:, 1:-1, :-2, 1:-1]) / 2.0
    duz_dz = (uz[:, 1:-1, 1:-1, 2:] - uz[:, 1:-1, 1:-1, :-2]) / 2.0

    div = dux_dx + duy_dy + duz_dz  # (B, D-2, H-2, W-2)

    # Mask to fluid cells only (mask == 0 is fluid)
    fluid_mask = (mask[:, 0, 1:-1, 1:-1, 1:-1] == 0).float()
    n_fluid = fluid_mask.sum(dim=(1, 2, 3)).clamp(min=1.0)

    abs_div = (div.abs() * fluid_mask).sum(dim=(1, 2, 3)) / n_fluid
    return abs_div
