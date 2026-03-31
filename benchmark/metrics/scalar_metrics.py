"""Scalar output evaluation metrics."""

from __future__ import annotations

import torch


def scalar_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """RMSE per scalar output.

    Args:
        pred, target: (B, N) tensors.
    Returns:
        (N,) tensor of per-output RMSE values.
    """
    return ((pred - target) ** 2).mean(dim=0).sqrt()


def scalar_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MAE per scalar output.

    Args:
        pred, target: (B, N) tensors.
    Returns:
        (N,) tensor.
    """
    return (pred - target).abs().mean(dim=0)


def scalar_r2(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """R² (coefficient of determination) per scalar output.

    Args:
        pred, target: (B, N) tensors.
    Returns:
        (N,) tensor.
    """
    ss_res = ((target - pred) ** 2).sum(dim=0)
    ss_tot = ((target - target.mean(dim=0, keepdim=True)) ** 2).sum(dim=0)
    return 1.0 - ss_res / ss_tot.clamp(min=1e-8)
