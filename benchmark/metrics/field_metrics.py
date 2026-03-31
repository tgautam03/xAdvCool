"""Field-level evaluation metrics."""

from __future__ import annotations

import torch


def relative_l2_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample relative L2 error: ||pred - target||_2 / ||target||_2.

    Args:
        pred, target: (B, C, D, H, W) tensors.
    Returns:
        (B,) tensor of relative L2 errors.
    """
    diff = (pred - target).reshape(pred.shape[0], -1)
    tgt = target.reshape(target.shape[0], -1)
    return diff.norm(dim=1) / tgt.norm(dim=1).clamp(min=1e-8)


def per_channel_relative_l2(
    pred: torch.Tensor,
    target: torch.Tensor,
    channel_names: list[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Relative L2 error per output channel.

    Args:
        pred, target: (B, C, D, H, W) tensors.
        channel_names: Optional names for each channel.
    Returns:
        Dict mapping channel name → (B,) tensor of per-sample errors.
    """
    C = pred.shape[1]
    if channel_names is None:
        channel_names = [f"channel_{i}" for i in range(C)]

    results = {}
    for i, name in enumerate(channel_names):
        p = pred[:, i : i + 1]  # (B, 1, D, H, W)
        t = target[:, i : i + 1]
        results[name] = relative_l2_error(p, t)
    return results


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample RMSE averaged over all channels and spatial dims.

    Args:
        pred, target: (B, C, D, H, W) tensors.
    Returns:
        (B,) tensor.
    """
    diff_sq = (pred - target) ** 2
    return diff_sq.reshape(pred.shape[0], -1).mean(dim=1).sqrt()
