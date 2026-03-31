"""Per-channel normalization statistics for field data."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import torch

from .h5_utils import resolve_h5_paths, build_shard_index

BUFFER = 10  # inlet/outlet buffer cells on x-axis


@dataclass
class NormStats:
    """Per-channel mean and std for each field."""

    mean: dict[str, np.ndarray] = field(default_factory=dict)
    std: dict[str, np.ndarray] = field(default_factory=dict)

    def save(self, path: str) -> None:
        data = {
            "mean": {k: v.tolist() for k, v in self.mean.items()},
            "std": {k: v.tolist() for k, v in self.std.items()},
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> NormStats:
        data = json.loads(Path(path).read_text())
        return cls(
            mean={k: np.array(v, dtype=np.float32) for k, v in data["mean"].items()},
            std={k: np.array(v, dtype=np.float32) for k, v in data["std"].items()},
        )


def compute_norm_stats(
    h5_path: str | list[str],
    sample_ids: list[str],
    fields: tuple[str, ...] = ("velocity", "temperature", "pressure", "mask", "heat_source"),
    max_samples: int = 200,
    cache_dir: str = "benchmark",
) -> NormStats:
    """Compute per-channel mean/std from a subset of training samples.

    Supports sharded HDF5 files. Results are cached to disk.
    """
    # Check cache
    id_hash = hashlib.md5("_".join(sorted(sample_ids)).encode()).hexdigest()[:12]
    cache_path = Path(cache_dir) / f"norm_stats_{id_hash}.json"
    if cache_path.exists():
        return NormStats.load(str(cache_path))

    # Build shard index
    h5_paths = resolve_h5_paths(h5_path)
    shard_index = build_shard_index(h5_paths)

    # Filter to sample_ids that exist in shards
    valid_ids = [sid for sid in sample_ids if sid in shard_index]

    rng = np.random.RandomState(42)
    subset = rng.choice(valid_ids, size=min(max_samples, len(valid_ids)), replace=False)

    # Accumulate per-sample statistics
    sample_means: dict[str, list[np.ndarray]] = {f: [] for f in fields}
    sample_vars: dict[str, list[np.ndarray]] = {f: [] for f in fields}

    # Group samples by shard for efficient I/O
    shard_groups: dict[str, list[str]] = {}
    for sid in subset:
        path = shard_index[sid]
        shard_groups.setdefault(path, []).append(sid)

    for path, sids in shard_groups.items():
        with h5py.File(path, "r") as f:
            for sid in sids:
                grp = f[sid]
                for fname in fields:
                    arr = grp[fname][()].astype(np.float32)
                    arr = arr[BUFFER:-BUFFER]

                    if arr.ndim == 4:  # velocity
                        flat = arr.reshape(-1, arr.shape[-1])  # (N, 3)
                        sample_means[fname].append(flat.mean(axis=0))
                        sample_vars[fname].append(flat.var(axis=0))
                    else:
                        sample_means[fname].append(np.array([arr.mean()]))
                        sample_vars[fname].append(np.array([arr.var()]))

    stats = NormStats()
    for fname in fields:
        sm = np.stack(sample_means[fname])  # (n_samples, n_channels)
        sv = np.stack(sample_vars[fname])
        # Global mean = mean of sample means (exact when sample sizes are equal)
        global_mean = sm.mean(axis=0).astype(np.float32)
        # Global var = mean of within-sample vars + var of sample means
        global_var = sv.mean(axis=0) + sm.var(axis=0)
        global_std = np.sqrt(global_var).astype(np.float32)
        global_std = np.maximum(global_std, 1e-8)
        stats.mean[fname] = global_mean
        stats.std[fname] = global_std

    stats.save(str(cache_path))
    return stats


def normalize(field: torch.Tensor, field_name: str, stats: NormStats) -> torch.Tensor:
    """Normalize a field tensor using precomputed stats.

    Args:
        field: (C, D, H, W) or (D, H, W) tensor.
        field_name: Key into stats.mean/std.
        stats: Precomputed normalization statistics.
    """
    mean = torch.tensor(stats.mean[field_name], dtype=field.dtype, device=field.device)
    std = torch.tensor(stats.std[field_name], dtype=field.dtype, device=field.device)
    if field.ndim == 4:  # (C, D, H, W)
        mean = mean.view(-1, 1, 1, 1)
        std = std.view(-1, 1, 1, 1)
    elif field.ndim == 3:  # (D, H, W) — single channel
        mean = mean.squeeze()
        std = std.squeeze()
    return (field - mean) / std


def denormalize(field: torch.Tensor, field_name: str, stats: NormStats) -> torch.Tensor:
    """Inverse of normalize."""
    mean = torch.tensor(stats.mean[field_name], dtype=field.dtype, device=field.device)
    std = torch.tensor(stats.std[field_name], dtype=field.dtype, device=field.device)
    if field.ndim == 4:
        mean = mean.view(-1, 1, 1, 1)
        std = std.view(-1, 1, 1, 1)
    elif field.ndim == 3:
        mean = mean.squeeze()
        std = std.squeeze()
    return field * std + mean
