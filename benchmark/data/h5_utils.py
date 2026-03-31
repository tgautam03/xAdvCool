"""HDF5 shard utilities shared across data modules."""

from __future__ import annotations

from pathlib import Path

import h5py


def resolve_h5_paths(h5_path: str | list[str]) -> list[str]:
    """Resolve HDF5 path(s) — supports a single file, a list of files,
    or a glob pattern (e.g., 'dataset/shards/*.h5')."""
    if isinstance(h5_path, list):
        return h5_path
    p = Path(h5_path)
    if p.is_file():
        return [str(p)]
    # Try glob from parent
    matches = sorted(p.parent.glob(p.name))
    if matches:
        return [str(m) for m in matches]
    return [h5_path]  # fallback, will fail at open time with a clear error


def build_shard_index(h5_paths: list[str]) -> dict[str, str]:
    """Build a mapping from sample_id → h5_path across all shards."""
    index = {}
    for path in h5_paths:
        with h5py.File(path, "r") as f:
            for key in f.keys():
                index[key] = path
    return index
