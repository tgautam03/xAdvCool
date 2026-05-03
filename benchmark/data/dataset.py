"""PyTorch Dataset for xAdvCool 3D conjugate heat transfer data."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .h5_utils import resolve_h5_paths, build_shard_index
from .normalization import NormStats, normalize

BUFFER = 10  # inlet/outlet buffer cells on x-axis

# Scalar columns used as model inputs
INPUT_SCALAR_COLS = ["feature_size", "spacing", "tau_fluid", "u_inlet_val", "heat_power"]

# Scalar targets (engineering metrics)
TARGET_SCALAR_COLS = [
    "R_th", "Nu", "f_friction", "P_pump", "COP", "Q_vol",
    "T_max_surface", "T_avg_surface", "delta_T_max", "sigma_T", "fluid_temp_rise",
]


def _load_field_from_h5(grp: h5py.Group, fname: str, crop_buffer: bool) -> np.ndarray:
    """Read a single field from HDF5, crop, and reshape to (C, D, H, W)."""
    arr = grp[fname][()].astype(np.float32)
    if crop_buffer:
        arr = arr[BUFFER:-BUFFER]
    if arr.ndim == 4:  # velocity-like (X, Y, Z, C) → (C, X, Y, Z)
        arr = arr.transpose(3, 0, 1, 2)
    else:  # scalar (X, Y, Z) → (1, X, Y, Z)
        arr = arr[np.newaxis]
    return arr


class CHTDataset(Dataset):
    """HDF5-backed dataset for 3D conjugate heat transfer fields.

    Supports two tasks:
      - "field_prediction": input fields + scalars → output 3D fields
      - "scalar_prediction": input fields + scalars → engineering scalar metrics

    Supports sharded HDF5: pass a list of paths or a glob pattern.

    Set ``cache_dir`` to enable a one-time preprocessing cache: each sample
    is decompressed, cropped, normalized, and saved as an individual ``.pt``
    file. Subsequent epochs load from the cache (~10× faster than gzip HDF5).
    """

    def __init__(
        self,
        h5_path: str | list[str],
        metadata_path: str,
        sample_ids: list[str],
        task: str = "field_prediction",
        input_fields: tuple[str, ...] = ("mask", "heat_source"),
        target_fields: tuple[str, ...] = ("velocity", "temperature", "pressure"),
        scalar_input_cols: tuple[str, ...] = INPUT_SCALAR_COLS,
        scalar_target_cols: tuple[str, ...] = TARGET_SCALAR_COLS,
        crop_buffer: bool = True,
        norm_stats: Optional[NormStats] = None,
        cache_dir: Optional[str] = None,
    ):
        self.h5_paths = resolve_h5_paths(h5_path)
        self.task = task
        self.input_fields = input_fields
        self.target_fields = target_fields
        self.scalar_input_cols = list(scalar_input_cols)
        self.scalar_target_cols = list(scalar_target_cols)
        self.crop_buffer = crop_buffer
        self.norm_stats = norm_stats

        # Build shard index: sample_id → which h5 file
        self._shard_index = build_shard_index(self.h5_paths)

        # Load metadata and filter to requested sample_ids that exist in shards
        df = pd.read_parquet(metadata_path)
        valid_ids = set(sample_ids) & set(self._shard_index.keys())
        df = df[df["sample_id"].isin(valid_ids)].reset_index(drop=True)
        self.sample_ids = df["sample_id"].tolist()
        self.scalar_inputs = df[self.scalar_input_cols].values.astype(np.float32)
        self.scalar_targets = df[self.scalar_target_cols].values.astype(np.float32)

        # Map each sample to its shard path for fast lookup
        self._sample_shards = [self._shard_index[sid] for sid in self.sample_ids]

        # Preprocessing cache
        self._cache_dir: Optional[Path] = None
        if cache_dir is not None:
            self._cache_dir = Path(cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            self._build_cache()

        # Lazy HDF5 handles per shard (only needed when cache is off)
        self._h5_handles: dict[str, h5py.File] = {}

    def _cache_path(self, sid: str) -> Path:
        return self._cache_dir / f"{sid}.pt"

    def _build_cache(self) -> None:
        """One-time pass: decompress HDF5 → .pt files."""
        missing = [i for i, sid in enumerate(self.sample_ids)
                    if not self._cache_path(sid).exists()]
        if not missing:
            return

        print(f"Caching {len(missing)} samples to {self._cache_dir} ...")
        # Group by shard for sequential HDF5 reads
        shard_groups: dict[str, list[int]] = {}
        for i in missing:
            path = self._sample_shards[i]
            shard_groups.setdefault(path, []).append(i)

        for path, indices in shard_groups.items():
            with h5py.File(path, "r") as f:
                for i in tqdm(indices, desc=f"Caching {Path(path).name}", leave=False):
                    sid = self.sample_ids[i]
                    grp = f[sid]
                    entry = self._process_sample(grp, i)
                    torch.save(entry, self._cache_path(sid))

    def _process_sample(self, grp: h5py.Group, idx: int) -> dict:
        """Read, crop, normalize a single sample and return as tensors."""
        input_channels = []
        for fname in self.input_fields:
            arr = _load_field_from_h5(grp, fname, self.crop_buffer)
            t = torch.from_numpy(arr)
            if self.norm_stats and fname in self.norm_stats.mean:
                t = normalize(t, fname, self.norm_stats)
            input_channels.append(t)

        entry = {
            "input_fields": torch.cat(input_channels, dim=0),
            "scalar_params": torch.from_numpy(self.scalar_inputs[idx].copy()),
        }

        if self.task == "field_prediction":
            target_channels = []
            for fname in self.target_fields:
                arr = _load_field_from_h5(grp, fname, self.crop_buffer)
                t = torch.from_numpy(arr)
                if self.norm_stats and fname in self.norm_stats.mean:
                    t = normalize(t, fname, self.norm_stats)
                target_channels.append(t)
            entry["target_fields"] = torch.cat(target_channels, dim=0)
        elif self.task == "scalar_prediction":
            entry["target_scalars"] = torch.from_numpy(self.scalar_targets[idx].copy())

        return entry

    def _get_h5(self, path: str) -> h5py.File:
        if path not in self._h5_handles:
            self._h5_handles[path] = h5py.File(path, "r")
        return self._h5_handles[path]

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        sid = self.sample_ids[idx]

        # --- Fast path: load from .pt cache ---
        if self._cache_dir is not None:
            entry = torch.load(self._cache_path(sid), weights_only=True)
            input_dict = {
                "fields": entry["input_fields"],
                "scalar_params": entry["scalar_params"],
            }
            if self.task == "field_prediction":
                target_dict = {"fields": entry["target_fields"]}
            else:
                target_dict = {"scalars": entry["target_scalars"]}
            return input_dict, target_dict

        # --- Slow path: read from HDF5 (gzip decompression) ---
        h5 = self._get_h5(self._sample_shards[idx])
        grp = h5[sid]

        input_channels = []
        for fname in self.input_fields:
            arr = _load_field_from_h5(grp, fname, self.crop_buffer)
            t = torch.from_numpy(arr)
            if self.norm_stats and fname in self.norm_stats.mean:
                t = normalize(t, fname, self.norm_stats)
            input_channels.append(t)

        input_dict = {
            "fields": torch.cat(input_channels, dim=0),
            "scalar_params": torch.from_numpy(self.scalar_inputs[idx]),
        }

        target_dict = {}
        if self.task == "field_prediction":
            target_channels = []
            for fname in self.target_fields:
                arr = _load_field_from_h5(grp, fname, self.crop_buffer)
                t = torch.from_numpy(arr)
                if self.norm_stats and fname in self.norm_stats.mean:
                    t = normalize(t, fname, self.norm_stats)
                target_channels.append(t)
            target_dict["fields"] = torch.cat(target_channels, dim=0)
        elif self.task == "scalar_prediction":
            target_dict["scalars"] = torch.from_numpy(self.scalar_targets[idx])
        else:
            raise ValueError(f"Unknown task: {self.task}")

        return input_dict, target_dict

    def __del__(self):
        for h5 in self._h5_handles.values():
            h5.close()
