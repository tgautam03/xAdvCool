"""PyTorch Dataset for xAdvCool 3D conjugate heat transfer data."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

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


class CHTDataset(Dataset):
    """On-the-fly HDF5 dataset for 3D conjugate heat transfer fields.

    Supports two tasks:
      - "field_prediction": input fields + scalars → output 3D fields
      - "scalar_prediction": input fields + scalars → engineering scalar metrics

    Supports sharded HDF5: pass a list of paths or a glob pattern.
    HDF5 file handles are opened lazily per DataLoader worker.
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

        # Lazy HDF5 handles per shard (opened per worker)
        self._h5_handles: dict[str, h5py.File] = {}

    def _get_h5(self, path: str) -> h5py.File:
        if path not in self._h5_handles:
            self._h5_handles[path] = h5py.File(path, "r")
        return self._h5_handles[path]

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        sid = self.sample_ids[idx]
        h5 = self._get_h5(self._sample_shards[idx])
        grp = h5[sid]

        # --- Load and preprocess input fields ---
        input_channels = []
        for fname in self.input_fields:
            arr = grp[fname][()].astype(np.float32)
            if self.crop_buffer:
                arr = arr[BUFFER:-BUFFER]
            if arr.ndim == 4:  # velocity-like (X, Y, Z, C) → (C, X, Y, Z)
                arr = arr.transpose(3, 0, 1, 2)
            else:  # scalar (X, Y, Z) → (1, X, Y, Z)
                arr = arr[np.newaxis]
            t = torch.from_numpy(arr)
            if self.norm_stats and fname in self.norm_stats.mean:
                t = normalize(t, fname, self.norm_stats)
            input_channels.append(t)

        input_dict = {
            "fields": torch.cat(input_channels, dim=0),  # (C_in, D, H, W)
            "scalar_params": torch.from_numpy(self.scalar_inputs[idx]),  # (S,)
        }

        # --- Load targets ---
        target_dict = {}
        if self.task == "field_prediction":
            target_channels = []
            for fname in self.target_fields:
                arr = grp[fname][()].astype(np.float32)
                if self.crop_buffer:
                    arr = arr[BUFFER:-BUFFER]
                if arr.ndim == 4:
                    arr = arr.transpose(3, 0, 1, 2)
                else:
                    arr = arr[np.newaxis]
                t = torch.from_numpy(arr)
                if self.norm_stats and fname in self.norm_stats.mean:
                    t = normalize(t, fname, self.norm_stats)
                target_channels.append(t)
            target_dict["fields"] = torch.cat(target_channels, dim=0)  # (C_out, D, H, W)
        elif self.task == "scalar_prediction":
            target_dict["scalars"] = torch.from_numpy(self.scalar_targets[idx])  # (N,)
        else:
            raise ValueError(f"Unknown task: {self.task}")

        return input_dict, target_dict

    def __del__(self):
        for h5 in self._h5_handles.values():
            h5.close()
