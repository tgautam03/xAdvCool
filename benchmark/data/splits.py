"""Deterministic train/val/test splits for the xAdvCool benchmark."""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from .h5_utils import resolve_h5_paths


def _valid_sample_ids(metadata_path: str, h5_path: str | list[str]) -> pd.DataFrame:
    """Return metadata rows whose sample_id exists in HDF5 file(s) and converged."""
    df = pd.read_parquet(metadata_path)
    df = df[df["converged"] == True].reset_index(drop=True)  # noqa: E712

    h5_paths = resolve_h5_paths(h5_path)
    h5_keys: set[str] = set()
    for path in h5_paths:
        with h5py.File(path, "r") as f:
            h5_keys.update(f.keys())

    df = df[df["sample_id"].isin(h5_keys)].reset_index(drop=True)
    return df


def _subsample(
    splits: dict[str, list[str]],
    max_samples: int | None,
    seed: int,
) -> dict[str, list[str]]:
    """Subsample the training split (val/test kept intact for fair comparison)."""
    if max_samples is None or max_samples <= 0:
        return splits
    rng = np.random.RandomState(seed)
    result = dict(splits)
    if len(result["train"]) > max_samples:
        result["train"] = rng.choice(
            result["train"], size=max_samples, replace=False
        ).tolist()
    return result


def get_canonical_splits(
    metadata_path: str = "dataset/metadata.parquet",
    h5_path: str | list[str] = "dataset/data.h5",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    max_samples: int | None = None,
) -> dict[str, list[str]]:
    """70/15/15 split stratified by design_name.

    Args:
        max_samples: If set, subsample the training set to this many samples.
            Useful for prototyping on large datasets.

    Returns {"train": [...sample_ids], "val": [...], "test": [...]}.
    """
    df = _valid_sample_ids(metadata_path, h5_path)

    # First split: train vs (val+test)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - train_ratio, random_state=seed)
    train_idx, rest_idx = next(sss1.split(df, df["design_name"]))

    # Second split: val vs test (equal halves of the remainder)
    df_rest = df.iloc[rest_idx]
    relative_val = val_ratio / (1.0 - train_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1.0 - relative_val, random_state=seed)
    val_idx_rel, test_idx_rel = next(sss2.split(df_rest, df_rest["design_name"]))

    splits = {
        "train": df.iloc[train_idx]["sample_id"].tolist(),
        "val": df_rest.iloc[val_idx_rel]["sample_id"].tolist(),
        "test": df_rest.iloc[test_idx_rel]["sample_id"].tolist(),
    }
    return _subsample(splits, max_samples, seed)


def get_ood_splits(
    metadata_path: str = "dataset/metadata.parquet",
    h5_path: str | list[str] = "dataset/data.h5",
    held_out_geometry: str = "Gyroid TPMS",
    val_ratio: float = 0.15,
    seed: int = 42,
    max_samples: int | None = None,
) -> dict[str, list[str]]:
    """Leave-one-geometry-out split for OOD generalization (Task C).

    Train/val from 4 geometry types, test_ood from the held-out type.
    Returns {"train": [...], "val": [...], "test_ood": [...]}.
    """
    df = _valid_sample_ids(metadata_path, h5_path)

    ood_mask = df["design_name"] == held_out_geometry
    df_ood = df[ood_mask]
    df_in = df[~ood_mask]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(sss.split(df_in, df_in["design_name"]))

    splits = {
        "train": df_in.iloc[train_idx]["sample_id"].tolist(),
        "val": df_in.iloc[val_idx]["sample_id"].tolist(),
        "test_ood": df_ood["sample_id"].tolist(),
    }
    return _subsample(splits, max_samples, seed)


def get_geometry_only_splits(
    metadata_path: str = "dataset/metadata.parquet",
    h5_path: str | list[str] = "dataset/data.h5",
    heat_source_type: str = "uniform",
    seed: int = 42,
    max_samples: int | None = None,
) -> dict[str, list[str]]:
    """Subset for Task B: geometry-only prediction with fixed operating conditions.

    Filters to a single heat_source_type, then applies 70/15/15 split.
    Returns {"train": [...], "val": [...], "test": [...]}.
    """
    df = _valid_sample_ids(metadata_path, h5_path)
    df = df[df["heat_source_type"] == heat_source_type].reset_index(drop=True)

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, rest_idx = next(sss1.split(df, df["design_name"]))

    df_rest = df.iloc[rest_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_idx_rel, test_idx_rel = next(sss2.split(df_rest, df_rest["design_name"]))

    splits = {
        "train": df.iloc[train_idx]["sample_id"].tolist(),
        "val": df_rest.iloc[val_idx_rel]["sample_id"].tolist(),
        "test": df_rest.iloc[test_idx_rel]["sample_id"].tolist(),
    }
    return _subsample(splits, max_samples, seed)
