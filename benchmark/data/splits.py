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
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,
    seed: int = 42,
    max_samples: int | None = None,
) -> dict[str, list[str]]:
    """Split stratified by design_name: test first (fixed), then val from train pool.

    Step 1: Carve out a fixed test set (same across all models for a given seed).
    Step 2: Split the remaining pool into train/val (val used for early stopping).

    Args:
        test_ratio: Fraction held out for testing (fixed across models).
        val_ratio: Fraction of *original* data used for validation.
        max_samples: If set, subsample the training set to this many samples.

    Returns {"train": [...sample_ids], "val": [...], "test": [...]}.
    """
    df = _valid_sample_ids(metadata_path, h5_path)

    # Step 1: Fixed test split (identical across all models for a given seed)
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trainval_idx, test_idx = next(sss_test.split(df, df["design_name"]))

    # Step 2: Val from the train pool (for early stopping)
    df_trainval = df.iloc[trainval_idx]
    relative_val = val_ratio / (1.0 - test_ratio)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=relative_val, random_state=seed)
    train_idx_rel, val_idx_rel = next(sss_val.split(df_trainval, df_trainval["design_name"]))

    splits = {
        "train": df_trainval.iloc[train_idx_rel]["sample_id"].tolist(),
        "val": df_trainval.iloc[val_idx_rel]["sample_id"].tolist(),
        "test": df.iloc[test_idx]["sample_id"].tolist(),
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

    Test is the entire held-out geometry (fixed). Val is carved from
    the remaining 4 geometry types (for early stopping).
    Returns {"train": [...], "val": [...], "test_ood": [...]}.
    """
    df = _valid_sample_ids(metadata_path, h5_path)

    # Step 1: Fixed test — entire held-out geometry
    ood_mask = df["design_name"] == held_out_geometry
    df_ood = df[ood_mask]
    df_in = df[~ood_mask]

    # Step 2: Val from the in-distribution pool (for early stopping)
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
    test_ratio: float = 0.15,
    val_ratio: float = 0.15,
    seed: int = 42,
    max_samples: int | None = None,
) -> dict[str, list[str]]:
    """Subset for Task B: geometry-only prediction with fixed operating conditions.

    Filters to a single heat_source_type, then splits: test first (fixed),
    then val from the remaining pool (for early stopping).
    Returns {"train": [...], "val": [...], "test": [...]}.
    """
    df = _valid_sample_ids(metadata_path, h5_path)
    df = df[df["heat_source_type"] == heat_source_type].reset_index(drop=True)

    # Step 1: Fixed test split
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trainval_idx, test_idx = next(sss_test.split(df, df["design_name"]))

    # Step 2: Val from the train pool (for early stopping)
    df_trainval = df.iloc[trainval_idx]
    relative_val = val_ratio / (1.0 - test_ratio)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=relative_val, random_state=seed)
    train_idx_rel, val_idx_rel = next(sss_val.split(df_trainval, df_trainval["design_name"]))

    splits = {
        "train": df_trainval.iloc[train_idx_rel]["sample_id"].tolist(),
        "val": df_trainval.iloc[val_idx_rel]["sample_id"].tolist(),
        "test": df.iloc[test_idx]["sample_id"].tolist(),
    }
    return _subsample(splits, max_samples, seed)
