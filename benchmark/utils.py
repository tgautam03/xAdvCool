"""Config loading, seeding, and logging utilities."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(config_path: str, overrides: list[str] | None = None) -> dict:
    """Load a YAML config, merging with defaults.yaml if present.

    Args:
        config_path: Path to the experiment YAML config.
        overrides: List of "key.subkey=value" strings for CLI overrides.
    """
    config_dir = Path(config_path).parent

    # Load defaults if they exist
    defaults_path = config_dir / "defaults.yaml"
    cfg = {}
    if defaults_path.exists():
        with open(defaults_path) as f:
            cfg = yaml.safe_load(f) or {}

    # Load experiment config and merge (experiment overrides defaults)
    with open(config_path) as f:
        exp_cfg = yaml.safe_load(f) or {}
    cfg = _deep_merge(cfg, exp_cfg)

    # Apply CLI overrides
    if overrides:
        for ov in overrides:
            key, val = ov.split("=", 1)
            _set_nested(cfg, key.split("."), _parse_value(val))

    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def _set_nested(d: dict, keys: list[str], value) -> None:
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _parse_value(s: str):
    """Try to parse a string as int, float, bool, or leave as string."""
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def seed_everything(seed: int = 42) -> None:
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_output_dir(output_dir: str) -> Path:
    """Create output directory and return Path."""
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p
