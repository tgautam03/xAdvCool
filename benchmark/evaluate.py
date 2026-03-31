"""Evaluation entry point for xAdvCool benchmark models.

Usage:
    python -m benchmark.evaluate --config benchmark/configs/task_a_unet3d.yaml \
        --checkpoint results/task_a_unet3d/best_model.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from benchmark.data.dataset import CHTDataset
from benchmark.data.normalization import NormStats
from benchmark.data.splits import get_canonical_splits, get_ood_splits, get_geometry_only_splits
from benchmark.metrics.field_metrics import relative_l2_error, per_channel_relative_l2, rmse
from benchmark.metrics.scalar_metrics import scalar_r2, scalar_rmse, scalar_mae
from benchmark.metrics.physics import mass_conservation_violation
from benchmark.models.base import MODEL_REGISTRY
from benchmark.utils import load_config, seed_everything

# Register models
import benchmark.models.trivial  # noqa: F401


def _build_splits(cfg: dict) -> dict[str, list[str]]:
    data_cfg = cfg["data"]
    split_type = data_cfg.get("split", "canonical")
    h5_path = data_cfg["h5_path"]
    meta_path = data_cfg["metadata_path"]
    seed = cfg["training"].get("seed", 42)

    if split_type == "canonical":
        return get_canonical_splits(meta_path, h5_path, seed=seed)
    elif split_type.startswith("ood_"):
        geometry = split_type[4:].replace("_", " ")
        return get_ood_splits(meta_path, h5_path, held_out_geometry=geometry, seed=seed)
    elif split_type == "geometry_only":
        hs_type = data_cfg.get("heat_source_type", "uniform")
        return get_geometry_only_splits(meta_path, h5_path, heat_source_type=hs_type, seed=seed)
    else:
        raise ValueError(f"Unknown split type: {split_type}")


def evaluate(cfg: dict, checkpoint_path: str, split: str = "test") -> dict:
    seed_everything(cfg["training"].get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load norm stats from checkpoint directory
    ckpt_dir = Path(checkpoint_path).parent
    norm_stats_path = ckpt_dir / "norm_stats.json"
    norm_stats = NormStats.load(str(norm_stats_path)) if norm_stats_path.exists() else None

    # Get split sample IDs
    splits = _build_splits(cfg)
    if split == "test_ood" and "test_ood" in splits:
        sample_ids = splits["test_ood"]
    elif split in splits:
        sample_ids = splits[split]
    else:
        raise ValueError(f"Split '{split}' not found. Available: {list(splits.keys())}")

    print(f"Evaluating on {len(sample_ids)} samples ({split} split)")

    # Dataset
    data_cfg = cfg["data"]
    ds = CHTDataset(
        h5_path=data_cfg["h5_path"],
        metadata_path=data_cfg["metadata_path"],
        sample_ids=sample_ids,
        task=cfg["task"],
        input_fields=tuple(data_cfg.get("input_fields", ["mask", "heat_source"])),
        target_fields=tuple(data_cfg.get("target_fields", ["velocity", "temperature", "pressure"])),
        crop_buffer=data_cfg.get("crop_buffer", True),
        norm_stats=norm_stats,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    # Load model
    model_name = cfg["model"]
    model_cls = MODEL_REGISTRY[model_name]
    model_args = cfg.get("model_args", {})
    model_args["task"] = cfg["task"]
    model = model_cls(**model_args).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    target_key = "fields" if cfg["task"] == "field_prediction" else "scalars"

    # Collect predictions
    all_rel_l2 = []
    all_rmse = []
    all_channel_l2: dict[str, list] = {}
    all_div = []
    all_pred_scalars = []
    all_tgt_scalars = []

    channel_names = []
    for f in data_cfg.get("target_fields", ["velocity", "temperature", "pressure"]):
        if f == "velocity":
            channel_names.extend(["velocity_x", "velocity_y", "velocity_z"])
        else:
            channel_names.append(f)

    with torch.no_grad():
        for input_dict, target_dict in loader:
            input_dict = {k: v.to(device) for k, v in input_dict.items()}
            target = target_dict[target_key].to(device)
            out = model(input_dict)
            pred = out[target_key]

            if target_key == "fields":
                all_rel_l2.append(relative_l2_error(pred, target).cpu())
                all_rmse.append(rmse(pred, target).cpu())

                ch_l2 = per_channel_relative_l2(pred, target, channel_names)
                for name, val in ch_l2.items():
                    all_channel_l2.setdefault(name, []).append(val.cpu())

                # Physics: mass conservation on velocity channels (first 3)
                vel_pred = pred[:, :3]
                mask_input = input_dict["fields"][:, :1]  # mask is first input channel
                all_div.append(mass_conservation_violation(vel_pred, mask_input).cpu())
            else:
                all_pred_scalars.append(pred.cpu())
                all_tgt_scalars.append(target.cpu())

    # Aggregate results
    results = {
        "model": model_name,
        "task": cfg["task"],
        "split": split,
        "n_samples": len(sample_ids),
        "n_params": sum(p.numel() for p in model.parameters()),
    }

    if target_key == "fields":
        rel_l2 = torch.cat(all_rel_l2)
        results["metrics"] = {
            "overall_rel_l2": {"mean": rel_l2.mean().item(), "std": rel_l2.std().item()},
            "overall_rmse": {
                "mean": torch.cat(all_rmse).mean().item(),
                "std": torch.cat(all_rmse).std().item(),
            },
            "mass_conservation_violation": {
                "mean": torch.cat(all_div).mean().item(),
                "std": torch.cat(all_div).std().item(),
            },
        }
        for name in channel_names:
            vals = torch.cat(all_channel_l2[name])
            results["metrics"][f"{name}_rel_l2"] = {
                "mean": vals.mean().item(),
                "std": vals.std().item(),
            }
    else:
        pred_all = torch.cat(all_pred_scalars)
        tgt_all = torch.cat(all_tgt_scalars)
        from benchmark.data.dataset import TARGET_SCALAR_COLS

        r2 = scalar_r2(pred_all, tgt_all)
        rmse_vals = scalar_rmse(pred_all, tgt_all)
        mae_vals = scalar_mae(pred_all, tgt_all)

        results["metrics"] = {}
        for i, col in enumerate(TARGET_SCALAR_COLS):
            results["metrics"][col] = {
                "r2": r2[i].item(),
                "rmse": rmse_vals[i].item(),
                "mae": mae_vals[i].item(),
            }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate an xAdvCool benchmark model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    results = evaluate(cfg, args.checkpoint, args.split)

    # Print results
    print(json.dumps(results, indent=2))

    # Save
    output_path = args.output or str(Path(args.checkpoint).parent / f"eval_{args.split}.json")
    Path(output_path).write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
