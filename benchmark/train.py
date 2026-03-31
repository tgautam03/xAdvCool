"""Training entry point for xAdvCool benchmark models.

Usage:
    python -m benchmark.train --config benchmark/configs/task_a_unet3d.yaml
    python -m benchmark.train --config benchmark/configs/task_a_unet3d.yaml --override training.epochs=50
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from benchmark.data.dataset import CHTDataset
from benchmark.data.normalization import compute_norm_stats
from benchmark.data.splits import get_canonical_splits, get_ood_splits, get_geometry_only_splits
from benchmark.metrics.field_metrics import relative_l2_error
from benchmark.models.base import MODEL_REGISTRY
from benchmark.utils import load_config, seed_everything, setup_output_dir

# Import all models so they register themselves
import benchmark.models.trivial  # noqa: F401


def _build_splits(cfg: dict) -> dict[str, list[str]]:
    data_cfg = cfg["data"]
    split_type = data_cfg.get("split", "canonical")
    h5_path = data_cfg["h5_path"]
    meta_path = data_cfg["metadata_path"]
    seed = cfg["training"].get("seed", 42)
    max_samples = data_cfg.get("max_samples", None)

    if split_type == "canonical":
        return get_canonical_splits(meta_path, h5_path, seed=seed, max_samples=max_samples)
    elif split_type.startswith("ood_"):
        geometry = split_type[4:].replace("_", " ")
        return get_ood_splits(meta_path, h5_path, held_out_geometry=geometry, seed=seed, max_samples=max_samples)
    elif split_type == "geometry_only":
        hs_type = data_cfg.get("heat_source_type", "uniform")
        return get_geometry_only_splits(meta_path, h5_path, heat_source_type=hs_type, seed=seed, max_samples=max_samples)
    else:
        raise ValueError(f"Unknown split type: {split_type}")


def _build_dataset(cfg: dict, sample_ids: list[str], norm_stats) -> CHTDataset:
    data_cfg = cfg["data"]
    return CHTDataset(
        h5_path=data_cfg["h5_path"],
        metadata_path=data_cfg["metadata_path"],
        sample_ids=sample_ids,
        task=cfg["task"],
        input_fields=tuple(data_cfg.get("input_fields", ["mask", "heat_source"])),
        target_fields=tuple(data_cfg.get("target_fields", ["velocity", "temperature", "pressure"])),
        crop_buffer=data_cfg.get("crop_buffer", True),
        norm_stats=norm_stats,
    )


def train(cfg: dict) -> None:
    train_cfg = cfg["training"]
    seed_everything(train_cfg.get("seed", 42))
    output_dir = setup_output_dir(train_cfg.get("output_dir", "results/default"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Save config
    (output_dir / "config.yaml").write_text(
        json.dumps(cfg, indent=2, default=str)
    )

    # Splits
    splits = _build_splits(cfg)
    train_ids = splits["train"]
    val_ids = splits.get("val", splits.get("test_ood", []))

    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")

    # Normalization stats
    norm_stats = compute_norm_stats(
        cfg["data"]["h5_path"],
        train_ids,
        fields=tuple(cfg["data"].get("input_fields", ["mask", "heat_source"])
                      + cfg["data"].get("target_fields", ["velocity", "temperature", "pressure"])),
        cache_dir=str(output_dir),
    )

    # Datasets & loaders
    train_ds = _build_dataset(cfg, train_ids, norm_stats)
    val_ds = _build_dataset(cfg, val_ids, norm_stats)

    num_workers = train_cfg.get("num_workers", 4)
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.get("batch_size", 4),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.get("batch_size", 4),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    # Model
    model_name = cfg["model"]
    model_cls = MODEL_REGISTRY[model_name]
    model_args = cfg.get("model_args", {})
    model_args["task"] = cfg["task"]
    model = model_cls(**model_args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_name} | Params: {n_params:,}")

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )
    epochs = train_cfg.get("epochs", 100)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss
    criterion = nn.MSELoss()

    # Mixed precision
    use_amp = train_cfg.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Logging
    writer = SummaryWriter(log_dir=str(output_dir / "tb"))

    # Training loop
    best_val_loss = float("inf")
    target_key = "fields" if cfg["task"] == "field_prediction" else "scalars"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for batch_idx, (input_dict, target_dict) in enumerate(train_loader):
            input_dict = {k: v.to(device) for k, v in input_dict.items()}
            target = target_dict[target_key].to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(input_dict)
                loss = criterion(out[target_key], target)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_loader)
        elapsed = time.time() - t0

        # Validation
        model.eval()
        val_loss = 0.0
        val_rel_l2 = 0.0
        n_val = 0

        with torch.no_grad():
            for input_dict, target_dict in val_loader:
                input_dict = {k: v.to(device) for k, v in input_dict.items()}
                target = target_dict[target_key].to(device)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(input_dict)
                    loss = criterion(out[target_key], target)

                val_loss += loss.item()
                if target_key == "fields":
                    val_rel_l2 += relative_l2_error(out["fields"], target).sum().item()
                n_val += target.shape[0]

        val_loss /= max(len(val_loader), 1)
        val_rel_l2 /= max(n_val, 1)

        # Log
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        if target_key == "fields":
            writer.add_scalar("val/rel_l2", val_rel_l2, epoch)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val RelL2: {val_rel_l2:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": cfg,
            }, output_dir / "best_model.pt")

        checkpoint_every = train_cfg.get("checkpoint_every", 20)
        if epoch % checkpoint_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": cfg,
            }, output_dir / f"checkpoint_epoch{epoch}.pt")

    writer.close()
    # Save norm stats alongside model
    norm_stats.save(str(output_dir / "norm_stats.json"))
    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    print(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train an xAdvCool benchmark model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--override", nargs="*", default=[], help="Config overrides: key.subkey=value")
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    train(cfg)


if __name__ == "__main__":
    main()
