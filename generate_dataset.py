"""
Dataset Generator for Cold Plate Topology Optimization
=======================================================
Generates mask -> [velocity, pressure, temperature] pairs for five design types:
  1. Straight Channels
  2. Pin-Fin Staggered
  3. Gyroid TPMS
  4. Schwarz P TPMS
  5. Schwarz D TPMS

Storage: HDF5 for tensor data + Parquet metadata index (HuggingFace-ready).
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import time

import h5py
import numpy as np
import pandas as pd
from scipy.stats import qmc

from geometry import (generate_mask, CORE_NX, CORE_NY, CORE_NZ, BC_SOLID,
                      HEAT_SOURCE_REGISTRY, HEAT_SOURCE_NAMES)
from simulation import run_simulation


def _get_hardware_info(device):
    """Collect hardware metadata for reproducibility and environmental reporting."""
    info = {
        "cpu": platform.processor() or platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }
    if device == "cuda":
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader,nounits"],
                text=True, timeout=5,
            ).strip().split("\n")[0].split(", ")
            info["gpu_name"] = out[0]
            info["gpu_memory_mb"] = int(out[1])
            info["gpu_driver"] = out[2]
        except Exception:
            info["gpu_name"] = "unknown"
    return info

# ---------------------------------------------------------------------------
# Hydraulic diameter utility (mirrors simulation.py lines 44-59)
# ---------------------------------------------------------------------------
def compute_hydraulic_diameter(mask):
    buf = 10
    core = mask[buf:-buf, 1:-1, 4:-2]
    fluid_mask = (core != BC_SOLID)
    solid_mask = (core == BC_SOLID)
    V_fluid = int(np.sum(fluid_mask))
    faces_x = np.sum(fluid_mask[1:, :, :] & solid_mask[:-1, :, :]) + np.sum(solid_mask[1:, :, :] & fluid_mask[:-1, :, :])
    faces_y = np.sum(fluid_mask[:, 1:, :] & solid_mask[:, :-1, :]) + np.sum(solid_mask[:, 1:, :] & fluid_mask[:, :-1, :])
    faces_z = np.sum(fluid_mask[:, :, 1:] & solid_mask[:, :, :-1]) + np.sum(solid_mask[:, :, 1:] & fluid_mask[:, :, :-1])
    A_wetted = int(faces_x + faces_y + faces_z)
    Dh = 4.0 * V_fluid / A_wetted if A_wetted > 0 else float(mask.shape[1])
    return Dh, V_fluid, A_wetted


def compute_porosity(mask):
    buf = 10
    core = mask[buf:-buf, :, :]
    return 1.0 - np.sum(core == BC_SOLID) / core.size


# ---------------------------------------------------------------------------
# Macroscopic performance metrics (post-simulation)
# ---------------------------------------------------------------------------
def compute_macroscopic_metrics(results, mask, params, Dh, A_wetted):
    """Compute thermal-hydraulic performance metrics from converged fields.

    Returns a dict of scalar quantities:
      R_th, Nu, f, P_pump, COP, Q_vol, T_max_surface, T_avg_surface,
      delta_T_max, sigma_T, fluid_temp_rise
    """
    u_np = results["u"]          # (nx, ny, nz, 3)
    rho_np = results["rho"]      # (nx, ny, nz)
    T_np = results["T"]          # (nx, ny, nz)
    nx, ny, nz = mask.shape

    t_inlet = 25.0  # inlet temperature (same default as simulation.py)

    # --- Flow rate (volumetric) at outlet ---
    # Outlet is at x = nx-2; sum x-velocities over fluid cells
    outlet_mask = (mask[-2, :, :] != BC_SOLID)
    Q_vol = float(np.sum(u_np[-2, :, :, 0][outlet_mask]))

    # --- Pressure drop ---
    rho_in_avg = float(np.mean(rho_np[1, :, :]))
    rho_out_avg = float(np.mean(rho_np[-2, :, :]))
    delta_P = (rho_in_avg - rho_out_avg) / 3.0

    # --- Pumping power ---
    P_pump = abs(delta_P * Q_vol)

    # --- Heated surface temperature stats (floor at z=0, core region only) ---
    buf = 10
    T_surface = T_np[buf:buf + CORE_NX, :CORE_NY, 0]
    T_max_surface = float(np.max(T_surface))
    T_avg_surface = float(np.mean(T_surface))
    T_min_surface = float(np.min(T_surface))
    delta_T_max = T_max_surface - T_min_surface
    sigma_T = float(np.std(T_surface))

    # --- Fluid temperature rise (outlet avg - inlet) ---
    # Average T over fluid cells at inlet and outlet slices
    inlet_fluid = (mask[1, :, :] != BC_SOLID)
    outlet_fluid = outlet_mask
    T_inlet_avg = float(np.mean(T_np[1, :, :][inlet_fluid])) if np.any(inlet_fluid) else t_inlet
    T_outlet_avg = float(np.mean(T_np[-2, :, :][outlet_fluid])) if np.any(outlet_fluid) else t_inlet
    fluid_temp_rise = T_outlet_avg - T_inlet_avg

    # --- Total heat input (lattice units) ---
    # Q_heat = heat_power * sum(heat_source_mask) per timestep
    # For thermal resistance we use the effective total heat as heat_power * N_heated_cells
    heat_power = params["heat_power"]
    Q_heat = heat_power * Q_vol * fluid_temp_rise if fluid_temp_rise > 0 else 1e-12

    # --- Thermal resistance ---
    # R_th = (T_max_surface - T_inlet) / Q_heat_total
    # Use a heat balance: Q_absorbed = Q_vol * fluid_temp_rise (lattice units)
    Q_absorbed = abs(Q_vol * fluid_temp_rise) if abs(Q_vol * fluid_temp_rise) > 1e-12 else 1e-12
    R_th = (T_max_surface - t_inlet) / Q_absorbed

    # --- Average heat transfer coefficient and Nusselt number ---
    # h = Q_absorbed / (A_wetted * (T_wall_avg - T_fluid_avg))
    # T_fluid_avg = average T over all fluid cells in core
    core_fluid = (mask[buf:-buf, 1:-1, 4:-2] != BC_SOLID)
    T_fluid_avg = float(np.mean(T_np[buf:-buf, 1:-1, 4:-2][core_fluid])) if np.any(core_fluid) else t_inlet
    dT_conv = T_avg_surface - T_fluid_avg
    if abs(dT_conv) > 1e-6 and A_wetted > 0:
        h_avg = Q_absorbed / (A_wetted * abs(dT_conv))
    else:
        h_avg = 0.0
    # k_fluid in LBM = cs^2 * (tau_th - 0.5) where tau_th = 0.5 + (tau-0.5)/7
    tau_th_fluid = 0.5 + (params["tau_fluid"] - 0.5) / 7.0
    k_fluid = (1.0 / 3.0) * (tau_th_fluid - 0.5)
    Nu = (h_avg * Dh / k_fluid) if k_fluid > 1e-12 else 0.0

    # --- Fanning friction factor ---
    # f = delta_P * Dh / (2 * L * rho_avg * u_avg^2)
    L = nx - 2 * buf  # core length in lattice units
    u_inlet = params["u_inlet_val"]
    rho_avg = (rho_in_avg + rho_out_avg) / 2.0
    f_friction = (abs(delta_P) * Dh) / (2.0 * L * rho_avg * u_inlet ** 2) if u_inlet > 1e-12 else 0.0

    # --- COP ---
    COP = Q_absorbed / P_pump if P_pump > 1e-12 else 0.0

    return {
        "R_th": float(R_th),
        "Nu": float(Nu),
        "f_friction": float(f_friction),
        "P_pump": float(P_pump),
        "COP": float(COP),
        "Q_vol": float(Q_vol),
        "T_max_surface": float(T_max_surface),
        "T_avg_surface": float(T_avg_surface),
        "delta_T_max": float(delta_T_max),
        "sigma_T": float(sigma_T),
        "fluid_temp_rise": float(fluid_temp_rise),
    }


# ---------------------------------------------------------------------------
# Realistic parameter ranges per design type
# ---------------------------------------------------------------------------
# Each range is [low, high] for Latin Hypercube Sampling.
# Dimensions: feature_size, spacing, tau_fluid, u_inlet_val, heat_power
DESIGN_CONFIGS = {
    "Straight Channels": {
        "design_idx": 1,
        "ranges": {
            "feature_size": (2.0, 8.0),    # channel width 4-16 px
            "spacing":      (1.5, 7.0),    # channel count  6-28
            "tau_fluid":    (0.505, 0.65),  # nu = 0.0017 - 0.050
            "u_inlet_val":  (0.01, 0.05),   # conservative: keeps Ma < 0.15 after constriction acceleration
            "heat_power":   (0.05, 0.20),
        },
    },
    "Pin-Fin Staggered": {
        "design_idx": 2,
        "ranges": {
            "feature_size": (1.5, 5.0),     # pin diameter 3-10 px
            "spacing":      (2.0, 7.0),     # pitch 6-21 px
            "tau_fluid":    (0.505, 0.65),
            "u_inlet_val":  (0.01, 0.05),
            "heat_power":   (0.05, 0.20),
        },
    },
    "Gyroid TPMS": {
        "design_idx": 3,
        "ranges": {
            "feature_size": (4.0, 8.5),     # isovalue C: enough porosity for flow
            "spacing":      (2.0, 7.0),     # cell size 31-73 px
            "tau_fluid":    (0.505, 0.65),
            "u_inlet_val":  (0.01, 0.05),
            "heat_power":   (0.05, 0.20),
        },
    },
    "Schwarz P TPMS": {
        "design_idx": 4,
        "ranges": {
            "feature_size": (4.0, 8.5),
            "spacing":      (2.0, 7.0),
            "tau_fluid":    (0.505, 0.65),
            "u_inlet_val":  (0.01, 0.05),
            "heat_power":   (0.05, 0.20),
        },
    },
    "Schwarz D TPMS": {
        "design_idx": 5,
        "ranges": {
            "feature_size": (4.0, 8.5),
            "spacing":      (2.0, 7.0),
            "tau_fluid":    (0.505, 0.65),
            "u_inlet_val":  (0.01, 0.05),
            "heat_power":   (0.05, 0.20),
        },
    },
}

# ---------------------------------------------------------------------------
# Sample generation via Latin Hypercube
# ---------------------------------------------------------------------------
def generate_samples(design_name, n_samples, seed=42):
    """Return a list of parameter dicts for one design type.

    Each geometry is paired with every heat source type, so the total number
    of samples returned is ``n_samples * len(HEAT_SOURCE_NAMES)``.
    """
    cfg = DESIGN_CONFIGS[design_name]
    param_names = list(cfg["ranges"].keys())
    lows = np.array([cfg["ranges"][p][0] for p in param_names])
    highs = np.array([cfg["ranges"][p][1] for p in param_names])

    sampler = qmc.LatinHypercube(d=len(param_names), seed=seed)
    raw = sampler.random(n=n_samples)
    scaled = qmc.scale(raw, lows, highs)

    samples = []
    for i in range(n_samples):
        base = {name: float(scaled[i, j]) for j, name in enumerate(param_names)}
        base["design_idx"] = cfg["design_idx"]
        base["design_name"] = design_name
        # Pair each geometry configuration with every heat source type
        for hs_name in HEAT_SOURCE_NAMES:
            params = dict(base)
            params["heat_source_type"] = hs_name
            samples.append(params)
    return samples


# ---------------------------------------------------------------------------
# Validity check: ensure geometry is not blocked and porosity is reasonable
# ---------------------------------------------------------------------------
MIN_POROSITY = 0.15
MAX_POROSITY = 0.90

def is_valid_geometry(mask):
    porosity = compute_porosity(mask)
    if porosity < MIN_POROSITY or porosity > MAX_POROSITY:
        return False, porosity
    # Check inlet/outlet are not fully blocked
    buf = 10
    inlet_slice = mask[buf, 1:-1, 4:-2]
    outlet_slice = mask[-(buf + 1), 1:-1, 4:-2]
    if np.all(inlet_slice == BC_SOLID) or np.all(outlet_slice == BC_SOLID):
        return False, porosity
    return True, porosity


# ---------------------------------------------------------------------------
# Main dataset generation
# ---------------------------------------------------------------------------
def generate_dataset(output_dir, samples_per_design=50, max_steps=30000,
                     device="cuda", seed=42, resume=True,
                     snapshot_every=0):
    os.makedirs(output_dir, exist_ok=True)
    h5_path = os.path.join(output_dir, "data.h5")
    meta_path = os.path.join(output_dir, "metadata.parquet")

    hw_info = _get_hardware_info(device)
    dataset_start_time = time.time()

    # Collect existing sample IDs for resume support and rebuild parquet
    # from HDF5 if it's missing (e.g. script was stopped before first write)
    existing_ids = set()
    if resume and os.path.exists(h5_path):
        with h5py.File(h5_path, "r") as f:
            existing_ids = set(f.keys())
            if not os.path.exists(meta_path) and len(existing_ids) > 0:
                print(f"Rebuilding metadata parquet from {len(existing_ids)} existing samples...")
                rows = []
                for sid in existing_ids:
                    row = {k: (v.item() if hasattr(v, 'item') else v)
                           for k, v in f[sid].attrs.items()}
                    row["sample_id"] = sid
                    rows.append(row)
                pd.DataFrame(rows).to_parquet(meta_path, index=False)
        print(f"Resuming: {len(existing_ids)} samples already in {h5_path}")

    # Build full sample list.  Each design's LHS points are shuffled
    # independently so that the first N samples already span the full
    # parameter range, then interleaved across designs for an even mix.
    rng = np.random.default_rng(seed)
    per_design = {}
    for name in DESIGN_CONFIGS:
        samples = generate_samples(name, samples_per_design, seed=seed)
        rng.shuffle(samples)
        per_design[name] = samples
    all_samples = []
    max_len = max(len(v) for v in per_design.values())
    for i in range(max_len):
        for name in DESIGN_CONFIGS:
            if i < len(per_design[name]):
                all_samples.append(per_design[name][i])

    metadata_rows = []
    total = len(all_samples)
    done = 0
    skipped = 0
    failed = 0

    for i, params in enumerate(all_samples):
        # Deterministic sample ID from parameters
        param_str = json.dumps(params, sort_keys=True)
        sample_id = f"{params['design_name'].replace(' ', '_')}_{hashlib.md5(param_str.encode()).hexdigest()[:8]}"

        if sample_id in existing_ids:
            done += 1
            continue

        # Generate geometry mask
        state = {
            "design_type": params["design_idx"],
            "feature_size": params["feature_size"],
            "spacing": params["spacing"],
            "buffer_size": 10,
        }
        mask = generate_mask(state)

        # Validate
        valid, porosity = is_valid_geometry(mask)
        if not valid:
            skipped += 1
            print(f"[{i+1}/{total}] SKIP {sample_id} (porosity={porosity:.2%})")
            continue

        # Compute Re before running
        Dh, V_fluid, A_wetted = compute_hydraulic_diameter(mask)
        nu = (params["tau_fluid"] - 0.5) / 3.0
        Re = params["u_inlet_val"] * Dh / nu

        # Build 3D heat source array from the 2D pattern
        nx, ny, nz = mask.shape
        buf = 10
        hs_name = params["heat_source_type"]
        hs_2d = HEAT_SOURCE_REGISTRY[hs_name](CORE_NX, CORE_NY, seed=hash(sample_id) & 0xFFFFFFFF)
        heat_source_3d = np.zeros((nx, ny, nz), dtype=np.float32)
        # Apply only to the core region floor (z=0 solid base), not into
        # the inlet/outlet buffer zones
        heat_source_3d[buf:buf+CORE_NX, :CORE_NY, 0] = hs_2d

        print(f"[{i+1}/{total}] RUN  {sample_id}  Re={Re:.1f}  porosity={porosity:.2%}  heat={hs_name}")

        try:
            t0 = time.time()
            results = run_simulation(
                mask,
                max_steps=max_steps,
                tau_fluid=params["tau_fluid"],
                u_inlet_val=params["u_inlet_val"],
                heat_power=params["heat_power"],
                heat_source=heat_source_3d,
                tol_ke=1e-5,
                tol_T=1e-5,
                device=device,
                silent=True,
            )
            elapsed = time.time() - t0
        except Exception as e:
            failed += 1
            print(f"  FAILED: {e}")
            continue

        # Check for divergence
        u_np = results["u"]
        max_v = np.max(np.linalg.norm(u_np, axis=-1))
        if np.isnan(max_v) or max_v > 0.5:
            failed += 1
            print(f"  DIVERGED (max_v={max_v:.4f})")
            continue

        # Compute output statistics
        T_np = results["T"]
        rho_np = results["rho"]
        pressure = rho_np / 3.0  # p = rho * cs^2, cs^2 = 1/3

        # Compute macroscopic performance metrics
        metrics = compute_macroscopic_metrics(results, mask, params, Dh, A_wetted)

        # Write to HDF5 (one group per sample, compressed)
        with h5py.File(h5_path, "a") as f:
            grp = f.create_group(sample_id)
            grp.create_dataset("mask", data=mask.astype(np.int8), compression="gzip", compression_opts=2)
            grp.create_dataset("heat_source", data=heat_source_3d, compression="gzip", compression_opts=2)
            grp.create_dataset("velocity", data=u_np.astype(np.float32), compression="gzip", compression_opts=2)
            grp.create_dataset("pressure", data=pressure.astype(np.float32), compression="gzip", compression_opts=2)
            grp.create_dataset("temperature", data=T_np.astype(np.float32), compression="gzip", compression_opts=2)
            # Store scalar metadata as attributes
            for k, v in params.items():
                grp.attrs[k] = v
            grp.attrs["porosity"] = porosity
            grp.attrs["Dh"] = Dh
            grp.attrs["Re"] = Re
            grp.attrs["max_velocity"] = float(max_v)
            grp.attrs["max_temperature"] = float(np.max(T_np))
            grp.attrs["mean_temperature"] = float(np.mean(T_np))
            grp.attrs["grid_shape"] = list(mask.shape)
            grp.attrs["elapsed_s"] = elapsed
            grp.attrs["sim_steps"] = results["steps"]
            grp.attrs["converged"] = results["converged"]
            n_cells = int(np.prod(mask.shape))
            grp.attrs["grid_cells"] = n_cells
            grp.attrs["total_lattice_updates"] = n_cells * results["steps"]
            grp.attrs["mlups"] = (n_cells * results["steps"]) / (elapsed * 1e6) if elapsed > 0 else 0.0
            grp.attrs["gpu_name"] = hw_info.get("gpu_name", "unknown")
            for mk, mv in metrics.items():
                grp.attrs[mk] = mv

        existing_ids.add(sample_id)
        done += 1

        # Snapshot: safe copy for training while generation continues
        if snapshot_every > 0 and done % snapshot_every == 0:
            snap_h5 = os.path.join(output_dir, "data_snapshot.h5")
            snap_meta = os.path.join(output_dir, "metadata_snapshot.parquet")
            shutil.copy2(h5_path, snap_h5)
            # Write current metadata to snapshot parquet
            df_snap = pd.DataFrame(metadata_rows)
            if resume and os.path.exists(meta_path):
                df_old = pd.read_parquet(meta_path)
                df_snap = pd.concat([df_old, df_snap], ignore_index=True).drop_duplicates(
                    subset=["sample_id"], keep="last"
                )
            df_snap.to_parquet(snap_meta, index=False)
            print(f"  SNAPSHOT saved ({done} samples) -> {snap_h5}")

        # Append metadata row
        row = {
            "sample_id": sample_id,
            "design_name": params["design_name"],
            "design_idx": params["design_idx"],
            "heat_source_type": params["heat_source_type"],
            "feature_size": params["feature_size"],
            "spacing": params["spacing"],
            "tau_fluid": params["tau_fluid"],
            "u_inlet_val": params["u_inlet_val"],
            "heat_power": params["heat_power"],
            "porosity": porosity,
            "Dh": Dh,
            "Re": Re,
            "max_velocity": float(max_v),
            "max_temperature": float(np.max(T_np)),
            "mean_temperature": float(np.mean(T_np)),
            "pressure_drop": float(np.mean(rho_np[1, :, :]) - np.mean(rho_np[-2, :, :])) / 3.0,
            "elapsed_s": elapsed,
            "sim_steps": results["steps"],
            "converged": results["converged"],
            "grid_nx": mask.shape[0],
            "grid_ny": mask.shape[1],
            "grid_nz": mask.shape[2],
            "grid_cells": int(np.prod(mask.shape)),
            "total_lattice_updates": int(np.prod(mask.shape)) * results["steps"],
            "mlups": (int(np.prod(mask.shape)) * results["steps"]) / (elapsed * 1e6) if elapsed > 0 else 0.0,
            "gpu_name": hw_info.get("gpu_name", "unknown"),
            # Macroscopic performance metrics
            "R_th": metrics["R_th"],
            "Nu": metrics["Nu"],
            "f_friction": metrics["f_friction"],
            "P_pump": metrics["P_pump"],
            "COP": metrics["COP"],
            "Q_vol": metrics["Q_vol"],
            "T_max_surface": metrics["T_max_surface"],
            "T_avg_surface": metrics["T_avg_surface"],
            "delta_T_max": metrics["delta_T_max"],
            "sigma_T": metrics["sigma_T"],
            "fluid_temp_rise": metrics["fluid_temp_rise"],
        }
        metadata_rows.append(row)

        status = "CONVERGED" if results["converged"] else "NOT CONVERGED"
        print(f"  {status} at step {results['steps']}/{max_steps} in {elapsed:.1f}s  "
              f"dKE={results['d_ke']:.1e}  dT={results['dT_conv']:.1e}  "
              f"max_v={max_v:.4f}  Nu={metrics['Nu']:.2f}  R_th={metrics['R_th']:.4f}")

        # Write metadata parquet after every sample so it's always up to date
        df_new = pd.DataFrame(metadata_rows)
        if resume and os.path.exists(meta_path):
            df_old = pd.read_parquet(meta_path)
            df = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(
                subset=["sample_id"], keep="last"
            )
        else:
            df = df_new
        df.to_parquet(meta_path, index=False)

    dataset_wall_time = time.time() - dataset_start_time

    print(f"\nDataset generation complete.")
    print(f"  Completed: {done}  Skipped: {skipped}  Failed: {failed}")
    print(f"  HDF5: {h5_path}")
    print(f"  Metadata: {meta_path}")

    # Simulation cost summary
    if metadata_rows:
        total_sim_s = sum(r["elapsed_s"] for r in metadata_rows)
        total_steps = sum(r["sim_steps"] for r in metadata_rows)
        total_lups = sum(r["total_lattice_updates"] for r in metadata_rows)
        avg_mlups = (total_lups / total_sim_s / 1e6) if total_sim_s > 0 else 0
        n_converged = sum(1 for r in metadata_rows if r["converged"])
        print(f"\n  --- Simulation Cost Summary ---")
        print(f"  Hardware:              {hw_info.get('gpu_name', 'N/A')}")
        print(f"  Total wall time:       {dataset_wall_time:.0f}s ({dataset_wall_time/3600:.1f}h)")
        print(f"  Total simulation time: {total_sim_s:.0f}s ({total_sim_s/3600:.1f}h)")
        print(f"  Total timesteps:       {total_steps:,}")
        print(f"  Total lattice updates: {total_lups:.3e}")
        print(f"  Average throughput:    {avg_mlups:.1f} MLUPS")
        print(f"  Converged:             {n_converged}/{len(metadata_rows)}")
        print(f"  Avg time/sample:       {total_sim_s/len(metadata_rows):.1f}s")

    return h5_path, meta_path



def write_dataset_card(output_dir):
    """Generate a HuggingFace-compatible README.md dataset card."""
    card = f"""\
---
license: cc-by-4.0
task_categories:
  - image-to-image
tags:
  - physics
  - simulation
  - cfd
  - thermal
  - topology-optimization
  - lattice-boltzmann
  - heat-sink
  - cold-plate
pretty_name: Cold Plate Topology CFD Dataset
size_categories:
  - 1K<n<10K
---

# Cold Plate Topology CFD Dataset

A dataset of 3D conjugate heat transfer simulations for cold plate / heat sink
topology optimization. Each sample maps a **geometry mask** to converged
**velocity**, **pressure**, and **temperature** fields computed via the Lattice
Boltzmann Method (D3Q19, BGK collision).

## Design Types

| ID | Type | Geometry Parameters |
|----|------|-------------------|
| 1 | Straight Channels | channel width, channel count |
| 2 | Pin-Fin Staggered | pin diameter, pitch |
| 3 | Gyroid TPMS | isovalue (porosity), unit cell size |
| 4 | Schwarz P TPMS | isovalue (porosity), unit cell size |
| 5 | Schwarz D TPMS | isovalue (porosity), unit cell size |

## Heat Source Patterns

Each geometry is simulated with all 10 heat source types applied to the
floor (z=0) of the cold plate, confined to the core region (no heating in
inlet/outlet buffer zones):

| Pattern | Description |
|---------|-------------|
| `uniform` | Uniform heat flux across the full floor |
| `dual_core` | Dual-core CPU with L3 cache strip and background leakage |
| `quad_core` | 2x2 quad-core CPU with shared L3 cache cross |
| `gpu_die` | GPU shader array with HBM controller strips on two edges |
| `chiplet` | Multi-chiplet package (2x3 compute dies + I/O die) |
| `igbt_half_bridge` | IGBT half-bridge module (3 IGBT + 3 diode dies) |
| `soc_heterogeneous` | big.LITTLE SoC with GPU block, memory controller, NPU |
| `gaussian_hotspots` | 1-3 smooth Gaussian hotspots (seed-varied) |
| `hotspot_on_background` | Uniform background with 1-2 intense ALU/FPU hotspots |
| `peripheral_ring` | FPGA-style: high power I/O ring, cool interior |

## Data Format

- **`data.h5`** (HDF5): One group per sample containing:
  - `mask`: `int8 (nx, ny, nz)` -- boundary condition labels
    (0=fluid, 1=solid, 2=inlet, 3=outlet)
  - `velocity`: `float32 (nx, ny, nz, 3)` -- velocity vector field
  - `pressure`: `float32 (nx, ny, nz)` -- pressure field (rho/3)
  - `temperature`: `float32 (nx, ny, nz)` -- temperature field
  - `heat_source`: `float32 (nx, ny, nz)` -- applied heat source field
  - Attributes: all simulation parameters and computed statistics
- **`metadata.parquet`**: Tabular index with per-sample parameters, Reynolds
  number, porosity, hydraulic diameter, pressure drop, heat source type, and timing.

## Grid

- Domain: `{CORE_NX}x{CORE_NY}x{CORE_NZ}` core + 10-cell inlet/outlet buffers
- Total grid per sample: `{CORE_NX + 20}x{CORE_NY}x{CORE_NZ}`

## Reynolds Number Range

Re in [~20, ~1000], spanning the laminar regime relevant for electronics
cooling cold plates.

## Loading Example

```python
import h5py
import pandas as pd

# Browse metadata
meta = pd.read_parquet("metadata.parquet")
print(meta[["sample_id", "design_name", "Re", "porosity"]].head())

# Load a single sample
with h5py.File("data.h5", "r") as f:
    sample = f["Straight_Channels_a1b2c3d4"]
    mask = sample["mask"][:]
    velocity = sample["velocity"][:]
    pressure = sample["pressure"][:]
    temperature = sample["temperature"][:]
    Re = sample.attrs["Re"]
```

## Citation

If you use this dataset, please cite:

```bibtex
@misc{{cold_plate_cfd_dataset,
  title={{Cold Plate Topology CFD Dataset}},
  year={{2026}},
  url={{https://huggingface.co/datasets/YOUR_USERNAME/cold-plate-cfd}},
}}
```
"""
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(card)
    print(f"Dataset card written to {readme_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate cold plate CFD dataset")
    parser.add_argument("--output-dir", default="dataset", help="Output directory")
    parser.add_argument("--samples-per-design", type=int, default=50,
                        help="Number of LHS samples per design type")
    parser.add_argument("--max-steps", type=int, default=30000,
                        help="Max LBM timesteps per simulation")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh, ignore existing data")
    parser.add_argument("--snapshot-every", type=int, default=0,
                        help="Save a safe snapshot copy every N samples (0=off)")
    args = parser.parse_args()

    # Generate dataset card
    write_dataset_card(args.output_dir)

    # Run generation
    generate_dataset(
        output_dir=args.output_dir,
        samples_per_design=args.samples_per_design,
        max_steps=args.max_steps,
        device=args.device,
        seed=args.seed,
        resume=not args.no_resume,
        snapshot_every=args.snapshot_every,
    )
