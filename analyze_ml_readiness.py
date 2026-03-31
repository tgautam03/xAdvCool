"""
ML-Readiness Analysis for Cold Plate CFD Dataset
==================================================
Evaluates whether the dataset has sufficient complexity, diversity, and
inter-sample variation for machine learning benchmarks.

Analyses:
  1. Baseline learnability  -- linear & tree baselines; establishes floor
  2. Sample uniqueness       -- pairwise distances, nearest-neighbor gaps
  3. Geometry diversity      -- mask topology metrics, SSIM distributions
  4. Field complexity        -- spatial frequency spectra, gradient stats
  5. Input-output difficulty -- non-linear dependence, conditional variance
  6. Train/test shift        -- stratified split quality checks

All figures follow NeurIPS 2-column format (same style as analyze_dataset.py).

Usage:
    python analyze_ml_readiness.py
    python analyze_ml_readiness.py --data dataset/data.h5 --meta dataset/metadata.parquet
    python analyze_ml_readiness.py --data dataset/data.h5 --field-analysis
"""
import argparse
import os
import sys
import warnings

import h5py
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats as sp_stats
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ===================================================================
# PLOT STYLE (same as analyze_dataset.py)
# ===================================================================

COL_W = 3.25
PAGE_W = 6.75
MAX_H = 9.0
PALETTE = ["#332288", "#88CCEE", "#44AA99", "#117733", "#999933", "#CC6677", "#882255"]

BC_SOLID = 1


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.title_fontsize": 8,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "axes.linewidth": 0.6,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "figure.dpi": 150,
        "figure.constrained_layout.use": True,
    })


def _save(fig, out_dir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  Saved: {name}.pdf / .png")


INPUT_COLS = ["feature_size", "spacing", "tau_fluid", "u_inlet_val", "heat_power"]
OUTPUT_COLS = ["R_th", "Nu", "f_friction", "P_pump", "COP", "Q_vol",
               "T_max_surface", "T_avg_surface", "delta_T_max", "sigma_T",
               "fluid_temp_rise"]
CONTEXT_COLS = ["porosity", "Dh", "Re"]

PRETTY = {
    "R_th": r"$R_{\mathrm{th}}$", "Nu": r"$\mathrm{Nu}$",
    "f_friction": r"$f$", "P_pump": r"$P_{\mathrm{pump}}$",
    "COP": r"$\mathrm{COP}$", "Q_vol": r"$\dot{Q}$",
    "T_max_surface": r"$T_{\max,s}$", "T_avg_surface": r"$\bar{T}_s$",
    "delta_T_max": r"$\Delta T_{\max}$", "sigma_T": r"$\sigma_T$",
    "fluid_temp_rise": r"$\Delta T_f$",
}

SHORT_DESIGN = {
    "Straight Channels": "Straight",
    "Pin-Fin Staggered": "Pin-Fin",
    "Gyroid TPMS": "Gyroid",
    "Schwarz P TPMS": "Schwarz P",
    "Schwarz D TPMS": "Schwarz D",
}


def _label(col):
    return PRETTY.get(col, col)


def _short(name):
    return SHORT_DESIGN.get(name, name[:10])


def _design_style(designs):
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    return {d: (PALETTE[i % len(PALETTE)], markers[i % len(markers)])
            for i, d in enumerate(sorted(designs))}


# ===================================================================
# 1. BASELINE LEARNABILITY
# ===================================================================

def baseline_learnability(df, out_dir):
    """Train simple baselines (Ridge, RF, GBT) to quantify task difficulty."""
    print("\n" + "=" * 70)
    print("BASELINE LEARNABILITY")
    print("=" * 70)

    avail_in = [c for c in INPUT_COLS + CONTEXT_COLS if c in df.columns]
    avail_out = [c for c in OUTPUT_COLS if c in df.columns]

    if len(avail_in) < 2 or len(avail_out) < 1:
        print("  Insufficient columns for baseline analysis.")
        return None

    # Add design as one-hot features
    X_df = df[avail_in].copy()
    if "design_name" in df.columns:
        dummies = pd.get_dummies(df["design_name"], prefix="design", dtype=float)
        X_df = pd.concat([X_df, dummies], axis=1)

    X = X_df.values.astype(np.float64)
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    # 5-fold CV for each target
    models = {
        "Ridge": lambda: Ridge(alpha=1.0),
        "RF": lambda: RandomForestRegressor(n_estimators=100, max_depth=10,
                                             random_state=42, n_jobs=-1),
        "GBT": lambda: GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                                   learning_rate=0.1, random_state=42),
    }

    results = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for target in avail_out:
        y = df[target].values.astype(np.float64)
        valid = ~np.isnan(y) & ~np.isinf(y)
        X_valid = X_scaled[valid]
        y_valid = y[valid]

        if len(y_valid) < 10:
            continue

        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y_valid.reshape(-1, 1)).ravel()

        for model_name, model_fn in models.items():
            r2_scores = []
            mae_scores = []
            for train_idx, test_idx in kf.split(X_valid):
                model = model_fn()
                model.fit(X_valid[train_idx], y_scaled[train_idx])
                y_pred = model.predict(X_valid[test_idx])
                r2_scores.append(r2_score(y_scaled[test_idx], y_pred))
                # MAE in original scale
                y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()
                y_true_orig = y_valid[test_idx]
                mae_scores.append(mean_absolute_error(y_true_orig, y_pred_orig))

            results.append({
                "target": target,
                "model": model_name,
                "r2_mean": np.mean(r2_scores),
                "r2_std": np.std(r2_scores),
                "mae_mean": np.mean(mae_scores),
                "mae_std": np.std(mae_scores),
                "nrmae": np.mean(mae_scores) / (y_valid.max() - y_valid.min() + 1e-12),
            })

    results_df = pd.DataFrame(results)

    # Print summary
    print(f"\n  5-Fold CV results ({len(df)} samples, {len(avail_in)}+design features):\n")
    print(f"  {'Target':20s} {'Model':6s} {'R2':>12s} {'NRMAE':>10s}")
    print("  " + "-" * 52)
    for target in avail_out:
        sub = results_df[results_df["target"] == target]
        for _, row in sub.iterrows():
            print(f"  {target:20s} {row['model']:6s} "
                  f"{row['r2_mean']:+.3f}+/-{row['r2_std']:.3f} "
                  f"{row['nrmae']:.4f}")
        if len(sub) > 0:
            print()

    # Classify difficulty
    print("  Task difficulty classification (based on best baseline R2):")
    for target in avail_out:
        sub = results_df[results_df["target"] == target]
        if len(sub) == 0:
            continue
        best_r2 = sub["r2_mean"].max()
        if best_r2 > 0.95:
            difficulty = "EASY (R2 > 0.95) -- trivial for simple models"
        elif best_r2 > 0.80:
            difficulty = "MODERATE (0.80 < R2 < 0.95) -- room for DL improvement"
        elif best_r2 > 0.50:
            difficulty = "CHALLENGING (0.50 < R2 < 0.80) -- good benchmark target"
        else:
            difficulty = "HARD (R2 < 0.50) -- needs field-level features or DL"
        print(f"    {target:20s}  best R2={best_r2:+.3f}  {difficulty}")

    # --- Fig: Baseline R2 heatmap ---
    pivot = results_df.pivot(index="target", columns="model", values="r2_mean")
    pivot = pivot.reindex(columns=["Ridge", "RF", "GBT"])
    pivot = pivot.loc[[t for t in avail_out if t in pivot.index]]

    fig, ax = plt.subplots(figsize=(COL_W * 1.2, COL_W * 1.6))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-0.5, vmax=1.0,
                   aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([_label(c) for c in pivot.index])
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = "white" if val < 0.3 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=color)
    fig.colorbar(im, ax=ax, shrink=0.6, label=r"$R^2$ (5-fold CV)")
    ax.set_title("Baseline Learnability", fontsize=9)
    _save(fig, out_dir, "ml_baseline_r2")

    # --- Fig: NRMAE bar chart ---
    fig, ax = plt.subplots(figsize=(PAGE_W, COL_W * 0.85))
    targets_sorted = [t for t in avail_out if t in results_df["target"].values]
    x_pos = np.arange(len(targets_sorted))
    width = 0.25
    for k, (model_name, color) in enumerate(zip(["Ridge", "RF", "GBT"],
                                                  [PALETTE[0], PALETTE[2], PALETTE[5]])):
        sub = results_df[results_df["model"] == model_name]
        nrmae_vals = []
        for t in targets_sorted:
            row = sub[sub["target"] == t]
            nrmae_vals.append(row["nrmae"].values[0] if len(row) > 0 else 0)
        ax.bar(x_pos + k * width, nrmae_vals, width, label=model_name,
               color=color, alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.set_xticks(x_pos + width)
    ax.set_xticklabels([_label(t) for t in targets_sorted], rotation=45, ha="right")
    ax.set_ylabel("NRMAE (lower = easier)")
    ax.legend(frameon=True, fancybox=False, edgecolor="0.7")
    ax.set_title("Normalized MAE by Target and Baseline Model")
    _save(fig, out_dir, "ml_baseline_nrmae")

    results_df.to_csv(os.path.join(out_dir, "ml_baseline_results.csv"), index=False)
    print(f"  Saved: ml_baseline_results.csv")

    return results_df


# ===================================================================
# 2. SAMPLE UNIQUENESS
# ===================================================================

def sample_uniqueness(df, out_dir):
    """Measure pairwise distances in output space to quantify inter-sample variation."""
    print("\n" + "=" * 70)
    print("SAMPLE UNIQUENESS ANALYSIS")
    print("=" * 70)

    avail_out = [c for c in OUTPUT_COLS if c in df.columns]
    if len(avail_out) < 3:
        print("  Insufficient output columns.")
        return

    Y = df[avail_out].values.astype(np.float64)
    valid = ~np.any(np.isnan(Y) | np.isinf(Y), axis=1)
    Y = Y[valid]
    design_names = df.loc[valid, "design_name"].values if "design_name" in df.columns else None

    scaler = StandardScaler()
    Y_scaled = scaler.fit_transform(Y)

    # Pairwise Euclidean distances in normalized output space
    dists = pdist(Y_scaled, metric="euclidean")
    dist_matrix = squareform(dists)

    # Nearest-neighbor distances (exclude self)
    np.fill_diagonal(dist_matrix, np.inf)
    nn_dists = dist_matrix.min(axis=1)
    np.fill_diagonal(dist_matrix, 0)

    print(f"  Samples: {len(Y_scaled)}")
    print(f"  Output dimensions: {len(avail_out)}")
    print(f"  Pairwise distance stats (normalized):")
    print(f"    Mean: {np.mean(dists):.3f}")
    print(f"    Std:  {np.std(dists):.3f}")
    print(f"    Min:  {np.min(dists):.3f}")
    print(f"    Max:  {np.max(dists):.3f}")
    print(f"  Nearest-neighbor distance stats:")
    print(f"    Mean: {np.mean(nn_dists):.3f}")
    print(f"    Std:  {np.std(nn_dists):.3f}")
    print(f"    Min:  {np.min(nn_dists):.4f}  (most similar pair)")
    print(f"    Max:  {np.max(nn_dists):.3f}  (most isolated sample)")

    # Duplicate detection
    near_dupes = np.sum(nn_dists < 0.1)
    print(f"  Near-duplicates (NN dist < 0.1): {near_dupes} ({100*near_dupes/len(Y_scaled):.1f}%)")

    # --- Fig: 2x2 uniqueness diagnostics ---
    fig, axes = plt.subplots(2, 2, figsize=(PAGE_W, PAGE_W * 0.7))

    # (a) Pairwise distance histogram
    ax = axes[0, 0]
    ax.hist(dists, bins=60, edgecolor="white", linewidth=0.3, alpha=0.85,
            color=PALETTE[0])
    ax.axvline(np.mean(dists), color=PALETTE[5], ls="--", lw=0.8, label=f"Mean={np.mean(dists):.2f}")
    ax.set_xlabel("Pairwise distance (normalized output space)")
    ax.set_ylabel("Count")
    ax.set_title("(a) Pairwise Distance Distribution")
    ax.legend(fontsize=6)

    # (b) NN distance histogram
    ax = axes[0, 1]
    ax.hist(nn_dists, bins=40, edgecolor="white", linewidth=0.3, alpha=0.85,
            color=PALETTE[2])
    ax.axvline(0.1, color=PALETTE[5], ls=":", lw=0.8, label="Near-dup threshold")
    ax.set_xlabel("Nearest-neighbor distance")
    ax.set_ylabel("Count")
    ax.set_title("(b) Nearest-Neighbor Distances")
    ax.legend(fontsize=6)

    # (c) t-SNE embedding
    ax = axes[1, 0]
    perplexity = min(30, len(Y_scaled) - 1)
    if len(Y_scaled) > 5:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                     init="pca", learning_rate="auto")
        Z = tsne.fit_transform(Y_scaled)
        if design_names is not None:
            designs = sorted(set(design_names))
            dstyle = _design_style(designs)
            for d in designs:
                mask = design_names == d
                ax.scatter(Z[mask, 0], Z[mask, 1], s=8, alpha=0.6,
                           color=dstyle[d][0], marker=dstyle[d][1],
                           label=_short(d), rasterized=True)
            ax.legend(fontsize=5, loc="best", frameon=True, fancybox=False, edgecolor="0.7")
        else:
            ax.scatter(Z[:, 0], Z[:, 1], s=6, alpha=0.5, color=PALETTE[0], rasterized=True)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("(c) t-SNE of Output Space")

    # (d) PCA explained variance of outputs
    ax = axes[1, 1]
    pca = PCA()
    pca.fit(Y_scaled)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    k = len(cumvar)
    ax.bar(range(1, k + 1), pca.explained_variance_ratio_, color=PALETTE[0],
           alpha=0.7, edgecolor="white", linewidth=0.3, label="Individual")
    ax2 = ax.twinx()
    ax2.plot(range(1, k + 1), cumvar, "o-", color=PALETTE[5], markersize=3, lw=0.9,
             label="Cumulative")
    ax2.axhline(0.95, color=PALETTE[3], ls="--", lw=0.6)
    n_95 = int(np.searchsorted(cumvar, 0.95)) + 1
    ax2.set_ylabel("Cumulative variance", fontsize=7)
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title(f"(d) Output PCA ({n_95} PCs for 95%)")
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=5, loc="center right")

    _save(fig, out_dir, "ml_sample_uniqueness")

    # Intrinsic dimensionality estimate (using PCA 95%)
    print(f"  Intrinsic dimensionality (PCA 95%): {n_95} / {len(avail_out)} output dims")


# ===================================================================
# 3. GEOMETRY DIVERSITY
# ===================================================================

def geometry_diversity(h5_path, df, out_dir, max_samples=200):
    """Quantify structural diversity of geometry masks."""
    print("\n" + "=" * 70)
    print("GEOMETRY DIVERSITY")
    print("=" * 70)

    with h5py.File(h5_path, "r") as f:
        all_ids = sorted(f.keys())
    ids = all_ids[:max_samples]
    n = len(ids)

    # Extract mid-plane geometry features
    mid_z_masks = []
    porosities = []
    surface_areas = []
    euler_chars = []

    with h5py.File(h5_path, "r") as f:
        for sid in ids:
            mask = f[sid]["mask"][()]
            nz = mask.shape[2]
            mid_z = nz // 2

            solid = (mask == BC_SOLID)
            # Core region (exclude buffer)
            buf = 10
            core = solid[buf:-buf, :, :]

            # Porosity
            porosity = 1.0 - np.mean(core)
            porosities.append(porosity)

            # Surface area (solid-fluid interface faces)
            faces_x = np.sum(core[1:] != core[:-1])
            faces_y = np.sum(core[:, 1:] != core[:, :-1])
            faces_z = np.sum(core[:, :, 1:] != core[:, :, :-1])
            sa = faces_x + faces_y + faces_z
            surface_areas.append(sa)

            # 2D Euler characteristic of mid-plane (connected components - holes)
            slice_2d = solid[buf:-buf, :, mid_z].astype(np.int32)
            # Approximate via vertices - edges + faces (for binary image)
            V = np.sum(slice_2d)
            E_h = np.sum(slice_2d[:, :-1] & slice_2d[:, 1:])
            E_v = np.sum(slice_2d[:-1, :] & slice_2d[1:, :])
            F = np.sum(slice_2d[:-1, :-1] & slice_2d[1:, :-1] &
                        slice_2d[:-1, 1:] & slice_2d[1:, 1:])
            euler = V - E_h - E_v + F
            euler_chars.append(euler)

            # Flattened mid-plane for pairwise comparison
            mid_z_masks.append(solid[buf:-buf, :, mid_z].astype(np.float32).flatten())

    porosities = np.array(porosities)
    surface_areas = np.array(surface_areas)
    euler_chars = np.array(euler_chars)
    mid_z_stack = np.array(mid_z_masks)

    # Pairwise Hamming distances between mid-plane masks
    hamming_dists = pdist(mid_z_stack, metric="hamming")

    design_names = []
    for sid in ids:
        row = df[df["sample_id"] == sid] if "sample_id" in df.columns else pd.DataFrame()
        if len(row) > 0:
            design_names.append(row.iloc[0]["design_name"])
        else:
            design_names.append("Unknown")
    design_names = np.array(design_names)

    # Intra-class vs inter-class distances
    designs_unique = sorted(set(design_names))
    intra_dists = []
    inter_dists = []
    dist_matrix = squareform(hamming_dists)
    for i in range(n):
        for j in range(i + 1, n):
            if design_names[i] == design_names[j]:
                intra_dists.append(dist_matrix[i, j])
            else:
                inter_dists.append(dist_matrix[i, j])

    print(f"  Samples analyzed: {n}")
    print(f"  Porosity range: [{porosities.min():.3f}, {porosities.max():.3f}]")
    print(f"  Surface area range: [{surface_areas.min():.0f}, {surface_areas.max():.0f}]")
    print(f"  Euler characteristic range: [{euler_chars.min():.0f}, {euler_chars.max():.0f}]")
    print(f"  Hamming distance (mid-plane):")
    print(f"    Intra-class mean: {np.mean(intra_dists):.4f}")
    print(f"    Inter-class mean: {np.mean(inter_dists):.4f}")
    print(f"    Ratio (inter/intra): {np.mean(inter_dists)/(np.mean(intra_dists)+1e-12):.2f}")

    # --- Fig: 2x2 geometry diversity ---
    fig, axes = plt.subplots(2, 2, figsize=(PAGE_W, PAGE_W * 0.7))
    dstyle = _design_style(designs_unique)

    # (a) Porosity vs surface area by design
    ax = axes[0, 0]
    for d in designs_unique:
        mask_d = design_names == d
        ax.scatter(porosities[mask_d], surface_areas[mask_d], s=10, alpha=0.6,
                   color=dstyle[d][0], marker=dstyle[d][1], label=_short(d),
                   rasterized=True)
    ax.set_xlabel("Porosity")
    ax.set_ylabel("Solid-fluid interface area")
    ax.set_title("(a) Porosity vs Surface Area")
    ax.legend(fontsize=5, frameon=True, fancybox=False, edgecolor="0.7")

    # (b) Euler characteristic distribution
    ax = axes[0, 1]
    for d in designs_unique:
        mask_d = design_names == d
        ax.hist(euler_chars[mask_d], bins=20, alpha=0.5, color=dstyle[d][0],
                label=_short(d))
    ax.set_xlabel("2D Euler characteristic (mid-plane)")
    ax.set_ylabel("Count")
    ax.set_title("(b) Topological Complexity")
    ax.legend(fontsize=5, frameon=True, fancybox=False, edgecolor="0.7")

    # (c) Intra vs inter class Hamming distance
    ax = axes[1, 0]
    bins = np.linspace(0, max(np.max(intra_dists), np.max(inter_dists)), 50)
    ax.hist(intra_dists, bins=bins, alpha=0.6, color=PALETTE[0],
            label=f"Intra-class (mean={np.mean(intra_dists):.3f})", density=True)
    ax.hist(inter_dists, bins=bins, alpha=0.6, color=PALETTE[5],
            label=f"Inter-class (mean={np.mean(inter_dists):.3f})", density=True)
    ax.set_xlabel("Hamming distance (mid-plane mask)")
    ax.set_ylabel("Density")
    ax.set_title("(c) Geometry Separability")
    ax.legend(fontsize=5, frameon=True, fancybox=False, edgecolor="0.7")

    # (d) t-SNE of geometry
    ax = axes[1, 1]
    if n > 5:
        perp = min(30, n - 1)
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                     init="pca", learning_rate="auto")
        Z = tsne.fit_transform(mid_z_stack)
        for d in designs_unique:
            mask_d = design_names == d
            ax.scatter(Z[mask_d, 0], Z[mask_d, 1], s=10, alpha=0.6,
                       color=dstyle[d][0], marker=dstyle[d][1], label=_short(d),
                       rasterized=True)
        ax.legend(fontsize=5, frameon=True, fancybox=False, edgecolor="0.7")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("(d) t-SNE of Geometry Masks")

    _save(fig, out_dir, "ml_geometry_diversity")


# ===================================================================
# 4. FIELD COMPLEXITY
# ===================================================================

def field_complexity(h5_path, df, out_dir, max_samples=100):
    """Analyze spatial complexity of output fields (gradients, spectra)."""
    print("\n" + "=" * 70)
    print("FIELD COMPLEXITY ANALYSIS")
    print("=" * 70)

    with h5py.File(h5_path, "r") as f:
        all_ids = sorted(f.keys())
    ids = all_ids[:max_samples]
    n = len(ids)

    # Per-sample statistics
    T_grad_means = []
    T_grad_maxes = []
    V_grad_means = []
    T_spectral_centroids = []
    T_spectral_bandwidths = []
    T_entropies = []
    design_labels = []

    with h5py.File(h5_path, "r") as f:
        for sid in ids:
            mask = f[sid]["mask"][()]
            T = f[sid]["temperature"][()].astype(np.float64)
            u = f[sid]["velocity"][()]
            nz = mask.shape[2]
            mid_z = nz // 2

            T_slice = T[:, :, mid_z]
            V_slice = np.linalg.norm(u[:, :, mid_z, :], axis=-1)
            solid_2d = (mask[:, :, mid_z] == BC_SOLID)

            # Gradient magnitude (finite differences)
            dTdx = np.diff(T_slice, axis=0)
            dTdy = np.diff(T_slice, axis=1)
            # Trim to common shape
            min_r, min_c = min(dTdx.shape[0], dTdy.shape[0]), min(dTdx.shape[1], dTdy.shape[1])
            grad_T = np.sqrt(dTdx[:min_r, :min_c]**2 + dTdy[:min_r, :min_c]**2)

            dVdx = np.diff(V_slice, axis=0)
            dVdy = np.diff(V_slice, axis=1)
            grad_V = np.sqrt(dVdx[:min_r, :min_c]**2 + dVdy[:min_r, :min_c]**2)

            T_grad_means.append(np.mean(grad_T))
            T_grad_maxes.append(np.max(grad_T))
            V_grad_means.append(np.mean(grad_V))

            # 2D FFT of temperature (power spectrum)
            T_centered = T_slice - T_slice.mean()
            fft2 = np.fft.fft2(T_centered)
            power = np.abs(fft2)**2
            # Radial average
            ny_fft, nx_fft = power.shape
            ky = np.fft.fftfreq(ny_fft)
            kx = np.fft.fftfreq(nx_fft)
            KX, KY = np.meshgrid(kx, ky)
            K_mag = np.sqrt(KX**2 + KY**2)
            # Spectral centroid (mean frequency weighted by power)
            total_power = np.sum(power)
            if total_power > 0:
                centroid = np.sum(K_mag * power) / total_power
                bandwidth = np.sqrt(np.sum((K_mag - centroid)**2 * power) / total_power)
            else:
                centroid, bandwidth = 0.0, 0.0
            T_spectral_centroids.append(centroid)
            T_spectral_bandwidths.append(bandwidth)

            # Spatial entropy of temperature (discretized)
            T_fluid = T_slice[~solid_2d]
            if len(T_fluid) > 10:
                hist, _ = np.histogram(T_fluid, bins=50, density=True)
                hist = hist[hist > 0]
                bin_width = (T_fluid.max() - T_fluid.min()) / 50 if T_fluid.max() > T_fluid.min() else 1.0
                entropy = -np.sum(hist * bin_width * np.log(hist * bin_width + 1e-30))
            else:
                entropy = 0.0
            T_entropies.append(entropy)

            row = df[df["sample_id"] == sid] if "sample_id" in df.columns else pd.DataFrame()
            design_labels.append(row.iloc[0]["design_name"] if len(row) > 0 else "Unknown")

    T_grad_means = np.array(T_grad_means)
    T_grad_maxes = np.array(T_grad_maxes)
    V_grad_means = np.array(V_grad_means)
    T_spectral_centroids = np.array(T_spectral_centroids)
    T_spectral_bandwidths = np.array(T_spectral_bandwidths)
    T_entropies = np.array(T_entropies)
    design_labels = np.array(design_labels)

    print(f"  Samples analyzed: {n}")
    print(f"  Temperature gradient (mean): min={T_grad_means.min():.4f}, "
          f"max={T_grad_means.max():.4f}, CV={np.std(T_grad_means)/(np.mean(T_grad_means)+1e-12):.3f}")
    print(f"  Temperature gradient (max):  min={T_grad_maxes.min():.2f}, "
          f"max={T_grad_maxes.max():.2f}")
    print(f"  Velocity gradient (mean):    min={V_grad_means.min():.5f}, "
          f"max={V_grad_means.max():.5f}")
    print(f"  Spectral centroid (T):       min={T_spectral_centroids.min():.4f}, "
          f"max={T_spectral_centroids.max():.4f}")
    print(f"  Temperature entropy:         min={T_entropies.min():.3f}, "
          f"max={T_entropies.max():.3f}")

    # --- Fig: 2x3 field complexity ---
    fig, axes = plt.subplots(2, 3, figsize=(PAGE_W, PAGE_W * 0.55))
    designs_unique = sorted(set(design_labels))
    dstyle = _design_style(designs_unique)

    # (a) T gradient mean distribution
    ax = axes[0, 0]
    for d in designs_unique:
        m = design_labels == d
        ax.hist(T_grad_means[m], bins=20, alpha=0.5, color=dstyle[d][0], label=_short(d))
    ax.set_xlabel(r"Mean $|\nabla T|$ (mid-plane)")
    ax.set_ylabel("Count")
    ax.set_title("(a) Thermal Gradient")

    # (b) V gradient mean distribution
    ax = axes[0, 1]
    for d in designs_unique:
        m = design_labels == d
        ax.hist(V_grad_means[m], bins=20, alpha=0.5, color=dstyle[d][0], label=_short(d))
    ax.set_xlabel(r"Mean $|\nabla |\mathbf{u}||$ (mid-plane)")
    ax.set_ylabel("Count")
    ax.set_title("(b) Velocity Gradient")

    # (c) Temperature entropy
    ax = axes[0, 2]
    for d in designs_unique:
        m = design_labels == d
        ax.hist(T_entropies[m], bins=20, alpha=0.5, color=dstyle[d][0], label=_short(d))
    ax.set_xlabel("Temperature entropy (fluid cells)")
    ax.set_ylabel("Count")
    ax.set_title("(c) Spatial Entropy")
    ax.legend(fontsize=4.5, frameon=True, fancybox=False, edgecolor="0.7")

    # (d) Spectral centroid vs bandwidth
    ax = axes[1, 0]
    for d in designs_unique:
        m = design_labels == d
        ax.scatter(T_spectral_centroids[m], T_spectral_bandwidths[m], s=8, alpha=0.6,
                   color=dstyle[d][0], marker=dstyle[d][1], label=_short(d), rasterized=True)
    ax.set_xlabel("Spectral centroid")
    ax.set_ylabel("Spectral bandwidth")
    ax.set_title("(d) Frequency Content")

    # (e) Grad T mean vs Grad V mean (cross-field complexity)
    ax = axes[1, 1]
    for d in designs_unique:
        m = design_labels == d
        ax.scatter(T_grad_means[m], V_grad_means[m], s=8, alpha=0.6,
                   color=dstyle[d][0], marker=dstyle[d][1], label=_short(d), rasterized=True)
    ax.set_xlabel(r"Mean $|\nabla T|$")
    ax.set_ylabel(r"Mean $|\nabla |\mathbf{u}||$")
    ax.set_title("(e) Cross-Field Complexity")

    # (f) Max gradient vs entropy
    ax = axes[1, 2]
    for d in designs_unique:
        m = design_labels == d
        ax.scatter(T_grad_maxes[m], T_entropies[m], s=8, alpha=0.6,
                   color=dstyle[d][0], marker=dstyle[d][1], label=_short(d), rasterized=True)
    ax.set_xlabel(r"Max $|\nabla T|$")
    ax.set_ylabel("Temperature entropy")
    ax.set_title("(f) Sharpness vs Entropy")
    ax.legend(fontsize=4.5, frameon=True, fancybox=False, edgecolor="0.7")

    _save(fig, out_dir, "ml_field_complexity")


# ===================================================================
# 5. INPUT-OUTPUT DIFFICULTY
# ===================================================================

def input_output_difficulty(df, out_dir):
    """Measure non-linear dependence and conditional variance between inputs and outputs."""
    print("\n" + "=" * 70)
    print("INPUT-OUTPUT RELATIONSHIP DIFFICULTY")
    print("=" * 70)

    avail_in = [c for c in INPUT_COLS if c in df.columns]
    avail_out = [c for c in OUTPUT_COLS if c in df.columns]

    if len(avail_in) < 2 or len(avail_out) < 2:
        print("  Insufficient columns.")
        return

    # Non-linearity: compare Spearman (captures monotonic) vs Pearson (linear only)
    # Large gap = non-linear relationship
    print("\n  Non-linearity detection (|Spearman| - |Pearson| gap):")
    print(f"  {'Input':20s} {'Output':20s} {'Pearson':>8s} {'Spearman':>8s} {'Gap':>8s} {'Type':>12s}")
    print("  " + "-" * 80)

    nonlinearity_data = []
    for ic in avail_in:
        for oc in avail_out:
            x = df[ic].values.astype(np.float64)
            y = df[oc].values.astype(np.float64)
            valid = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
            if np.sum(valid) < 10:
                continue
            pearson = abs(np.corrcoef(x[valid], y[valid])[0, 1])
            spearman = abs(sp_stats.spearmanr(x[valid], y[valid]).statistic)
            gap = spearman - pearson

            if gap > 0.1:
                rel_type = "NON-LINEAR"
            elif spearman > 0.5:
                rel_type = "linear"
            elif spearman > 0.2:
                rel_type = "weak"
            else:
                rel_type = "negligible"

            nonlinearity_data.append({
                "input": ic, "output": oc,
                "pearson": pearson, "spearman": spearman, "gap": gap,
                "type": rel_type,
            })

    nonlinearity_df = pd.DataFrame(nonlinearity_data)
    nonlinear = nonlinearity_df[nonlinearity_df["gap"] > 0.05].sort_values("gap", ascending=False)
    for _, row in nonlinear.head(15).iterrows():
        print(f"  {row['input']:20s} {row['output']:20s} "
              f"{row['pearson']:8.3f} {row['spearman']:8.3f} {row['gap']:8.3f} {row['type']:>12s}")

    n_nonlinear = len(nonlinearity_df[nonlinearity_df["gap"] > 0.1])
    n_total = len(nonlinearity_df)
    print(f"\n  Non-linear pairs (gap > 0.1): {n_nonlinear}/{n_total} "
          f"({100*n_nonlinear/n_total:.1f}%)")

    # Conditional variance: bin inputs and measure output variance within bins
    print("\n  Conditional variance analysis (heteroscedasticity):")
    print(f"  {'Output':20s} {'CV_within':>10s} {'CV_between':>10s} {'Ratio':>8s} {'Verdict':>12s}")
    print("  " + "-" * 65)

    hetero_data = []
    for oc in avail_out:
        y = df[oc].values.astype(np.float64)
        valid = ~(np.isnan(y) | np.isinf(y))
        if np.sum(valid) < 20:
            continue
        y_valid = y[valid]

        # Group by design and input bins
        within_vars = []
        between_means = []
        if "design_name" in df.columns:
            for design in df["design_name"].unique():
                sub_y = y[valid & (df["design_name"] == design).values]
                if len(sub_y) > 3:
                    within_vars.append(np.var(sub_y))
                    between_means.append(np.mean(sub_y))

        if len(within_vars) > 1:
            cv_within = np.sqrt(np.mean(within_vars)) / (np.abs(np.mean(y_valid)) + 1e-12)
            cv_between = np.std(between_means) / (np.abs(np.mean(y_valid)) + 1e-12)
            ratio = cv_within / (cv_between + 1e-12)

            if ratio > 0.8:
                verdict = "HIGH within"
            elif ratio > 0.3:
                verdict = "BALANCED"
            else:
                verdict = "Design-driven"

            print(f"  {oc:20s} {cv_within:10.4f} {cv_between:10.4f} {ratio:8.3f} {verdict:>12s}")
            hetero_data.append({"output": oc, "cv_within": cv_within,
                                "cv_between": cv_between, "ratio": ratio})

    # --- Fig: 2x2 difficulty plots ---
    fig, axes = plt.subplots(2, 2, figsize=(PAGE_W, PAGE_W * 0.7))

    # (a) Pearson vs Spearman heatmap
    ax = axes[0, 0]
    if len(nonlinearity_df) > 0:
        gap_pivot = nonlinearity_df.pivot(index="output", columns="input", values="gap")
        gap_pivot = gap_pivot.reindex(index=[o for o in avail_out if o in gap_pivot.index],
                                       columns=[i for i in avail_in if i in gap_pivot.columns])
        if gap_pivot.shape[0] > 0 and gap_pivot.shape[1] > 0:
            im = ax.imshow(gap_pivot.values, cmap="YlOrRd", vmin=0, vmax=0.3,
                           aspect="auto", interpolation="nearest")
            ax.set_xticks(range(len(gap_pivot.columns)))
            ax.set_xticklabels(gap_pivot.columns, rotation=45, ha="right", fontsize=6)
            ax.set_yticks(range(len(gap_pivot.index)))
            ax.set_yticklabels([_label(c) for c in gap_pivot.index], fontsize=6)
            for i in range(gap_pivot.shape[0]):
                for j in range(gap_pivot.shape[1]):
                    val = gap_pivot.values[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5)
            fig.colorbar(im, ax=ax, shrink=0.7, label="|Spearman|-|Pearson|")
    ax.set_title("(a) Non-linearity Map")

    # (b) Spearman heatmap
    ax = axes[0, 1]
    if len(nonlinearity_df) > 0:
        spear_pivot = nonlinearity_df.pivot(index="output", columns="input", values="spearman")
        spear_pivot = spear_pivot.reindex(index=[o for o in avail_out if o in spear_pivot.index],
                                           columns=[i for i in avail_in if i in spear_pivot.columns])
        if spear_pivot.shape[0] > 0 and spear_pivot.shape[1] > 0:
            im = ax.imshow(spear_pivot.values, cmap="Blues", vmin=0, vmax=1.0,
                           aspect="auto", interpolation="nearest")
            ax.set_xticks(range(len(spear_pivot.columns)))
            ax.set_xticklabels(spear_pivot.columns, rotation=45, ha="right", fontsize=6)
            ax.set_yticks(range(len(spear_pivot.index)))
            ax.set_yticklabels([_label(c) for c in spear_pivot.index], fontsize=6)
            for i in range(spear_pivot.shape[0]):
                for j in range(spear_pivot.shape[1]):
                    val = spear_pivot.values[i, j]
                    if not np.isnan(val):
                        color = "white" if val > 0.6 else "black"
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                fontsize=5, color=color)
            fig.colorbar(im, ax=ax, shrink=0.7, label="|Spearman|")
    ax.set_title("(b) Monotonic Dependence")

    # (c) Conditional variance bars
    ax = axes[1, 0]
    if hetero_data:
        hdf = pd.DataFrame(hetero_data)
        x_pos = np.arange(len(hdf))
        ax.bar(x_pos - 0.15, hdf["cv_within"], 0.3, label="Within-design CV",
               color=PALETTE[0], alpha=0.8)
        ax.bar(x_pos + 0.15, hdf["cv_between"], 0.3, label="Between-design CV",
               color=PALETTE[5], alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([_label(o) for o in hdf["output"]], rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("CV")
        ax.legend(fontsize=5, frameon=True, fancybox=False, edgecolor="0.7")
    ax.set_title("(c) Conditional Variance")

    # (d) Multi-target correlation (output interdependence)
    ax = axes[1, 1]
    out_available = [c for c in avail_out if c in df.columns]
    if len(out_available) > 2:
        out_corr = df[out_available].corr(method="spearman").abs()
        # Mask diagonal
        mask_tri = np.triu(np.ones_like(out_corr, dtype=bool), k=1)
        upper_vals = out_corr.values[mask_tri]
        ax.hist(upper_vals, bins=30, edgecolor="white", linewidth=0.3,
                alpha=0.85, color=PALETTE[2])
        ax.axvline(np.mean(upper_vals), color=PALETTE[5], ls="--", lw=0.8,
                   label=f"Mean={np.mean(upper_vals):.2f}")
        ax.set_xlabel("|Spearman| between output pairs")
        ax.set_ylabel("Count")
        ax.legend(fontsize=6)
    ax.set_title("(d) Output Interdependence")

    _save(fig, out_dir, "ml_io_difficulty")


# ===================================================================
# 6. TRAIN/TEST SPLIT QUALITY
# ===================================================================

def split_quality(df, out_dir):
    """Evaluate quality of stratified train/test splits."""
    print("\n" + "=" * 70)
    print("TRAIN/TEST SPLIT QUALITY")
    print("=" * 70)

    avail_out = [c for c in OUTPUT_COLS if c in df.columns]
    avail_in = [c for c in INPUT_COLS if c in df.columns]
    if "design_name" not in df.columns or len(avail_out) < 2:
        print("  Insufficient data for split analysis.")
        return

    n = len(df)

    # Strategy 1: Random 80/20
    rng = np.random.RandomState(42)
    idx = rng.permutation(n)
    split_point = int(0.8 * n)
    train_random = df.iloc[idx[:split_point]]
    test_random = df.iloc[idx[split_point:]]

    # Strategy 2: Stratified by design (ensure proportional representation)
    train_strat_parts = []
    test_strat_parts = []
    for design in df["design_name"].unique():
        sub = df[df["design_name"] == design]
        sub_idx = rng.permutation(len(sub))
        sp = int(0.8 * len(sub))
        train_strat_parts.append(sub.iloc[sub_idx[:sp]])
        test_strat_parts.append(sub.iloc[sub_idx[sp:]])
    train_strat = pd.concat(train_strat_parts)
    test_strat = pd.concat(test_strat_parts)

    # Strategy 3: Leave-one-design-out (hardest generalization test)
    designs = sorted(df["design_name"].unique())

    def _distribution_shift(train, test, cols):
        """Compute KS statistic and mean shift between train/test distributions."""
        shifts = {}
        for col in cols:
            t1 = train[col].dropna().values
            t2 = test[col].dropna().values
            if len(t1) < 5 or len(t2) < 5:
                continue
            ks_stat, ks_p = sp_stats.ks_2samp(t1, t2)
            mean_shift = abs(t1.mean() - t2.mean()) / (t1.std() + 1e-12)
            shifts[col] = {"ks": ks_stat, "ks_p": ks_p, "mean_shift": mean_shift}
        return shifts

    # Evaluate splits
    print("\n  Strategy 1: Random 80/20")
    shifts_random = _distribution_shift(train_random, test_random, avail_out + avail_in)
    max_ks = max((v["ks"] for v in shifts_random.values()), default=0)
    print(f"    Max KS statistic: {max_ks:.3f}")
    print(f"    Train design distribution:")
    for d in designs:
        n_train = len(train_random[train_random["design_name"] == d])
        n_test = len(test_random[test_random["design_name"] == d])
        print(f"      {_short(d):12s}  train={n_train:3d}  test={n_test:3d}")

    print("\n  Strategy 2: Stratified by design 80/20")
    shifts_strat = _distribution_shift(train_strat, test_strat, avail_out + avail_in)
    max_ks_strat = max((v["ks"] for v in shifts_strat.values()), default=0)
    print(f"    Max KS statistic: {max_ks_strat:.3f}")

    print("\n  Strategy 3: Leave-one-design-out (generalization)")
    lodo_results = []
    for held_out in designs:
        train_lodo = df[df["design_name"] != held_out]
        test_lodo = df[df["design_name"] == held_out]
        shifts_lodo = _distribution_shift(train_lodo, test_lodo, avail_out)
        avg_ks = np.mean([v["ks"] for v in shifts_lodo.values()])
        avg_shift = np.mean([v["mean_shift"] for v in shifts_lodo.values()])
        lodo_results.append({"held_out": held_out, "n_test": len(test_lodo),
                             "avg_ks": avg_ks, "avg_shift": avg_shift})
        print(f"    Hold out {_short(held_out):12s}: n_test={len(test_lodo):3d}, "
              f"avg KS={avg_ks:.3f}, avg mean shift={avg_shift:.2f}")

    # --- Fig: 2x2 split diagnostics ---
    fig, axes = plt.subplots(2, 2, figsize=(PAGE_W, PAGE_W * 0.7))

    # (a) KS statistics comparison: random vs stratified
    ax = axes[0, 0]
    cols_common = sorted(set(shifts_random.keys()) & set(shifts_strat.keys()))
    if cols_common:
        ks_random = [shifts_random[c]["ks"] for c in cols_common]
        ks_strat = [shifts_strat[c]["ks"] for c in cols_common]
        x_pos = np.arange(len(cols_common))
        ax.bar(x_pos - 0.15, ks_random, 0.3, label="Random", color=PALETTE[0], alpha=0.8)
        ax.bar(x_pos + 0.15, ks_strat, 0.3, label="Stratified", color=PALETTE[2], alpha=0.8)
        ax.axhline(0.2, color=PALETTE[5], ls=":", lw=0.7, label="Concern threshold")
        ax.set_xticks(x_pos)
        ax.set_xticklabels([c[:8] for c in cols_common], rotation=45, ha="right", fontsize=5)
        ax.set_ylabel("KS statistic")
        ax.legend(fontsize=5, frameon=True, fancybox=False, edgecolor="0.7")
    ax.set_title("(a) Distribution Shift (KS)")

    # (b) Leave-one-design-out shift
    ax = axes[0, 1]
    if lodo_results:
        lodo_df = pd.DataFrame(lodo_results)
        x_pos = np.arange(len(lodo_df))
        ax.bar(x_pos, lodo_df["avg_ks"], color=PALETTE[5], alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([_short(d) for d in lodo_df["held_out"]], rotation=45, ha="right")
        ax.set_ylabel("Avg KS statistic")
        ax.axhline(0.3, color="0.3", ls=":", lw=0.7)
    ax.set_title("(b) Leave-One-Design-Out Shift")

    # (c) Output QQ plot: train vs test (stratified, first 4 outputs)
    ax = axes[1, 0]
    plot_cols = avail_out[:4]
    for k, col in enumerate(plot_cols):
        t1 = np.sort(train_strat[col].dropna().values)
        t2 = np.sort(test_strat[col].dropna().values)
        # Interpolate to common length
        common_len = min(len(t1), len(t2))
        q1 = np.quantile(t1, np.linspace(0, 1, common_len))
        q2 = np.quantile(t2, np.linspace(0, 1, common_len))
        ax.scatter(q1, q2, s=3, alpha=0.5, color=PALETTE[k % len(PALETTE)],
                   label=_label(col), rasterized=True)
    lims = ax.get_xlim()
    ax.plot(lims, lims, "k--", lw=0.5, zorder=0)
    ax.set_xlabel("Train quantiles")
    ax.set_ylabel("Test quantiles")
    ax.legend(fontsize=5, frameon=True, fancybox=False, edgecolor="0.7")
    ax.set_title("(c) QQ: Train vs Test (Stratified)")

    # (d) Design balance in splits
    ax = axes[1, 1]
    x_pos = np.arange(len(designs))
    train_counts = [len(train_strat[train_strat["design_name"] == d]) / len(train_strat) for d in designs]
    test_counts = [len(test_strat[test_strat["design_name"] == d]) / len(test_strat) for d in designs]
    full_counts = [len(df[df["design_name"] == d]) / len(df) for d in designs]
    w = 0.25
    ax.bar(x_pos - w, full_counts, w, label="Full", color=PALETTE[3], alpha=0.8)
    ax.bar(x_pos, train_counts, w, label="Train", color=PALETTE[0], alpha=0.8)
    ax.bar(x_pos + w, test_counts, w, label="Test", color=PALETTE[5], alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([_short(d) for d in designs], rotation=45, ha="right")
    ax.set_ylabel("Proportion")
    ax.legend(fontsize=5, frameon=True, fancybox=False, edgecolor="0.7")
    ax.set_title("(d) Design Balance in Splits")

    _save(fig, out_dir, "ml_split_quality")


# ===================================================================
# 7. ML READINESS SUMMARY
# ===================================================================

def readiness_summary(df, baseline_results, out_dir):
    """Generate a single-page readiness scorecard."""
    print("\n" + "=" * 70)
    print("ML READINESS SCORECARD")
    print("=" * 70)

    checks = {}

    # 1. Sample count
    n = len(df)
    n_designs = df["design_name"].nunique() if "design_name" in df.columns else 1
    checks["Sample count"] = f"{n} samples, {n_designs} design classes"
    checks["Samples per class"] = f"min={df['design_name'].value_counts().min()}, " \
                                   f"max={df['design_name'].value_counts().max()}" \
        if "design_name" in df.columns else "N/A"

    # 2. Output diversity
    avail_out = [c for c in OUTPUT_COLS if c in df.columns]
    cvs = {c: df[c].std() / (df[c].mean() + 1e-12) for c in avail_out}
    low_cv = [c for c, v in cvs.items() if abs(v) < 0.15]
    checks["Output diversity (CV)"] = f"All CV >= 0.15" if not low_cv else \
        f"LOW CV in: {', '.join(low_cv)}"

    # 3. Dynamic range (orders of magnitude)
    for c in avail_out:
        vals = df[c].dropna()
        if len(vals) > 0 and vals.min() > 0:
            drange = np.log10(vals.max() / vals.min())
            if drange > 1:
                checks[f"Dynamic range ({c})"] = f"{drange:.1f} decades"

    # 4. Baseline headroom
    if baseline_results is not None and len(baseline_results) > 0:
        best_per_target = baseline_results.groupby("target")["r2_mean"].max()
        easy_targets = best_per_target[best_per_target > 0.95].index.tolist()
        hard_targets = best_per_target[best_per_target < 0.5].index.tolist()
        checks["Easy targets (R2>0.95)"] = ", ".join(easy_targets) if easy_targets else "None"
        checks["Hard targets (R2<0.5)"] = ", ".join(hard_targets) if hard_targets else "None"
        avg_best_r2 = best_per_target.mean()
        checks["Avg best baseline R2"] = f"{avg_best_r2:.3f}"

    # 5. Convergence quality
    if "converged" in df.columns:
        conv_rate = df["converged"].mean()
        checks["Convergence rate"] = f"{100*conv_rate:.1f}%"

    # Print scorecard
    for key, val in checks.items():
        status = "OK" if "LOW" not in str(val) and "None" != str(val) else "CHECK"
        print(f"  [{status:5s}] {key:30s}: {val}")

    # Save scorecard
    with open(os.path.join(out_dir, "ml_readiness_scorecard.txt"), "w") as fh:
        fh.write("ML READINESS SCORECARD\n")
        fh.write("=" * 60 + "\n")
        for key, val in checks.items():
            fh.write(f"  {key:30s}: {val}\n")
    print(f"\n  Saved: ml_readiness_scorecard.txt")


# ===================================================================
# MAIN
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ML-readiness analysis for cold plate CFD dataset")
    parser.add_argument("--data", type=str, default="dataset/data.h5",
                        help="Path to HDF5 dataset")
    parser.add_argument("--meta", type=str, default=None,
                        help="Path to metadata parquet")
    parser.add_argument("--out", type=str, default="analysis_results",
                        help="Output directory")
    parser.add_argument("--field-analysis", action="store_true",
                        help="Run field-level complexity analysis (reads HDF5 fields)")
    parser.add_argument("--max-field-samples", type=int, default=100,
                        help="Max samples for field analysis")
    parser.add_argument("--max-geo-samples", type=int, default=200,
                        help="Max samples for geometry diversity")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Error: {args.data} not found.")
        sys.exit(1)

    setup_style()
    os.makedirs(args.out, exist_ok=True)

    # Auto-detect metadata
    meta_path = args.meta
    if meta_path is None:
        candidate = os.path.join(os.path.dirname(args.data), "metadata.parquet")
        if os.path.exists(candidate):
            meta_path = candidate

    if meta_path and os.path.exists(meta_path):
        print(f"Loading metadata: {meta_path}")
        df = pd.read_parquet(meta_path)
    else:
        print("No parquet found, reading from HDF5 attributes...")
        rows = []
        with h5py.File(args.data, "r") as f:
            for sid in f.keys():
                row = dict(f[sid].attrs)
                row["sample_id"] = sid
                rows.append(row)
        df = pd.DataFrame(rows)

    print(f"Dataset: {len(df)} samples")

    # Run analyses
    baseline_results = baseline_learnability(df, args.out)
    sample_uniqueness(df, args.out)
    input_output_difficulty(df, args.out)
    split_quality(df, args.out)

    # These read HDF5 fields -- heavier
    geometry_diversity(args.data, df, args.out, max_samples=args.max_geo_samples)

    if args.field_analysis:
        field_complexity(args.data, df, args.out, max_samples=args.max_field_samples)

    readiness_summary(df, baseline_results, args.out)

    print("\n" + "=" * 70)
    print("ML READINESS ANALYSIS COMPLETE")
    print(f"All figures saved to: {args.out}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
