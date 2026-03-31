"""
Dataset Analysis & Validation for Cold Plate CFD Dataset
=========================================================
Validates physical accuracy, checks dataset diversity, and generates
publication-quality figures for a NeurIPS dataset paper.

All figures are saved as both PDF (vector, for paper) and PNG (for quick
preview).  Figures are sized for NeurIPS 2-column format:
  - Single column:  3.25 in  (82.55 mm)
  - Full page:      6.75 in  (171.45 mm)
  - Max height:     9.0  in  (228.6 mm)

Usage:
    python analyze_dataset.py                            # default: dataset/
    python analyze_dataset.py --data dataset/data.h5 --meta dataset/metadata.parquet
    python analyze_dataset.py --data dataset/data.h5 --field-analysis   # expensive PCA on fields
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
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from scipy import stats as sp_stats

# ===================================================================
# PLOT STYLE CONFIGURATION (NeurIPS publication quality)
# ===================================================================

# NeurIPS column widths in inches
COL_W = 3.25       # single column
PAGE_W = 6.75      # full width
MAX_H = 9.0        # max page height

# Colorblind-safe palette (Tol's muted, 7 colours)
PALETTE = ["#332288", "#88CCEE", "#44AA99", "#117733", "#999933", "#CC6677", "#882255"]

def setup_style():
    """Configure matplotlib for publication-quality output."""
    plt.rcParams.update({
        # Font
        "font.family":        "serif",
        "font.serif":         ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
        "mathtext.fontset":   "cm",
        "font.size":          8,
        "axes.titlesize":     9,
        "axes.labelsize":     8,
        "xtick.labelsize":    7,
        "ytick.labelsize":    7,
        "legend.fontsize":    7,
        "legend.title_fontsize": 8,
        # Lines & markers
        "lines.linewidth":    1.0,
        "lines.markersize":   3,
        "scatter.marker":     "o",
        # Axes
        "axes.linewidth":     0.6,
        "axes.grid":          False,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        # Ticks
        "xtick.major.width":  0.6,
        "ytick.major.width":  0.6,
        "xtick.minor.width":  0.4,
        "ytick.minor.width":  0.4,
        "xtick.direction":    "in",
        "ytick.direction":    "in",
        # Save
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.02,
        "figure.dpi":         150,
        # Layout
        "figure.constrained_layout.use": True,
    })


def _save(fig, out_dir, name):
    """Save figure as PDF + PNG and close."""
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  Saved: {name}.pdf / .png")


def _design_style(designs):
    """Return a dict mapping design name -> (color, marker)."""
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    return {d: (PALETTE[i % len(PALETTE)], markers[i % len(markers)])
            for i, d in enumerate(sorted(designs))}


# ===================================================================
# CONSTANTS
# ===================================================================

BC_SOLID = 1
CORE_NX, CORE_NY = 256, 256
BUF = 10

THRESHOLDS = {
    "mass_conservation":  {"pass": 0.01, "warn": 0.05},
    "mach_number":        {"pass": 0.15, "warn": 0.30},
    "density_deviation":  {"pass": 0.05, "warn": 0.10},
    "temp_lower_bound":   {"pass": 0.0,  "warn": -0.5},
    "nusselt_low":        1.0,
    "nusselt_high":       500.0,
    "f_re_high":          200.0,
    "divergence_free":    {"pass": 0.001, "warn": 0.01},
}

INPUT_COLS = ["feature_size", "spacing", "tau_fluid", "u_inlet_val", "heat_power"]
OUTPUT_COLS = ["R_th", "Nu", "f_friction", "P_pump", "COP", "Q_vol",
               "T_max_surface", "T_avg_surface", "delta_T_max", "sigma_T", "fluid_temp_rise"]
CONTEXT_COLS = ["porosity", "Dh", "Re", "max_velocity", "max_temperature", "mean_temperature"]

# Pretty labels (LaTeX)
PRETTY = {
    "feature_size": "Feature size",
    "spacing": "Spacing",
    "tau_fluid": r"$\tau_f$",
    "u_inlet_val": r"$u_{\mathrm{in}}$",
    "heat_power": r"$\dot{q}$",
    "R_th": r"$R_{\mathrm{th}}$",
    "Nu": r"$\mathrm{Nu}$",
    "f_friction": r"$f$",
    "P_pump": r"$P_{\mathrm{pump}}$",
    "COP": r"$\mathrm{COP}$",
    "Q_vol": r"$\dot{Q}$",
    "T_max_surface": r"$T_{\max,s}$",
    "T_avg_surface": r"$\bar{T}_s$",
    "delta_T_max": r"$\Delta T_{\max}$",
    "sigma_T": r"$\sigma_T$",
    "fluid_temp_rise": r"$\Delta T_f$",
    "porosity": r"$\phi$",
    "Dh": r"$D_h$",
    "Re": r"$\mathrm{Re}$",
    "max_velocity": r"$u_{\max}$",
    "max_temperature": r"$T_{\max}$",
    "mean_temperature": r"$\bar{T}$",
    "mass_err": r"$\epsilon_{\dot{m}}$",
    "mach": r"$\mathrm{Ma}$",
    "rho_dev": r"$|\rho - 1|_{\max}$",
    "T_lower_bound": r"$T_{\min} - T_{\mathrm{in}}$",
    "delta_P": r"$\Delta P$",
    "mean_div": r"$\langle|\nabla\cdot\mathbf{u}|\rangle$",
    "energy_err": r"$\epsilon_E$",
    "f_Re": r"$f \cdot \mathrm{Re}$",
}


def _label(col):
    return PRETTY.get(col, col)


# Short design labels for compact legends
SHORT_DESIGN = {
    "Straight Channels": "Straight",
    "Pin-Fin Staggered": "Pin-Fin",
    "Gyroid TPMS": "Gyroid",
    "Schwarz P TPMS": "Schwarz P",
    "Schwarz D TPMS": "Schwarz D",
}


def _short(name):
    return SHORT_DESIGN.get(name, name[:10])


# ===================================================================
# 1. PER-SAMPLE PHYSICAL VALIDATION
# ===================================================================

def validate_sample(h5_path, sample_id):
    """Run all physical checks on a single sample. Returns a dict of results."""
    with h5py.File(h5_path, "r") as f:
        grp = f[sample_id]
        mask = grp["mask"][()]
        u = grp["velocity"][()]
        T = grp["temperature"][()]
        P = grp["pressure"][()]
        hs = grp["heat_source"][()]
        attrs = dict(grp.attrs)

    nx, ny, nz = mask.shape
    fluid = (mask != BC_SOLID)
    t_inlet = 25.0
    checks = {"sample_id": sample_id}

    # 1. Mass conservation
    rho = P * 3.0
    inlet_fluid = (mask[1, :, :] != BC_SOLID)
    outlet_fluid = (mask[-2, :, :] != BC_SOLID)
    mdot_in = float(np.sum(rho[1, :, :][inlet_fluid] * u[1, :, :, 0][inlet_fluid]))
    mdot_out = float(np.sum(rho[-2, :, :][outlet_fluid] * u[-2, :, :, 0][outlet_fluid]))
    mass_err = abs(mdot_in - mdot_out) / (abs(mdot_in) + 1e-12)
    checks["mass_err"] = mass_err
    checks["mass_pass"] = mass_err < THRESHOLDS["mass_conservation"]["pass"]
    checks["mass_warn"] = mass_err < THRESHOLDS["mass_conservation"]["warn"]

    # 2. Mach number
    v_mag = np.linalg.norm(u, axis=-1)
    max_v = float(np.max(v_mag[fluid])) if np.any(fluid) else 0.0
    mach = max_v * np.sqrt(3.0)
    checks["mach"] = mach
    checks["mach_pass"] = mach < THRESHOLDS["mach_number"]["pass"]
    checks["mach_warn"] = mach < THRESHOLDS["mach_number"]["warn"]

    # 3. Density deviation (Mach-aware: compressibility scales as O(Ma²))
    rho_dev = float(np.max(np.abs(rho[fluid] - 1.0))) if np.any(fluid) else 0.0
    checks["rho_dev"] = rho_dev
    rho_dev_expected = max(THRESHOLDS["density_deviation"]["pass"], 2.0 * mach ** 2)
    rho_dev_warn_th = max(THRESHOLDS["density_deviation"]["warn"], 3.0 * mach ** 2)
    checks["rho_dev_pass"] = rho_dev < rho_dev_expected
    checks["rho_dev_warn"] = rho_dev < rho_dev_warn_th

    # 4. Temperature lower bound
    T_min_fluid = float(np.min(T[fluid])) if np.any(fluid) else t_inlet
    T_lb = T_min_fluid - t_inlet
    checks["T_lower_bound"] = T_lb
    checks["T_lb_pass"] = T_lb >= THRESHOLDS["temp_lower_bound"]["pass"]
    checks["T_lb_warn"] = T_lb >= THRESHOLDS["temp_lower_bound"]["warn"]

    # 5. Pressure drop sign
    rho_in = float(np.mean(rho[1, :, :]))
    rho_out = float(np.mean(rho[-2, :, :]))
    delta_P = (rho_in - rho_out) / 3.0
    checks["delta_P"] = delta_P
    checks["delta_P_pass"] = delta_P > 0

    # 6. Nusselt range
    Nu = attrs.get("Nu", 0.0)
    checks["Nu"] = Nu
    checks["Nu_pass"] = THRESHOLDS["nusselt_low"] <= Nu <= THRESHOLDS["nusselt_high"]

    # 7. f*Re range
    f = attrs.get("f_friction", 0.0)
    Re = attrs.get("Re", 1.0)
    f_Re = f * Re
    checks["f_Re"] = f_Re
    checks["f_Re_pass"] = 0 < f_Re < THRESHOLDS["f_re_high"]

    # 8. Velocity divergence
    ux, uy, uz = u[:, :, :, 0], u[:, :, :, 1], u[:, :, :, 2]
    dux_dx = (ux[2:, 1:-1, 1:-1] - ux[:-2, 1:-1, 1:-1]) / 2.0
    duy_dy = (uy[1:-1, 2:, 1:-1] - uy[1:-1, :-2, 1:-1]) / 2.0
    duz_dz = (uz[1:-1, 1:-1, 2:] - uz[1:-1, 1:-1, :-2]) / 2.0
    div_u = dux_dx + duy_dy + duz_dz
    interior_fluid = fluid[1:-1, 1:-1, 1:-1]
    if np.any(interior_fluid):
        mean_div = float(np.mean(np.abs(div_u[interior_fluid])))
        max_div = float(np.max(np.abs(div_u[interior_fluid])))
    else:
        mean_div, max_div = 0.0, 0.0
    checks["mean_div"] = mean_div
    checks["max_div"] = max_div
    checks["div_pass"] = mean_div < THRESHOLDS["divergence_free"]["pass"]
    checks["div_warn"] = mean_div < THRESHOLDS["divergence_free"]["warn"]

    # 9. Energy balance
    # Heat injected per timestep: sum over cells of (1 - 0.5/tau) * heat_power * mask_val
    # At steady state this equals convective transport: Q_vol * (T_out - T_in)
    # Heat source is applied at z=0 (solid base), so use tau_th_solid
    heat_power = attrs.get("heat_power", 0.0)
    tau_th_fluid = 0.5 + (attrs.get("tau_fluid", 0.52) - 0.5) / 7.0
    tau_th_solid = 0.5 + 628.0 * (tau_th_fluid - 0.5)
    Q_source_rate = (1.0 - 0.5 / tau_th_solid) * heat_power * float(np.sum(hs))

    T_inlet_avg = float(np.mean(T[1, :, :][inlet_fluid])) if np.any(inlet_fluid) else t_inlet
    T_outlet_avg = float(np.mean(T[-2, :, :][outlet_fluid])) if np.any(outlet_fluid) else t_inlet
    Q_vol = attrs.get("Q_vol", 0.0)
    Q_conv = abs(Q_vol * (T_outlet_avg - T_inlet_avg))

    energy_err = abs(Q_source_rate - Q_conv) / (Q_source_rate + 1e-12) if Q_source_rate > 1e-12 else 0.0
    checks["Q_source_rate"] = Q_source_rate
    checks["Q_conv"] = Q_conv
    checks["energy_err"] = energy_err
    checks["energy_pass"] = energy_err < 0.20
    checks["energy_warn"] = energy_err < 0.40

    return checks


# ===================================================================
# 2. PHYSICAL VALIDATION REPORT & FIGURE
# ===================================================================

def physical_report(checks_df, out_dir):
    """Print summary and generate validation distributions figure."""
    pass_cols = [c for c in checks_df.columns if c.endswith("_pass")]

    print("\n" + "=" * 70)
    print("PHYSICAL VALIDATION REPORT")
    print("=" * 70)

    n = len(checks_df)
    summary_rows = []
    check_names = [c.replace("_pass", "") for c in pass_cols]
    for name in check_names:
        p_col = f"{name}_pass"
        w_col = f"{name}_warn"
        n_pass = int(checks_df[p_col].sum())
        n_warn = int(checks_df[w_col].sum()) - n_pass if w_col in checks_df.columns else 0
        n_fail = n - n_pass - n_warn
        summary_rows.append({"check": name, "PASS": n_pass, "WARN": n_warn, "FAIL": n_fail})
        status = "OK" if n_fail == 0 else "ISSUES"
        print(f"  {name:25s}  PASS={n_pass:4d}  WARN={n_warn:4d}  FAIL={n_fail:4d}  [{status}]")

    # --- Figure: 3x3 validation histograms ---
    fig, axes = plt.subplots(3, 3, figsize=(PAGE_W, PAGE_W * 0.75))

    metric_cols = ["mass_err", "mach", "rho_dev", "T_lower_bound", "delta_P",
                   "mean_div", "energy_err", "Nu", "f_Re"]

    threshold_lines = {
        "mass_err":    [(0.01, "pass"), (0.05, "fail")],
        "mach":        [(0.15, "pass"), (0.30, "fail")],
        "energy_err":  [(0.05, "pass"), (0.10, "fail")],
        "mean_div":    [(0.001, "pass"), (0.01, "fail")],
        "rho_dev":     [(0.05, "pass"), (0.10, "fail")],
    }

    for ax, col in zip(axes.flat, metric_cols):
        data = checks_df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) == 0:
            ax.set_xlabel(_label(col))
            continue
        ax.hist(data, bins=40, edgecolor="white", linewidth=0.3,
                alpha=0.85, color=PALETTE[0])
        ax.set_xlabel(_label(col))
        ax.set_ylabel("Count")

        if col in threshold_lines:
            for val, kind in threshold_lines[col]:
                color = "#117733" if kind == "pass" else "#CC3311"
                style = "--" if kind == "pass" else ":"
                ax.axvline(val, color=color, ls=style, lw=0.8, zorder=5)

    _save(fig, out_dir, "physical_validation")

    # Worst offenders
    worst = checks_df.nlargest(5, "energy_err")[["sample_id", "energy_err", "mass_err", "mach", "mean_div"]]
    print("\n  Top 5 worst energy balance:")
    print(worst.to_string(index=False))

    return pd.DataFrame(summary_rows)


# ===================================================================
# 3. DATASET DIVERSITY & COVERAGE
# ===================================================================

def coverage_analysis(df, out_dir):
    """Analyze parameter space coverage and output diversity."""
    print("\n" + "=" * 70)
    print("DATASET DIVERSITY & COVERAGE")
    print("=" * 70)

    # --- Sample counts ---
    print("\n  Sample counts by design type:")
    design_counts = df["design_name"].value_counts()
    for name, count in design_counts.items():
        print(f"    {name:25s}  {count:4d}")

    if "heat_source_type" in df.columns:
        print("\n  Sample counts by heat source type:")
        hs_counts = df["heat_source_type"].value_counts()
        for name, count in hs_counts.items():
            print(f"    {name:25s}  {count:4d}")

    # --- Output stats ---
    avail_output = [c for c in OUTPUT_COLS if c in df.columns]
    print("\n  Output metric statistics:")
    stats_table = df[avail_output].describe().T
    stats_table["CV"] = stats_table["std"] / (stats_table["mean"].abs() + 1e-12)
    stats_table["skew"] = df[avail_output].skew()
    print(stats_table[["count", "mean", "std", "min", "25%", "50%", "75%", "max", "CV", "skew"]].to_string())

    low_cv = stats_table[stats_table["CV"] < 0.1]
    if len(low_cv) > 0:
        print(f"\n  WARNING: Low diversity (CV < 0.1) in: {list(low_cv.index)}")
    else:
        print("\n  All output metrics have CV >= 0.1 (good diversity).")

    # --- Fig: Input pairwise scatter matrix ---
    avail_input = [c for c in INPUT_COLS if c in df.columns]
    if avail_input and "design_name" in df.columns:
        designs = sorted(df["design_name"].unique())
        dstyle = _design_style(designs)
        n_inp = len(avail_input)

        fig, axes = plt.subplots(n_inp, n_inp, figsize=(PAGE_W, PAGE_W))
        for i, pi in enumerate(avail_input):
            for j, pj in enumerate(avail_input):
                ax = axes[i, j]
                if i == j:
                    # Marginal histogram
                    for d in designs:
                        sub = df[df["design_name"] == d]
                        ax.hist(sub[pi], bins=20, alpha=0.45,
                                color=dstyle[d][0], label=_short(d))
                else:
                    for d in designs:
                        sub = df[df["design_name"] == d]
                        ax.scatter(sub[pj], sub[pi], s=4, alpha=0.4,
                                   color=dstyle[d][0], marker=dstyle[d][1],
                                   rasterized=True)
                # Axis labels on edges only
                if i == n_inp - 1:
                    ax.set_xlabel(_label(pj))
                else:
                    ax.set_xticklabels([])
                if j == 0:
                    ax.set_ylabel(_label(pi))
                else:
                    ax.set_yticklabels([])

        handles = [plt.Line2D([0], [0], marker=dstyle[d][1], color=dstyle[d][0],
                               ls="", markersize=4, label=_short(d)) for d in designs]
        fig.legend(handles=handles, loc="upper right", frameon=True,
                   fancybox=False, edgecolor="0.7", fontsize=6,
                   bbox_to_anchor=(0.98, 0.98))
        _save(fig, out_dir, "input_coverage")

    # --- Fig: Output violin plots by design ---
    if avail_output and "design_name" in df.columns:
        designs_sorted = sorted(df["design_name"].unique())
        dstyle = _design_style(designs_sorted)
        n_out = len(avail_output)
        ncols = 4
        nrows = (n_out + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(PAGE_W, 2.2 * nrows))
        axes_flat = axes.flatten()

        for idx, col in enumerate(avail_output):
            ax = axes_flat[idx]
            data_groups = []
            group_colors = []
            labels = []
            for d in designs_sorted:
                vals = df[df["design_name"] == d][col].dropna().values
                if len(vals) > 0:
                    data_groups.append(vals)
                    group_colors.append(dstyle[d][0])
                    labels.append(_short(d))

            if data_groups:
                parts = ax.violinplot(data_groups, showmedians=True,
                                      showextrema=False)
                for i, body in enumerate(parts["bodies"]):
                    body.set_facecolor(group_colors[i])
                    body.set_edgecolor("black")
                    body.set_linewidth(0.4)
                    body.set_alpha(0.7)
                parts["cmedians"].set_color("black")
                parts["cmedians"].set_linewidth(0.8)
                ax.set_xticks(range(1, len(labels) + 1))
                ax.set_xticklabels(labels, rotation=45, ha="right")

            ax.set_ylabel(_label(col))

        for idx in range(len(avail_output), len(axes_flat)):
            axes_flat[idx].set_visible(False)

        _save(fig, out_dir, "output_distributions")


def correlation_analysis(df, out_dir):
    """Compute and plot input-output Spearman correlation heatmap."""
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    all_cols = INPUT_COLS + CONTEXT_COLS + OUTPUT_COLS
    avail = [c for c in all_cols if c in df.columns]
    if len(avail) < 4:
        print("  Insufficient columns for correlation analysis.")
        return

    corr = df[avail].corr(method="spearman")
    labels = [_label(c) for c in avail]

    fig, ax = plt.subplots(figsize=(PAGE_W, PAGE_W * 0.92))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal",
                   interpolation="nearest")

    ax.set_xticks(range(len(avail)))
    ax.set_yticks(range(len(avail)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)

    # Annotate cells
    for i in range(len(avail)):
        for j in range(len(avail)):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=4.5, color=color)

    # Divider lines between input / context / output blocks
    n_in = sum(1 for c in INPUT_COLS if c in avail)
    n_ctx = sum(1 for c in CONTEXT_COLS if c in avail)
    for pos in [n_in - 0.5, n_in + n_ctx - 0.5]:
        ax.axhline(pos, color="black", lw=0.6)
        ax.axvline(pos, color="black", lw=0.6)

    cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
    cbar.ax.set_ylabel(r"Spearman $\rho$", rotation=270, labelpad=12)

    _save(fig, out_dir, "correlation_matrix")

    # Print strong cross-correlations
    avail_in = [c for c in INPUT_COLS + CONTEXT_COLS if c in df.columns]
    avail_out = [c for c in OUTPUT_COLS if c in df.columns]
    print("\n  Strongest input->output correlations (|r| > 0.4):")
    pairs = []
    for ic in avail_in:
        for oc in avail_out:
            r = corr.loc[ic, oc] if ic in corr.index and oc in corr.columns else 0
            if abs(r) > 0.4:
                pairs.append((ic, oc, r))
    pairs.sort(key=lambda x: -abs(x[2]))
    for ic, oc, r in pairs[:20]:
        print(f"    {ic:20s} -> {oc:20s}  r={r:+.3f}")
    if not pairs:
        print("    (none found)")


# ===================================================================
# 4. PHYSICS RELATIONSHIP PLOTS
# ===================================================================

def physics_plots(df, out_dir):
    """Generate physics scatter plots: Nu vs Re, f vs Re, Pareto front, etc."""
    print("\n" + "=" * 70)
    print("PHYSICS RELATIONSHIP PLOTS")
    print("=" * 70)

    designs = sorted(df["design_name"].unique()) if "design_name" in df.columns else ["all"]
    dstyle = _design_style(designs)

    # --- Main 3x3 grid ---
    fig, axes = plt.subplots(3, 3, figsize=(PAGE_W, PAGE_W * 0.88))

    plot_specs = [
        ("Re", "Nu",            False, True,  None),
        ("Re", "f_friction",    True,  True,  "moody"),
        ("Re", "P_pump",        False, True,  None),
        ("Re", "R_th",          False, False,  None),
        ("Re", "COP",           False, True,  None),
        ("porosity", "Nu",      False, False,  None),
        ("heat_power", "T_max_surface", False, False, None),
        ("porosity", "f_friction",      False, True,  None),
        ("P_pump", "R_th",      True,  True,  "pareto"),
    ]

    for ax, (xcol, ycol, logx, logy, special) in zip(axes.flat, plot_specs):
        if xcol not in df.columns or ycol not in df.columns:
            continue
        for d in designs:
            sub = df[df["design_name"] == d] if "design_name" in df.columns else df
            x = sub[xcol].replace([np.inf, -np.inf], np.nan).dropna()
            y = sub.loc[x.index, ycol].replace([np.inf, -np.inf], np.nan).dropna()
            x = x.loc[y.index]
            ax.scatter(x, y, s=6, alpha=0.5, color=dstyle[d][0],
                       marker=dstyle[d][1], rasterized=True, label=_short(d))
        ax.set_xlabel(_label(xcol))
        ax.set_ylabel(_label(ycol))

        if logx:
            ax.set_xscale("log")
        if logy:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ymin = df[ycol].replace([np.inf, -np.inf], np.nan).dropna()
                if len(ymin) > 0 and ymin.min() > 0:
                    ax.set_yscale("log")

        # Reference lines for f vs Re (Moody chart laminar limits)
        if special == "moody" and "Re" in df.columns:
            re_range = np.logspace(
                np.log10(max(df["Re"].min(), 1)),
                np.log10(df["Re"].max()), 100)
            ax.plot(re_range, 16.0 / re_range, color="0.3", ls="--", lw=0.8,
                    label=r"$16/\mathrm{Re}$ (pipe)", zorder=1)
            ax.plot(re_range, 24.0 / re_range, color="0.3", ls=":", lw=0.8,
                    label=r"$24/\mathrm{Re}$ (plates)", zorder=1)
            ax.legend(fontsize=5, loc="upper right", frameon=True,
                      fancybox=False, edgecolor="0.8")

    # Single shared legend at bottom
    handles = [plt.Line2D([0], [0], marker=dstyle[d][1], color=dstyle[d][0],
                           ls="", markersize=4, label=_short(d)) for d in designs]
    fig.legend(handles=handles, loc="lower center", ncol=len(designs),
               frameon=True, fancybox=False, edgecolor="0.7",
               bbox_to_anchor=(0.5, -0.02))

    _save(fig, out_dir, "physics_plots")

    # --- Standalone Pareto front figure (single-column, for paper main body) ---
    if "P_pump" in df.columns and "R_th" in df.columns:
        fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.85))
        for d in designs:
            sub = df[df["design_name"] == d] if "design_name" in df.columns else df
            x = sub["P_pump"].replace([np.inf, -np.inf], np.nan).dropna()
            y = sub.loc[x.index, "R_th"].replace([np.inf, -np.inf], np.nan).dropna()
            x = x.loc[y.index]
            ax.scatter(x, y, s=8, alpha=0.55, color=dstyle[d][0],
                       marker=dstyle[d][1], rasterized=True, label=_short(d))
        ax.set_xlabel(_label("P_pump"))
        ax.set_ylabel(_label("R_th"))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=6, frameon=True, fancybox=False, edgecolor="0.7")
        _save(fig, out_dir, "pareto_front")


# ===================================================================
# 5. FIELD VARIABILITY (PCA)
# ===================================================================

def field_variability_analysis(h5_path, sample_ids, out_dir, max_samples=100):
    """Analyze inter-sample field variability via per-voxel variance and PCA."""
    print("\n" + "=" * 70)
    print("FIELD VARIABILITY ANALYSIS")
    print("=" * 70)

    ids = sample_ids[:max_samples]
    n = len(ids)
    print(f"  Analyzing {n} samples (max {max_samples})...")

    with h5py.File(h5_path, "r") as f:
        shape = f[ids[0]]["temperature"].shape
    nx, ny, nz = shape

    # Running mean + variance (Welford)
    T_mean = np.zeros(shape, dtype=np.float64)
    T_M2 = np.zeros(shape, dtype=np.float64)
    V_mean = np.zeros(shape, dtype=np.float64)
    V_M2 = np.zeros(shape, dtype=np.float64)

    mid_z = nz // 2
    slice_size = nx * ny
    T_slices = np.zeros((n, slice_size), dtype=np.float32)
    V_slices = np.zeros((n, slice_size), dtype=np.float32)

    with h5py.File(h5_path, "r") as f:
        for i, sid in enumerate(ids):
            T = f[sid]["temperature"][()].astype(np.float64)
            u = f[sid]["velocity"][()]
            v_mag = np.linalg.norm(u, axis=-1).astype(np.float64)

            delta_T = T - T_mean
            T_mean += delta_T / (i + 1)
            T_M2 += delta_T * (T - T_mean)

            delta_V = v_mag - V_mean
            V_mean += delta_V / (i + 1)
            V_M2 += delta_V * (v_mag - V_mean)

            T_slices[i] = T[:, :, mid_z].flatten()
            V_slices[i] = v_mag[:, :, mid_z].flatten()

    T_var = T_M2 / max(n - 1, 1)
    V_var = V_M2 / max(n - 1, 1)

    # --- Fig: Variance heatmaps ---
    fig, axes = plt.subplots(2, 3, figsize=(PAGE_W, PAGE_W * 0.55))
    slices_info = [
        (T_var[:, :, 0],        r"$\mathrm{Var}[T]$, $z{=}0$"),
        (T_var[:, :, mid_z],    r"$\mathrm{Var}[T]$, $z{=}\mathrm{mid}$"),
        (T_var[:, ny // 2, :],  r"$\mathrm{Var}[T]$, $y{=}\mathrm{mid}$"),
        (V_var[:, :, 0],        r"$\mathrm{Var}[|\mathbf{u}|]$, $z{=}0$"),
        (V_var[:, :, mid_z],    r"$\mathrm{Var}[|\mathbf{u}|]$, $z{=}\mathrm{mid}$"),
        (V_var[:, ny // 2, :],  r"$\mathrm{Var}[|\mathbf{u}|]$, $y{=}\mathrm{mid}$"),
    ]
    for ax, (data, title) in zip(axes.flat, slices_info):
        im = ax.imshow(data.T, origin="lower", aspect="auto", cmap="inferno",
                       rasterized=True)
        cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
        cbar.ax.tick_params(labelsize=5)
        ax.set_title(title, fontsize=7)
        ax.tick_params(labelsize=5)
    _save(fig, out_dir, "field_variance")

    # --- Fig: PCA explained variance ---
    for field_name, slices_data in [("Temperature", T_slices), ("Velocity", V_slices)]:
        centered = slices_data - slices_data.mean(axis=0)
        try:
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            explained = (S ** 2) / np.sum(S ** 2)
            cumulative = np.cumsum(explained)

            n_90 = int(np.searchsorted(cumulative, 0.90)) + 1
            n_95 = int(np.searchsorted(cumulative, 0.95)) + 1
            n_99 = int(np.searchsorted(cumulative, 0.99)) + 1
            print(f"  {field_name} PCA (z=mid slice): {n_90} for 90%, "
                  f"{n_95} for 95%, {n_99} for 99% variance")
        except Exception as e:
            print(f"  {field_name} PCA failed: {e}")
            continue

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(PAGE_W, COL_W * 0.7))
        k = min(50, len(explained))
        ax1.bar(range(1, k + 1), explained[:k], color=PALETTE[0], width=0.8,
                edgecolor="white", linewidth=0.3)
        ax1.set_xlabel("Component index")
        ax1.set_ylabel("Explained variance ratio")

        ax2.plot(range(1, k + 1), cumulative[:k], "o-", markersize=2.5,
                 color=PALETTE[0], lw=0.9)
        for threshold, label, color in [(0.90, "90%", "#117733"),
                                         (0.95, "95%", "#999933"),
                                         (0.99, "99%", "#CC6677")]:
            ax2.axhline(threshold, color=color, ls="--", lw=0.6)
            ax2.text(k * 0.92, threshold + 0.01, label, fontsize=6,
                     color=color, va="bottom", ha="right")
        ax2.set_xlabel("Number of components")
        ax2.set_ylabel("Cumulative explained variance")
        ax2.set_ylim(0, 1.05)
        _save(fig, out_dir, f"pca_{field_name.lower()}")


# ===================================================================
# 6. OUTLIER DETECTION
# ===================================================================

def outlier_analysis(df, out_dir):
    """Flag statistical outliers using IQR method."""
    print("\n" + "=" * 70)
    print("OUTLIER DETECTION")
    print("=" * 70)

    avail = [c for c in OUTPUT_COLS if c in df.columns]
    outlier_flags = pd.DataFrame(index=df.index)
    for col in avail:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_flags[col] = (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)

    total_outliers = outlier_flags.any(axis=1).sum()
    print(f"  Samples flagged as outliers in at least one metric: {total_outliers}/{len(df)}")
    print("\n  Outlier counts per metric:")
    for col in avail:
        n_out = outlier_flags[col].sum()
        print(f"    {col:25s}  {n_out:4d}  ({100*n_out/len(df):.1f}%)")

    # --- Fig: Outlier scatter plots ---
    pairs = [("Nu", "f_friction"), ("R_th", "COP"),
             ("P_pump", "R_th"), ("Re", "Nu")]
    fig, axes = plt.subplots(2, 2, figsize=(PAGE_W, PAGE_W * 0.62))

    for ax, (xcol, ycol) in zip(axes.flat, pairs):
        if xcol not in df.columns or ycol not in df.columns:
            continue
        is_out = (outlier_flags[[xcol, ycol]].any(axis=1)
                  if xcol in avail and ycol in avail
                  else pd.Series(False, index=df.index))
        normal = df[~is_out]
        outliers = df[is_out]

        ax.scatter(normal[xcol], normal[ycol], s=5, alpha=0.35,
                   color=PALETTE[1], rasterized=True, label="Normal",
                   zorder=2)
        ax.scatter(outliers[xcol], outliers[ycol], s=12, alpha=0.85,
                   color=PALETTE[5], marker="x", lw=0.7, label="Outlier",
                   zorder=3)
        ax.set_xlabel(_label(xcol))
        ax.set_ylabel(_label(ycol))
        ax.legend(fontsize=6, frameon=True, fancybox=False, edgecolor="0.7")

    _save(fig, out_dir, "outliers")


# ===================================================================
# 7. STATISTICAL SUMMARY TABLE (LaTeX-ready)
# ===================================================================

def statistical_summary(df, out_dir):
    """Generate LaTeX-ready summary tables."""
    print("\n" + "=" * 70)
    print("STATISTICAL SUMMARY (per design type)")
    print("=" * 70)

    avail = [c for c in OUTPUT_COLS if c in df.columns]
    if not avail:
        print("  No output columns found.")
        return

    full_stats = df[avail].describe().T
    full_stats["CV"] = full_stats["std"] / (full_stats["mean"].abs() + 1e-12)
    print("\n  Full dataset:")
    print(full_stats[["count", "mean", "std", "min", "25%", "50%", "75%", "max", "CV"]].to_string())

    # Per design type
    if "design_name" in df.columns:
        for design in sorted(df["design_name"].unique()):
            sub = df[df["design_name"] == design]
            if len(sub) < 2:
                continue
            s = sub[avail].describe().T
            s["CV"] = s["std"] / (s["mean"].abs() + 1e-12)
            print(f"\n  {design} (n={len(sub)}):")
            print(s[["mean", "std", "min", "max", "CV"]].to_string())

    # Save CSV
    full_stats.to_csv(os.path.join(out_dir, "summary_stats.csv"))

    # Save LaTeX table
    latex_cols = ["mean", "std", "min", "50%", "max", "CV"]
    rename = {c: _label(c) for c in avail}
    lt = full_stats[latex_cols].rename(index=rename)
    lt.columns = ["Mean", "Std", "Min", "Median", "Max", "CV"]

    # Format numbers
    def _fmt(x):
        if abs(x) >= 1000 or (abs(x) < 0.01 and x != 0):
            return f"{x:.2e}"
        return f"{x:.3f}"

    lt_str = lt.map(_fmt)
    try:
        latex = lt_str.to_latex(escape=False)
    except ImportError:
        # Build LaTeX table manually without jinja2
        cols = lt_str.columns.tolist()
        header = " & ".join([""] + [str(c) for c in cols]) + r" \\"
        lines = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule", header, r"\midrule"]
        for idx, row in lt_str.iterrows():
            lines.append(" & ".join([str(idx)] + [str(row[c]) for c in cols]) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        latex = "\n".join(lines)
    tex_path = os.path.join(out_dir, "summary_stats.tex")
    with open(tex_path, "w") as fh:
        fh.write(latex)

    print(f"\n  Saved: summary_stats.csv / .tex")


# ===================================================================
# 8. REPRESENTATIVE FIELD VISUALIZATIONS
# ===================================================================

def representative_fields(h5_path, df, out_dir):
    """Plot 2D mid-plane slices for one representative sample per design."""
    print("\n" + "=" * 70)
    print("REPRESENTATIVE FIELD VISUALIZATIONS")
    print("=" * 70)

    if "design_name" not in df.columns:
        return

    designs = sorted(df["design_name"].unique())
    n_designs = len(designs)

    fig, axes = plt.subplots(n_designs, 4, figsize=(PAGE_W, 1.45 * n_designs))
    if n_designs == 1:
        axes = axes[np.newaxis, :]

    field_cmaps = [
        ("mask",        r"Geometry",                     "gray_r",   False),
        ("velocity",    r"$|\mathbf{u}|$",               "coolwarm", True),
        ("temperature", r"$T$",                          "inferno",  False),
        ("pressure",    r"$p$",                          "viridis",  True),
    ]

    with h5py.File(h5_path, "r") as f:
        for row, design in enumerate(designs):
            sub = df[df["design_name"] == design]
            # Pick median-Nu sample
            if "Nu" in sub.columns and len(sub) > 0:
                median_nu = sub["Nu"].median()
                idx = (sub["Nu"] - median_nu).abs().idxmin()
                sid = sub.loc[idx, "sample_id"] if "sample_id" in sub.columns else list(f.keys())[0]
            else:
                sid = list(sub.index)[0] if len(sub) > 0 else list(f.keys())[0]

            if sid not in f:
                continue

            grp = f[sid]
            mask = grp["mask"][()]
            u = grp["velocity"][()]
            T = grp["temperature"][()]
            P = grp["pressure"][()]
            nz = mask.shape[2]
            mid_z = nz // 2
            solid_2d = (mask[:, :, mid_z] == BC_SOLID)

            raw_fields = {
                "mask": solid_2d.astype(float),
                "velocity": np.linalg.norm(u[:, :, mid_z, :], axis=-1),
                "temperature": T[:, :, mid_z].copy(),
                "pressure": P[:, :, mid_z].copy(),
            }
            # Mask solids in fluid fields
            for key in ("velocity", "pressure"):
                raw_fields[key][solid_2d] = np.nan

            for col, (key, title, cmap, mask_solid) in enumerate(field_cmaps):
                ax = axes[row, col]
                data = raw_fields[key]
                im = ax.imshow(data.T, origin="lower", aspect="equal",
                               cmap=cmap, rasterized=True)
                cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
                cbar.ax.tick_params(labelsize=5)

                if col == 0:
                    ax.set_ylabel(_short(design), fontsize=7, fontweight="bold")
                if row == 0:
                    ax.set_title(title, fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])

    _save(fig, out_dir, "representative_fields")


# ===================================================================
# MAIN
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze and validate the cold plate CFD dataset")
    parser.add_argument("--data", type=str, default="dataset/data.h5",
                        help="Path to HDF5 dataset")
    parser.add_argument("--meta", type=str, default=None,
                        help="Path to metadata parquet (auto-detected if omitted)")
    parser.add_argument("--out", type=str, default="analysis_results",
                        help="Output directory for plots/reports")
    parser.add_argument("--field-analysis", action="store_true",
                        help="Run expensive field variability / PCA analysis")
    parser.add_argument("--max-validate", type=int, default=0,
                        help="Max samples for per-sample validation (0=all)")
    parser.add_argument("--max-field-samples", type=int, default=100,
                        help="Max samples for field PCA analysis")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Error: {args.data} not found.")
        sys.exit(1)

    setup_style()
    os.makedirs(args.out, exist_ok=True)

    # Auto-detect metadata parquet
    meta_path = args.meta
    if meta_path is None:
        candidate = os.path.join(os.path.dirname(args.data), "metadata.parquet")
        if os.path.exists(candidate):
            meta_path = candidate

    # Load metadata
    if meta_path and os.path.exists(meta_path):
        print(f"Loading metadata: {meta_path}")
        df = pd.read_parquet(meta_path)
    else:
        print("No parquet found, reading metadata from HDF5 attributes...")
        rows = []
        with h5py.File(args.data, "r") as f:
            for sid in f.keys():
                row = dict(f[sid].attrs)
                row["sample_id"] = sid
                rows.append(row)
        df = pd.DataFrame(rows)

    n_samples = len(df)
    print(f"Dataset: {n_samples} samples")

    # Per-sample physical validation
    with h5py.File(args.data, "r") as f:
        all_ids = sorted(f.keys())
    validate_ids = all_ids if args.max_validate == 0 else all_ids[:args.max_validate]

    print(f"\nRunning physical validation on {len(validate_ids)} samples...")
    checks_list = []
    for i, sid in enumerate(validate_ids):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Validating {i+1}/{len(validate_ids)}...")
        try:
            checks = validate_sample(args.data, sid)
            checks_list.append(checks)
        except Exception as e:
            print(f"  ERROR on {sid}: {e}")
    checks_df = pd.DataFrame(checks_list)

    # Reports and plots
    physical_report(checks_df, args.out)
    coverage_analysis(df, args.out)
    correlation_analysis(df, args.out)
    physics_plots(df, args.out)
    outlier_analysis(df, args.out)
    statistical_summary(df, args.out)
    representative_fields(args.data, df, args.out)

    if args.field_analysis:
        field_variability_analysis(args.data, all_ids, args.out,
                                   max_samples=args.max_field_samples)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print(f"All figures saved to: {args.out}/")
    print("  PDF (vector) files for paper, PNG for preview.")
    print("  LaTeX table: summary_stats.tex")
    print("=" * 70)


if __name__ == "__main__":
    main()
