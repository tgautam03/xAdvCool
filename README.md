# xAdvCool: 3D Conjugate Heat Transfer Dataset & Benchmark

xAdvCool is a GPU-accelerated Lattice Boltzmann (LBM) solver and ML benchmark suite for steady-state 3D conjugate heat transfer (CHT) in cold-plate heat sinks. The solver — built on NVIDIA Warp with a D3Q19 fluid lattice and a coupled thermal lattice — is used to generate a public dataset of **8,071 converged simulations** spanning **5 geometry families** (Straight Channels, Pin-Fin Staggered, Gyroid TPMS, Schwarz P TPMS, Schwarz D TPMS) crossed with **10 heat-source patterns** (uniform, dual_core, quad_core, gpu_die, chiplet, igbt_half_bridge, soc_heterogeneous, gaussian_hotspots, hotspot_on_background, peripheral_ring) on a 148×128×32 lattice. Each sample stores geometry mask, heat-source map, 3D velocity, pressure, and temperature fields as HDF5, plus scalar metadata (Re, pressure drop, R_th, Nu, …) as Parquet. On top of the dataset, the [benchmark/](benchmark/) suite provides three tasks (full-input, geometry-only, leave-one-geometry-out OOD) and six surrogate models (MLP, 3D U-Net, FNO-3D, DeepONet, ViT-3D, MeshGraphNet). Targeted at **NeurIPS 2026 Datasets & Benchmarks**.

---

## Repo layout

```
xAdvCool/
├── src/                        Core physics & geometry (Warp/CUDA)
│   ├── fluid_engine3d.py         D3Q19 fluid solver
│   ├── thermal_engine3d.py       Conjugate heat transfer solver
│   ├── geometry.py               Geometry generation (TPMS, pin-fin, channels)
│   └── simulation.py             LBM simulation runner
├── generate_dataset.py         Dataset generation pipeline
├── validate_analytical.py      Analytical validation suite (pytest-compatible)
├── analyze_dataset.py          Dataset statistics & publication figures
├── generate_croissant.py       Croissant metadata for HuggingFace/NeurIPS
├── dataset_viewer.py           Interactive 3D field viewer (PyVista)
├── benchmark/                  ML benchmark suite
│   ├── train.py                  Training entry point
│   ├── evaluate.py               Evaluation entry point
│   ├── data/                     Dataset loader, splits, normalization
│   ├── models/                   6 surrogate models
│   ├── metrics/                  Field, scalar, and physics metrics
│   └── configs/                  YAML configs (one per task × model)
└── dataset/                    Generated artifacts (not in git)
    ├── data.h5                   Fields (~53 GB at full size)
    ├── metadata.parquet          Per-sample scalars
    ├── failed_samples.json       Resume tracker for the generator
    └── cache/                    Per-sample tensor cache built by training
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install warp-lang numpy scipy tqdm h5py pandas pyvista scikit-learn
pip install torch torchvision   # for benchmark
```

Requires Python 3.10+ and an NVIDIA GPU with CUDA for the solver and benchmark.

---

## 1. Validate the solver

```bash
python validate_analytical.py                  # run all tests directly
python -m pytest validate_analytical.py -v     # or via pytest
```

Runs 10 analytical test cases (1D semi-infinite conduction, advection-diffusion Gaussian, conjugate slab, volumetric heat source, rectangular Poiseuille, Taylor-Green decay, Kuwabara pin-fin drag, Couette flow, grid convergence, Mach-number sensitivity). Most thresholds are <3% error; Kuwabara and Mach are <5%. Prints a pass/fail summary to stdout — no files are written.

---

## 2. Reproduce the dataset

```bash
python generate_dataset.py --samples-per-design 200 --device cuda
```

Useful flags: `--output-dir dataset` (default), `--max-steps 30000`, `--seed 42`, `--no-resume`, `--snapshot-every 0`.

**What runs:** for each (geometry × heat-source) pair, samples random design parameters, builds the geometry via [src/geometry.py](src/geometry.py), runs the coupled fluid+thermal LBM in [src/simulation.py](src/simulation.py) until residuals converge, then writes the result.

**What gets produced** (under `--output-dir`, default `dataset/`):
- `data.h5` — one HDF5 group per sample with `mask`, `heat_source`, `velocity` (3-channel), `pressure`, `temperature` on the 148×128×32 grid.
- `metadata.parquet` — one row per sample: design parameters, Re, pressure drop, R_th, Nu, runtime, MLUPS, convergence flag.
- `failed_samples.json` — IDs of samples that diverged or errored; skipped on resume.

**Resume:** re-running the same command picks up where it left off. Already-written sample IDs are detected from the HDF5 file; failed IDs are read from the JSON and not retried.

---

## 3. Analyze the dataset

```bash
python analyze_dataset.py                       # default: dataset/data.h5 → analysis_results/
python analyze_dataset.py --data path/to/data.h5 --out my_analysis --field-analysis
```

Reads `dataset/data.h5` + `dataset/metadata.parquet` and writes plots (PDF + PNG) and summary CSVs into `--out` (default `analysis_results/`): correlation matrix, input coverage, output distributions, outliers, Pareto front, physical validation, representative fields, and `summary_stats.csv` / `summary_stats.tex`. Use `--field-analysis` for the heavier per-field statistics.

---

## 4. View samples interactively

```bash
python dataset_viewer.py                        # default: dataset/data.h5, sample 0
python dataset_viewer.py --data path/to/data.h5 --sample 5
```

Opens a PyVista window with sliders/keys to step through samples and toggle fields. Read-only; nothing is written.

---

## 5. Train a model

Configs live in [benchmark/configs/](benchmark/configs/), one per (task, model). Naming convention: `task_<a|b|c[_geom]>_<model>.yaml`. Each leaf config inherits from `defaults.yaml` and sets `training.output_dir`.

```bash
# Smoke test (cheap baseline)
python -m benchmark.train --config benchmark/configs/task_a_trivial.yaml

# Real training
python -m benchmark.train --config benchmark/configs/task_a_unet3d.yaml

# Prototype with fewer samples / shorter run
python -m benchmark.train --config benchmark/configs/task_a_unet3d.yaml \
    --override training.epochs=5 data.max_samples=100
```

**Inputs:** `dataset/data.h5` + `dataset/metadata.parquet` (paths set in [benchmark/configs/defaults.yaml](benchmark/configs/defaults.yaml)).

**Caching:** on first epoch, per-sample preprocessed tensors are written under `dataset/cache/` (one `.pt` per sample, keyed by content hash). Subsequent runs reuse them — wipe this directory if you change preprocessing.

**Outputs** (under `training.output_dir`, e.g. `results/task_a_unet3d/`):
- `config.yaml` — frozen, fully-resolved config used for the run.
- `best_model.pt` — checkpoint with lowest validation loss.
- `checkpoint_epoch{N}.pt` — periodic snapshots (every `training.checkpoint_every`, default 20 epochs).
- `norm_stats.json` — per-field mean/std used for normalization.
- `tb/` — TensorBoard event files (`tensorboard --logdir results/task_a_unet3d/tb`).

---

## 6. Evaluate a checkpoint

```bash
python -m benchmark.evaluate \
    --config benchmark/configs/task_a_unet3d.yaml \
    --checkpoint results/task_a_unet3d/best_model.pt \
    --split test
```

Optional: `--output path/to/results.json` (default writes `eval_<split>.json` next to the checkpoint).

Loads the checkpoint plus its sibling `norm_stats.json`, runs the requested split through the model, and writes a JSON of metrics: per-field NRMSE / R² / max-error, scalar metrics (R_th, pressure drop, Nu), and physics-consistency checks (mass conservation, energy balance).

---

## Benchmark tasks

| Task | Description | Split |
|------|-------------|-------|
| **A** | Full-input field prediction (mask + heat source + scalars → fields) | 70/15/15 stratified |
| **B** | Geometry-only prediction (mask → fields, uniform-heat subset) | 70/15/15 stratified |
| **C** | OOD generalization (leave-one-geometry-out) | 4 train geometries / 1 held out |

Task C provides one config per held-out geometry: `task_c_{straight,pinfin,gyroid,schwarzp,schwarzd}_unet3d.yaml`.

## Benchmark models

| Model | Scalar conditioning | Params |
|-------|--------------------|--------|
| MLPBaseline | Concat | small |
| 3D U-Net | FiLM | ~22.7M |
| FNO-3D | Broadcast-concat | ~18.9M |
| DeepONet | Branch-trunk concat | ~953K |
| ViT-3D | Condition token | ~7.5M |
| MeshGraphNet | Node broadcast | ~745K |

---

## Physics

- **Fluid:** D3Q19 lattice, BGK collision, bounce-back boundaries.
- **Thermal:** Conjugate heat transfer with separate fluid/solid relaxation times.
- **Coolant:** Water (Pr=7.0) through copper (k_ratio=628.0).
- **Convergence:** Monitored via kinetic-energy and temperature residuals.

Generation throughput on a single RTX 3090: ~718 MLUPS, ~12.4s per sample, ~11h total wall time for the released 8,071-sample dataset.

---

## License

MIT
