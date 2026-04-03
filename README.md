# xAdvCool: 3D Conjugate Heat Transfer Dataset & Benchmark

A GPU-accelerated Lattice Boltzmann solver for generating large-scale 3D Conjugate Heat Transfer (CHT) datasets, with an ML benchmark suite for surrogate modeling of cold plate heat sinks.

Built on NVIDIA Warp. Targeting **NeurIPS 2026 Datasets & Benchmarks**.

---

## Dataset

Over 1,000 steady-state simulations of water cooling through copper cold plates:
- **5 geometry types:** Straight Channels, Pin-Fin Staggered, Gyroid TPMS, Schwarz P TPMS, Schwarz D TPMS
- **10 heat source patterns:** uniform, dual_core, quad_core, gpu_die, chiplet, igbt_half_bridge, soc_heterogeneous, gaussian_hotspots, hotspot_on_background, peripheral_ring
- **Per sample:** geometry mask, heat source, velocity (3D vector), pressure, temperature fields on a 148x128x32 lattice
- **Storage:** HDF5 (fields) + Parquet (scalar metadata)

---

## Project Structure

```
xAdvCool/
├── src/                        # Core physics & geometry
│   ├── fluid_engine3d.py       #   D3Q19 fluid solver (Warp/CUDA)
│   ├── thermal_engine3d.py     #   Conjugate heat transfer solver (Warp/CUDA)
│   ├── geometry.py             #   Geometry generation (TPMS, pins, channels)
│   └── simulation.py           #   LBM simulation runner
├── generate_dataset.py         # Dataset generation pipeline
├── generate_croissant.py       # Croissant metadata for HuggingFace/NeurIPS
├── validate_analytical.py      # Analytical validation suite (10 test cases, pytest-compatible)
├── analyze_dataset.py          # Dataset statistics & publication figures
├── analyze_ml_readiness.py     # ML readiness analysis (learnability, diversity)
├── dataset_viewer.py           # Interactive 3D field viewer (PyVista)
├── benchmark/                  # ML benchmark suite
│   ├── train.py                #   Training entry point
│   ├── evaluate.py             #   Evaluation entry point
│   ├── data/                   #   Dataset, splits, normalization
│   ├── models/                 #   7 models (see below)
│   ├── metrics/                #   Field, scalar, and physics metrics
│   └── configs/                #   YAML configs per task/model
└── dataset/                    # Generated data (not in git)
    ├── data.h5
    └── metadata.parquet
```

---

## Usage

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA support

### Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install warp-lang numpy scipy tqdm h5py pandas pyvista scikit-learn
pip install torch torchvision  # for benchmark
```

### 1. Validate the solver analytically
```bash
python validate_analytical.py              # run all 10 test cases directly
python -m pytest validate_analytical.py -v # or via pytest
```

### 2. Generate the dataset
```bash
python generate_dataset.py --samples-per-design 200 --device cuda
```
Supports resume: re-run the same command to continue from where it stopped. Previously failed simulations are tracked in `dataset/failed_samples.json` and skipped.

### 3. View dataset samples interactively
```bash
python dataset_viewer.py                        # default: dataset/data.h5
python dataset_viewer.py --data path/to/data.h5 --sample 5
```

### 4. Analyze the dataset
```bash
python analyze_dataset.py       # physical validation plots & stats
python analyze_ml_readiness.py  # ML readiness evaluation
```

### 5. Generate Croissant metadata
```bash
python generate_croissant.py    # writes croissant.json with checksums
```

### 6. Train & evaluate ML models
```bash
# Quick smoke test with trivial baselines
python -m benchmark.train --config benchmark/configs/task_a_trivial.yaml

# Train a real model
python -m benchmark.train --config benchmark/configs/task_a_unet3d.yaml

# Prototype with fewer samples/epochs
python -m benchmark.train --config benchmark/configs/task_a_unet3d.yaml \
    --override training.epochs=5 data.max_samples=100

# Evaluate
python -m benchmark.evaluate --config benchmark/configs/task_a_unet3d.yaml \
    --checkpoint results/task_a_unet3d/best_model.pt
```

---

## Benchmark Tasks

| Task | Description | Split |
|------|-------------|-------|
| **A** | Full-input field prediction (mask + heat source + scalars -> fields) | 70/15/15 stratified |
| **B** | Geometry-only prediction (mask -> fields, uniform heat source subset) | 70/15/15 stratified |
| **C** | OOD generalization (leave-one-geometry-out) | 4 train / 1 held-out |

## Benchmark Models

| Model | Scalar injection | Params |
|-------|-----------------|--------|
| MLPBaseline | Concat | small |
| 3D U-Net | FiLM conditioning | ~22.7M |
| FNO-3D | Broadcast-concat | ~18.9M |
| DeepONet | Branch-trunk concat | ~953K |
| ViT-3D | Condition token | ~7.5M |
| MeshGraphNet | Node broadcast | ~745K |

---

## Analytical Validation

11 test cases validate the LBM solvers against known analytical solutions:

**Thermal solver:**
| Test | Reference solution | Threshold |
|------|--------------------|-----------|
| 1D semi-infinite conduction | `T(x,t) = T_s * erfc(x / 2*sqrt(alpha*t))` | < 3% |
| Advection-diffusion Gaussian | Gaussian spreading with drift | < 3% |
| Conjugate slab (CHT) | Steady-state two-material conduction | < 3% |
| Volumetric heat source | Poisson equation (parabolic profile) | < 3% |

**Fluid solver:**
| Test | Reference solution | Threshold |
|------|--------------------|-----------|
| 3D rectangular Poiseuille | Analytical duct flow profile | < 3% |
| Taylor-Green vortex decay | Exponential viscous decay | < 3% |
| Couette flow | Linear velocity profile | < 3% |
| Kuwabara pin-fin drag | Kuwabara cell model | < 5% |

**Convergence & sensitivity:**
| Test | What it checks | Threshold |
|------|---------------|-----------|
| Poiseuille grid convergence | Spatial convergence order | >= 1.8 (expect ~2.0) |
| Taylor-Green convergence | Temporal convergence order | >= 1.8 |
| Mach number sensitivity | Error at high Ma | < 5% |

---

## Physics Engine

- **Fluid:** D3Q19 lattice, BGK collision, bounce-back boundaries
- **Thermal:** Conjugate heat transfer with separate fluid/solid relaxation times
- **Coolant:** Water (Pr=7.0) through copper (k_ratio=628.0)
- **Convergence:** Monitored via kinetic energy and temperature residuals

---

Dataset generation complete.
  Completed: 8071  Skipped: 1411  Failed: 518
  HDF5: dataset/data.h5
  Metadata: dataset/metadata.parquet

  --- Simulation Cost Summary ---
  Hardware:              NVIDIA GeForce RTX 3090
  Total wall time:       41030s (11.4h)
  Total simulation time: 36952s (10.3h)
  Total timesteps:       43,759,100
  Total lattice updates: 2.653e+13
  Average throughput:    717.9 MLUPS
  Converged:             2977/2977
  Avg time/sample:       12.4s
(.venv) rvn@atom:~/repos/xAdvCool$ 


## License

CC-BY-4.0
