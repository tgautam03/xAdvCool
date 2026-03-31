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

- Domain: `256x256x64` core + 10-cell inlet/outlet buffers
- Total grid per sample: `276x256x64`

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
@misc{cold_plate_cfd_dataset,
  title={Cold Plate Topology CFD Dataset},
  year={2026},
  url={https://huggingface.co/datasets/YOUR_USERNAME/cold-plate-cfd},
}
```
