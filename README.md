# xAdvCool: Advanced 3D Conjugate Heat Transfer LBM Solver

**xAdvCool** is a high-performance, GPU-accelerated 3D Lattice Boltzmann Method (LBM) solver designed for simulating **Conjugate Heat Transfer (CHT)** in microchannel cooling systems. Built on NVIDIA Warp, it enables rapid simulation of complex fluid-thermal interactions in intricate 3D geometries, specifically tailored for generating large-scale datasets for machine learning benchmarks like NeurIPS.

---

## 🚀 Key Features

### 1. High-Performance GPU Engine
*   **NVIDIA Warp Backend:** Utilizes JIT-compiled CUDA kernels for massive parallelism on modern GPUs.
*   **D3Q19 Lattice:** Employs the standard 19-velocity 3D lattice for an optimal balance of accuracy and computational efficiency.
*   **Scalable Domains:** Capable of simulating millions of voxels with low memory overhead.

### 2. Advanced Collision Physics
*   **Entropic KBC Solver:** Implements the Karlin-Bösch-Chikatamarla (KBC) collision model. This entropic stabilizer ensures numerical stability at extremely low viscosities (High Reynolds numbers) without adding artificial dissipation.
*   **Standard BGK:** Includes the classic single-relaxation-time (BGK) operator for laminar, high-viscosity regimes.
*   **CHT-Coupled Thermal Solver:** Simulates heat conduction in solids and convection in fluids with direct thermal coupling at the interface.

### 3. Mathematical 3D Geometry Engine
Replaces traditional CAD with implicitly defined Triply Periodic Minimal Surfaces (TPMS) and structured pins:
*   **Gyroid:** $\sin(x)\cos(y) + \sin(y)\cos(z) + \sin(z)\cos(x) = C$
*   **Schwarz Primitive:** $\cos(x) + \cos(y) + \cos(z) = C$
*   **Schwarz Diamond:** $\sin(x)\sin(y)\sin(z) + ... = C$
*   **Pin-Fin Staggered:** Classic high-efficiency cooling pillars.
*   **Straight Channels:** Baseline laminar microchannels.

### 4. Realistic Thermal boundary Conditions
*   **Processor Heat Maps:** Simulates realistic heat generation profiles (e.g., Dual-Core + L3 Cache layouts).
*   **Conjugate Coupling:** Handles different thermal relaxation times for fluid (water) and solid (Silicon) domains.

---

## 📊 Performance Metrics

The solver automatically calculates and tracks key hydraulic and thermal engineering parameters:
*   **Reynolds Number (Re):** Calculated dynamically using porosity and hydraulic diameter.
*   **Flow Rate (Q) & Pressure Drop (dP):** For hydraulic resistance analysis.
*   **Permeability (k):** Quantitative measure of flow conductance.
*   **Maximum Temperature (MaxT) & Average Temperature:** For thermal effectiveness.
*   **Pumping Power:** Energy required to drive the flow.

---

## 🛠️ Getting Started

### Prerequisites
*   Python 3.10+
*   NVIDIA GPU (with CUDA support)

### Installation
1. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install warp-lang numpy pyvista scipy tqdm
```

### Usage
1.  **Generate Geometry:** Run and interact with the 3D designer.
    ```bash
    python geometry.py
    ```
2.  **Run Simulation:** Execute the multi-physics solver.
    ```bash
    python simulation.py
    ```
3.  **Visualize Results:** Inspect the 3D temperature and velocity fields.
    ```bash
    python visualization.py
    ```

---

## 🎯 Target Use Case: NeurIPS Benchmarks
The codebase is structured to be the backbone of a **NeurIPS Datasets and Benchmarks** submission. By leveraging the fast TPMS generation and stable KBC solver, researchers can generate tens of thousands of unique 3D CHT samples to train next-generation Surrogate Models such as FNOs (Fourier Neural Operators) and GNNs (Graph Neural Networks).
