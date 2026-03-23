# NeurIPS 2026: The xAdvCool Master Plan

This document outlines the end-to-end execution strategy for a "Datasets & Benchmarks" paper using the xAdvCool solver.

## Phase 1: Solver Scaling and Validation (Weeks 1-2)
1.  **[1.1] Python-Only Validation & Analytical Benchmarks**: 
    -   **Analytical Solutions**: Verify the solver against classic solutions where $T$ and $U$ are known:
        -   **Poiseuille Flow**: For velocity profile $(\nu, \Delta P, U)$ consistency.
        -   **Graetz Problem**: For heat transfer coefficient ($\text{Nu}$) in a laminar internal flow.
        -   **Slab Conduction**: For steady-state 3D heat spreading in the Silicon base.
    -   **Cross-Code LBM Validation (Python)**:
        -   [XLB](https://github.com/autodesk-research/XLB): Use its Warp backend for a direct comparison with a differentiable Python-based LBM.
        -   [LBMpy](https://github.com/m-reiter/lbmpy): A powerful Python kernel generator that is the gold standard for LBM accuracy research.
    -   Target: Prove xAdvCool is within < 3% of analytical predictions for simplified cases.
2.  **[1.2] Headless Batch Script Development**: 
    -   Create a CLI for `simulation.py` to automate geometry-to-result pipelines.
    -   Implement checkpointing and graceful failure handling (specifically for LBM "blow-ups" at high Re).
3.  **[1.3] Performance Optimization**: 
    -   Ensure NVIDIA Warp kernels are fully JIT-compiled and persistent.
    -   Measure throughput (Target: < 30 seconds per $256^3$ steady-state sample).

## Phase 2: Mass Dataset Production (Weeks 3-5)
1.  **[2.1] Pilot Generation (1,000 Samples)**:
    -   Generate a small diverse set across all Design Types (Gyroid, Schwarz, Pins).
    -   Conduct sanity checks on the results ($T$ should increase from inlet to outlet, $\rho$ should drop).
2.  **[2.2] Main Production Run (20,000 Samples)**:
    -   Execute the randomized parameter sweep (Re, Pr, Heat Power).
    -   Split the data: 15k Train, 2k Val, 3k Test.
3.  **[2.3] Out-of-Distribution (OOD) Set (5,000 Samples)**:
    -   Generate geometries **never seen** in the training set (e.g., Neovius surfaces or hybrid designs).
    -   This is the "stress test" for neural operators.

## Phase 3: Benchmarking Suite Implementation (Weeks 6-9)
1.  **[3.1] Data Pre-processing**: 
    -   Convert `.npy` files to a more efficient ML-ready format (e.g., `.zarr`, `.h5`, or custom PyTorch/TensorFlow datasets).
    -   Compute normalized error weights based on voxels near the interface.
2.  **[3.2] Baseline Training**: 
    -   **FNO-3D**: Train a Fourier Neural Operator for full-field prediction.
    -   **U-Net / Vision Transformer**: Implement standard CNN/ViT baselines for 3D physics.
    -   **MeshGraphNet**: Convert voxels to graph nodes for the wetted surface for GNN comparison.
3.  **[3.3] Physics-Consistency Analysis**: 
    -   Measure how often models violate the conservation of energy or continuity.
    -   Create the "Physics-Consistency Score" (lower is better).

## Phase 4: Paper Drafting and Visualization (Weeks 10-12)
1.  **[4.1] High-Impact Visualization**:
    -   Render "hero shots" of the most efficient cooling designs discovered in the dataset.
    -   Create side-by-side error maps showing where FNO struggles (e.g., high-curvature regions).
2.  **[4.2] Writing the Paper (NeurIPS Template)**: 
    -   **Introduction**: Frame the work around "Sustainable AI" and the necessity of efficient thermal management for data centers.
    -   **Related Work**: Bridge the gap between classical CFD and current surrogate modeling (FNO, DeepONet).
    -   **The xAdvCool Benchmark**: Formally define the tasks (e.g., "Given a geometry $M$ and heat flux $Q$, predict $T(x,y,z)$").
3.  **[4.3] Supplementary Material**: 
    -   Full code release on GitHub with `poetry`/`pip` installation scripts.
    -   Hosting the full dataset on HuggingFace or Zenodo with a clear Data Loader.

## Success Criteria for "Instant Accept":
-   **Scale**: At least 25,000 distinct 3D simulations.
-   **Complexity**: Prove that models trained on simple pins fail on TPMS (proving the need for our data).
-   **Reliability**: Provide the FEM validation results as proof of physical accuracy.
-   **Ethics & Impact**: Discuss how our dataset will lower the carbon footprint of future compute hardware.
