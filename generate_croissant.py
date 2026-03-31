"""Generate croissant.json metadata file for the xAdvCool dataset.

Usage:
    python generate_croissant.py
    python generate_croissant.py --data-dir dataset --output croissant.json
"""

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_croissant(data_dir: str = "dataset", output: str = "croissant.json"):
    data_dir = Path(data_dir)
    h5_path = data_dir / "data.h5"
    parquet_path = data_dir / "metadata.parquet"

    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    print(f"Computing checksums...")
    h5_sha256 = sha256_file(str(h5_path))
    parquet_sha256 = sha256_file(str(parquet_path))
    print(f"  data.h5:           {h5_sha256}")
    print(f"  metadata.parquet:  {parquet_sha256}")

    croissant = {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "sc": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/",
            "dct": "https://purl.org/dc/terms/",
        },
        "@type": "sc:Dataset",
        "name": "xAdvCool",
        "description": (
            "A benchmark dataset of 3D conjugate heat transfer (CHT) simulations "
            "for data-driven surrogate modeling of cold plate heat sinks. Contains "
            "Lattice Boltzmann simulations across 5 topology types (Straight Channels, "
            "Pin-Fin Staggered, Gyroid TPMS, Schwarz P TPMS, Schwarz D TPMS) and "
            "10 heat source patterns, with water cooling through copper cold plates. "
            "Each sample maps a geometry mask and operating conditions to steady-state "
            "velocity, pressure, and temperature fields on a 148x128x32 lattice."
        ),
        "dct:conformsTo": "http://mlcommons.org/croissant/1.0",
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "url": "https://huggingface.co/datasets/tgautam03/xAdvCool",
        "version": "1.0.0",
        "citation": (
            "Gautam, T. (2026). xAdvCool: A Benchmark Dataset for 3D Conjugate "
            "Heat Transfer Surrogate Modeling. NeurIPS Datasets and Benchmarks Track."
        ),
        "creator": {"@type": "Person", "name": "Tushar Gautam"},
        "datePublished": "2026-03-31",
        "keywords": [
            "physics",
            "simulation",
            "cfd",
            "conjugate-heat-transfer",
            "lattice-boltzmann",
            "cold-plate",
            "topology-optimization",
            "surrogate-modeling",
        ],
        "distribution": [
            {
                "@type": "sc:FileObject",
                "@id": "data.h5",
                "name": "data.h5",
                "description": (
                    "HDF5 file containing all simulation samples. Each top-level group "
                    "is a sample keyed by sample_id, containing 5 datasets (mask, "
                    "heat_source, velocity, pressure, temperature) and scalar attributes."
                ),
                "contentUrl": "https://huggingface.co/datasets/tgautam03/xAdvCool/resolve/main/data.h5",
                "encodingFormat": "application/x-hdf5",
                "sha256": h5_sha256,
            },
            {
                "@type": "sc:FileObject",
                "@id": "metadata.parquet",
                "name": "metadata.parquet",
                "description": "Parquet metadata index with scalar parameters and performance metrics for each sample.",
                "contentUrl": "https://huggingface.co/datasets/tgautam03/xAdvCool/resolve/main/metadata.parquet",
                "encodingFormat": "application/x-parquet",
                "sha256": parquet_sha256,
            },
        ],
        "recordSet": [
            {
                "@type": "cr:RecordSet",
                "@id": "samples",
                "name": "samples",
                "description": (
                    "Individual CHT simulation samples. 3D field data is stored in the "
                    "HDF5 file; scalar metadata is in the Parquet file. Linked by sample_id."
                ),
                "field": [
                    # --- Parquet fields (scalar metadata) ---
                    _field("sample_id", "Unique sample identifier (also the HDF5 group key)", "sc:Text", "metadata.parquet"),
                    _field("design_name", "Geometry topology type: Straight Channels, Pin-Fin Staggered, Gyroid TPMS, Schwarz P TPMS, or Schwarz D TPMS", "sc:Text", "metadata.parquet"),
                    _field("heat_source_type", "Heat source pattern name (e.g., uniform, dual_core, gpu_die)", "sc:Text", "metadata.parquet"),
                    _field("feature_size", "Geometric feature size parameter (lattice units)", "sc:Float", "metadata.parquet"),
                    _field("spacing", "Geometric spacing parameter (lattice units)", "sc:Float", "metadata.parquet"),
                    _field("tau_fluid", "BGK relaxation time for fluid (controls viscosity: nu = (tau - 0.5)/3)", "sc:Float", "metadata.parquet"),
                    _field("u_inlet_val", "Inlet velocity magnitude (lattice units)", "sc:Float", "metadata.parquet"),
                    _field("heat_power", "Total heat power input (lattice units)", "sc:Float", "metadata.parquet"),
                    _field("Re", "Reynolds number based on hydraulic diameter", "sc:Float", "metadata.parquet"),
                    _field("Nu", "Nusselt number (dimensionless heat transfer coefficient)", "sc:Float", "metadata.parquet"),
                    _field("R_th", "Thermal resistance (lattice units)", "sc:Float", "metadata.parquet"),
                    _field("f_friction", "Darcy friction factor", "sc:Float", "metadata.parquet"),
                    _field("COP", "Coefficient of Performance (thermal performance / pumping cost)", "sc:Float", "metadata.parquet"),
                    _field("converged", "Whether the simulation reached steady-state convergence", "sc:Boolean", "metadata.parquet"),
                    # --- HDF5 fields (3D tensors) ---
                    _field("mask", "3D geometry mask (int8, 148x128x32). Values: 0=solid, 1=fluid, 2=inlet, 3=outlet", "sc:Integer", "data.h5"),
                    _field("heat_source", "3D heat source distribution (float32, 148x128x32)", "sc:Float", "data.h5"),
                    _field("velocity", "3D steady-state velocity field (float32, 148x128x32x3, channels: x,y,z)", "sc:Float", "data.h5"),
                    _field("pressure", "3D steady-state pressure field (float32, 148x128x32)", "sc:Float", "data.h5"),
                    _field("temperature", "3D steady-state temperature field (float32, 148x128x32)", "sc:Float", "data.h5"),
                ],
            }
        ],
    }

    with open(output, "w") as f:
        json.dump(croissant, f, indent=2)

    print(f"Wrote {output}")


def _field(name: str, description: str, data_type: str, file_id: str) -> dict:
    return {
        "@type": "cr:Field",
        "@id": f"samples/{name}",
        "name": name,
        "description": description,
        "dataType": data_type,
        "source": {
            "fileObject": {"@id": file_id},
            "extract": {"column": name},
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate croissant.json for xAdvCool")
    parser.add_argument("--data-dir", default="dataset", help="Directory containing data.h5 and metadata.parquet")
    parser.add_argument("--output", default="croissant.json", help="Output path")
    args = parser.parse_args()
    generate_croissant(args.data_dir, args.output)
