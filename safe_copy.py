"""
Safe Dataset Copy
Waits until the HDF5 file is not being written to, then copies it.

Usage:
    python safe_copy.py                          # default paths
    python safe_copy.py --src dataset --dst dataset/snapshot
"""
import argparse
import os
import shutil
import time

import h5py


def is_h5_readable(path):
    """Try opening the HDF5 in read mode — fails if a writer holds it."""
    try:
        with h5py.File(path, "r") as f:
            n = len(f.keys())
        return True, n
    except Exception:
        return False, 0


def safe_copy(src_dir, dst_dir):
    h5_src = os.path.join(src_dir, "data.h5")
    meta_src = os.path.join(src_dir, "metadata.parquet")

    if not os.path.exists(h5_src):
        print(f"No HDF5 found at {h5_src}")
        return

    # Poll until the file is safe to read
    print(f"Checking {h5_src} ...")
    for attempt in range(30):
        ok, n_samples = is_h5_readable(h5_src)
        if ok:
            break
        print(f"  File is busy, retrying in 2s ... (attempt {attempt + 1}/30)")
        time.sleep(2)
    else:
        print("File remained busy after 60s — aborting.")
        return

    os.makedirs(dst_dir, exist_ok=True)
    h5_dst = os.path.join(dst_dir, "data.h5")
    meta_dst = os.path.join(dst_dir, "metadata.parquet")

    print(f"File is safe ({n_samples} samples). Copying ...")
    shutil.copy2(h5_src, h5_dst)
    if os.path.exists(meta_src):
        shutil.copy2(meta_src, meta_dst)

    # Verify the copy
    ok, n_verify = is_h5_readable(h5_dst)
    if ok and n_verify == n_samples:
        print(f"Done. Copied {n_samples} samples to {dst_dir}")
    else:
        print(f"WARNING: verification failed (expected {n_samples}, got {n_verify})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Safe dataset copy")
    parser.add_argument("--src", default="dataset", help="Source directory")
    parser.add_argument("--dst", default="dataset/snapshot", help="Destination directory")
    args = parser.parse_args()
    safe_copy(args.src, args.dst)
