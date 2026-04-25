"""
Merge per-rank ESM2-15B chunk files into final arrays.
Run after all workers in launch_esm2_15b.sh complete.

Usage:
    python precompute/merge_esm2_15b.py
"""

import os
import numpy as np
from glob import glob

CHUNK_DIR = "/workspace/embeddings"

JOBS = [
    {"key": "gb1_nsite",    "n": 149361, "out": "/workspace/protein/v1/data/gb1/embeddings_esm2_15b_4site.npy"},
    {"key": "trpb_nsite",   "n": 160000, "out": "/workspace/protein/v1/data/trpb/embeddings_esm2_15b_4site.npy"},
    {"key": "tev_meanpool", "n": 159132, "out": "/workspace/protein/v1/data/tev/embeddings_esm2_15b_meanpool.npy"},
    {"key": "tev_nsite",    "n": 159132, "out": "/workspace/protein/v1/data/tev/embeddings_esm2_15b_4site.npy"},
    {"key": "t7_meanpool",  "n": 6725,   "out": "/workspace/protein/v1/data/t7/embeddings_esm2_15b_meanpool.npy"},
    {"key": "t7_nsite",     "n": 6725,   "out": "/workspace/protein/v1/data/t7/embeddings_esm2_15b_3site.npy"},
]

for job in JOBS:
    key    = job["key"]
    chunks = sorted(glob(os.path.join(CHUNK_DIR, f"{key}_chunk*.npz")))
    if not chunks:
        print(f"\n{key}: no chunks found, skipping.")
        continue

    print(f"\n=== {key}: {len(chunks)} chunks ===")
    first = np.load(chunks[0])
    dim   = first["embeddings"].shape[1]
    dtype = first["embeddings"].dtype
    print(f"  dim={dim}  dtype={dtype}")

    out = np.zeros((job["n"], dim), dtype=dtype)
    for path in chunks:
        data = np.load(path)
        lo, hi = int(data["lo"]), int(data["hi"])
        print(f"  {os.path.basename(path)}  [{lo}:{hi}]  shape={data['embeddings'].shape}")
        out[lo:hi] = data["embeddings"]

    tmp_path = f"/tmp/{key}.npy"
    np.save(tmp_path, out)
    size_gb = os.path.getsize(tmp_path) / 1e9
    print(f"  Saved: {tmp_path}  ({size_gb:.2f} GB)")
    print(f"  Send:  runpodctl send {tmp_path}")
