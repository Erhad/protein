"""
Merge per-rank ESM2-15B chunk files into final (N, 5120) float16 array.
Run after all workers in launch_esm2_15b.sh complete.

Usage:
    python precompute/merge_esm2_15b.py
"""

import os
import numpy as np
from glob import glob

CHUNK_DIR = "/workspace/embeddings"

LANDSCAPES = [
    {"name": "gb1",  "n": 149361, "out": "/workspace/protein/v1/data/gb1/embeddings_esm2_15b_meanpool.npy"},
    {"name": "trpb", "n": 160000, "out": "/workspace/protein/v1/data/trpb/embeddings_esm2_15b_meanpool.npy"},
]

for lc in LANDSCAPES:
    name   = lc["name"]
    chunks = sorted(glob(os.path.join(CHUNK_DIR, f"{name}_15b_chunk*.npz")))
    if not chunks:
        print(f"\n{name}: no chunks found, skipping.")
        continue

    print(f"\n=== {name}: {len(chunks)} chunks ===")
    first = np.load(chunks[0])
    dim   = first["embeddings"].shape[1]
    print(f"  dim={dim}")

    out = np.zeros((lc["n"], dim), dtype=np.float16)
    for path in chunks:
        data = np.load(path)
        lo, hi = int(data["lo"]), int(data["hi"])
        print(f"  {os.path.basename(path)}  [{lo}:{hi}]  shape={data['embeddings'].shape}")
        out[lo:hi] = data["embeddings"]

    os.makedirs(os.path.dirname(lc["out"]), exist_ok=True)
    np.save(lc["out"], out)
    print(f"  Saved: {lc['out']}  ({os.path.getsize(lc['out'])/1e9:.2f} GB)")
    print(f"  Send:  runpodctl send {lc['out']}")
