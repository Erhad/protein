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
OUT_PATH  = "/workspace/protein/v1/data/gb1/embeddings_esm2_15b_meanpool.npy"
N         = 149361

chunks = sorted(glob(os.path.join(CHUNK_DIR, "gb1_15b_chunk*.npz")))
print(f"Found {len(chunks)} chunks")

first = np.load(chunks[0])
dim   = first["embeddings"].shape[1]
print(f"dim={dim}")

out = np.zeros((N, dim), dtype=np.float16)

for path in chunks:
    data = np.load(path)
    lo, hi = int(data["lo"]), int(data["hi"])
    print(f"  {os.path.basename(path)}  [{lo}:{hi}]  shape={data['embeddings'].shape}")
    out[lo:hi] = data["embeddings"]

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
np.save(OUT_PATH, out)
print(f"\nSaved: {OUT_PATH}  ({os.path.getsize(OUT_PATH)/1e9:.2f} GB)")
print("Now send to your Mac:")
print(f"  runpodctl send {OUT_PATH}")
