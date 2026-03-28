"""
Merge per-rank chunk npz files into final embeddings_esm2_15b_4site.npz.
Run after launch_embeddings.sh completes.

Usage:
    python precompute/merge_embeddings.py

Then rsync the two npz files to your Mac:
    rsync -avz --progress root@<pod-ip>:/workspace/data/ /Users/aleksanderheino/protein/v1/data/
"""

import os
import numpy as np
from glob import glob

CHUNK_DIR = "/workspace/embeddings"
OUT_DIR   = "/workspace/data"

TASKS = [
    {"name": "gb1",  "n": 149361},
    {"name": "trpb", "n": 160000},
]

for task in TASKS:
    name = task["name"]
    chunks = sorted(glob(os.path.join(CHUNK_DIR, f"{name}_chunk*.npz")))
    print(f"\n=== {name}: {len(chunks)} chunks ===")

    # Auto-detect dim from first chunk
    first = np.load(chunks[0])
    dim = first["embeddings"].shape[1]
    print(f"  dim={dim} (auto-detected)")

    out = np.zeros((task["n"], dim), dtype=np.float32)

    for path in chunks:
        data = np.load(path)
        lo, hi = int(data["lo"]), int(data["hi"])
        print(f"  {os.path.basename(path)}  [{lo}:{hi}]  shape={data['embeddings'].shape}")
        out[lo:hi] = data["embeddings"]

    out_path = os.path.join(OUT_DIR, f"{name}/embeddings_esmc600m_4site.npy")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, out)
    print(f"Saved: {out_path}  ({os.path.getsize(out_path)/1e9:.2f} GB)")

print("\nAll merged. Now rsync to your Mac.")
