"""
Watch for completed chunk sets and immediately merge + send each job.
Run in a second pod terminal while workers are running.

Usage:
    python precompute/watch_and_send.py --world 6
    python precompute/watch_and_send.py --world 6 --model esmc
"""

import argparse
import os
import subprocess
import time

import numpy as np
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument("--world", type=int, default=6)
parser.add_argument("--model", choices=["esm2_15b", "esmc"], default="esm2_15b")
args = parser.parse_args()

CHUNK_DIR = "/workspace/embeddings"

JOBS_ESM2_15B = [
    {"key": "gb1_nsite",    "n": 149361, "out": "/workspace/protein/v1/data/gb1/embeddings_esm2_15b_4site.npy"},
    {"key": "trpb_nsite",   "n": 160000, "out": "/workspace/protein/v1/data/trpb/embeddings_esm2_15b_4site.npy"},
    {"key": "tev_meanpool", "n": 159132, "out": "/workspace/protein/v1/data/tev/embeddings_esm2_15b_meanpool.npy"},
    {"key": "tev_nsite",    "n": 159132, "out": "/workspace/protein/v1/data/tev/embeddings_esm2_15b_4site.npy"},
    {"key": "t7_meanpool",  "n": 6725,   "out": "/workspace/protein/v1/data/t7/embeddings_esm2_15b_meanpool.npy"},
    {"key": "t7_nsite",     "n": 6725,   "out": "/workspace/protein/v1/data/t7/embeddings_esm2_15b_3site.npy"},
]

JOBS_ESMC = [
    {"key": "gb1_meanpool",  "n": 149361, "out": "/workspace/protein/v1/data/gb1/embeddings_esmc600m_meanpool.npy"},
    {"key": "trpb_meanpool", "n": 160000, "out": "/workspace/protein/v1/data/trpb/embeddings_esmc600m_meanpool.npy"},
    {"key": "tev_meanpool",  "n": 159132, "out": "/workspace/protein/v1/data/tev/embeddings_esmc600m_meanpool.npy"},
    {"key": "tev_nsite",     "n": 159132, "out": "/workspace/protein/v1/data/tev/embeddings_esmc600m_4site.npy"},
    {"key": "t7_meanpool",   "n": 6725,   "out": "/workspace/protein/v1/data/t7/embeddings_esmc600m_meanpool.npy"},
    {"key": "t7_nsite",      "n": 6725,   "out": "/workspace/protein/v1/data/t7/embeddings_esmc600m_3site.npy"},
]

JOBS      = JOBS_ESM2_15B if args.model == "esm2_15b" else JOBS_ESMC
chunk_sfx = "_chunk" if args.model == "esm2_15b" else "_esmc_chunk"

done = set()
print(f"Watching {len(JOBS)} jobs (world={args.world}, model={args.model})...", flush=True)

while len(done) < len(JOBS):
    for job in JOBS:
        key = job["key"]
        if key in done:
            continue

        chunks = sorted(glob(f"{CHUNK_DIR}/{key}{chunk_sfx}*.npz"))
        if len(chunks) < args.world:
            continue

        print(f"\n=== {key}: all {args.world} chunks found — merging ===", flush=True)
        first = np.load(chunks[0])
        dim, dtype = first["embeddings"].shape[1], first["embeddings"].dtype
        out = np.zeros((job["n"], dim), dtype=dtype)
        for path in chunks:
            data = np.load(path)
            lo, hi = int(data["lo"]), int(data["hi"])
            out[lo:hi] = data["embeddings"]

        # Save to /tmp (local SSD) to avoid /workspace network volume I/O errors
        send_path = f"/tmp/{key}.npy"
        np.save(send_path, out)
        size_gb = os.path.getsize(send_path) / 1e9
        print(f"  Saved: {send_path}  ({size_gb:.2f} GB)", flush=True)
        subprocess.run(["runpodctl", "send", send_path])
        os.remove(send_path)  # free /tmp space after send

        done.add(key)
        print(f"  Done. {len(done)}/{len(JOBS)} jobs complete.", flush=True)

    time.sleep(30)

print("\nAll jobs sent!")
