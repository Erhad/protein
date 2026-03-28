"""
RunPod multi-GPU script: compute ESMc-600M embeddings for GB1 and TrpB.

Each process handles one GPU + one slice of sequences (data parallelism).
Model is tiny (~1.2GB) so all 4 GPUs load independently.

Usage (run all 4 in parallel):
    pip install esm datasets
    python runpod_esmc600m.py --rank 0 --world 4 &
    python runpod_esmc600m.py --rank 1 --world 4 &
    python runpod_esmc600m.py --rank 2 --world 4 &
    python runpod_esmc600m.py --rank 3 --world 4 &

Or use the launch script:
    bash launch_esmc600m.sh

Output chunks (merged later with merge_embeddings.py):
    /workspace/embeddings/gb1_chunk<rank>.npy
    /workspace/embeddings/trpb_chunk<rank>.npy
"""

import os, time, argparse
import numpy as np
import torch
from datasets import load_dataset
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

parser = argparse.ArgumentParser()
parser.add_argument("--rank",  type=int, default=0)
parser.add_argument("--world", type=int, default=4)
args = parser.parse_args()

RANK  = args.rank
WORLD = args.world

n_gpus = torch.cuda.device_count()
gpu_id = RANK % n_gpus
DEVICE = f"cuda:{gpu_id}" if n_gpus > 0 else "cpu"
OUT_DIR = "/workspace/embeddings"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"[rank {RANK}/{WORLD}] GPU={gpu_id} ({DEVICE})", flush=True)
print(f"Loading ESMc-600M...", flush=True)
t0 = time.time()
model = ESMC.from_pretrained("esmc_600m").to(DEVICE)
model.eval()
print(f"Loaded in {time.time()-t0:.1f}s", flush=True)


def embed_4site(sequences: list, sites: list, rank: int) -> np.ndarray:
    # Detect hidden dim
    with torch.no_grad():
        pt  = model.encode(ESMProtein(sequence=sequences[0]))
        out = model.logits(pt, LogitsConfig(return_embeddings=True))
        hidden_dim = out.embeddings.shape[-1]

    print(f"  hidden_dim={hidden_dim}  4site_dim={len(sites)*hidden_dim}", flush=True)

    n   = len(sequences)
    dim = len(sites) * hidden_dim
    result = np.zeros((n, dim), dtype=np.float32)

    t0 = time.time()
    for i, seq in enumerate(sequences):
        with torch.no_grad():
            pt  = model.encode(ESMProtein(sequence=seq))
            out = model.logits(pt, LogitsConfig(return_embeddings=True))
            emb = out.embeddings[0].float()  # (L+2, hidden_dim)
            site_vec = torch.cat([emb[s + 1] for s in sites])
            result[i] = site_vec.cpu().numpy()

        if i % 2000 == 0 and i > 0:
            elapsed = time.time() - t0
            rate    = i / elapsed
            eta     = (n - i) / rate / 60
            print(f"  [rank {rank}] {i:>7,}/{n:,}  {rate:.1f} seq/s  ETA {eta:.0f}min", flush=True)

    return result


LANDSCAPES = [
    {
        "name":  "gb1",
        "hf":    "SaProtHub/Dataset-GB1-fitness",
        "sites": [38, 39, 40, 53],
        "n":     149361,
    },
    {
        "name":  "trpb",
        "hf":    "SaProtHub/Dataset-TrpB_fitness_landsacpe",
        "sites": [182, 183, 226, 227],
        "n":     160000,
    },
]

for lc in LANDSCAPES:
    name  = lc["name"]
    out_path = f"{OUT_DIR}/{name}_chunk{RANK:02d}.npy"

    if os.path.exists(out_path):
        print(f"\n[rank {RANK}] {name} chunk already exists, skipping.", flush=True)
        continue

    print(f"\n[rank {RANK}] === {name.upper()} ===", flush=True)
    ds   = load_dataset(lc["hf"], split="train")
    seqs = [s.replace("*", "").strip() for s in ds["protein"]]

    # Slice for this rank
    n_total = len(seqs)
    lo = (n_total * RANK) // WORLD
    hi = (n_total * (RANK + 1)) // WORLD
    my_seqs = seqs[lo:hi]
    print(f"  slice [{lo}:{hi}]  ({len(my_seqs)} sequences)", flush=True)

    t0 = time.time()
    emb = embed_4site(my_seqs, lc["sites"], RANK)
    print(f"  Done in {time.time()-t0:.0f}s  shape={emb.shape}", flush=True)

    # Save with lo/hi metadata via npz
    np.savez(out_path.replace(".npy", ".npz"), embeddings=emb, lo=lo, hi=hi)
    print(f"  Saved: {out_path.replace('.npy','.npz')}", flush=True)
    del emb

print(f"\n[rank {RANK}] All done!", flush=True)
