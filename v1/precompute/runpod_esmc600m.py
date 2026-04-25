"""
RunPod multi-GPU: compute ESMc-600M embeddings (meanpool and N-site) for all landscapes.

Model is ~1.2GB — all GPUs load independently.

Usage (via launch_esmc600m.sh):
    python precompute/runpod_esmc600m.py --rank 0 --world 4 --only gb1_meanpool tev_nsite
    python precompute/runpod_esmc600m.py --rank 0 --world 4          # all jobs

Chunk outputs (merged by merge_esmc600m.py):
    /workspace/embeddings/{key}_esmc_chunk{rank:02d}.npz
"""

import argparse
import os
import time

import numpy as np
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein

# ── Job definitions ────────────────────────────────────────────────────────────
JOBS = [
    {
        "key":     "gb1_meanpool",
        "csv":     "/workspace/protein/v1/data/gb1/gb1_fitness.csv",
        "hf":      None,
        "seq_col": "protein",
        "mode":    "meanpool",
        "sites":   None,
        "n":       149361,
        "out":     "/workspace/protein/v1/data/gb1/embeddings_esmc600m_meanpool.npy",
        "batch":   128,
    },
    {
        "key":     "trpb_meanpool",
        "csv":     None,
        "hf":      "SaProtHub/Dataset-TrpB_fitness_landsacpe",
        "seq_col": "protein",
        "mode":    "meanpool",
        "sites":   None,
        "n":       160000,
        "out":     "/workspace/protein/v1/data/trpb/embeddings_esmc600m_meanpool.npy",
        "batch":   128,
    },
    {
        "key":     "tev_meanpool",
        "csv":     "/workspace/protein/v1/data/tev/tev_fitness.csv",
        "hf":      None,
        "seq_col": "protein",
        "mode":    "meanpool",
        "sites":   None,
        "n":       159132,
        "out":     "/workspace/protein/v1/data/tev/embeddings_esmc600m_meanpool.npy",
        "batch":   128,
    },
    {
        "key":     "tev_nsite",
        "csv":     "/workspace/protein/v1/data/tev/tev_fitness.csv",
        "hf":      None,
        "seq_col": "protein",
        "mode":    "nsite",
        "sites":   [145, 147, 166, 169],
        "n":       159132,
        "out":     "/workspace/protein/v1/data/tev/embeddings_esmc600m_4site.npy",
        "batch":   128,
    },
    {
        "key":     "t7_meanpool",
        "csv":     "/workspace/protein/v1/data/t7/t7_fitness.csv",
        "hf":      None,
        "seq_col": "protein",
        "mode":    "meanpool",
        "sites":   None,
        "n":       6725,
        "out":     "/workspace/protein/v1/data/t7/embeddings_esmc600m_meanpool.npy",
        "batch":   16,  # length-883 sequences
    },
    {
        "key":     "t7_nsite",
        "csv":     "/workspace/protein/v1/data/t7/t7_fitness.csv",
        "hf":      None,
        "seq_col": "protein",
        "mode":    "nsite",
        "sites":   [747, 755, 757],
        "n":       6725,
        "out":     "/workspace/protein/v1/data/t7/embeddings_esmc600m_3site.npy",
        "batch":   16,
    },
]

# ── Setup ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--rank",  type=int, default=0)
parser.add_argument("--world", type=int, default=4)
parser.add_argument("--only",  nargs="+", default=None)
args = parser.parse_args()

RANK  = args.rank
WORLD = args.world

n_gpus = torch.cuda.device_count()
gpu_id = RANK % n_gpus
DEVICE = f"cuda:{gpu_id}" if n_gpus > 0 else "cpu"
OUT_DIR = "/workspace/embeddings"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"[rank {RANK}/{WORLD}] GPU={gpu_id} ({DEVICE})", flush=True)
print("Loading ESMc-600M ...", flush=True)
t0 = time.time()
model = ESMC.from_pretrained("esmc_600m").to(DEVICE)
model.eval()
print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

# Detect hidden dim from a dummy forward pass
with torch.no_grad():
    _pt  = model.encode(ESMProtein(sequence="ACDE"))
    _out = model.forward(sequence_tokens=_pt.sequence.unsqueeze(0).to(DEVICE))
    HIDDEN_DIM = _out.embeddings.shape[-1]
print(f"ESMc hidden_dim={HIDDEN_DIM}", flush=True)


def _encode_batch(sequences: list):
    """Encode sequences to token tensors (same length required for stack)."""
    tokens = [model.encode(ESMProtein(sequence=s)).sequence for s in sequences]
    lengths = [t.shape[0] for t in tokens]
    if len(set(lengths)) == 1:
        return torch.stack(tokens).to(DEVICE), True
    return tokens, False  # variable length — process individually


def embed_meanpool(sequences: list, rank: int, batch: int) -> np.ndarray:
    n      = len(sequences)
    result = np.zeros((n, HIDDEN_DIM), dtype=np.float32)
    t0     = time.time()

    for start in range(0, n, batch):
        batch_seqs = sequences[start : start + batch]
        tokens, same_len = _encode_batch(batch_seqs)

        if same_len:
            with torch.no_grad():
                out = model.forward(sequence_tokens=tokens)
            emb = out.embeddings.float()  # (B, L+2, dim)
            # positions 1..-1 are residues; 0=BOS, -1=EOS
            pooled = emb[:, 1:-1, :].mean(dim=1).cpu().numpy()
            result[start : start + len(batch_seqs)] = pooled
        else:
            for j, tok in enumerate(tokens):
                with torch.no_grad():
                    out = model.forward(sequence_tokens=tok.unsqueeze(0).to(DEVICE))
                emb = out.embeddings[0].float()  # (L+2, dim)
                result[start + j] = emb[1:-1, :].mean(dim=0).cpu().numpy()

        if start % (batch * 20) == 0 and start > 0:
            elapsed = time.time() - t0
            rate    = start / elapsed
            eta     = (n - start) / rate / 60 if rate > 0 else 0
            print(f"  [rank {rank}] {min(start+batch,n):>7,}/{n:,}  "
                  f"{rate:.1f} seq/s  ETA {eta:.0f}min", flush=True)

    return result


def embed_nsite(sequences: list, sites: list, rank: int, batch: int) -> np.ndarray:
    dim    = len(sites) * HIDDEN_DIM
    n      = len(sequences)
    result = np.zeros((n, dim), dtype=np.float32)
    t0     = time.time()

    for start in range(0, n, batch):
        batch_seqs = sequences[start : start + batch]
        tokens, same_len = _encode_batch(batch_seqs)

        if same_len:
            with torch.no_grad():
                out = model.forward(sequence_tokens=tokens)
            emb = out.embeddings.float()  # (B, L+2, dim)
            for j in range(len(batch_seqs)):
                result[start + j] = torch.cat([emb[j, s + 1] for s in sites]).cpu().numpy()
        else:
            for j, tok in enumerate(tokens):
                with torch.no_grad():
                    out = model.forward(sequence_tokens=tok.unsqueeze(0).to(DEVICE))
                emb = out.embeddings[0].float()
                result[start + j] = torch.cat([emb[s + 1] for s in sites]).cpu().numpy()

        if start % (batch * 20) == 0 and start > 0:
            elapsed = time.time() - t0
            rate    = start / elapsed
            eta     = (n - start) / rate / 60 if rate > 0 else 0
            print(f"  [rank {rank}] {min(start+batch,n):>7,}/{n:,}  "
                  f"{rate:.1f} seq/s  ETA {eta:.0f}min", flush=True)

    return result


# ── Load data helper ───────────────────────────────────────────────────────────
import pandas as pd
from datasets import load_dataset

def load_sequences(job: dict) -> list:
    if job["csv"]:
        df = pd.read_csv(job["csv"])
        return df[job["seq_col"]].str.strip().str.replace("*", "", regex=False).tolist()
    ds = load_dataset(job["hf"], split="train")
    return [s.replace("*", "").strip() for s in ds[job["seq_col"]]]


# ── Main loop ──────────────────────────────────────────────────────────────────
for job in JOBS:
    key = job["key"]
    if args.only and key not in args.only:
        continue

    chunk_path = f"{OUT_DIR}/{key}_esmc_chunk{RANK:02d}.npz"

    if os.path.exists(chunk_path):
        print(f"\n[rank {RANK}] {key}: chunk exists, skipping.", flush=True)
        continue

    print(f"\n[rank {RANK}] === {key} ({job['mode']}) ===", flush=True)

    all_seqs = load_sequences(job)
    n_total  = len(all_seqs)
    lo = (n_total * RANK) // WORLD
    hi = (n_total * (RANK + 1)) // WORLD
    my_seqs = all_seqs[lo:hi]
    print(f"  slice [{lo}:{hi}]  ({len(my_seqs):,} sequences)", flush=True)

    t_start = time.time()
    if job["mode"] == "meanpool":
        emb = embed_meanpool(my_seqs, RANK, job["batch"])
    else:
        emb = embed_nsite(my_seqs, job["sites"], RANK, job["batch"])

    print(f"  Done in {time.time()-t_start:.0f}s  shape={emb.shape}", flush=True)
    np.savez(chunk_path, embeddings=emb, lo=lo, hi=hi)
    print(f"  Saved: {chunk_path}", flush=True)

print(f"\n[rank {RANK}] All done!", flush=True)
