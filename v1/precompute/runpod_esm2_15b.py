"""
RunPod multi-GPU: compute ESM2-15B embeddings (meanpool and N-site) for all landscapes.

Each process handles one GPU + one slice of sequences (data parallelism).
Model is ~30GB fp16 — use 4x A100-40GB or A100-80GB.

Usage (via launch_esm2_15b.sh):
    python precompute/runpod_esm2_15b.py --rank 0 --world 4 --only gb1_nsite tev_meanpool
    python precompute/runpod_esm2_15b.py --rank 0 --world 4          # all jobs

Chunk outputs (merged by merge_esm2_15b.py):
    /workspace/embeddings/{key}_chunk{rank:02d}.npz
"""

import argparse
import os
import time

import numpy as np
import torch
from transformers import AutoTokenizer, EsmModel

MODEL_NAME = "facebook/esm2_t48_15B_UR50D"
EMB_DIM    = 5120

# ── Job definitions ────────────────────────────────────────────────────────────
# key: unique id used for chunk naming and --only filtering
# mode: "meanpool" (all residues, avg) or "nsite" (specific positions, concat)
# batch: sequences per forward pass — reduce for long sequences
JOBS = [
    {
        "key":     "gb1_nsite",
        "csv":     "/workspace/protein/v1/data/gb1/gb1_fitness.csv",
        "hf":      None,
        "seq_col": "protein",
        "mode":    "nsite",
        "sites":   [38, 39, 40, 53],
        "n":       149361,
        "out":     "/workspace/protein/v1/data/gb1/embeddings_esm2_15b_4site.npy",
        "batch":   16,
    },
    {
        "key":     "trpb_nsite",
        "csv":     None,
        "hf":      "SaProtHub/Dataset-TrpB_fitness_landsacpe",
        "seq_col": "protein",
        "mode":    "nsite",
        "sites":   [182, 183, 226, 227],
        "n":       160000,
        "out":     "/workspace/protein/v1/data/trpb/embeddings_esm2_15b_4site.npy",
        "batch":   16,
    },
    {
        "key":     "tev_meanpool",
        "csv":     "/workspace/protein/v1/data/tev/tev_fitness.csv",
        "hf":      None,
        "seq_col": "protein",
        "mode":    "meanpool",
        "sites":   None,
        "n":       159132,
        "out":     "/workspace/protein/v1/data/tev/embeddings_esm2_15b_meanpool.npy",
        "batch":   16,
    },
    {
        "key":     "tev_nsite",
        "csv":     "/workspace/protein/v1/data/tev/tev_fitness.csv",
        "hf":      None,
        "seq_col": "protein",
        "mode":    "nsite",
        "sites":   [145, 147, 166, 169],
        "n":       159132,
        "out":     "/workspace/protein/v1/data/tev/embeddings_esm2_15b_4site.npy",
        "batch":   16,
    },
    {
        "key":     "t7_meanpool",
        "csv":     "/workspace/protein/v1/data/t7/t7_fitness.csv",
        "hf":      None,
        "seq_col": "protein",
        "mode":    "meanpool",
        "sites":   None,
        "n":       6725,
        "out":     "/workspace/protein/v1/data/t7/embeddings_esm2_15b_meanpool.npy",
        "batch":   4,  # length-883 sequences — reduce batch to fit VRAM
    },
    {
        "key":     "t7_nsite",
        "csv":     "/workspace/protein/v1/data/t7/t7_fitness.csv",
        "hf":      None,
        "seq_col": "protein",
        "mode":    "nsite",
        "sites":   [747, 755, 757],
        "n":       6725,
        "out":     "/workspace/protein/v1/data/t7/embeddings_esm2_15b_3site.npy",
        "batch":   4,
    },
]

# ── Setup ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--rank",  type=int, default=0)
parser.add_argument("--world", type=int, default=4)
parser.add_argument("--only",  nargs="+", default=None,
                    help="Job keys to run, e.g. --only gb1_nsite tev_meanpool")
args = parser.parse_args()

RANK  = args.rank
WORLD = args.world

n_gpus = torch.cuda.device_count()
gpu_id = RANK % n_gpus
DEVICE = f"cuda:{gpu_id}" if n_gpus > 0 else "cpu"
OUT_DIR = "/workspace/embeddings"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"[rank {RANK}/{WORLD}] GPU={gpu_id} ({DEVICE})", flush=True)
print(f"Loading {MODEL_NAME} ...", flush=True)
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = EsmModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, use_safetensors=True)
model.to(DEVICE).eval()
print(f"Loaded in {time.time()-t0:.1f}s", flush=True)


def embed_meanpool(sequences: list, rank: int, batch: int) -> np.ndarray:
    n = len(sequences)
    result = np.zeros((n, EMB_DIM), dtype=np.float16)
    t0 = time.time()

    for start in range(0, n, batch):
        batch_seqs = sequences[start : start + batch]
        enc = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=False)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc)

        hidden = out.last_hidden_state          # (B, L, 5120) fp16
        mask   = enc["attention_mask"].float()  # (B, L)
        mask[:, 0] = 0.0                        # exclude BOS
        seq_lens = enc["attention_mask"].sum(dim=1)
        for b in range(len(batch_seqs)):
            mask[b, seq_lens[b] - 1] = 0.0     # exclude EOS

        mask   = mask.unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        result[start : start + len(batch_seqs)] = pooled.cpu().numpy().astype(np.float16)

        if start % (batch * 20) == 0 and start > 0:
            elapsed = time.time() - t0
            rate    = start / elapsed
            eta     = (n - start) / rate / 60 if rate > 0 else 0
            print(f"  [rank {rank}] {min(start+batch,n):>7,}/{n:,}  "
                  f"{rate:.1f} seq/s  ETA {eta:.0f}min", flush=True)

    return result


def embed_nsite(sequences: list, sites: list, rank: int, batch: int) -> np.ndarray:
    dim = len(sites) * EMB_DIM
    n   = len(sequences)
    result = np.zeros((n, dim), dtype=np.float16)
    t0 = time.time()

    for start in range(0, n, batch):
        batch_seqs = sequences[start : start + batch]
        enc = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=False)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc)

        hidden = out.last_hidden_state.float().cpu().numpy()  # (B, L, 5120)
        for j in range(len(batch_seqs)):
            # +1 offset: position 0 is BOS, residues start at 1
            result[start + j] = np.concatenate([hidden[j, s + 1] for s in sites])

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

    out_path   = job["out"]
    chunk_path = f"{OUT_DIR}/{key}_chunk{RANK:02d}.npz"

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
