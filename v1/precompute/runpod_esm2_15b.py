"""
RunPod multi-GPU script: compute ESM2-15B mean-pool embeddings for GB1.

Each process handles one GPU + one slice of sequences (data parallelism).
Model is ~30GB fp16, so one copy per GPU — use 4x A100-80GB or similar.

Mean pool: average over all residue positions, excluding BOS and EOS tokens.
Output dim: 5120 (ESM2-15B hidden size).

Usage (launched automatically by launch_esm2_15b.sh):
    python precompute/runpod_esm2_15b.py --rank 0 --world 4 &
    python precompute/runpod_esm2_15b.py --rank 1 --world 4 &
    python precompute/runpod_esm2_15b.py --rank 2 --world 4 &
    python precompute/runpod_esm2_15b.py --rank 3 --world 4 &

Output chunks (merged later with merge_esm2_15b.py):
    /workspace/embeddings/gb1_15b_chunk<rank>.npz
"""

import argparse
import os
import time

import numpy as np
import torch
from transformers import AutoTokenizer, EsmModel

MODEL_NAME = "facebook/esm2_t48_15B_UR50D"
EMB_DIM    = 5120
BATCH      = 16   # sequences per forward pass — 15B is ~30GB fp16, 16 seqs of 56aa fits easily

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
print(f"Loading {MODEL_NAME}...", flush=True)
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = EsmModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
model.to(DEVICE)
model.eval()
print(f"Loaded in {time.time()-t0:.1f}s", flush=True)


def embed_meanpool(sequences: list, rank: int) -> np.ndarray:
    n = len(sequences)
    result = np.zeros((n, EMB_DIM), dtype=np.float16)
    t0 = time.time()

    for start in range(0, n, BATCH):
        batch_seqs = sequences[start : start + BATCH]
        enc = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc)

        hidden = out.last_hidden_state   # (B, L, 5120) fp16
        mask   = enc["attention_mask"].float()  # (B, L)

        # Exclude BOS (position 0) and EOS (last real token) from the mean
        mask[:, 0] = 0.0
        seq_lens = enc["attention_mask"].sum(dim=1)  # includes BOS + residues + EOS
        for b in range(len(batch_seqs)):
            mask[b, seq_lens[b] - 1] = 0.0

        mask   = mask.unsqueeze(-1)                          # (B, L, 1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # (B, 5120)
        result[start : start + len(batch_seqs)] = pooled.cpu().float().numpy().astype(np.float16)

        if start % (BATCH * 20) == 0 and start > 0:
            elapsed = time.time() - t0
            rate    = (start + BATCH) / elapsed
            eta     = (n - start) / rate / 60
            print(f"  [rank {rank}] {min(start+BATCH,n):>7,}/{n:,}  "
                  f"{rate:.1f} seq/s  ETA {eta:.0f}min", flush=True)

    return result


# ── GB1 ───────────────────────────────────────────────────────────────────────
import pandas as pd

CSV_PATH = "/workspace/protein/v1/data/gb1/gb1_fitness.csv"
OUT_PATH = f"{OUT_DIR}/gb1_15b_chunk{RANK:02d}.npz"

if os.path.exists(OUT_PATH):
    print(f"\n[rank {RANK}] chunk already exists, skipping.", flush=True)
else:
    print(f"\n[rank {RANK}] === GB1 ===", flush=True)
    df   = pd.read_csv(CSV_PATH)
    seqs = df["protein"].str.strip().str.replace("*", "", regex=False).tolist()
    n_total = len(seqs)

    lo = (n_total * RANK) // WORLD
    hi = (n_total * (RANK + 1)) // WORLD
    my_seqs = seqs[lo:hi]
    print(f"  slice [{lo}:{hi}]  ({len(my_seqs):,} sequences)", flush=True)

    t_start = time.time()
    emb = embed_meanpool(my_seqs, RANK)
    print(f"  Done in {time.time()-t_start:.0f}s  shape={emb.shape}", flush=True)

    np.savez(OUT_PATH, embeddings=emb, lo=lo, hi=hi)
    print(f"  Saved: {OUT_PATH}", flush=True)

print(f"\n[rank {RANK}] All done!", flush=True)
