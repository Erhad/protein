"""
RunPod multi-GPU: compute ESMc-600M embeddings for all landscapes.

Single forward pass per protein — both meanpool and nsite extracted from the
same hidden states.

Usage:
    python precompute/runpod_esmc600m.py --rank 0 --world 4
    python precompute/runpod_esmc600m.py --rank 0 --world 4 --only tev t7

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

PROTEINS = [
    {
        "name":    "gb1",
        "csv":     "/workspace/protein/v1/data/gb1/gb1_fitness.csv",
        "hf":      None,
        "seq_col": "protein",
        "n":       149361,
        "batch":   128,
        "outputs": [
            # nsite already computed — skip
            {
                "key":   "gb1_meanpool",
                "mode":  "meanpool",
                "sites": None,
                "out":   "/workspace/protein/v1/data/gb1/embeddings_esmc600m_meanpool.npy",
            },
        ],
    },
    {
        "name":    "trpb",
        "csv":     None,
        "hf":      "SaProtHub/Dataset-TrpB_fitness_landsacpe",
        "seq_col": "protein",
        "n":       160000,
        "batch":   128,
        "outputs": [
            # nsite already computed — skip
            {
                "key":   "trpb_meanpool",
                "mode":  "meanpool",
                "sites": None,
                "out":   "/workspace/protein/v1/data/trpb/embeddings_esmc600m_meanpool.npy",
            },
        ],
    },
    {
        "name":    "tev",
        "csv":     "/workspace/protein/v1/data/tev/tev_fitness.csv",
        "hf":      None,
        "seq_col": "protein",
        "n":       159132,
        "batch":   128,
        "outputs": [
            {
                "key":   "tev_meanpool",
                "mode":  "meanpool",
                "sites": None,
                "out":   "/workspace/protein/v1/data/tev/embeddings_esmc600m_meanpool.npy",
            },
            {
                "key":   "tev_nsite",
                "mode":  "nsite",
                "sites": [145, 147, 166, 169],
                "out":   "/workspace/protein/v1/data/tev/embeddings_esmc600m_4site.npy",
            },
        ],
    },
    {
        "name":    "t7",
        "csv":     "/workspace/protein/v1/data/t7/t7_fitness.csv",
        "hf":      None,
        "seq_col": "protein",
        "n":       6725,
        "batch":   16,  # length-883 sequences
        "outputs": [
            {
                "key":   "t7_meanpool",
                "mode":  "meanpool",
                "sites": None,
                "out":   "/workspace/protein/v1/data/t7/embeddings_esmc600m_meanpool.npy",
            },
            {
                "key":   "t7_nsite",
                "mode":  "nsite",
                "sites": [747, 755, 757],
                "out":   "/workspace/protein/v1/data/t7/embeddings_esmc600m_3site.npy",
            },
        ],
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

# Detect hidden dim
with torch.no_grad():
    _pt  = model.encode(ESMProtein(sequence="ACDE"))
    _out = model.forward(sequence_tokens=_pt.sequence.unsqueeze(0).to(DEVICE))
    HIDDEN_DIM = _out.embeddings.shape[-1]
print(f"ESMc hidden_dim={HIDDEN_DIM}", flush=True)


def embed_protein(sequences: list, protein: dict, rank: int) -> dict[str, np.ndarray]:
    """One forward pass per batch — extract all outputs from the same hidden states."""
    n     = len(sequences)
    batch = protein["batch"]
    outs  = protein["outputs"]

    results = {}
    for o in outs:
        if o["mode"] == "meanpool":
            results[o["key"]] = np.zeros((n, HIDDEN_DIM), dtype=np.float32)
        else:
            results[o["key"]] = np.zeros((n, len(o["sites"]) * HIDDEN_DIM), dtype=np.float32)

    t0 = time.time()
    for start in range(0, n, batch):
        batch_seqs = sequences[start : start + batch]

        tokens = [model.encode(ESMProtein(sequence=s)).sequence for s in batch_seqs]
        lengths = [t.shape[0] for t in tokens]
        same_len = len(set(lengths)) == 1

        if same_len:
            tok_tensor = torch.stack(tokens).to(DEVICE)
            with torch.no_grad():
                out = model.forward(sequence_tokens=tok_tensor)
            emb = out.embeddings.float()  # (B, L+2, dim)

            for o in outs:
                key = o["key"]
                if o["mode"] == "meanpool":
                    pooled = emb[:, 1:-1, :].mean(dim=1).cpu().numpy()
                    results[key][start : start + len(batch_seqs)] = pooled
                else:
                    for j in range(len(batch_seqs)):
                        results[key][start + j] = torch.cat(
                            [emb[j, s + 1] for s in o["sites"]]
                        ).cpu().numpy()
        else:
            for j, tok in enumerate(tokens):
                with torch.no_grad():
                    out = model.forward(sequence_tokens=tok.unsqueeze(0).to(DEVICE))
                emb = out.embeddings[0].float()  # (L+2, dim)
                for o in outs:
                    key = o["key"]
                    if o["mode"] == "meanpool":
                        results[key][start + j] = emb[1:-1, :].mean(dim=0).cpu().numpy()
                    else:
                        results[key][start + j] = torch.cat(
                            [emb[s + 1] for s in o["sites"]]
                        ).cpu().numpy()

        if start % (batch * 20) == 0 and start > 0:
            elapsed = time.time() - t0
            rate    = start / elapsed
            eta     = (n - start) / rate / 60 if rate > 0 else 0
            print(f"  [rank {rank}] {min(start+batch,n):>7,}/{n:,}  "
                  f"{rate:.1f} seq/s  ETA {eta:.0f}min", flush=True)

    return results


# ── Load data helper ───────────────────────────────────────────────────────────
import pandas as pd
from datasets import load_dataset

def load_sequences(protein: dict) -> list:
    if protein["csv"]:
        df = pd.read_csv(protein["csv"])
        return df[protein["seq_col"]].str.strip().str.replace("*", "", regex=False).tolist()
    ds = load_dataset(protein["hf"], split="train")
    return [s.replace("*", "").strip() for s in ds[protein["seq_col"]]]


# ── Main loop ──────────────────────────────────────────────────────────────────
for protein in PROTEINS:
    name = protein["name"]
    if args.only and name not in args.only:
        continue

    pending = [o for o in protein["outputs"]
               if not os.path.exists(f"{OUT_DIR}/{o['key']}_esmc_chunk{RANK:02d}.npz")]
    if not pending:
        print(f"\n[rank {RANK}] {name}: all chunks exist, skipping.", flush=True)
        continue

    print(f"\n[rank {RANK}] === {name} ({[o['key'] for o in pending]}) ===", flush=True)

    all_seqs = load_sequences(protein)
    n_total  = len(all_seqs)
    lo = (n_total * RANK) // WORLD
    hi = (n_total * (RANK + 1)) // WORLD
    my_seqs = all_seqs[lo:hi]
    print(f"  slice [{lo}:{hi}]  ({len(my_seqs):,} sequences)", flush=True)

    protein_pending = {**protein, "outputs": pending}
    t_start = time.time()
    results = embed_protein(my_seqs, protein_pending, RANK)
    print(f"  Done in {time.time()-t_start:.0f}s", flush=True)

    for o in pending:
        chunk_path = f"{OUT_DIR}/{o['key']}_esmc_chunk{RANK:02d}.npz"
        np.savez(chunk_path, embeddings=results[o["key"]], lo=lo, hi=hi)
        print(f"  Saved: {chunk_path}", flush=True)

print(f"\n[rank {RANK}] All done!", flush=True)
