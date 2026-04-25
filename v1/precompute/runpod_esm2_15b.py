"""
RunPod multi-GPU: compute ESM2-15B embeddings for all landscapes.

Single forward pass per protein — both meanpool and nsite extracted from the
same hidden states. Each GPU processes a slice of each protein's sequences.

Usage:
    python precompute/runpod_esm2_15b.py --rank 0 --world 4
    python precompute/runpod_esm2_15b.py --rank 0 --world 4 --only tev t7

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

# ── Protein definitions ────────────────────────────────────────────────────────
# Each protein is processed in ONE forward pass; all listed outputs are extracted
# from the same hidden states.
# Skip outputs whose file already exists on disk.
PROTEINS = [
    {
        "name":    "gb1",
        "csv":     "/workspace/protein/v1/data/gb1/gb1_fitness.csv",
        "hf":      None,
        "seq_col": "protein",
        "n":       149361,
        "batch":   None,  # auto
        "outputs": [
            # meanpool already computed — skip by not listing it here
            {
                "key":   "gb1_nsite",
                "mode":  "nsite",
                "sites": [38, 39, 40, 53],
                "out":   "/workspace/protein/v1/data/gb1/embeddings_esm2_15b_4site.npy",
            },
        ],
    },
    {
        "name":    "trpb",
        "csv":     None,
        "hf":      "SaProtHub/Dataset-TrpB_fitness_landsacpe",
        "seq_col": "protein",
        "n":       160000,
        "batch":   None,  # auto
        "outputs": [
            # meanpool already computed — skip
            {
                "key":   "trpb_nsite",
                "mode":  "nsite",
                "sites": [182, 183, 226, 227],
                "out":   "/workspace/protein/v1/data/trpb/embeddings_esm2_15b_4site.npy",
            },
        ],
    },
    {
        "name":    "tev",
        "csv":     "/workspace/protein/v1/data/tev/tev_fitness.csv",
        "hf":      None,
        "seq_col": "protein",
        "n":       159132,
        "batch":   None,  # auto
        "outputs": [
            {
                "key":   "tev_meanpool",
                "mode":  "meanpool",
                "sites": None,
                "out":   "/workspace/protein/v1/data/tev/embeddings_esm2_15b_meanpool.npy",
            },
            {
                "key":   "tev_nsite",
                "mode":  "nsite",
                "sites": [145, 147, 166, 169],
                "out":   "/workspace/protein/v1/data/tev/embeddings_esm2_15b_4site.npy",
            },
        ],
    },
    {
        "name":    "t7",
        "csv":     "/workspace/protein/v1/data/t7/t7_fitness.csv",
        "hf":      None,
        "seq_col": "protein",
        "n":       6725,
        "batch":   None,  # auto
        "outputs": [
            {
                "key":   "t7_meanpool",
                "mode":  "meanpool",
                "sites": None,
                "out":   "/workspace/protein/v1/data/t7/embeddings_esm2_15b_meanpool.npy",
            },
            {
                "key":   "t7_nsite",
                "mode":  "nsite",
                "sites": [747, 755, 757],
                "out":   "/workspace/protein/v1/data/t7/embeddings_esm2_15b_3site.npy",
            },
        ],
    },
]

# ── Setup ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--rank",  type=int, default=0)
parser.add_argument("--world", type=int, default=4)
parser.add_argument("--only",  nargs="+", default=None,
                    help="Protein names to run, e.g. --only tev t7")
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
model = EsmModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    use_safetensors=True,
    attn_implementation="flash_attention_2",
)
model.to(DEVICE).eval()
print(f"Loaded in {time.time()-t0:.1f}s", flush=True)

MODEL_VRAM_GB = 30.0  # ESM2-15B fp16

def auto_batch(seq_len: int) -> int:
    """Fill available VRAM with as large a batch as safely fits."""
    if not torch.cuda.is_available():
        return 4
    total_gb  = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9
    free_gb   = total_gb - MODEL_VRAM_GB
    # Rough activation budget: seq_len * EMB_DIM * 4 bytes * ~12 (layers + attn) per sequence
    bytes_per_seq = seq_len * EMB_DIM * 4 * 12
    batch = max(1, int(free_gb * 1e9 * 0.8 / bytes_per_seq))  # 80% of free VRAM
    return min(batch, 512)  # cap at 512


def embed_protein(sequences: list, protein: dict, rank: int) -> dict[str, np.ndarray]:
    """One forward pass per batch — extract all outputs from the same hidden states."""
    n      = len(sequences)
    seq_len = len(sequences[0])
    batch  = protein["batch"] or auto_batch(seq_len)
    print(f"  batch={batch} (seq_len={seq_len})", flush=True)
    outs   = protein["outputs"]

    results = {}
    for o in outs:
        if o["mode"] == "meanpool":
            results[o["key"]] = np.zeros((n, EMB_DIM), dtype=np.float16)
        else:
            results[o["key"]] = np.zeros((n, len(o["sites"]) * EMB_DIM), dtype=np.float16)

    t0 = time.time()
    for start in range(0, n, batch):
        batch_seqs = sequences[start : start + batch]
        enc = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=False)
        enc = {k: v.to(DEVICE) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc)

        hidden   = out.last_hidden_state          # (B, L, 5120) fp16
        bsz      = len(batch_seqs)
        mask     = enc["attention_mask"].float()
        seq_lens = mask.sum(dim=1)

        for o in outs:
            key = o["key"]
            if o["mode"] == "meanpool":
                m = mask.clone()
                m[:, 0] = 0.0
                for b in range(bsz):
                    m[b, int(seq_lens[b]) - 1] = 0.0
                m = m.unsqueeze(-1)
                pooled = (hidden * m).sum(dim=1) / m.sum(dim=1)
                results[key][start : start + bsz] = pooled.cpu().numpy().astype(np.float16)
            else:
                h = hidden.float().cpu().numpy()
                for j in range(bsz):
                    results[key][start + j] = np.concatenate([h[j, s + 1] for s in o["sites"]])

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

    # Filter outputs to only those not yet chunked by this rank
    pending = [o for o in protein["outputs"]
               if not os.path.exists(f"{OUT_DIR}/{o['key']}_chunk{RANK:02d}.npz")]
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

    # Run with only pending outputs
    protein_pending = {**protein, "outputs": pending}
    t_start = time.time()
    results = embed_protein(my_seqs, protein_pending, RANK)
    print(f"  Done in {time.time()-t_start:.0f}s", flush=True)

    for o in pending:
        chunk_path = f"{OUT_DIR}/{o['key']}_chunk{RANK:02d}.npz"
        np.savez(chunk_path, embeddings=results[o["key"]], lo=lo, hi=hi)
        print(f"  Saved: {chunk_path}", flush=True)

print(f"\n[rank {RANK}] All done!", flush=True)
