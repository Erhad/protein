"""
Multi-GPU ESM2-15B embedding script for RunPod / Vast.ai.

Each process handles one GPU + one data chunk. Run 4 processes in parallel:

    python runpod_embeddings_15B.py --rank 0 --world 4
    python runpod_embeddings_15B.py --rank 1 --world 4
    python runpod_embeddings_15B.py --rank 2 --world 4
    python runpod_embeddings_15B.py --rank 3 --world 4

Or use the launcher (recommended):

    bash launch_embeddings.sh

Output: /workspace/embeddings/chunk_<rank>.npz per process
Final merge: python merge_embeddings.py

Requirements (install once):
    pip install "transformers<4.48.2" "huggingface_hub<1.0" accelerate datasets
"""

import argparse
import os
import time
import numpy as np
import torch
from datasets import load_dataset
from transformers import EsmModel, EsmTokenizer

MODEL_ID = "facebook/esm2_t48_15B_UR50D"
OUT_DIR  = "/workspace/embeddings"

# ── Data config ───────────────────────────────────────────────────────────────
TASKS = [
    {
        "name":    "gb1",
        "dataset": "SaProtHub/Dataset-GB1-fitness",
        "seq_col": "protein",
        "sites":   [38, 39, 40, 53],   # V39 D40 G41 V54 (0-indexed)
        "n":       149361,
    },
    {
        "name":    "trpb",
        "dataset": "SaProtHub/Dataset-TrpB_fitness_landsacpe",
        "seq_col": "protein",
        "sites":   [182, 183, 226, 227],
        "n":       160000,
    },
]


def embed_chunk(sequences, sites, model, tokenizer, device, batch_size=32):
    n   = len(sequences)
    dim = len(sites) * 5120
    out = np.zeros((n, dim), dtype=np.float32)

    for start in range(0, n, batch_size):
        batch = [str(s).replace("*", "").strip()
                 for s in sequences[start: start + batch_size]]

        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=False
        ).to(device)

        with torch.no_grad():
            reps = model(**inputs).last_hidden_state.float().cpu().numpy()

        for j, rep in enumerate(reps):
            out[start + j] = np.concatenate([rep[s + 1] for s in sites])

        if start % 2000 == 0:
            pct = 100 * min(start + batch_size, n) / n
            print(f"  [{start + batch_size:>7,}/{n:,}] {pct:.1f}%", flush=True)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank",  type=int, required=True, help="This process index (0-based)")
    parser.add_argument("--world", type=int, required=True, help="Total number of processes")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Assign GPU
    n_gpus = torch.cuda.device_count()
    gpu_id = args.rank % n_gpus
    device = torch.device(f"cuda:{gpu_id}")
    print(f"Rank {args.rank}/{args.world} → GPU {gpu_id} ({torch.cuda.get_device_name(gpu_id)})", flush=True)

    # Load model onto this GPU (fits on single 80GB A100 in fp16)
    print(f"Loading {MODEL_ID} in fp16...", flush=True)
    t0 = time.time()
    tokenizer = EsmTokenizer.from_pretrained(MODEL_ID)
    model = EsmModel.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map={"": gpu_id},   # pin entire model to this GPU
    )
    model.eval()
    print(f"Loaded in {time.time()-t0:.0f}s", flush=True)

    # Split all tasks across ranks
    # Flatten: [(task, start, end), ...] then assign by rank
    all_chunks = []
    for task in TASKS:
        ds = load_dataset(task["dataset"], split="train")
        seqs = [str(s).replace("*", "").strip() for s in ds[task["seq_col"]]]
        chunk = len(seqs) // args.world
        for r in range(args.world):
            lo = r * chunk
            hi = len(seqs) if r == args.world - 1 else (r + 1) * chunk
            all_chunks.append((task["name"], task["sites"], seqs, lo, hi))

    my_chunks = [c for i, c in enumerate(all_chunks) if i % args.world == args.rank]

    for (name, sites, seqs, lo, hi) in my_chunks:
        out_path = os.path.join(OUT_DIR, f"{name}_chunk{args.rank:02d}.npz")
        if os.path.exists(out_path):
            print(f"Already done: {out_path}, skipping.", flush=True)
            continue

        print(f"\n=== {name} [{lo}:{hi}] ===", flush=True)
        t0 = time.time()
        emb = embed_chunk(seqs[lo:hi], sites, model, tokenizer, device, args.batch_size)
        print(f"Done in {time.time()-t0:.0f}s  shape={emb.shape}", flush=True)

        np.savez_compressed(out_path, embeddings=emb, lo=lo, hi=hi)
        print(f"Saved: {out_path}  ({os.path.getsize(out_path)/1e6:.0f} MB)", flush=True)

    print(f"\nRank {args.rank} finished.", flush=True)


if __name__ == "__main__":
    main()
