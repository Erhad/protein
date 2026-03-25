"""
Colab script: compute ESM2-15B embeddings for GB1 and TrpB.

Requirements:
  - Colab Pro+ with A100 GPU (40GB VRAM)
  - Google Drive mounted at /content/drive/MyDrive/protein/v1/

Uses HuggingFace transformers to load directly in fp16 (~30GB),
avoiding the fair-esm approach that materializes fp32 first (~80GB crash).

Output dimensions (float32 npz):
  GB1:  (149361, 20480)  — 4 sites × 5120
  TrpB: (160000, 20480)  — 4 sites × 5120
"""

# ── Cell 1: Install & mount ───────────────────────────────────────────────────
# !pip install -q transformers accelerate
# from google.colab import drive
# drive.mount('/content/drive')

# ── Cell 2: Imports ───────────────────────────────────────────────────────────
import os
import time
import numpy as np
import pandas as pd
import torch
from transformers import EsmModel, EsmTokenizer

DRIVE_ROOT = "/content/drive/MyDrive/protein/v1"   # adjust if needed
DATA_DIR   = os.path.join(DRIVE_ROOT, "data")
MODEL_ID   = "facebook/esm2_t48_15B_UR50D"

# ── Cell 3: Load model ────────────────────────────────────────────────────────
# Loads directly in fp16 — never materializes fp32, peak RAM ~30GB not 80GB.
print(f"Loading {MODEL_ID} in fp16 (downloads ~30GB first time)...")
t0 = time.time()

tokenizer = EsmTokenizer.from_pretrained(MODEL_ID)
model = EsmModel.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,   # load straight to fp16
    device_map="auto",           # auto-place on GPU; spills to CPU if needed
)
model.eval()
print(f"Loaded in {time.time()-t0:.0f}s")

# ── Cell 4: Embedding function ────────────────────────────────────────────────
def embed_4site(sequences: list, sites: list, batch_size: int = 8) -> np.ndarray:
    """
    Extract and concatenate per-residue ESM2 embeddings at 4 mutation sites.

    sites: 0-indexed residue positions in the sequence.
    HuggingFace tokens: [CLS] res1 res2 ... resL [EOS]
    So residue i (0-indexed) is at token position i+1.
    """
    n   = len(sequences)
    dim = len(sites) * 5120
    out = np.zeros((n, dim), dtype=np.float32)

    for start in range(0, n, batch_size):
        batch = [str(s).replace("*", "").strip() for s in sequences[start : start + batch_size]]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        # last_hidden_state: (B, L+2, 5120) — fp16 on GPU
        reps = outputs.last_hidden_state.float().cpu().numpy()

        for j, rep in enumerate(reps):
            # token 0 = CLS, tokens 1..L = residues, token L+1 = EOS
            site_vecs = [rep[s + 1] for s in sites]
            out[start + j] = np.concatenate(site_vecs)

        if start % 5000 == 0:
            pct = 100 * min(start + batch_size, n) / n
            print(f"  {min(start + batch_size, n):>7,}/{n:,}  ({pct:.1f}%)")

    return out


# ── Cell 5: GB1 ───────────────────────────────────────────────────────────────
print("\n=== GB1 ===")
df_gb1    = pd.read_csv(os.path.join(DATA_DIR, "gb1/gb1_fitness.csv"))
seqs_gb1  = df_gb1["protein"].tolist()
sites_gb1 = [38, 39, 40, 53]   # V39, D40, G41, V54 (0-indexed)

t0 = time.time()
emb_gb1 = embed_4site(seqs_gb1, sites_gb1, batch_size=8)
print(f"Done in {time.time()-t0:.0f}s  shape={emb_gb1.shape}")

out_gb1 = os.path.join(DATA_DIR, "gb1/embeddings_esm2_15b_4site.npz")
np.savez_compressed(out_gb1, embeddings=emb_gb1)
print(f"Saved: {out_gb1}  ({os.path.getsize(out_gb1)/1e6:.0f} MB)")


# ── Cell 6: TrpB ──────────────────────────────────────────────────────────────
print("\n=== TrpB ===")
df_trpb    = pd.read_csv(os.path.join(DATA_DIR, "trpb/trpb_fitness.csv"))
seqs_trpb  = df_trpb["protein"].tolist()
sites_trpb = [182, 183, 226, 227]

t0 = time.time()
emb_trpb = embed_4site(seqs_trpb, sites_trpb, batch_size=8)
print(f"Done in {time.time()-t0:.0f}s  shape={emb_trpb.shape}")

out_trpb = os.path.join(DATA_DIR, "trpb/embeddings_esm2_15b_4site.npz")
np.savez_compressed(out_trpb, embeddings=emb_trpb)
print(f"Saved: {out_trpb}  ({os.path.getsize(out_trpb)/1e6:.0f} MB)")

print("\nAll done. Transfer npz files from Drive to your Mac via rsync.")
