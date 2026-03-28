"""
Colab script: compute ESM3-sm embeddings for GB1 and TrpB.

Requirements:
  - Colab Pro+ with A100 GPU (40GB VRAM)
  - Google Drive mounted at /content/drive/MyDrive/protein/v1/

Model: ESM3-sm-open-v1 (1.4B params, hidden_dim=1536)
  - Loads in bfloat16: ~2.8 GB VRAM — fits easily on A100 40GB
  - No quantization required; bfloat16 is sufficient

Output dimensions (float32 npz):
  GB1:  (149361, 6144)  — 4 sites × 1536
  TrpB: (160000, 6144)  — 4 sites × 1536
"""

# ── Cell 1: Install & mount ───────────────────────────────────────────────────
# !pip install -q esm datasets
# from google.colab import drive
# drive.mount('/content/drive')

# ── Cell 2: Imports ───────────────────────────────────────────────────────────
import os
import time
import numpy as np
import torch
from datasets import load_dataset

DRIVE_ROOT = "/content/drive/MyDrive/protein/v1"   # adjust if needed
DATA_DIR   = os.path.join(DRIVE_ROOT, "data")

# ── Cell 3: Load ESM3-sm ──────────────────────────────────────────────────────
# Patch: newer huggingface_hub removed reset_sessions, which esm still imports.
import huggingface_hub.utils as _hf_utils
if not hasattr(_hf_utils, "reset_sessions"):
    _hf_utils.reset_sessions = lambda: None

from esm.models.esm3 import ESM3
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

HIDDEN_DIM = 1536
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading ESM3-sm-open-v1 in bfloat16 (~3 GB VRAM)...")
t0 = time.time()
model = ESM3.from_pretrained("esm3_sm_open_v1").to(device=device, dtype=torch.bfloat16)
model.eval()
tokenizer = EsmSequenceTokenizer()
print(f"Loaded in {time.time()-t0:.0f}s on {device}")

# ── Cell 4: Embedding function ────────────────────────────────────────────────
def embed_4site(sequences: list, sites: list, batch_size: int = 64) -> np.ndarray:
    """
    Extract and concatenate per-residue ESM3 embeddings at 4 mutation sites.

    sites: 0-indexed residue positions.
    ESM3 tokens: [BOS] res1 res2 ... resL [EOS]
    So residue i (0-indexed) is at token position i+1.
    """
    n   = len(sequences)
    dim = len(sites) * HIDDEN_DIM
    out = np.zeros((n, dim), dtype=np.float32)

    for start in range(0, n, batch_size):
        batch = [str(s).replace("*", "").strip() for s in sequences[start : start + batch_size]]

        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        input_ids = enc["input_ids"].to(device)

        with torch.no_grad():
            output = model(sequence_tokens=input_ids, return_embeddings=True)

        reps = output.embeddings.float().cpu().numpy()  # (B, L+2, 1536)

        for j in range(len(batch)):
            out[start + j] = np.concatenate([reps[j, s + 1] for s in sites])

        if start % 5000 == 0:
            pct = 100 * min(start + batch_size, n) / n
            print(f"  {min(start + batch_size, n):>7,}/{n:,}  ({pct:.1f}%)")

    return out


# ── Cell 5: GB1 ───────────────────────────────────────────────────────────────
print("\n=== GB1 ===")
gb1       = load_dataset("SaProtHub/Dataset-GB1-fitness", split="train")
seqs_gb1  = [s.replace("*", "").strip() for s in gb1["protein"]]
sites_gb1 = [38, 39, 40, 53]   # V39, D40, G41, V54 (0-indexed)

t0 = time.time()
emb_gb1 = embed_4site(seqs_gb1, sites_gb1, batch_size=64)
print(f"Done in {time.time()-t0:.0f}s  shape={emb_gb1.shape}")

out_gb1 = os.path.join(DATA_DIR, "gb1/embeddings_esm3_sm_4site.npz")
os.makedirs(os.path.dirname(out_gb1), exist_ok=True)
np.savez_compressed(out_gb1, embeddings=emb_gb1)
print(f"Saved: {out_gb1}  ({os.path.getsize(out_gb1)/1e6:.0f} MB)")


# ── Cell 6: TrpB ──────────────────────────────────────────────────────────────
print("\n=== TrpB ===")
trpb       = load_dataset("SaProtHub/Dataset-TrpB_fitness_landsacpe", split="train")
seqs_trpb  = [s.replace("*", "").strip() for s in trpb["protein"]]
sites_trpb = [182, 183, 226, 227]

t0 = time.time()
emb_trpb = embed_4site(seqs_trpb, sites_trpb, batch_size=64)
print(f"Done in {time.time()-t0:.0f}s  shape={emb_trpb.shape}")

out_trpb = os.path.join(DATA_DIR, "trpb/embeddings_esm3_sm_4site.npz")
os.makedirs(os.path.dirname(out_trpb), exist_ok=True)
np.savez_compressed(out_trpb, embeddings=emb_trpb)
print(f"Saved: {out_trpb}  ({os.path.getsize(out_trpb)/1e6:.0f} MB)")

print("\nAll done. Transfer npz files from Drive to your Mac via rsync.")
