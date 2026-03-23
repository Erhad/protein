# ============================================================
# ESM2-650M Embeddings — GB1 & TrpB
# Run this on Google Colab (GPU runtime)
# Output: gb1_embeddings_esm2_650m_4site.npz
#         trpb_embeddings_esm2_650m_4site.npz
# ============================================================

# ── Cell 1: Check GPU ────────────────────────────────────────
import subprocess
result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                        '--format=csv,noheader'], capture_output=True, text=True)
print(result.stdout.strip())

import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── Cell 2: Install dependencies ─────────────────────────────
subprocess.run(['pip', 'install', 'fair-esm', 'datasets', '-q'], check=True)

# ── Cell 3: Download datasets from HuggingFace ───────────────
import csv, os
from datasets import load_dataset

print("Downloading GB1...")
gb1 = load_dataset("SaProtHub/Dataset-GB1-fitness", split="train")
print(f"  GB1: {len(gb1):,} sequences, cols: {gb1.column_names}")

print("Downloading TrpB...")
trpb = load_dataset("SaProtHub/Dataset-TrpB_fitness_landsacpe", split="train")
print(f"  TrpB: {len(trpb):,} sequences, cols: {trpb.column_names}")

gb1_seqs  = gb1["protein"]
trpb_seqs = trpb["protein"]

# ── Cell 4: Load ESM2-650M ────────────────────────────────────
import esm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading ESM2-650M on {device}...")
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.eval().to(device)
batch_converter = alphabet.get_batch_converter()
print("Model loaded.")

# ── Cell 5: Embedding function ────────────────────────────────
import numpy as np, time

def get_batch_size(seq_len: int) -> int:
    """Scale batch size with sequence length to fit in VRAM."""
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    if vram_gb >= 35:      # A100 40GB / H100
        return 512 if seq_len < 100 else 256
    elif vram_gb >= 14:    # V100 16GB / T4
        return 512 if seq_len < 100 else 64
    else:
        return 32

def compute_embeddings_4site(sequences: list, sites: list, label: str) -> np.ndarray:
    n = len(sequences)
    out = np.zeros((n, len(sites) * 1280), dtype=np.float32)
    seq_len = len(sequences[0].replace("*", "").strip())
    batch_size = get_batch_size(seq_len)
    print(f"  {label}: {n:,} seqs, len={seq_len}, batch={batch_size}")

    t0 = time.time()
    for start in range(0, n, batch_size):
        batch = sequences[start : start + batch_size]
        data = [(str(i), s.replace("*", "").strip()) for i, s in enumerate(batch)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)

        with torch.no_grad():
            result = model(tokens, repr_layers=[33], return_contacts=False)

        reps = result["representations"][33].float().cpu().numpy()
        for j, rep in enumerate(reps):
            out[start + j] = np.concatenate([rep[s + 1] for s in sites])

        if (start // batch_size) % 50 == 0:
            pct = 100 * (start + len(batch)) / n
            elapsed = time.time() - t0
            rate = (start + len(batch)) / elapsed
            eta = (n - start - len(batch)) / rate if rate > 0 else 0
            print(f"    {start+len(batch):>7,}/{n:,} ({pct:.1f}%)  "
                  f"{rate:.0f} seq/s  ETA {eta/60:.1f} min")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed/60:.1f} min  ({n/elapsed:.0f} seq/s)")
    return out

# ── Cell 6: Compute GB1 embeddings ───────────────────────────
# V39, D40, G41, V54 → 0-indexed positions 38, 39, 40, 53
GB1_SITES = [38, 39, 40, 53]
gb1_emb = compute_embeddings_4site(gb1_seqs, GB1_SITES, "GB1")
assert gb1_emb.shape == (len(gb1_seqs), 5120), f"Shape error: {gb1_emb.shape}"
np.savez_compressed("gb1_embeddings_esm2_650m_4site.npz", embeddings=gb1_emb)
print(f"Saved GB1: {gb1_emb.shape}  {os.path.getsize('gb1_embeddings_esm2_650m_4site.npz')/1e6:.0f} MB")

# ── Cell 7: Compute TrpB embeddings ──────────────────────────
# Variable sites at 0-indexed positions 182, 183, 226, 227
TRPB_SITES = [182, 183, 226, 227]
trpb_emb = compute_embeddings_4site(trpb_seqs, TRPB_SITES, "TrpB")
assert trpb_emb.shape == (len(trpb_seqs), 5120), f"Shape error: {trpb_emb.shape}"
np.savez_compressed("trpb_embeddings_esm2_650m_4site.npz", embeddings=trpb_emb)
print(f"Saved TrpB: {trpb_emb.shape}  {os.path.getsize('trpb_embeddings_esm2_650m_4site.npz')/1e6:.0f} MB")

# ── Cell 8: Download files ────────────────────────────────────
try:
    from google.colab import files
    print("Downloading gb1_embeddings_esm2_650m_4site.npz ...")
    files.download("gb1_embeddings_esm2_650m_4site.npz")
    print("Downloading trpb_embeddings_esm2_650m_4site.npz ...")
    files.download("trpb_embeddings_esm2_650m_4site.npz")
except ImportError:
    print("Not in Colab — files saved locally.")

print("\nAll done.")
