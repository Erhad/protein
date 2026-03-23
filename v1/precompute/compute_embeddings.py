"""
Compute and cache ESM2-650M embeddings for GFP, GB1, and TrpB.

Usage:
    python compute_embeddings.py gfp
    python compute_embeddings.py gb1
    python compute_embeddings.py trpb
    python compute_embeddings.py gfp gb1 trpb   # all

Outputs (float32 .npz, keyed by 'embeddings'):
    data/gfp/embeddings_esm2_650m_meanpool.npz   shape (58417, 1280)
    data/gb1/embeddings_esm2_650m_4site.npz      shape (149361, 5120)
    data/trpb/embeddings_esm2_650m_4site.npz     shape (160000, 5120)
"""

import os
import sys
import time

import esm
import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # v1/

DATASETS = {
    "gfp": {
        "csv": "data/gfp/sequences_scores.csv",
        "seq_col": "mutated_protein",
        "out": "data/gfp/embeddings_esm2_650m_meanpool.npz",
        "mode": "meanpool",
        "sites": None,
        "expected_dim": 1280,
    },
    "gb1": {
        "csv": "data/gb1/gb1_fitness.csv",
        "seq_col": "protein",
        "out": "data/gb1/embeddings_esm2_650m_4site.npz",
        "mode": "4site",
        "sites": [38, 39, 40, 53],   # V39, D40, G41, V54 (0-indexed)
        "expected_dim": 5120,
    },
    "trpb": {
        "csv": "data/trpb/trpb_fitness.csv",
        "seq_col": "protein",
        "out": "data/trpb/embeddings_esm2_650m_4site.npz",
        "mode": "4site",
        "sites": [182, 183, 226, 227],
        "expected_dim": 5120,
    },
}

BATCH_SIZE = 32
ESM_MODEL = "esm2_t33_650M_UR50D"
REPR_LAYER = 33   # last layer of esm2_t33


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(device: torch.device):
    print(f"Loading {ESM_MODEL} …")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter


def embed_sequences(
    sequences: list[str],
    model,
    batch_converter,
    device: torch.device,
    mode: str,
    sites: list[int] | None,
) -> np.ndarray:
    n = len(sequences)
    if mode == "meanpool":
        out = np.zeros((n, 1280), dtype=np.float32)
    else:
        out = np.zeros((n, len(sites) * 1280), dtype=np.float32)

    for batch_start in range(0, n, BATCH_SIZE):
        batch_seqs = sequences[batch_start : batch_start + BATCH_SIZE]
        data = [(str(i), str(seq).replace("*", "").strip()) for i, seq in enumerate(batch_seqs)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)

        with torch.no_grad():
            result = model(tokens, repr_layers=[REPR_LAYER], return_contacts=False)

        # result["representations"][REPR_LAYER]: (B, L+2, 1280)  (+2 for BOS/EOS)
        reps = result["representations"][REPR_LAYER].float().cpu().numpy()

        for j, rep in enumerate(reps):
            # rep shape: (seq_len + 2, 1280)  — index 0 is BOS, 1..L are residues, L+1 is EOS
            idx = batch_start + j
            if mode == "meanpool":
                seq_len = len(str(batch_seqs[j]).replace("*", "").strip())
                out[idx] = rep[1 : seq_len + 1].mean(axis=0)
            else:
                # Extract embeddings at the 4 targeted sites (1-indexed in token space = site+1)
                site_vecs = [rep[s + 1] for s in sites]
                out[idx] = np.concatenate(site_vecs)

        if (batch_start // BATCH_SIZE) % 20 == 0:
            pct = 100 * (batch_start + len(batch_seqs)) / n
            print(f"  {batch_start + len(batch_seqs):>7,}/{n:,}  ({pct:.1f}%)")

    return out


def run(name: str, device: torch.device, model, batch_converter):
    cfg = DATASETS[name]
    csv_path = os.path.join(ROOT, cfg["csv"])
    out_path = os.path.join(ROOT, cfg["out"])

    print(f"\n{'='*55}")
    print(f"Dataset : {name.upper()}")
    print(f"Mode    : {cfg['mode']}  (expected dim: {cfg['expected_dim']})")
    print(f"CSV     : {csv_path}")
    print(f"Out     : {out_path}")

    df = pd.read_csv(csv_path)
    sequences = df[cfg["seq_col"]].tolist()
    print(f"Sequences: {len(sequences):,}")

    t0 = time.time()
    embeddings = embed_sequences(
        sequences, model, batch_converter, device,
        mode=cfg["mode"], sites=cfg["sites"],
    )
    elapsed = time.time() - t0

    assert embeddings.shape == (len(sequences), cfg["expected_dim"]), (
        f"Shape mismatch: {embeddings.shape} vs expected ({len(sequences)}, {cfg['expected_dim']})"
    )

    np.savez_compressed(out_path, embeddings=embeddings)
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f"  Saved : {embeddings.shape}  {size_mb:.1f} MB  ({elapsed:.0f}s)")


def main():
    targets = sys.argv[1:] if sys.argv[1:] else list(DATASETS.keys())
    unknown = [t for t in targets if t not in DATASETS]
    if unknown:
        print(f"Unknown: {unknown}. Choose from: {list(DATASETS.keys())}")
        sys.exit(1)

    device = get_device()
    print(f"Device: {device}")
    model, batch_converter = load_model(device)

    for name in targets:
        run(name, device, model, batch_converter)

    print("\nAll embeddings computed.")


if __name__ == "__main__":
    main()
