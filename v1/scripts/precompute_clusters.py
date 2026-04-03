"""
Precompute HDBSCAN cluster labels for a landscape and save to disk.

Pipeline:
  1. ZS filter: keep top 25% by ev-esm-esmif_score
  2. PCA 1152→50
  3. UMAP 50→15  (n_neighbors=30, min_dist=0, McInnes clustering params)
  4. HDBSCAN mcs=500

Output: data/{landscape}/cluster_labels_hdbscan_mcs{mcs}.npy
  Array of length N:
    -2  = filtered out (bottom 75% ZS)
    -1  = HDBSCAN noise
    0.. = cluster id

Usage:
    python scripts/precompute_clusters.py --landscape gb1
"""
import argparse, gc, os, sys, time
import numpy as np
import pandas as pd
import torch
import umap
import hdbscan

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

LANDSCAPE_CFG = {
    "gb1": {
        "fitness_csv": "data/gb1/gb1_fitness.csv",
        "emb_mean":    "data/gb1/embeddings_esmc600m_4site_mean.npy",
        "zs_csv":      "data/li2024/results/zs_comb/all/GB1.csv",
        "sites":       [38, 39, 40, 53],
        "out":         "data/gb1/cluster_labels_hdbscan_mcs{mcs}.npy",
    },
    "gb1_esm2_15b": {
        "fitness_csv": "data/gb1/gb1_fitness.csv",
        "emb_mean":    "data/gb1/embeddings_esm2_15b_meanpool.npy",
        "zs_csv":      "data/li2024/results/zs_comb/all/GB1.csv",
        "sites":       [38, 39, 40, 53],
        "out":         "data/gb1/cluster_labels_esm2_15b_hdbscan_mcs{mcs}.npy",
    },
    "trpb": {
        "fitness_csv": "data/trpb/trpb_fitness.csv",
        "emb_mean":    None,
        "emb_4site":   "data/trpb/embeddings_esmc600m_4site.npy",
        "zs_csv":      "data/li2024/results/zs_comb/all/TrpB4.csv",
        "sites":       [182, 183, 226, 227],
        "out":         "data/trpb/cluster_labels_hdbscan_mcs{mcs}.npy",
    },
}

ZS_COL       = "ev-esm-esmif_score"
ZS_KEEP_PCT  = 25
PCA_DIMS     = 50
UMAP_DIMS    = 15
UMAP_N_NBRS  = 30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--landscape", required=True, choices=list(LANDSCAPE_CFG.keys()))
    parser.add_argument("--mcs", type=int, default=500)
    args = parser.parse_args()

    cfg = LANDSCAPE_CFG[args.landscape]
    out_path = os.path.join(ROOT, cfg["out"].format(mcs=args.mcs))

    print(f"Landscape: {args.landscape}  mcs={args.mcs}", flush=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    df = pd.read_csv(os.path.join(ROOT, cfg["fitness_csv"]))
    N  = len(df)

    if cfg.get("emb_mean"):
        emb = np.load(os.path.join(ROOT, cfg["emb_mean"]))
    else:
        emb_4s = np.load(os.path.join(ROOT, cfg["emb_4site"]))
        n, d4  = emb_4s.shape
        emb    = emb_4s.reshape(n, 4, d4//4).mean(axis=1)

    zs_df  = pd.read_csv(os.path.join(ROOT, cfg["zs_csv"]))
    sites  = cfg["sites"]
    aas    = df["protein"].apply(lambda s: ''.join(s[i] for i in sites))
    zs     = aas.map(dict(zip(zs_df["AAs"], zs_df[ZS_COL]))).to_numpy(dtype=np.float64, na_value=np.nan)
    zs     = np.where(np.isnan(zs), -np.inf, zs)

    print(f"Loaded {N:,} variants, emb shape {emb.shape}", flush=True)

    # ── ZS filter ─────────────────────────────────────────────────────────────
    thresh   = np.percentile(zs[np.isfinite(zs)], 100 - ZS_KEEP_PCT)
    keep_idx = np.where(np.isfinite(zs) & (zs >= thresh))[0]
    print(f"ZS top {ZS_KEEP_PCT}%: {len(keep_idx):,} variants kept", flush=True)

    sub_emb = emb[keep_idx].astype(np.float32)

    # ── PCA ───────────────────────────────────────────────────────────────────
    t0     = time.time()
    sample = sub_emb[np.random.choice(len(sub_emb), min(10_000, len(sub_emb)), replace=False)]
    center = torch.from_numpy(np.median(sample, axis=0))
    X      = torch.from_numpy(sub_emb)
    X_mps  = (X - center).to('mps')
    Omega  = torch.randn(sub_emb.shape[1], PCA_DIMS + 10, device='mps')
    Y      = (X_mps @ Omega).cpu()
    Q, _   = torch.linalg.qr(Y)
    B      = (Q.to('mps').T @ X_mps).cpu()
    _, S, Vh = torch.linalg.svd(B, full_matrices=False)
    V      = Vh[:PCA_DIMS].T
    pca_out = (X_mps @ V.to('mps')).cpu().numpy()
    del X, X_mps, Omega, Y, Q, B, S, Vh, V
    print(f"PCA {sub_emb.shape[1]}→{PCA_DIMS}d  ({time.time()-t0:.1f}s)", flush=True)

    # ── UMAP ──────────────────────────────────────────────────────────────────
    t0 = time.time()
    reducer = umap.UMAP(n_components=UMAP_DIMS, n_neighbors=UMAP_N_NBRS,
                        min_dist=0.0, metric='euclidean', n_jobs=1, low_memory=True)
    reduced = reducer.fit_transform(pca_out)
    del pca_out, reducer
    gc.collect()
    print(f"UMAP {PCA_DIMS}→{UMAP_DIMS}d  ({time.time()-t0:.1f}s)", flush=True)

    # ── HDBSCAN ───────────────────────────────────────────────────────────────
    t0 = time.time()
    cl  = hdbscan.HDBSCAN(min_cluster_size=args.mcs, min_samples=1,
                           algorithm='boruvka_balltree', core_dist_n_jobs=1)
    sub_labels = cl.fit_predict(reduced)
    del reduced, cl
    gc.collect()
    n_clusters = len(set(sub_labels) - {-1})
    n_noise    = (sub_labels == -1).sum()
    print(f"HDBSCAN: {n_clusters} clusters, {n_noise:,} noise  ({time.time()-t0:.1f}s)", flush=True)

    # ── Map back to full N ────────────────────────────────────────────────────
    labels = np.full(N, -2, dtype=np.int32)   # -2 = ZS-filtered
    labels[keep_idx] = sub_labels              # -1 = noise, 0+ = cluster

    np.save(out_path, labels)
    print(f"Saved → {out_path}", flush=True)

    # Summary
    for c in sorted(set(sub_labels) - {-1}):
        sz = (labels == c).sum()
        print(f"  cluster {c:3d}: {sz:,} variants")


if __name__ == "__main__":
    main()
