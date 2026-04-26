"""
Compute HDBSCAN cluster labels for ESM2-15B sitemean embeddings.

Matches parameters used for existing cluster label files (mcs500):
  - ZS filter: keep top 25% by ev_score
  - PCA → 50 dims
  - UMAP → 15 dims
  - HDBSCAN min_cluster_size=500

Usage:
    python precompute/compute_cluster_labels.py
"""

import os, sys, time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import umap
import hdbscan

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ZS_COL          = "ev_score"
ZS_KEEP_PCT     = 25        # keep top 25% by ZS score (matching existing labels)
PCA_DIMS        = 50
UMAP_DIMS       = 15
MIN_CLUSTER_SIZE = 500

PROTEINS = {
    "tev": {
        "fitness_csv": "data/tev/tev_fitness.csv",
        "emb":         "data/tev/embeddings_esmc600m_4site_mean.npy",
        "zs_csv":      "data/li2024/results/zs_comb/all/TEV.csv",
        "sites":       [145, 147, 166, 169],
        "out":         "data/tev/cluster_labels_hdbscan_mcs500.npy",
    },
    "t7": {
        "fitness_csv": "data/t7/t7_fitness.csv",
        "emb":         "data/t7/embeddings_esmc600m_3site_mean.npy",
        "zs_csv":      "data/li2024/results/zs_comb/all/T7.csv",
        "sites":       [747, 755, 757],
        "out":         "data/t7/cluster_labels_hdbscan_mcs500.npy",
    },
}


def compute_labels(protein: str, cfg: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  {protein.upper()}")
    print(f"{'='*60}")

    emb_path = os.path.join(ROOT, cfg["emb"])
    fit_path  = os.path.join(ROOT, cfg["fitness_csv"])
    zs_path   = os.path.join(ROOT, cfg["zs_csv"])
    out_path  = os.path.join(ROOT, cfg["out"])

    # Load
    print("Loading embeddings...", end=" ", flush=True)
    X = np.load(emb_path).astype(np.float32)
    df_fit = pd.read_csv(fit_path)
    df_zs  = pd.read_csv(zs_path)
    n_total = len(X)
    print(f"{X.shape}")

    # Build sequence → index map and get ZS scores
    seqs = df_fit["protein"].values
    sites = cfg["sites"]
    our_aas = np.array(["".join(seq[s] for s in sites) for seq in seqs])
    zs_map = dict(zip(df_zs["AAs"], df_zs[ZS_COL]))
    zs_scores = np.array([zs_map.get(aa, np.nan) for aa in our_aas])

    # ZS filter: mark bottom (100-ZS_KEEP_PCT)% as -2
    labels = np.full(n_total, -2, dtype=np.int32)
    valid_mask = np.isfinite(zs_scores)
    threshold  = np.nanpercentile(zs_scores, 100 - ZS_KEEP_PCT)
    keep_mask  = valid_mask & (zs_scores >= threshold)
    keep_idx   = np.where(keep_mask)[0]
    print(f"ZS filter: keeping top {ZS_KEEP_PCT}% → {keep_mask.sum()} / {n_total} variants")

    X_keep = X[keep_idx]

    # PCA — keep float32 throughout to avoid the 2× memory spike from float64
    print(f"PCA {X_keep.shape[1]}→{PCA_DIMS}...", end=" ", flush=True)
    t0 = time.time()
    X_pca = PCA(n_components=PCA_DIMS, random_state=42).fit_transform(X_keep)
    X_pca = X_pca.astype(np.float32)
    print(f"{time.time()-t0:.1f}s")

    # UMAP
    print(f"UMAP {PCA_DIMS}→{UMAP_DIMS}...", end=" ", flush=True)
    t0 = time.time()
    X_umap = umap.UMAP(
        n_components=UMAP_DIMS, n_neighbors=30, min_dist=0.0,
        metric="euclidean", random_state=42, low_memory=True,
    ).fit_transform(X_pca)
    print(f"{time.time()-t0:.1f}s")

    # HDBSCAN
    print(f"HDBSCAN mcs={MIN_CLUSTER_SIZE}...", end=" ", flush=True)
    t0 = time.time()
    cl = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE, min_samples=10,
        cluster_selection_method="eom", core_dist_n_jobs=-1,
    ).fit(X_umap)
    print(f"{time.time()-t0:.1f}s")

    cluster_ids = cl.labels_   # -1=noise, 0+=cluster
    n_clusters  = len(set(cluster_ids[cluster_ids >= 0]))
    n_noise     = (cluster_ids == -1).sum()
    print(f"Result: {n_clusters} clusters, {n_noise} noise points")

    # Write back into full-size label array (-2=filtered, -1=noise, 0+=cluster)
    labels[keep_idx] = cluster_ids
    np.save(out_path, labels)
    print(f"Saved → {cfg['out']}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--proteins", nargs="+", default=list(PROTEINS.keys()))
    args = parser.parse_args()

    for protein in args.proteins:
        compute_labels(protein, PROTEINS[protein])

    print("\nDone.")
