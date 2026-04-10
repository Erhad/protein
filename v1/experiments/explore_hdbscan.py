"""
explore_hdbscan.py — Explore HDBSCAN clustering on ZS-filtered ESMc embeddings.

Pipeline:
  1. Filter out bottom ZS_FILTER_PCT% by combined ZS score (likely dead proteins)
  2. PCA to 50 dims (HDBSCAN struggles in 1152-dim space)
  3. HDBSCAN with varying min_cluster_size
  4. Report: n_clusters, n_noise, cluster size distribution,
             which clusters contain the top-fitness variants

Usage:
    python experiments/explore_hdbscan.py --landscapes gb1 trpb
"""
import argparse, gc, os, sys, time
import numpy as np
import pandas as pd
import torch
import umap
import hdbscan
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LANDSCAPE_CFG = {
    "gb1": {
        "fitness_csv": "data/gb1/gb1_fitness.csv",
        "fitness_col": "label",
        "emb_mean":    "data/gb1/embeddings_esmc600m_4site_mean.npy",
        "emb_4site":   "data/gb1/embeddings_esmc600m_4site.npy",
        "zs_csv":      "data/li2024/results/zs_comb/all/GB1.csv",
        "sites":       [38, 39, 40, 53],
    },
    "trpb": {
        "fitness_csv": "data/trpb/trpb_fitness.csv",
        "fitness_col": "label",
        "emb_mean":    None,
        "emb_4site":   "data/trpb/embeddings_esmc600m_4site.npy",
        "zs_csv":      "data/li2024/results/zs_comb/all/TrpB4.csv",
        "sites":       [182, 183, 226, 227],
    },
}

ZS_COL       = "ev-esm-esmif_score"
ZS_FILTER_PCTS = [20, 40, 50]   # filter out bottom X% by ZS (keep top 80/60/50%)
PCA_DIMS     = 50               # intermediate PCA before UMAP (speeds up UMAP on 1152d)
UMAP_DIMS    = 15               # final dims fed to HDBSCAN
MIN_CLUSTER_SIZES = [100, 200, 500, 1000, 2000, 5000]


def load(landscape):
    cfg = LANDSCAPE_CFG[landscape]
    t = time.time(); df = pd.read_csv(os.path.join(ROOT, cfg["fitness_csv"]))
    print(f"fitness_csv: {time.time()-t:.1f}s", flush=True)
    fit = df[cfg["fitness_col"]].values.astype(np.float32)

    if cfg["emb_mean"]:
        t = time.time(); emb = np.load(os.path.join(ROOT, cfg["emb_mean"]))
        print(f"emb_mean: {time.time()-t:.1f}s", flush=True)
    else:
        t = time.time(); emb_4site = np.load(os.path.join(ROOT, cfg["emb_4site"]))
        print(f"emb_4site: {time.time()-t:.1f}s", flush=True)
        n, d4 = emb_4site.shape
        emb = emb_4site.reshape(n, 4, d4//4).mean(axis=1)

    t = time.time(); zs_df = pd.read_csv(os.path.join(ROOT, cfg["zs_csv"]))
    print(f"zs_csv: {time.time()-t:.1f}s", flush=True)
    sites  = cfg["sites"]
    t = time.time()
    aas    = df["protein"].str[sites[0]]
    for s in sites[1:]:
        aas = aas + df["protein"].str[s]
    zs     = aas.map(dict(zip(zs_df["AAs"], zs_df[ZS_COL]))).to_numpy(dtype=np.float64, na_value=np.nan)
    aas    = aas.values
    zs     = np.where(np.isnan(zs), -np.inf, zs)
    print(f"zs_map: {time.time()-t:.1f}s", flush=True)

    return fit, emb, zs, aas


DENSITY_THRESH = 0.30   # flag clusters where X% of members are top-N

def explore(landscape, fit, emb, zs):
    N          = len(fit)
    global_max = fit.max()
    t5         = np.percentile(fit, 95.0)
    t1         = np.percentile(fit, 99.0)
    t01        = np.percentile(fit, 99.9)
    n_top5     = (fit >= t5).sum()
    n_top1     = (fit >= t1).sum()
    n_top01    = (fit >= t01).sum()

    print(f"\n{'='*65}")
    print(f"  {landscape.upper()}  N={N:,}  max={global_max:.4f}")
    print(f"  top-5%={n_top5:,}  top-1%={n_top1:,}  top-0.1%={n_top01:,}")

    for filt_pct in ZS_FILTER_PCTS:
        # Filter pool
        if filt_pct > 0:
            thresh   = np.percentile(zs[np.isfinite(zs)], filt_pct)
            keep_idx = np.where(np.isfinite(zs) & (zs >= thresh))[0]
        else:
            keep_idx = np.where(np.isfinite(zs))[0]

        n_kept = len(keep_idx)
        print(f"\n  --- ZS filter: remove bottom {filt_pct}%  "
              f"→ {n_kept:,} variants remain ({100*n_kept/N:.0f}%) ---")

        # How many top-fitness variants are retained?
        kept_set   = set(keep_idx.tolist())
        top5_kept  = sum(1 for i in np.where(fit >= t5)[0]  if i in kept_set)
        top1_kept  = sum(1 for i in np.where(fit >= t1)[0]  if i in kept_set)
        top01_kept = sum(1 for i in np.where(fit >= t01)[0] if i in kept_set)
        print(f"  top-5% retained:  {top5_kept}/{n_top5}  ({100*top5_kept/n_top5:.0f}%)")
        print(f"  top-1% retained:  {top1_kept}/{n_top1}  ({100*top1_kept/n_top1:.0f}%)")
        print(f"  top-0.1% retained:{top01_kept}/{n_top01} ({100*top01_kept/n_top01:.0f}%)")

        # PCA (MPS) → UMAP pipeline
        sub_emb = emb[keep_idx].astype(np.float32)

        # Step 1: PCA 1152→50 on MPS
        t0 = time.time()
        sample = sub_emb[np.random.choice(len(sub_emb), min(10_000, len(sub_emb)), replace=False)]
        center = torch.from_numpy(np.median(sample, axis=0))
        X     = torch.from_numpy(sub_emb)
        X_mps = (X - center).to('mps')
        Omega = torch.randn(sub_emb.shape[1], PCA_DIMS + 10, device='mps')
        Y     = (X_mps @ Omega).cpu()
        Q, _  = torch.linalg.qr(Y)
        B     = (Q.to('mps').T @ X_mps).cpu()
        _, S, Vh = torch.linalg.svd(B, full_matrices=False)
        V     = Vh[:PCA_DIMS].T
        pca_out = (X_mps @ V.to('mps')).cpu().numpy()
        del X, X_mps, Omega, Y, Q, B, S, Vh, V
        print(f"  PCA {sub_emb.shape[1]}→{PCA_DIMS}d  ({time.time()-t0:.1f}s)", flush=True)

        # Step 2: UMAP 50→15
        t0 = time.time()
        reducer = umap.UMAP(n_components=UMAP_DIMS, n_neighbors=30, min_dist=0.0,
                            metric='euclidean', n_jobs=1, low_memory=True)
        reduced = reducer.fit_transform(pca_out)
        del pca_out, reducer
        print(f"  UMAP {PCA_DIMS}→{UMAP_DIMS}d  ({time.time()-t0:.1f}s)", flush=True)

        # HDBSCAN sweep
        print(f"\n  {'min_cls':>8} {'n_clust':>8} {'n_noise':>8} "
              f"{'noise%':>7} {'med_sz':>7} {'min_sz':>7} {'max_sz':>8} "
              f"{'top1%_cov':>10} {'top0.1%_cov':>12}")
        print(f"  {'-'*85}")

        for mcs in MIN_CLUSTER_SIZES:
            print(f"  HDBSCAN mcs={mcs:,}...", end=" ", flush=True)
            t0  = time.time()
            cl  = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=1,
                                   algorithm='boruvka_balltree',
                                   core_dist_n_jobs=1)
            lbl = cl.fit_predict(reduced)
            dt  = time.time() - t0

            unique_lbl  = [l for l in set(lbl) if l >= 0]
            n_clusters  = len(unique_lbl)
            n_noise     = (lbl == -1).sum()
            noise_pct   = 100 * n_noise / n_kept

            if n_clusters > 0:
                sizes = [int((lbl == l).sum()) for l in unique_lbl]
                med_sz, min_sz, max_sz = int(np.median(sizes)), min(sizes), max(sizes)
            else:
                med_sz = min_sz = max_sz = 0

            # Coverage: fraction of top-X% that land in ANY cluster (not noise)
            in_cluster_global = keep_idx[lbl >= 0]
            in_cluster_set    = set(in_cluster_global.tolist())
            top5_cov  = sum(1 for i in np.where(fit >= t5)[0]  if i in in_cluster_set)
            top1_cov  = sum(1 for i in np.where(fit >= t1)[0]  if i in in_cluster_set)
            top01_cov = sum(1 for i in np.where(fit >= t01)[0] if i in in_cluster_set)

            # Per-cluster density stats
            cluster_stats = []
            for l in unique_lbl:
                idx_c = keep_idx[lbl == l]
                sz    = len(idx_c)
                n5    = int((fit[idx_c] >= t5).sum())
                n1    = int((fit[idx_c] >= t1).sum())
                n01   = int((fit[idx_c] >= t01).sum())
                cluster_stats.append({
                    "cluster": l, "size": sz,
                    "max_fit": float(fit[idx_c].max()),
                    "mean_fit": float(fit[idx_c].mean()),
                    "n_top5": n5, "n_top1": n1, "n_top01": n01,
                    "pct5": 100*n5/sz, "pct1": 100*n1/sz, "pct01": 100*n01/sz,
                })
            cs = pd.DataFrame(cluster_stats) if cluster_stats else pd.DataFrame()

            # High-density clusters (any tier > DENSITY_THRESH)
            if not cs.empty:
                hd = cs[(cs.pct5 >= DENSITY_THRESH*100) |
                         (cs.pct1 >= DENSITY_THRESH*100) |
                         (cs.pct01 >= DENSITY_THRESH*100)].copy()
                n_hd = len(hd)
            else:
                hd, n_hd = pd.DataFrame(), 0

            print(f"  {mcs:>8,} {n_clusters:>8} {n_noise:>8,} "
                  f"{noise_pct:>6.1f}% {med_sz:>7,} {min_sz:>7,} {max_sz:>8,} "
                  f"{top1_cov:>4}/{n_top1:<4} ({100*top1_cov/max(1,n_top1):>3.0f}%)"
                  f"  {top01_cov:>3}/{n_top01} ({100*top01_cov/max(1,n_top01):>3.0f}%)"
                  f"  hd={n_hd}  {dt:.1f}s", flush=True)

            # Show high-density clusters
            if n_hd > 0:
                print(f"\n  [clusters with >{DENSITY_THRESH*100:.0f}% density in top-5/1/0.1%]")
                print(f"  {'cl':>4} {'size':>7} {'max_fit':>9} {'mean_fit':>9} "
                      f"{'n_top5':>7} {'%top5':>6} {'n_top1':>7} {'%top1':>6} "
                      f"{'n_top0.1':>8} {'%top0.1':>8}")
                hd_sorted = hd.sort_values("pct01", ascending=False)
                for _, row in hd_sorted.iterrows():
                    print(f"  {int(row.cluster):>4} {int(row['size']):>7,} "
                          f"{row.max_fit:>9.4f} {row.mean_fit:>9.4f} "
                          f"{int(row.n_top5):>7} {row.pct5:>5.1f}% "
                          f"{int(row.n_top1):>7} {row.pct1:>5.1f}% "
                          f"{int(row.n_top01):>8} {row.pct01:>7.1f}%")
                print()

            del cl, cs, hd
            gc.collect()

        del reduced, sub_emb
        gc.collect()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--landscapes", nargs="+", default=["gb1", "trpb"],
                        choices=list(LANDSCAPE_CFG.keys()))
    args = parser.parse_args()

    for ls in args.landscapes:
        print(f"\nLoading {ls}...", end=" ", flush=True)
        t0 = time.time()
        fit, emb, zs, aas = load(ls)
        print(f"done ({time.time()-t0:.1f}s)")
        explore(ls, fit, emb, zs)


if __name__ == "__main__":
    main()
