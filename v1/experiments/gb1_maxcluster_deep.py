"""
gb1_maxcluster_deep.py

Pipeline on GB1 top-25% ZS variants:
  1. Keep top 25% by ZS score
  2. PCA 1152→50 (MPS) → UMAP 50→15 → HDBSCAN sweep
  3. KNN analysis around the global max variant (FWAA) in original embedding space
  4. For each mcs: report which cluster FWAA lands in, and compare that cluster
     vs all others on mean fitness, % top-5/1/0.1%, density

Usage:
    python experiments/gb1_maxcluster_deep.py
"""
import gc, os, sys, time
import numpy as np
import pandas as pd
import torch
import umap
import hdbscan
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FITNESS_CSV  = os.path.join(ROOT, "data/gb1/gb1_fitness.csv")
EMB_MEAN     = os.path.join(ROOT, "data/gb1/embeddings_esmc600m_4site_mean.npy")
ZS_CSV       = os.path.join(ROOT, "data/li2024/results/zs_comb/all/GB1.csv")
SITES        = [38, 39, 40, 53]
ZS_COL       = "ev-esm-esmif_score"
ZS_KEEP_PCT  = 25          # keep top 25% by ZS
PCA_DIMS     = 50           # pre-UMAP noise reduction
UMAP_DIMS    = 15           # McInnes: 10-50 for clustering (not 2-3)
UMAP_N_NEIGHBORS = 30       # McInnes: 30 for clustering (not 15 viz default)
SKIP_UMAP    = False
MIN_CLUSTER_SIZES = [50, 100, 200, 500]
KNN_K        = 50


def load():
    df  = pd.read_csv(FITNESS_CSV)
    fit = df["label"].values.astype(np.float32)
    emb = np.load(EMB_MEAN)
    zs_df = pd.read_csv(ZS_CSV)
    aas = df["protein"].apply(lambda s: ''.join(s[i] for i in SITES))
    zs  = aas.map(dict(zip(zs_df["AAs"], zs_df[ZS_COL]))).to_numpy(dtype=np.float64, na_value=np.nan)
    zs  = np.where(np.isnan(zs), -np.inf, zs)
    return fit, emb, zs, aas.values, df


def pca_mps(sub_emb, n_dims):
    t0 = time.time()
    sample = sub_emb[np.random.choice(len(sub_emb), min(10_000, len(sub_emb)), replace=False)]
    center = torch.from_numpy(np.median(sample, axis=0))
    X     = torch.from_numpy(sub_emb)
    X_mps = (X - center).to('mps')
    Omega = torch.randn(sub_emb.shape[1], n_dims + 10, device='mps')
    Y     = (X_mps @ Omega).cpu()
    Q, _  = torch.linalg.qr(Y)
    B     = (Q.to('mps').T @ X_mps).cpu()
    _, S, Vh = torch.linalg.svd(B, full_matrices=False)
    V     = Vh[:n_dims].T
    out   = (X_mps @ V.to('mps')).cpu().numpy()
    del X, X_mps, Omega, Y, Q, B, S, Vh, V
    print(f"  PCA {sub_emb.shape[1]}→{n_dims}d  ({time.time()-t0:.1f}s)", flush=True)
    return out


def main():
    print("Loading GB1...", flush=True)
    fit, emb, zs, aas, df = load()
    N = len(fit)

    t5  = np.percentile(fit, 95.0)
    t1  = np.percentile(fit, 99.0)
    t01 = np.percentile(fit, 99.9)
    n_top5  = (fit >= t5).sum()
    n_top1  = (fit >= t1).sum()
    n_top01 = (fit >= t01).sum()

    # Global max
    max_idx = int(np.argmax(fit))
    max_aa  = aas[max_idx]
    print(f"N={N:,}  max={fit.max():.4f} ({max_aa})  top-5%={n_top5}  top-1%={n_top1}  top-0.1%={n_top01}")

    # --- ZS filter: keep top 25% ---
    thresh   = np.percentile(zs[np.isfinite(zs)], 100 - ZS_KEEP_PCT)
    keep_idx = np.where(np.isfinite(zs) & (zs >= thresh))[0]
    n_kept   = len(keep_idx)
    kept_set = set(keep_idx.tolist())
    print(f"\nKeep top {ZS_KEEP_PCT}% by ZS → {n_kept:,} variants ({100*n_kept/N:.0f}%)")
    print(f"  global max in pool: {max_idx in kept_set}  (ZS rank of max: {(zs >= zs[max_idx]).sum()}/{N})")
    print(f"  top-1%  retained: {sum(1 for i in np.where(fit>=t1)[0]  if i in kept_set)}/{n_top1}")
    print(f"  top-0.1% retained: {sum(1 for i in np.where(fit>=t01)[0] if i in kept_set)}/{n_top01}")

    sub_emb = emb[keep_idx].astype(np.float32)
    sub_fit = fit[keep_idx]

    # Local position of global max within filtered set
    local_max_pos = np.where(keep_idx == max_idx)[0]
    local_max_idx = int(local_max_pos[0]) if len(local_max_pos) else None
    print(f"  local index of global max in filtered set: {local_max_idx}")

    # ── KNN in original 1152d space ──────────────────────────────────────────
    print(f"\n── KNN (k={KNN_K}) around global max in original embedding space ──")
    t0 = time.time()
    nn = NearestNeighbors(n_neighbors=KNN_K+1, metric='cosine', n_jobs=1)
    nn.fit(sub_emb)
    dists, idxs = nn.kneighbors(sub_emb[[local_max_idx]])
    dists, idxs = dists[0][1:], idxs[0][1:]   # exclude self
    knn_fit = sub_fit[idxs]
    print(f"  time: {time.time()-t0:.1f}s")
    print(f"  dist range: {dists.min():.4f} – {dists.max():.4f}  (cosine)")
    print(f"  fitness of {KNN_K} nearest neighbors:")
    print(f"    mean={knn_fit.mean():.4f}  median={np.median(knn_fit):.4f}  max={knn_fit.max():.4f}")
    print(f"    top-5%:  {(knn_fit>=t5).sum()}/{KNN_K}  ({100*(knn_fit>=t5).mean():.0f}%)")
    print(f"    top-1%:  {(knn_fit>=t1).sum()}/{KNN_K}  ({100*(knn_fit>=t1).mean():.0f}%)")
    print(f"    top-0.1%:{(knn_fit>=t01).sum()}/{KNN_K}  ({100*(knn_fit>=t01).mean():.0f}%)")
    print(f"  top-10 nearest neighbors:")
    top10_idx = np.argsort(dists)[:10]
    for rank, i in enumerate(top10_idx):
        gi = keep_idx[idxs[i]]
        print(f"    #{rank+1}  dist={dists[i]:.4f}  fitness={sub_fit[idxs[i]]:.4f}  AA={aas[gi]}")

    # ── PCA (→ UMAP optional) ────────────────────────────────────────────────
    print(f"\n── PCA {'(no UMAP)' if SKIP_UMAP else '+ UMAP'} ──")
    pca_out = pca_mps(sub_emb, PCA_DIMS)
    gc.collect()

    if SKIP_UMAP:
        reduced = pca_out
        print(f"  skipping UMAP, clustering directly on PCA-{PCA_DIMS}", flush=True)
    else:
        t0 = time.time()
        reducer = umap.UMAP(n_components=UMAP_DIMS, n_neighbors=UMAP_N_NEIGHBORS,
                            min_dist=0.0, metric='euclidean', n_jobs=1, low_memory=True)
        reduced = reducer.fit_transform(pca_out)
        del reducer
        gc.collect()
        print(f"  UMAP {PCA_DIMS}→{UMAP_DIMS}d  ({time.time()-t0:.1f}s)", flush=True)
    del pca_out

    # ── HDBSCAN sweep ────────────────────────────────────────────────────────
    print(f"\n── HDBSCAN sweep ──")
    for mcs in MIN_CLUSTER_SIZES:
        print(f"\n  mcs={mcs}", flush=True)
        t0  = time.time()
        cl  = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=1,
                               algorithm='boruvka_balltree', core_dist_n_jobs=1)
        lbl = cl.fit_predict(reduced)
        dt  = time.time() - t0

        unique_lbl = [l for l in set(lbl) if l >= 0]
        n_clusters = len(unique_lbl)
        n_noise    = (lbl == -1).sum()
        print(f"  {n_clusters} clusters  {n_noise:,} noise ({100*n_noise/n_kept:.1f}%)  {dt:.1f}s")

        if local_max_idx is not None:
            max_cluster = lbl[local_max_idx]
            print(f"  global max (FWAA) → cluster {max_cluster} {'[NOISE]' if max_cluster==-1 else ''}")
        else:
            max_cluster = -99

        # Per-cluster stats
        rows = []
        for l in unique_lbl:
            idx_c = keep_idx[lbl == l]
            f_c   = fit[idx_c]
            sz    = len(idx_c)
            rows.append({
                "cl": l, "size": sz,
                "mean_fit": f_c.mean(), "max_fit": f_c.max(),
                "n5": (f_c>=t5).sum(), "n1": (f_c>=t1).sum(), "n01": (f_c>=t01).sum(),
                "pct5": 100*(f_c>=t5).mean(), "pct1": 100*(f_c>=t1).mean(),
                "pct01": 100*(f_c>=t01).mean(),
                "is_max_cluster": (l == max_cluster),
            })
        cs = pd.DataFrame(rows)

        # Max cluster detail
        if max_cluster >= 0 and len(cs):
            mc = cs[cs.cl == max_cluster].iloc[0]
            others = cs[cs.cl != max_cluster]
            print(f"\n  ┌─ Global max cluster (cl={max_cluster}) ─────────────────────────────")
            print(f"  │  size={int(mc['size']):,}  mean_fit={mc.mean_fit:.4f}  max_fit={mc.max_fit:.4f}")
            print(f"  │  top-5%:  {int(mc.n5):>4} / {int(mc['size'])}  ({mc.pct5:.1f}%)")
            print(f"  │  top-1%:  {int(mc.n1):>4} / {int(mc['size'])}  ({mc.pct1:.1f}%)")
            print(f"  │  top-0.1%:{int(mc.n01):>4} / {int(mc['size'])}  ({mc.pct01:.1f}%)")
            print(f"  └────────────────────────────────────────────────────────────────")
            if len(others):
                print(f"  All other clusters combined:")
                all_other_idx = keep_idx[np.isin(lbl, others.cl.values)]
                fo = fit[all_other_idx]
                print(f"    size={len(all_other_idx):,}  mean_fit={fo.mean():.4f}  max_fit={fo.max():.4f}")
                print(f"    top-5%:  {(fo>=t5).sum():>4} / {len(fo)}  ({100*(fo>=t5).mean():.1f}%)")
                print(f"    top-1%:  {(fo>=t1).sum():>4} / {len(fo)}  ({100*(fo>=t1).mean():.1f}%)")
                print(f"    top-0.1%:{(fo>=t01).sum():>4} / {len(fo)}  ({100*(fo>=t01).mean():.1f}%)")

            # Top 10 clusters by max_fit
            print(f"\n  Top 10 clusters by max_fit:")
            print(f"  {'cl':>4} {'size':>7} {'max_fit':>9} {'mean_fit':>9} "
                  f"{'%top5':>6} {'%top1':>6} {'%top0.1':>8}  max?")
            for _, row in cs.sort_values('max_fit', ascending=False).head(10).iterrows():
                flag = " ← GLOBAL MAX CLUSTER" if row.is_max_cluster else ""
                print(f"  {int(row.cl):>4} {int(row['size']):>7,} {row.max_fit:>9.4f} "
                      f"{row.mean_fit:>9.4f} {row.pct5:>5.1f}% {row.pct1:>5.1f}% "
                      f"{row.pct01:>7.1f}%{flag}")

        del cl, cs
        gc.collect()

    del reduced, sub_emb
    gc.collect()


if __name__ == "__main__":
    main()
