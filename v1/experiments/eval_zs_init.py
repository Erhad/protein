"""
eval_zs_init.py — Evaluate zero-shot initialization strategies for GB1 and TrpB.

Tests multiple strategies for selecting the initial 96 variants before any active learning.

Strategies tested:
  - random: uniform random baseline
  - zs_top_*: top-96 by each ZS predictor (current approach)
  - div_emb_*: greedy max-min diversity in ESMc embedding space, from top-K by ZS
  - div_ham_*: greedy max-min diversity in Hamming (sequence) space, from top-K by ZS
  - kmeans_*: K-means cluster embeddings (or top-K), pick best ZS from each cluster

Metrics for the initial 96-sample set:
  - norm_max_fitness: max(fitness[init]) / global_max
  - frac_top1pct:     fraction of top-1% variants captured in init
  - frac_top0p1pct:   fraction of top-0.1% variants captured in init

Usage:
    python experiments/eval_zs_init.py --landscapes gb1 trpb --n_seeds 20
    python experiments/eval_zs_init.py --landscapes gb1 --n_seeds 1   # quick smoke-test
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INITIAL_N = 96

LANDSCAPE_CFG = {
    "gb1": {
        "fitness_csv": "data/gb1/gb1_fitness.csv",
        "fitness_col": "label",
        "emb_4site":   "data/gb1/embeddings_esmc600m_4site.npy",
        "emb_mean":    "data/gb1/embeddings_esmc600m_4site_mean.npy",
        "zs_csv":      "data/li2024/results/zs_comb/all/GB1.csv",
        "sites":       [38, 39, 40, 53],
    },
    "trpb": {
        "fitness_csv": "data/trpb/trpb_fitness.csv",
        "fitness_col": "label",
        "emb_4site":   "data/trpb/embeddings_esmc600m_4site.npy",
        "emb_mean":    None,   # compute on-the-fly from 4-site
        "zs_csv":      "data/li2024/results/zs_comb/all/TrpB4.csv",
        "sites":       [183, 184, 227, 228],
    },
}

ZS_COLS = ["esm_score", "ev_score", "esmif_score", "ev-esm-esmif_score", "ed_score"]


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_landscape(landscape: str):
    cfg = LANDSCAPE_CFG[landscape]
    df = pd.read_csv(os.path.join(ROOT, cfg["fitness_csv"]))
    fitness = df[cfg["fitness_col"]].values.astype(np.float32)
    seqs    = df["protein"].values

    emb_4site = np.load(os.path.join(ROOT, cfg["emb_4site"]))
    if cfg["emb_mean"]:
        emb_mean = np.load(os.path.join(ROOT, cfg["emb_mean"]))
    else:
        n, d4 = emb_4site.shape
        d = d4 // 4
        emb_mean = emb_4site.reshape(n, 4, d).mean(axis=1)

    zs_df  = pd.read_csv(os.path.join(ROOT, cfg["zs_csv"]))
    sites  = cfg["sites"]
    aas    = np.array(["".join(seq[s] for s in sites) for seq in seqs])

    zs_scores = {}
    for col in ZS_COLS:
        if col in zs_df.columns:
            zs_map  = dict(zip(zs_df["AAs"], zs_df[col]))
            raw     = np.array([zs_map.get(a, np.nan) for a in aas], dtype=np.float64)
            zs_scores[col] = np.where(np.isnan(raw), -np.inf, raw)

    return fitness, aas, emb_mean, zs_scores


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(sel_idx, fitness):
    sel_fit    = fitness[np.asarray(sel_idx)]
    global_max = fitness.max()
    norm_max   = float(sel_fit.max() / global_max)

    t1   = np.percentile(fitness, 99.0)
    t01  = np.percentile(fitness, 99.9)
    n1   = max(1, (fitness >= t1).sum())
    n01  = max(1, (fitness >= t01).sum())
    frac1  = float((sel_fit >= t1).sum() / n1)
    frac01 = float((sel_fit >= t01).sum() / n01)
    return norm_max, frac1, frac01


# ──────────────────────────────────────────────────────────────────────────────
# Helper: get top-K candidates by a ZS predictor
# ──────────────────────────────────────────────────────────────────────────────

def _top_k(zs_scores, predictor, k):
    scores = zs_scores[predictor]
    finite_idx = np.where(np.isfinite(scores))[0]
    order      = np.argsort(scores[finite_idx])[::-1][:k]
    idx        = finite_idx[order]
    vals       = scores[idx]
    vmin, vmax = vals.min(), vals.max()
    norm       = (vals - vmin) / (vmax - vmin + 1e-8)
    return idx, vals, norm


# ──────────────────────────────────────────────────────────────────────────────
# Strategy: random
# ──────────────────────────────────────────────────────────────────────────────

def init_random(fitness, emb, zs, aas, rng, **kw):
    return rng.choice(len(fitness), size=INITIAL_N, replace=False)


# ──────────────────────────────────────────────────────────────────────────────
# Strategy: ZS top-N
# ──────────────────────────────────────────────────────────────────────────────

def init_zs_top(fitness, emb, zs, aas, predictor, rng=None, **kw):
    idx, _, _ = _top_k(zs, predictor, INITIAL_N)
    return idx


# ──────────────────────────────────────────────────────────────────────────────
# Strategy: greedy diversity in embedding space
# ──────────────────────────────────────────────────────────────────────────────

def init_div_emb(fitness, emb, zs, aas, predictor, k, alpha, rng=None, **kw):
    top_idx, _, zs_norm = _top_k(zs, predictor, k)
    top_emb  = emb[top_idx].astype(np.float32)

    n        = len(top_idx)
    n_sel    = min(INITIAL_N, n)
    selected = [0]
    min_dists = np.full(n, np.inf, dtype=np.float64)

    for _ in range(n_sel - 1):
        last = selected[-1]
        diff = top_emb - top_emb[last]
        new_dists = np.sqrt((diff * diff).sum(axis=1))
        np.minimum(min_dists, new_dists, out=min_dists)

        d_max = min_dists.max()
        dist_norm = min_dists / (d_max + 1e-8)
        score = alpha * zs_norm + (1.0 - alpha) * dist_norm
        score[selected] = -np.inf
        selected.append(int(np.argmax(score)))

    return top_idx[np.array(selected)]


# ──────────────────────────────────────────────────────────────────────────────
# Strategy: greedy diversity in Hamming (sequence) space
# ──────────────────────────────────────────────────────────────────────────────

def init_div_hamming(fitness, emb, zs, aas, predictor, k, alpha, rng=None, **kw):
    top_idx, _, zs_norm = _top_k(zs, predictor, k)
    aas_int  = np.array([[ord(c) for c in s] for s in aas[top_idx]], dtype=np.int16)

    n        = len(top_idx)
    n_sel    = min(INITIAL_N, n)
    selected = [0]
    min_dists = np.full(n, np.inf, dtype=np.float64)

    for _ in range(n_sel - 1):
        last = selected[-1]
        new_dists = (aas_int != aas_int[last]).sum(axis=1).astype(np.float64)
        np.minimum(min_dists, new_dists, out=min_dists)

        d_max = min_dists.max()
        dist_norm = min_dists / (d_max + 1e-8)
        score = alpha * zs_norm + (1.0 - alpha) * dist_norm
        score[selected] = -np.inf
        selected.append(int(np.argmax(score)))

    return top_idx[np.array(selected)]


# ──────────────────────────────────────────────────────────────────────────────
# Strategy: K-means cluster → best ZS per cluster
# ──────────────────────────────────────────────────────────────────────────────

def init_kmeans(fitness, emb, zs, aas, predictor, k_cands, rng_seed=42, **kw):
    scores      = zs[predictor]
    finite_idx  = np.where(np.isfinite(scores))[0]

    if k_cands is not None:
        order    = np.argsort(scores[finite_idx])[::-1][:k_cands]
        cand_idx = finite_idx[order]
    else:
        cand_idx = finite_idx

    cand_emb = emb[cand_idx].astype(np.float32)
    cand_zs  = scores[cand_idx]

    km     = MiniBatchKMeans(n_clusters=INITIAL_N, random_state=rng_seed,
                             n_init=5, max_iter=300)
    labels = km.fit_predict(cand_emb)

    selected     = []
    selected_set = set()
    for cid in range(INITIAL_N):
        mask = labels == cid
        if not mask.any():
            continue
        best = int(np.argmax(np.where(mask, cand_zs, -np.inf)))
        gi   = int(cand_idx[best])
        if gi not in selected_set:
            selected.append(gi)
            selected_set.add(gi)

    # Fill gaps if any cluster was empty / duplicate
    if len(selected) < INITIAL_N:
        order_fallback = np.argsort(cand_zs)[::-1]
        for i in order_fallback:
            gi = int(cand_idx[i])
            if gi not in selected_set:
                selected.append(gi)
                selected_set.add(gi)
            if len(selected) >= INITIAL_N:
                break

    return np.array(selected[:INITIAL_N])


# ──────────────────────────────────────────────────────────────────────────────
# Strategy: ZS top-K → farthest-point sampling only (no ZS weight, pure diversity)
# ──────────────────────────────────────────────────────────────────────────────

def init_fps_from_top_k(fitness, emb, zs, aas, predictor, k, rng=None, **kw):
    """Pure farthest-point sampling (FPS) from top-K ZS, no ZS weighting."""
    return init_div_emb(fitness, emb, zs, aas, predictor=predictor, k=k, alpha=0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────────────────────────────────────

def print_diagnostics(landscape, fitness, aas, zs):
    global_max = fitness.max()
    t1  = np.percentile(fitness, 99.0)
    t01 = np.percentile(fitness, 99.9)
    print(f"\n{'='*60}")
    print(f"  {landscape.upper()} diagnostics")
    print(f"  Total: {len(fitness):,}  |  Global max: {global_max:.4f}")
    print(f"  Top-1%  threshold: {t1:.4f}  ({(fitness >= t1).sum():,} variants)")
    print(f"  Top-0.1% threshold: {t01:.4f}  ({(fitness >= t01).sum():,} variants)")

    top_idx = np.argsort(fitness)[::-1][:5]
    print(f"\n  True top-5 variants:")
    for rank, idx in enumerate(top_idx):
        row = f"  #{rank+1}: {aas[idx]}  fit={fitness[idx]:.4f}"
        for col, scores in zs.items():
            if np.isfinite(scores[idx]):
                zs_rank = int((scores > scores[idx]).sum()) + 1
                short = col.replace("ev-esm-esmif_score", "comb").replace("_score", "")
                row += f"  {short}_rank={zs_rank:,}"
        print(row)
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Build strategy registry
# ──────────────────────────────────────────────────────────────────────────────

def build_strategies(available_preds):
    PRED = "ev-esm-esmif_score"
    strats = {}

    # Random baseline
    strats["random"] = lambda f, e, z, a, rng: init_random(f, e, z, a, rng)

    # ZS top-96 for each predictor
    for pred in available_preds:
        short = (pred.replace("ev-esm-esmif_score", "combined")
                     .replace("_score", ""))
        strats[f"zs_top_{short}"] = (
            lambda f, e, z, a, rng, p=pred: init_zs_top(f, e, z, a, predictor=p)
        )

    # Greedy embedding diversity (combined predictor, various k and alpha)
    for k in [200, 500, 1000, 2000, 5000, 10000]:
        for alpha in [0.0, 0.3, 0.5, 0.7, 0.9]:
            if alpha == 0.0 and k != 5000:
                continue   # pure FPS: only run a representative k
            tag = f"div_emb_k{k}_a{int(alpha*10)}"
            strats[tag] = (
                lambda f, e, z, a, rng, k_=k, al=alpha:
                    init_div_emb(f, e, z, a, predictor=PRED, k=k_, alpha=al)
            )

    # Greedy Hamming diversity (combined predictor)
    for k in [500, 2000, 10000, 50000]:
        for alpha in [0.0, 0.3, 0.5, 0.7]:
            if alpha == 0.0 and k != 10000:
                continue
            tag = f"div_ham_k{k}_a{int(alpha*10)}"
            strats[tag] = (
                lambda f, e, z, a, rng, k_=k, al=alpha:
                    init_div_hamming(f, e, z, a, predictor=PRED, k=k_, alpha=al)
            )

    # K-means: cluster top-K (and all) variants
    for k_cands in [500, 2000, 10000, None]:
        tag = f"kmeans_k{'all' if k_cands is None else k_cands}"
        strats[tag] = (
            lambda f, e, z, a, rng, kc=k_cands:
                init_kmeans(f, e, z, a, predictor=PRED, k_cands=kc,
                            rng_seed=int(rng.integers(0, 2**31)))
        )

    return strats


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_landscape(landscape, n_seeds, strategies):
    print(f"\nLoading {landscape}...", end=" ", flush=True)
    t0 = time.time()
    fitness, aas, emb, zs = load_landscape(landscape)
    print(f"done ({time.time()-t0:.1f}s)")

    print_diagnostics(landscape, fitness, aas, zs)

    results = {}
    n_strats = len(strategies)
    for si, (name, fn) in enumerate(strategies.items()):
        metrics_list = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            sel = fn(fitness, emb, zs, aas, rng)
            metrics_list.append(compute_metrics(sel, fitness))
        arr = np.array(metrics_list)
        results[name] = arr
        if (si + 1) % 10 == 0 or (si + 1) == n_strats:
            print(f"  [{si+1}/{n_strats}] {name:<30}  "
                  f"norm_max={arr[:,0].mean():.4f} ± {arr[:,0].std():.4f}",
                  flush=True)

    return results


def print_results_table(landscape, results, n_seeds):
    print(f"\n{'='*78}")
    print(f"  {landscape.upper()} — Initial {INITIAL_N} samples  (n_seeds={n_seeds})")
    print(f"{'='*78}")
    hdr = f"{'Strategy':<32}{'norm_max':>10}{'±':>7}{'top1%':>8}{'±':>7}{'top0.1%':>9}{'±':>7}"
    print(hdr)
    print("-" * 78)

    sorted_items = sorted(results.items(), key=lambda x: -x[1][:,0].mean())
    for name, arr in sorted_items:
        print(f"{name:<32}"
              f"{arr[:,0].mean():>10.4f}{arr[:,0].std():>7.4f}"
              f"{arr[:,1].mean():>8.4f}{arr[:,1].std():>7.4f}"
              f"{arr[:,2].mean():>9.4f}{arr[:,2].std():>7.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landscapes", nargs="+", default=["gb1", "trpb"],
                        choices=list(LANDSCAPE_CFG.keys()))
    parser.add_argument("--n_seeds", type=int, default=20,
                        help="Seeds for stochastic strategies (default 20)")
    args = parser.parse_args()

    for landscape in args.landscapes:
        # Probe which ZS predictors are available
        zs_df  = pd.read_csv(os.path.join(ROOT, LANDSCAPE_CFG[landscape]["zs_csv"]))
        avail  = [c for c in ZS_COLS if c in zs_df.columns]

        strats = build_strategies(avail)
        print(f"\nTesting {len(strats)} strategies on {landscape} (n_seeds={args.n_seeds})")

        results = evaluate_landscape(landscape, args.n_seeds, strats)
        print_results_table(landscape, results, args.n_seeds)


if __name__ == "__main__":
    main()
