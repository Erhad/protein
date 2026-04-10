"""
eval_zs_init_r2.py — Round 2: Generalizable strategies for ZS initialization.

Hypothesis: The K-too-small problem hurts TrpB (400 unique 4-site codes × 400 backgrounds;
best code IGGG is at ZS rank ~36k, so top-5000 misses it). For GB1 the sweet spot was k=5000
with alpha=0.3, but that was partly luck (VAAA happened to be in top-5000).

New strategies (all generalizable / parameter-free or adaptive):
  1. fps_all        — FPS from ALL variants (embedding, no ZS restriction), pure diversity
  2. div_emb_sqrtN  — Adaptive K = sqrt(N * INITIAL_N), alpha sweep
  3. div_emb_frac   — K as a fraction of N (10%, 25%, 50%), alpha sweep
  4. kmeans_prob    — K-means cluster + softmax(ZS) sampling within each cluster
                      (the "cluster then sample probabilistically by ZS" approach)
  5. Fine-tuned GB1 winners from R1: k=5000, alpha in [0.1, 0.2, 0.25]

Usage:
    python experiments/eval_zs_init_r2.py --landscapes gb1 trpb --n_seeds 20
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
        "sites":       [183, 184, 227, 228],
    },
}

ZS_COLS = ["esm_score", "ev_score", "esmif_score", "ev-esm-esmif_score", "ed_score"]
PRED = "ev-esm-esmif_score"


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
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def _get_zs(zs_scores, predictor=PRED):
    scores = zs_scores[predictor]
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    # Normalize to [0,1]
    finite_mask = np.isfinite(scores)
    s = scores[finite_mask]
    smin, smax = s.min(), s.max()
    norm = np.full_like(scores, 0.0)
    norm[finite_mask] = (scores[finite_mask] - smin) / (smax - smin + 1e-8)
    return scores, norm


def _greedy_fps_emb(emb, zs_norm, alpha, selected_start, n_total):
    """Greedy FPS with optional ZS weighting. emb: (N, d), zs_norm: (N,)"""
    n = len(emb)
    n_sel = min(INITIAL_N, n)
    selected = [selected_start]
    min_dists = np.full(n, np.inf, dtype=np.float32)

    for _ in range(n_sel - 1):
        last = selected[-1]
        diff = emb - emb[last]
        new_d = np.sqrt((diff * diff).sum(axis=1))
        np.minimum(min_dists, new_d, out=min_dists)

        dmax = min_dists.max()
        d_norm = min_dists / (dmax + 1e-8)
        score = alpha * zs_norm + (1.0 - alpha) * d_norm
        score[selected] = -np.inf
        selected.append(int(np.argmax(score)))

    return np.array(selected)


# ──────────────────────────────────────────────────────────────────────────────
# Strategy: random
# ──────────────────────────────────────────────────────────────────────────────

def init_random(fitness, emb, zs, aas, rng, **kw):
    return rng.choice(len(fitness), size=INITIAL_N, replace=False)


# ──────────────────────────────────────────────────────────────────────────────
# Strategy: ZS top-96 (baseline from R1)
# ──────────────────────────────────────────────────────────────────────────────

def init_zs_top(fitness, emb, zs, aas, rng=None, **kw):
    scores = zs[PRED]
    return np.argsort(scores)[::-1][:INITIAL_N]


# ──────────────────────────────────────────────────────────────────────────────
# Strategy: Greedy FPS from top-K ZS (R1 winner was k=5000, alpha=0.3 for GB1)
# ──────────────────────────────────────────────────────────────────────────────

def init_div_emb(fitness, emb, zs, aas, k, alpha, rng=None, **kw):
    scores, zs_norm = _get_zs(zs)
    finite_idx = np.where(np.isfinite(scores))[0]
    order = np.argsort(scores[finite_idx])[::-1][:k]
    top_k_idx = finite_idx[order]

    top_emb    = emb[top_k_idx].astype(np.float32)
    top_zs     = zs_norm[top_k_idx]

    local_sel = _greedy_fps_emb(top_emb, top_zs, alpha, 0, len(top_k_idx))
    return top_k_idx[local_sel]


# ──────────────────────────────────────────────────────────────────────────────
# Strategy: Adaptive K = sqrt(N * INITIAL_N) — generalizable, no manual tuning
# ──────────────────────────────────────────────────────────────────────────────

def init_div_emb_sqrtN(fitness, emb, zs, aas, alpha, rng=None, **kw):
    k = max(INITIAL_N, int(np.sqrt(len(fitness) * INITIAL_N)))
    return init_div_emb(fitness, emb, zs, aas, k=k, alpha=alpha)


# ──────────────────────────────────────────────────────────────────────────────
# Strategy: K = fraction of N — ensures large pools even for structured landscapes
# ──────────────────────────────────────────────────────────────────────────────

def init_div_emb_frac(fitness, emb, zs, aas, frac, alpha, rng=None, **kw):
    k = max(INITIAL_N, int(frac * len(fitness)))
    return init_div_emb(fitness, emb, zs, aas, k=k, alpha=alpha)


# ──────────────────────────────────────────────────────────────────────────────
# Strategy: Pure FPS from ALL variants — fully parameter-free
# ──────────────────────────────────────────────────────────────────────────────

def init_fps_all(fitness, emb, zs, aas, rng=None, **kw):
    """Pure farthest-point sampling from the entire landscape — no ZS bias, no K."""
    n = len(fitness)
    scores, _ = _get_zs(zs)
    start = int(np.argmax(np.where(np.isfinite(scores), scores, -np.inf)))
    selected = [start]
    min_dists = np.full(n, np.inf, dtype=np.float32)

    for _ in range(INITIAL_N - 1):
        last = selected[-1]
        diff = emb - emb[last]
        new_d = np.sqrt((diff * diff).sum(axis=1))
        np.minimum(min_dists, new_d, out=min_dists)
        min_dists[selected] = -np.inf
        selected.append(int(np.argmax(min_dists)))

    return np.array(selected)


# ──────────────────────────────────────────────────────────────────────────────
# Strategy: K-means cluster + softmax(ZS) selection within each cluster
#   The "cluster then probabilistically sample by ZS" idea.
#   Generalizable: adapts to landscape structure automatically.
#   n_clusters is a free parameter (tested: INITIAL_N, 2×, 4×, 8× INITIAL_N)
# ──────────────────────────────────────────────────────────────────────────────

def init_kmeans_prob(fitness, emb, zs, aas, n_clusters, temperature, rng, **kw):
    """
    1. K-means cluster all variants into n_clusters groups.
    2. Within each cluster, select 1 variant with prob proportional to softmax(ZS/T).
    3. If n_clusters > INITIAL_N, randomly pick INITIAL_N distinct clusters to sample from.
    """
    scores, _ = _get_zs(zs)

    km = MiniBatchKMeans(n_clusters=n_clusters,
                         random_state=int(rng.integers(0, 2**31)),
                         n_init=3, max_iter=200, batch_size=2048)
    labels = km.fit_predict(emb)

    # For each cluster, sample one variant proportional to softmax(ZS / T)
    cluster_selections = {}
    for cid in range(n_clusters):
        mask = np.where(labels == cid)[0]
        if len(mask) == 0:
            continue
        cand_zs = scores[mask]
        finite = np.isfinite(cand_zs)
        if not finite.any():
            # No ZS score → random
            cluster_selections[cid] = int(rng.choice(mask))
            continue
        z = cand_zs[finite]
        # Softmax with temperature
        z_shifted = (z - z.max()) / temperature
        probs = np.exp(z_shifted)
        probs /= probs.sum()
        chosen_local = int(rng.choice(len(z), p=probs))
        global_idx = int(mask[np.where(finite)[0][chosen_local]])
        cluster_selections[cid] = global_idx

    all_cids = list(cluster_selections.keys())
    if len(all_cids) <= INITIAL_N:
        selected = list(cluster_selections.values())
        # Fill remaining with best ZS from unselected variants
        sel_set = set(selected)
        order = np.argsort(scores)[::-1]
        for idx in order:
            if int(idx) not in sel_set:
                selected.append(int(idx))
                sel_set.add(int(idx))
            if len(selected) >= INITIAL_N:
                break
    else:
        # More clusters than budget → pick INITIAL_N clusters
        # Prefer clusters with highest max ZS score
        cluster_max_zs = {
            cid: scores[cluster_selections[cid]] if np.isfinite(scores[cluster_selections[cid]])
                 else -np.inf
            for cid in all_cids
        }
        top_cids = sorted(all_cids, key=lambda c: -cluster_max_zs[c])[:INITIAL_N]
        selected = [cluster_selections[c] for c in top_cids]

    return np.array(selected[:INITIAL_N])


# ──────────────────────────────────────────────────────────────────────────────
# Strategy: K-means (argmax ZS per cluster) — deterministic version
# ──────────────────────────────────────────────────────────────────────────────

def init_kmeans_argmax(fitness, emb, zs, aas, n_clusters, rng, **kw):
    """K-means cluster → pick argmax ZS per cluster. Deterministic (fixed seed)."""
    scores, _ = _get_zs(zs)

    km = MiniBatchKMeans(n_clusters=n_clusters,
                         random_state=int(rng.integers(0, 2**31)),
                         n_init=3, max_iter=200, batch_size=2048)
    labels = km.fit_predict(emb)

    selected = []
    sel_set  = set()
    for cid in range(n_clusters):
        mask = np.where(labels == cid)[0]
        if len(mask) == 0:
            continue
        best = int(mask[np.argmax(
            np.where(np.isfinite(scores[mask]), scores[mask], -np.inf)
        )])
        if best not in sel_set:
            selected.append(best)
            sel_set.add(best)

    if len(selected) < INITIAL_N:
        order = np.argsort(scores)[::-1]
        for idx in order:
            if int(idx) not in sel_set:
                selected.append(int(idx))
                sel_set.add(int(idx))
            if len(selected) >= INITIAL_N:
                break

    return np.array(selected[:INITIAL_N])


# ──────────────────────────────────────────────────────────────────────────────
# Strategy: R1 winner (for reference)
# ──────────────────────────────────────────────────────────────────────────────

def init_r1_winner(fitness, emb, zs, aas, rng=None, **kw):
    """R1 winner: div_emb_k5000_a3 (found VAAA at 0.7077 for GB1)."""
    return init_div_emb(fitness, emb, zs, aas, k=5000, alpha=0.3)


# ──────────────────────────────────────────────────────────────────────────────
# Build strategy registry
# ──────────────────────────────────────────────────────────────────────────────

def build_strategies():
    strats = {}

    # Baselines
    strats["random"]     = lambda f,e,z,a,rng: init_random(f, e, z, a, rng)
    strats["zs_top96"]   = lambda f,e,z,a,rng: init_zs_top(f, e, z, a)
    strats["r1_winner"]  = lambda f,e,z,a,rng: init_r1_winner(f, e, z, a)

    # ── R1 winner fine-tuning (alpha sweep at k=5000) ──────────────────────────
    for alpha in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        tag = f"div_emb_k5000_a{int(alpha*100):02d}"
        strats[tag] = (
            lambda f,e,z,a,rng, al=alpha:
                init_div_emb(f, e, z, a, k=5000, alpha=al)
        )

    # ── Adaptive K = sqrt(N * INITIAL_N) ─────────────────────────────────────
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.5]:
        tag = f"div_emb_sqrtN_a{int(alpha*10)}"
        strats[tag] = (
            lambda f,e,z,a,rng, al=alpha:
                init_div_emb_sqrtN(f, e, z, a, alpha=al)
        )

    # ── K as fraction of N ────────────────────────────────────────────────────
    for frac in [0.05, 0.10, 0.25, 0.50]:
        for alpha in [0.0, 0.1, 0.3]:
            tag = f"div_emb_f{int(frac*100):02d}pct_a{int(alpha*10)}"
            strats[tag] = (
                lambda f,e,z,a,rng, fr=frac, al=alpha:
                    init_div_emb_frac(f, e, z, a, frac=fr, alpha=al)
            )

    # ── Pure FPS from all (no K restriction, no ZS bias) ─────────────────────
    strats["fps_all"] = lambda f,e,z,a,rng: init_fps_all(f, e, z, a)

    # ── K-means + probabilistic ZS — the "cluster then sample by ZS" idea ────
    for n_clust in [INITIAL_N, INITIAL_N * 2, INITIAL_N * 4]:
        for temp in [0.01, 0.1, 1.0, 10.0]:
            tag = f"kmprob_k{n_clust}_t{str(temp).replace('.','p')}"
            strats[tag] = (
                lambda f,e,z,a,rng, nc=n_clust, t=temp:
                    init_kmeans_prob(f, e, z, a, n_clusters=nc,
                                     temperature=t, rng=rng)
            )

    # ── K-means + argmax ZS ───────────────────────────────────────────────────
    for n_clust in [INITIAL_N, INITIAL_N * 2, INITIAL_N * 4]:
        tag = f"kmargmax_k{n_clust}"
        strats[tag] = (
            lambda f,e,z,a,rng, nc=n_clust:
                init_kmeans_argmax(f, e, z, a, n_clusters=nc, rng=rng)
        )

    return strats


# ──────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────────────────────────────────────

def print_diagnostics(landscape, fitness, aas, zs):
    N = len(fitness)
    global_max = fitness.max()
    t1  = np.percentile(fitness, 99.0)
    t01 = np.percentile(fitness, 99.9)

    unique_codes = len(set(aas.tolist()))
    print(f"\n{'='*60}")
    print(f"  {landscape.upper()}: N={N:,}  max={global_max:.4f}  unique_4site={unique_codes:,}")
    print(f"  top-1% thresh: {t1:.4f}  ({(fitness>=t1).sum():,} variants)")
    print(f"  top-0.1% thresh: {t01:.4f}  ({(fitness>=t01).sum():,} variants)")

    # Adaptive K value
    sqrt_k = max(INITIAL_N, int(np.sqrt(N * INITIAL_N)))
    print(f"  sqrt(N*B)={sqrt_k:,}  frac10%={int(0.10*N):,}  frac25%={int(0.25*N):,}")

    # Check top-0.1% coverage at different K
    scores = zs[PRED]
    finite_idx = np.where(np.isfinite(scores))[0]
    order = np.argsort(scores[finite_idx])[::-1]
    top01_set = set(np.where(fitness >= t01)[0].tolist())
    for k in [500, sqrt_k, int(0.1*N), int(0.25*N), int(0.5*N)]:
        pool_set = set(finite_idx[order[:k]].tolist())
        overlap  = len(pool_set & top01_set)
        n_top01  = max(1, len(top01_set))
        print(f"  top-{k:,} ZS pool → contains {overlap}/{n_top01} top-0.1% variants ({100*overlap/n_top01:.1f}%)")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
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
        t_start = time.time()
        metrics_list = []
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            sel = fn(fitness, emb, zs, aas, rng)
            metrics_list.append(compute_metrics(sel, fitness))
        arr = np.array(metrics_list)
        elapsed = time.time() - t_start
        results[name] = arr
        if (si + 1) % 5 == 0 or (si + 1) == n_strats:
            print(f"  [{si+1}/{n_strats}] {name:<38}  "
                  f"norm_max={arr[:,0].mean():.4f}±{arr[:,0].std():.4f}  "
                  f"({elapsed:.1f}s)",
                  flush=True)

    return results


def print_results_table(landscape, results, n_seeds):
    print(f"\n{'='*80}")
    print(f"  {landscape.upper()} R2 — Initial {INITIAL_N} samples  (n_seeds={n_seeds})")
    print(f"{'='*80}")
    hdr = f"{'Strategy':<40}{'norm_max':>10}{'±':>6}{'top1%':>8}{'±':>6}{'top0.1%':>9}{'±':>6}"
    print(hdr)
    print("-" * 80)
    for name, arr in sorted(results.items(), key=lambda x: -x[1][:,0].mean()):
        print(f"{name:<40}"
              f"{arr[:,0].mean():>10.4f}{arr[:,0].std():>6.4f}"
              f"{arr[:,1].mean():>8.4f}{arr[:,1].std():>6.4f}"
              f"{arr[:,2].mean():>9.4f}{arr[:,2].std():>6.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landscapes", nargs="+", default=["gb1", "trpb"],
                        choices=list(LANDSCAPE_CFG.keys()))
    parser.add_argument("--n_seeds",  type=int, default=20)
    args = parser.parse_args()

    strats = build_strategies()
    print(f"Round 2: {len(strats)} strategies, n_seeds={args.n_seeds}")
    for landscape in args.landscapes:
        results = evaluate_landscape(landscape, args.n_seeds, strats)
        print_results_table(landscape, results, args.n_seeds)


if __name__ == "__main__":
    main()
