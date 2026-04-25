"""
Run a single experiment: one (landscape, method, batch_size, seed) combination.

Output: results/raw/{landscape}_{method}_{batch_size}.jsonl
  One JSON record per line. Each seed appends one record:
  {
    "landscape": "gfp",
    "method": "evolvepro",
    "batch_size": 96,
    "seed": 0,
    "selection_order": [4821, 302, ...]   # flat list, length = TOTAL_BUDGET
  }

From selection_order you can compute any metric at any n:
    labeled = selection_order[:n]
    max_fitness = fitness[labeled].max()

Usage:
    python experiments/run_single.py --landscape gfp --method evolvepro --batch_size 96 --seed 0
    python experiments/run_single.py --landscape gfp --method boes_ts   --batch_size 16 --seed 0
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INITIAL_N   = 96
TOTAL_BUDGET = 480

LANDSCAPE_CFG = {
    "gfp": {
        "fitness_csv": "data/gfp/sequences_scores.csv",
        "fitness_col": "score",
        "embeddings":  "data/gfp/embeddings_esm2_650m_meanpool.npz",
    },
    "gb1": {
        "fitness_csv": "data/gb1/gb1_fitness.csv",
        "fitness_col": "label",
        "embeddings":  "data/gb1/embeddings_esm2_650m_4site.npz",
    },
    "gb1_esm2": {
        "fitness_csv": "data/gb1/gb1_fitness.csv",
        "fitness_col": "label",
        "embeddings":  "data/gb1/embeddings_esm2_650m_4site.npz",
    },
    "gb1_esm2_mean": {
        "fitness_csv": "data/gb1/gb1_fitness.csv",
        "fitness_col": "label",
        "embeddings":  "data/gb1/embeddings_esm2_650m_4site_mean.npy",
    },
    "gb1_esm2_15b": {
        "fitness_csv": "data/gb1/gb1_fitness.csv",
        "fitness_col": "label",
        "embeddings":  "data/gb1/embeddings_esm2_15b_meanpool.npy",
    },
    "gb1_esm2_15b_c15b": {   # same embeddings, uses 15B-derived clusters for clinit
        "fitness_csv": "data/gb1/gb1_fitness.csv",
        "fitness_col": "label",
        "embeddings":  "data/gb1/embeddings_esm2_15b_meanpool.npy",
    },
    "gb1_esmc": {
        "fitness_csv": "data/gb1/gb1_fitness.csv",
        "fitness_col": "label",
        "embeddings":  "data/gb1/embeddings_esmc600m_4site.npy",
    },
    "trpb": {
        "fitness_csv": "data/trpb/trpb_fitness.csv",
        "fitness_col": "label",
        "embeddings":  "data/trpb/embeddings_esm2_650m_4site.npz",
    },
    "trpb_esmc": {
        "fitness_csv": "data/trpb/trpb_fitness.csv",
        "fitness_col": "label",
        "embeddings":  "data/trpb/embeddings_esmc600m_4site.npy",
    },
    "trpb_esm2_15b": {
        "fitness_csv": "data/trpb/trpb_fitness.csv",
        "fitness_col": "label",
        "embeddings":  "data/trpb/embeddings_esm2_15b_meanpool.npy",
    },
    "gb1_onehot": {
        "fitness_csv": "data/gb1/gb1_fitness.csv",
        "fitness_col": "label",
        "embeddings":  None,   # computed on-the-fly in load_landscape
    },
    "trpb_onehot": {
        "fitness_csv": "data/trpb/trpb_fitness.csv",
        "fitness_col": "label",
        "embeddings":  None,   # computed on-the-fly in load_landscape
    },
    "t7_onehot": {
        "fitness_csv": "data/t7/t7_fitness.csv",
        "fitness_col": "label",
        "embeddings":  None,
    },
    "t7_esm2_15b": {
        "fitness_csv": "data/t7/t7_fitness.csv",
        "fitness_col": "label",
        "embeddings":  "data/t7/embeddings_esm2_15b_meanpool.npy",
    },
    "tev_onehot": {
        "fitness_csv": "data/tev/tev_fitness.csv",
        "fitness_col": "label",
        "embeddings":  None,
    },
    "tev_esm2_15b": {
        "fitness_csv": "data/tev/tev_fitness.csv",
        "fitness_col": "label",
        "embeddings":  "data/tev/embeddings_esm2_15b_meanpool.npy",
    },
    # ── New: ESM2-15B N-site ───────────────────────────────────────────────────
    "gb1_esm2_15b_4site": {
        "fitness_csv": "data/gb1/gb1_fitness.csv",
        "fitness_col": "label",
        "embeddings":  "data/gb1/embeddings_esm2_15b_4site.npy",
    },
    "trpb_esm2_15b_4site": {
        "fitness_csv": "data/trpb/trpb_fitness.csv",
        "fitness_col": "label",
        "embeddings":  "data/trpb/embeddings_esm2_15b_4site.npy",
    },
    "tev_esm2_15b_4site": {
        "fitness_csv": "data/tev/tev_fitness.csv",
        "fitness_col": "label",
        "embeddings":  "data/tev/embeddings_esm2_15b_4site.npy",
    },
    "t7_esm2_15b_3site": {
        "fitness_csv": "data/t7/t7_fitness.csv",
        "fitness_col": "label",
        "embeddings":  "data/t7/embeddings_esm2_15b_3site.npy",
    },
    # ── New: ESMc meanpool ────────────────────────────────────────────────────
    "gb1_esmc_mean": {
        "fitness_csv": "data/gb1/gb1_fitness.csv",
        "fitness_col": "label",
        "embeddings":  "data/gb1/embeddings_esmc600m_meanpool.npy",
    },
    "trpb_esmc_mean": {
        "fitness_csv": "data/trpb/trpb_fitness.csv",
        "fitness_col": "label",
        "embeddings":  "data/trpb/embeddings_esmc600m_meanpool.npy",
    },
    "tev_esmc": {
        "fitness_csv": "data/tev/tev_fitness.csv",
        "fitness_col": "label",
        "embeddings":  "data/tev/embeddings_esmc600m_4site.npy",
    },
    "tev_esmc_mean": {
        "fitness_csv": "data/tev/tev_fitness.csv",
        "fitness_col": "label",
        "embeddings":  "data/tev/embeddings_esmc600m_meanpool.npy",
    },
    "t7_esmc": {
        "fitness_csv": "data/t7/t7_fitness.csv",
        "fitness_col": "label",
        "embeddings":  "data/t7/embeddings_esmc600m_3site.npy",
    },
    "t7_esmc_mean": {
        "fitness_csv": "data/t7/t7_fitness.csv",
        "fitness_col": "label",
        "embeddings":  "data/t7/embeddings_esmc600m_meanpool.npy",
    },
}


# ── Filename metadata lookup tables ────────────────────────────────────────────
# Maps landscape → embedding tag used in output filenames
LANDSCAPE_EMB = {
    "gb1":               "esm650m_4site",
    "gb1_esm2":          "esm650m_4site",
    "gb1_esm2_mean":     "esm650m_4site_mean",
    "gb1_esm2_15b":      "esm15b_mean",
    "gb1_esm2_15b_c15b": "esm15b_mean",
    "gb1_esmc":          "esmc600m_4site",
    "gb1_onehot":        "onehot",
    "trpb":              "esm650m_4site",
    "trpb_esmc":         "esmc600m_4site",
    "trpb_esm2_15b":     "esm15b_mean",
    "trpb_onehot":       "onehot",
    "t7_onehot":            "onehot",
    "t7_esm2_15b":          "esm15b_mean",
    "t7_esm2_15b_3site":    "esm15b_3site",
    "t7_esmc":              "esmc600m_3site",
    "t7_esmc_mean":         "esmc600m_mean",
    "tev_onehot":           "onehot",
    "tev_esm2_15b":         "esm15b_mean",
    "tev_esm2_15b_4site":   "esm15b_4site",
    "tev_esmc":             "esmc600m_4site",
    "tev_esmc_mean":        "esmc600m_mean",
    "gb1_esm2_15b_4site":   "esm15b_4site",
    "gb1_esmc_mean":        "esmc600m_mean",
    "trpb_esm2_15b_4site":  "esm15b_4site",
    "trpb_esmc_mean":       "esmc600m_mean",
}

# Maps method → (model_tag, acquisition_tag) used in output filenames
METHOD_MODEL_ACQ = {
    "random":          ("random",  "random"),
    "evolvepro":       ("rfk100",  "greedy"),
    "rf_ts":           ("rfk1",    "ts"),
    "rf_ts_k1":        ("rfk1",    "ts"),
    "rf_ts_k5":        ("rfk5",    "ts"),
    "rf_ts_k10":       ("rfk10",   "ts"),
    "rf_ts_k20":       ("rfk20",   "ts"),
    "rf_ts_k50":       ("rfk50",   "ts"),
    "dnn_ts":          ("dnn500",  "ts"),
    "dnn_ts_s":        ("dnn256",  "ts"),
    "dnn_greedy":      ("dnn500",  "greedy"),
    "dnn_ei":          ("dnn500",  "ei"),
    "dnn_pi":          ("dnn500",  "pi"),
    "dnn_ucb":         ("dnn500",  "ucb"),
    "alde_dnn":        ("dnn30",   "ts"),
    "alde_dnn_greedy": ("dnn30",   "greedy"),
    "boes_ts":         ("boes",    "ts"),
    "boes_ei":         ("boes",    "ei"),
}

def _make_run_name(landscape, method, batch_size, zs_predictor, cluster_init, double_mut_init):
    """Return canonical filename stem: {dataset}_{emb}_{model}_{acq}_{init}_{bs}"""
    dataset = landscape.split("_")[0]   # gb1 or trpb
    emb     = LANDSCAPE_EMB.get(landscape, landscape)
    model, acq = METHOD_MODEL_ACQ.get(method, (method, "unknown"))

    if zs_predictor:
        init = f"zs_{zs_predictor}"
    elif cluster_init:
        _15b_landscapes = {"gb1_esm2_15b_c15b", "trpb_esm2_15b"}
        init = "clinit_15b" if landscape in _15b_landscapes else "clinit_esmc"
    elif double_mut_init:
        init = "dmzs"
    else:
        init = "random"

    if model == "random":
        return f"{dataset}_random_{batch_size}"
    return f"{dataset}_{emb}_{model}_{acq}_{init}_{batch_size}"

_ALL_AAS = list("ACDEFGHIKLMNPQRSTVWY")
_GB1_SITES  = [38, 39, 40, 53]      # 0-indexed
_TRPB_SITES = [182, 183, 226, 227]  # 0-indexed (1-indexed: 183,184,227,228)
_T7_SITES   = [747, 755, 757]       # 0-indexed (1-indexed: 748,756,758)
_TEV_SITES  = [145, 147, 166, 169]  # 0-indexed (1-indexed: 146,148,167,170)

# ZS predictor configs per landscape
ZS_CFG = {
    "gb1":          {"csv": "data/li2024/results/zs_comb/all/GB1.csv",   "sites": _GB1_SITES},
    "gb1_esm2":      {"csv": "data/li2024/results/zs_comb/all/GB1.csv",   "sites": _GB1_SITES},
    "gb1_esm2_mean": {"csv": "data/li2024/results/zs_comb/all/GB1.csv",   "sites": _GB1_SITES},
    "gb1_esm2_15b":      {"csv": "data/li2024/results/zs_comb/all/GB1.csv", "sites": _GB1_SITES},
    "gb1_esm2_15b_c15b": {"csv": "data/li2024/results/zs_comb/all/GB1.csv", "sites": _GB1_SITES},
    "gb1_esmc":     {"csv": "data/li2024/results/zs_comb/all/GB1.csv",   "sites": _GB1_SITES},
    "gb1_onehot":   {"csv": "data/li2024/results/zs_comb/all/GB1.csv",   "sites": _GB1_SITES},
    "trpb":         {"csv": "data/li2024/results/zs_comb/all/TrpB4.csv", "sites": _TRPB_SITES},
    "trpb_esmc":     {"csv": "data/li2024/results/zs_comb/all/TrpB4.csv", "sites": _TRPB_SITES},
    "trpb_esm2_15b": {"csv": "data/li2024/results/zs_comb/all/TrpB4.csv", "sites": _TRPB_SITES},
    "trpb_onehot":   {"csv": "data/li2024/results/zs_comb/all/TrpB4.csv", "sites": _TRPB_SITES},
    "t7_onehot":     {"csv": "data/li2024/results/zs_comb/all/T7.csv",    "sites": _T7_SITES},
    "t7_esm2_15b":   {"csv": "data/li2024/results/zs_comb/all/T7.csv",    "sites": _T7_SITES},
    "tev_onehot":    {"csv": "data/li2024/results/zs_comb/all/TEV.csv",   "sites": _TEV_SITES},
    "tev_esm2_15b":  {"csv": "data/li2024/results/zs_comb/all/TEV.csv",   "sites": _TEV_SITES},
}

ZS_PREDICTORS = ["esm_score", "esmif_score", "ev_score", "ev-esm-esmif_score", "ed_score"]

WT_AAS = {
    "gb1":          "VDGV",
    "gb1_esm2":      "VDGV",
    "gb1_esm2_mean": "VDGV",
    "gb1_esm2_15b":      "VDGV",
    "gb1_esm2_15b_c15b": "VDGV",
    "gb1_esmc":     "VDGV",
    "gb1_onehot":   "VDGV",
    "trpb":         "VFVS",
    "trpb_esmc":     "VFVS",
    "trpb_esm2_15b": "VFVS",
    "trpb_onehot":   "VFVS",
    "t7_onehot":     "NRQ",
    "t7_esm2_15b":   "NRQ",
    "tev_onehot":    "TDHS",
    "tev_esm2_15b":  "TDHS",
}

def load_double_mut_zs_init(landscape: str, n: int, seqs: list, rng: np.random.Generator) -> list:
    """Li et al 'ds-ev + double': sample n from <=2-mutation variants, proportional to ev ZS score."""
    import pandas as pd
    cfg  = ZS_CFG[landscape]
    sites = cfg["sites"]
    wt   = WT_AAS[landscape]

    # Build AA codes and compute hamming distance to WT
    our_aas = np.array(["".join(seq[s] for s in sites) for seq in seqs])
    dists   = np.array([sum(a != b for a, b in zip(aa, wt)) for aa in our_aas])
    pool    = np.where(dists <= 2)[0]

    # Get ev ZS scores for pool
    zs_df   = pd.read_csv(os.path.join(ROOT, cfg["csv"]))
    zs_map  = dict(zip(zs_df["AAs"], zs_df["ev_score"]))
    scores  = np.array([zs_map.get(our_aas[i], -np.inf) for i in pool], dtype=np.float64)

    # Softmax-proportional sampling (doubly sampled = proportional to ZS)
    scores  = scores - scores.max()
    weights = np.exp(scores)
    weights /= weights.sum()

    chosen = rng.choice(len(pool), size=min(n, len(pool)), replace=False, p=weights)
    return pool[chosen].tolist()


CLUSTER_LABEL_PATHS = {
    "gb1":           "data/gb1/cluster_labels_hdbscan_mcs500.npy",
    "gb1_esm2":      "data/gb1/cluster_labels_hdbscan_mcs500.npy",
    "gb1_esm2_mean": "data/gb1/cluster_labels_hdbscan_mcs500.npy",
    "gb1_esm2_15b":      "data/gb1/cluster_labels_hdbscan_mcs500.npy",         # ESMc clusters (better)
    "gb1_esm2_15b_c15b": "data/gb1/cluster_labels_esm2_15b_hdbscan_mcs500.npy", # 15B clusters
    "gb1_esmc":      "data/gb1/cluster_labels_hdbscan_mcs500.npy",
    "gb1_onehot":    "data/gb1/cluster_labels_hdbscan_mcs500.npy",
    "trpb":          "data/trpb/cluster_labels_hdbscan_mcs500.npy",
    "trpb_esmc":     "data/trpb/cluster_labels_hdbscan_mcs500.npy",
    "trpb_esm2_15b": "data/trpb/cluster_labels_esm2_15b_hdbscan_mcs500.npy",  # 15B clusters
    "trpb_onehot":   "data/trpb/cluster_labels_hdbscan_mcs500.npy",
    "t7_onehot":     "data/t7/cluster_labels_hdbscan_mcs50.npy",
    "t7_esm2_15b":   "data/t7/cluster_labels_esm2_15b_hdbscan_mcs50.npy",
    "tev_onehot":    "data/tev/cluster_labels_hdbscan_mcs500.npy",
    "tev_esm2_15b":  "data/tev/cluster_labels_esm2_15b_hdbscan_mcs500.npy",
}

def load_cluster_init(landscape: str, n: int, rng: np.random.Generator) -> list:
    """Sample n indices with equal allocation across HDBSCAN clusters."""
    path = os.path.join(ROOT, CLUSTER_LABEL_PATHS[landscape])
    labels = np.load(path)                         # -2=filtered, -1=noise, 0+=cluster
    cluster_ids = sorted(set(labels[labels >= 0]))
    n_clusters  = len(cluster_ids)
    per_cluster = n // n_clusters
    remainder   = n - per_cluster * n_clusters

    selected = []
    for c in cluster_ids:
        idx = np.where(labels == c)[0]
        k   = min(per_cluster, len(idx))
        selected.extend(rng.choice(idx, k, replace=False).tolist())

    # Fill remainder randomly from all clustered variants
    if remainder > 0:
        clustered = np.where(labels >= 0)[0]
        pool = [i for i in clustered if i not in set(selected)]
        selected.extend(rng.choice(pool, remainder, replace=False).tolist())

    return selected


def load_zs_init(landscape: str, predictor: str, seqs: list) -> np.ndarray:
    """Returns indices of top-INITIAL_N variants sorted by ZS predictor (descending)."""
    import pandas as pd
    cfg = ZS_CFG[landscape]
    zs = pd.read_csv(os.path.join(ROOT, cfg["csv"]))
    sites = cfg["sites"]
    # Build 4-AA code for each variant in our dataset
    our_aas = ["".join(seq[s] for s in sites) for seq in seqs]
    # Map ZS score by AAs code
    zs_map = dict(zip(zs["AAs"], zs[predictor]))
    scores = np.array([zs_map.get(aa, -np.inf) for aa in our_aas])
    # Return top-INITIAL_N indices (highest ZS score = most promising)
    return np.argsort(scores)[::-1][:INITIAL_N]

def _seqs_to_onehot(seqs, sites=_GB1_SITES):
    aa_idx = {aa: i for i, aa in enumerate(_ALL_AAS)}
    out = np.zeros((len(seqs), len(sites) * len(_ALL_AAS)), dtype=np.float32)
    for i, seq in enumerate(seqs):
        for j, site in enumerate(sites):
            aa = seq[site]
            if aa in aa_idx:
                out[i, j * len(_ALL_AAS) + aa_idx[aa]] = 1.0
    return out

def load_landscape(landscape: str) -> tuple:
    import pandas as pd
    cfg = LANDSCAPE_CFG[landscape]
    df = pd.read_csv(os.path.join(ROOT, cfg["fitness_csv"]))
    fitness = df[cfg["fitness_col"]].values.astype(np.float32)
    if cfg["embeddings"] is None:
        # onehot encoding computed on-the-fly
        sites = ZS_CFG[landscape]["sites"] if landscape in ZS_CFG else _GB1_SITES
        emb = _seqs_to_onehot(df["protein"].values, sites=sites)
    else:
        emb_path = os.path.join(ROOT, cfg["embeddings"])
        emb = np.load(emb_path) if emb_path.endswith(".npy") else np.load(emb_path)["embeddings"]
        if emb.dtype == np.float16:
            emb = emb.astype(np.float32)
    assert len(emb) == len(fitness)
    return emb, fitness


def make_method(method: str, seed: int):
    if method == "evolvepro":
        from methods.evolvepro import EVOLVEpro
        return EVOLVEpro(seed=seed)
    elif method == "rf_greedy":
        from methods.rf_variants import RandomForestOptimizer
        return RandomForestOptimizer(seed=seed, acquisition="greedy")
    elif method == "rf_ucb":
        from methods.rf_variants import RandomForestOptimizer
        return RandomForestOptimizer(seed=seed, acquisition="ucb", beta=2.0)
    elif method == "rf_ts" or method == "rf_ts_k1":
        from methods.rf_variants import RandomForestOptimizer
        return RandomForestOptimizer(seed=seed, acquisition="ts", ts_k=1)
    elif method == "rf_ts_k5":
        from methods.rf_variants import RandomForestOptimizer
        return RandomForestOptimizer(seed=seed, acquisition="ts", ts_k=5)
    elif method == "rf_ts_k10":
        from methods.rf_variants import RandomForestOptimizer
        return RandomForestOptimizer(seed=seed, acquisition="ts", ts_k=10)
    elif method == "rf_ts_k20":
        from methods.rf_variants import RandomForestOptimizer
        return RandomForestOptimizer(seed=seed, acquisition="ts", ts_k=20)
    elif method == "rf_ts_k50":
        from methods.rf_variants import RandomForestOptimizer
        return RandomForestOptimizer(seed=seed, acquisition="ts", ts_k=50)
    elif method == "rf_ts_k5_eps10":
        from methods.rf_variants import RandomForestOptimizer
        return RandomForestOptimizer(seed=seed, acquisition="ts", ts_k=5, epsilon=0.10)
    elif method == "rf_ts_k5_eps20":
        from methods.rf_variants import RandomForestOptimizer
        return RandomForestOptimizer(seed=seed, acquisition="ts", ts_k=5, epsilon=0.20)
    elif method == "rf_ts_k5_eps30":
        from methods.rf_variants import RandomForestOptimizer
        return RandomForestOptimizer(seed=seed, acquisition="ts", ts_k=5, epsilon=0.30)
    elif method == "boes_ei":
        from methods.boes import BOES
        return BOES(seed=seed, acquisition="ei")
    elif method == "boes_ts":
        from methods.boes import BOES
        return BOES(seed=seed, acquisition="ts")
    elif method == "mutation_stats":
        from methods.mutation_stats import MutationStats
        return MutationStats(seed=seed)
    elif method == "dnn_ts":
        # DNN ensemble [input→500→150→50→1] + Thompson Sampling (ESMc-L config)
        from methods.dnn_ensemble import DNNEnsembleOptimizer
        return DNNEnsembleOptimizer(seed=seed, hidden=[500, 150, 50], acquisition="ts")
    elif method == "dnn_ts_s":
        # DNN ensemble [input→256→128→1] + TS, 70% bootstrap per member (ESMc-S config)
        from methods.dnn_ensemble import DNNEnsembleOptimizer
        return DNNEnsembleOptimizer(seed=seed, hidden=[256, 128], acquisition="ts", bootstrap_size=0.7)
    elif method == "dnn_greedy":
        from methods.dnn_ensemble import DNNEnsembleOptimizer
        return DNNEnsembleOptimizer(seed=seed, hidden=[500, 150, 50], acquisition="greedy")
    elif method == "dnn_ei":
        from methods.dnn_ensemble import DNNEnsembleOptimizer
        return DNNEnsembleOptimizer(seed=seed, hidden=[500, 150, 50], acquisition="ei")
    elif method == "dnn_pi":
        from methods.dnn_ensemble import DNNEnsembleOptimizer
        return DNNEnsembleOptimizer(seed=seed, hidden=[500, 150, 50], acquisition="pi")
    elif method == "dnn_ucb":
        from methods.dnn_ensemble import DNNEnsembleOptimizer
        return DNNEnsembleOptimizer(seed=seed, hidden=[500, 150, 50], acquisition="ucb", beta=2.0)
    elif method == "alde_dnn":
        # Li 2024 DNN_ENSEMBLE: [input→30→30→1] + proper batch TS (resample per item)
        from methods.dnn_ensemble import DNNEnsembleOptimizer
        return DNNEnsembleOptimizer(seed=seed, hidden=[30, 30], acquisition="ts")
    elif method == "alde_dnn_greedy":
        # Li 2024 DNN_ENSEMBLE with GREEDY acquisition (best for TrpB)
        from methods.dnn_ensemble import DNNEnsembleOptimizer
        return DNNEnsembleOptimizer(seed=seed, hidden=[30, 30], acquisition="greedy")
    else:
        raise ValueError(f"Unknown method: {method}")


def run(landscape: str, method: str, batch_size: int, seed: int,
        zs_predictor: str = None, cluster_init: bool = False,
        double_mut_init: bool = False) -> dict:
    # Check before doing any computation
    run_name = _make_run_name(landscape, method, batch_size, zs_predictor, cluster_init, double_mut_init)
    out_path = os.path.join(ROOT, "results", "raw", f"{run_name}.jsonl")
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing_seeds = {json.loads(line)["seed"] for line in f}
        if seed in existing_seeds:
            print(f"  [{run_name}|seed={seed}] already done, skipping.")
            return None

    print(f"[{run_name}|seed={seed}]", flush=True)

    np.random.seed(seed)
    import pandas as pd
    df = pd.read_csv(os.path.join(ROOT, LANDSCAPE_CFG[landscape]["fitness_csv"]))
    emb, fitness = load_landscape(landscape)
    n_total = len(fitness)
    opt = make_method(method, seed)
    rng = np.random.default_rng(seed)

    # Initial sample: cluster-stratified, double-mut ZS, ZS-guided, or random
    if cluster_init:
        labeled_idx = load_cluster_init(landscape, INITIAL_N, rng)
    elif double_mut_init:
        labeled_idx = load_double_mut_zs_init(landscape, INITIAL_N, df["protein"].values, rng)
    elif zs_predictor:
        labeled_idx = list(load_zs_init(landscape, zs_predictor, df["protein"].values).tolist())
    else:
        labeled_idx = list(rng.choice(n_total, size=INITIAL_N, replace=False).tolist())
    remaining_idx = list(set(range(n_total)) - set(labeled_idx))

    global_max = fitness.max()
    t_total = 0.0
    while len(labeled_idx) < TOTAL_BUDGET:
        actual_batch = min(batch_size, TOTAL_BUDGET - len(labeled_idx))

        X_train = emb[labeled_idx]
        y_train = fitness[labeled_idx]
        y_max   = y_train.max()
        y_train_norm = y_train / y_max if y_max > 0 else y_train
        X_pool  = emb[remaining_idx]

        t0 = time.time()
        opt.train(X_train, y_train_norm)
        pool_sel = opt.select(X_pool, batch_size=actual_batch)
        t_total += time.time() - t0

        selected_global = [remaining_idx[i] for i in pool_sel]
        labeled_idx    += selected_global
        remaining_set   = set(selected_global)
        remaining_idx   = [i for i in remaining_idx if i not in remaining_set]

        current_max = fitness[labeled_idx].max()
        print(f"  n={len(labeled_idx):4d}  "
              f"max={current_max:.4f}  "
              f"t={t_total:.1f}s", flush=True)
        if current_max >= global_max:
            break

    record = {
        "landscape":       landscape,
        "method":          method,
        "batch_size":      batch_size,
        "seed":            seed,
        "selection_order": labeled_idx,   # length = TOTAL_BUDGET
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "a") as f:
        f.write(json.dumps(record) + "\n")

    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--landscape",  required=True, choices=list(LANDSCAPE_CFG.keys()))
    parser.add_argument("--method",     required=True,
                        choices=["evolvepro", "rf_greedy", "rf_ucb",
                                 "rf_ts", "rf_ts_k1", "rf_ts_k5", "rf_ts_k10", "rf_ts_k20",
                                 "boes_ei", "boes_ts", "mutation_stats",
                                 "dnn_ts", "dnn_ts_s", "dnn_greedy", "dnn_ucb", "alde_dnn"])
    parser.add_argument("--batch_size",    required=True, type=int)
    parser.add_argument("--seed",          required=True, type=int)
    parser.add_argument("--zs_predictor",  default=None, choices=ZS_PREDICTORS,
                        help="Zero-shot predictor for guided initialization (default: random)")
    parser.add_argument("--cluster_init",   action="store_true",
                        help="Use cluster-stratified initialization (requires precomputed labels)")
    parser.add_argument("--double_mut_init", action="store_true",
                        help="Li et al ds-ev: sample from <=2-mut variants proportional to ev ZS score")
    args = parser.parse_args()
    run(args.landscape, args.method, args.batch_size, args.seed,
        args.zs_predictor, args.cluster_init, args.double_mut_init)
