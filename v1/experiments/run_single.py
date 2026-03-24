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
    "trpb": {
        "fitness_csv": "data/trpb/trpb_fitness.csv",
        "fitness_col": "label",
        "embeddings":  "data/trpb/embeddings_esm2_650m_4site.npz",
    },
}


def load_landscape(landscape: str) -> tuple:
    import pandas as pd
    cfg = LANDSCAPE_CFG[landscape]
    fitness = pd.read_csv(
        os.path.join(ROOT, cfg["fitness_csv"])
    )[cfg["fitness_col"]].values.astype(np.float32)
    emb = np.load(os.path.join(ROOT, cfg["embeddings"]))["embeddings"]
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
    elif method == "boes_ei":
        from methods.boes import BOES
        return BOES(seed=seed, acquisition="ei")
    elif method == "boes_ts":
        from methods.boes import BOES
        return BOES(seed=seed, acquisition="ts")
    elif method == "mutation_stats":
        from methods.mutation_stats import MutationStats
        return MutationStats(seed=seed)
    else:
        raise ValueError(f"Unknown method: {method}")


def run(landscape: str, method: str, batch_size: int, seed: int) -> dict:
    # Check before doing any computation
    out_path = os.path.join(
        ROOT, "results", "raw",
        f"{landscape}_{method}_{batch_size}.jsonl"
    )
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing_seeds = {json.loads(line)["seed"] for line in f}
        if seed in existing_seeds:
            print(f"  [{landscape}|{method}|bs={batch_size}|seed={seed}] already done, skipping.")
            return None

    print(f"[{landscape}|{method}|bs={batch_size}|seed={seed}]", flush=True)

    np.random.seed(seed)
    emb, fitness = load_landscape(landscape)
    n_total = len(fitness)
    opt = make_method(method, seed)
    rng = np.random.default_rng(seed)

    # Initial random sample
    labeled_idx   = list(rng.choice(n_total, size=INITIAL_N, replace=False).tolist())
    remaining_idx = list(set(range(n_total)) - set(labeled_idx))

    t_total = 0.0
    while len(labeled_idx) < TOTAL_BUDGET:
        actual_batch = min(batch_size, TOTAL_BUDGET - len(labeled_idx))

        X_train = emb[labeled_idx]
        y_train = fitness[labeled_idx]
        X_pool  = emb[remaining_idx]

        t0 = time.time()
        opt.train(X_train, y_train)
        pool_sel = opt.select(X_pool, batch_size=actual_batch)
        t_total += time.time() - t0

        selected_global = [remaining_idx[i] for i in pool_sel]
        labeled_idx    += selected_global
        remaining_set   = set(selected_global)
        remaining_idx   = [i for i in remaining_idx if i not in remaining_set]

        print(f"  n={len(labeled_idx):4d}  "
              f"max={fitness[labeled_idx].max():.4f}  "
              f"t={t_total:.1f}s", flush=True)

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
    parser.add_argument("--landscape",  required=True, choices=list(LANDSCAPE_CFG))
    parser.add_argument("--method",     required=True,
                        choices=["evolvepro", "rf_greedy", "rf_ucb",
                                 "rf_ts", "rf_ts_k1", "rf_ts_k5", "rf_ts_k10", "rf_ts_k20",
                                 "boes_ei", "boes_ts", "mutation_stats"])
    parser.add_argument("--batch_size", required=True, type=int)
    parser.add_argument("--seed",       required=True, type=int)
    args = parser.parse_args()
    run(args.landscape, args.method, args.batch_size, args.seed)
