"""
Benchmark and correctness check for TS selection optimizations.

Usage:
    python bench_ts.py
"""

import time
import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


# ── Two implementations of _select_ts ────────────────────────────────────────

def select_ts_original(all_preds, remaining_start, batch_size, k, rng):
    """Current implementation (non-vectorized mean)."""
    mask = np.ones(all_preds.shape[1], dtype=bool)
    remaining = np.arange(all_preds.shape[1])
    selected = []
    n_estimators = all_preds.shape[0]

    for _ in range(batch_size):
        if len(remaining) == 0:
            break
        idx = rng.choice(n_estimators, size=k, replace=False)
        scores = all_preds[idx][:, remaining].mean(axis=0)
        best_local = int(np.argmax(scores))
        best_global = int(remaining[best_local])
        selected.append(best_global)
        mask[best_global] = False
        remaining = np.where(mask)[0]

    return np.array(selected)


def select_ts_vectorized(all_preds, remaining_start, batch_size, k, rng):
    """Vectorized: precompute all score vectors, trivial loop."""
    n_estimators = all_preds.shape[0]

    # Pre-sample all tree subsets (per-row without replacement)
    idx_matrix = np.vstack([
        rng.choice(n_estimators, size=k, replace=False)
        for _ in range(batch_size)
    ])  # (batch_size, k)

    # Single op: (batch_size, k, n_pool) → mean → (batch_size, n_pool)
    score_matrix = all_preds[idx_matrix].mean(axis=1)

    remaining = np.arange(all_preds.shape[1])
    selected = []

    for i in range(batch_size):
        if len(remaining) == 0:
            break
        best_local = int(np.argmax(score_matrix[i][remaining]))
        best_global = int(remaining[best_local])
        selected.append(best_global)
        remaining = np.delete(remaining, best_local)

    return np.array(selected)


# ── Setup ─────────────────────────────────────────────────────────────────────

import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

print("Loading GB1 data...", flush=True)
import pandas as pd
fitness = pd.read_csv(os.path.join(ROOT, "data/gb1/gb1_fitness.csv"))["label"].values.astype(np.float32)
emb     = np.load(os.path.join(ROOT, "data/gb1/embeddings_esm2_650m_4site.npz"))["embeddings"]

SEED       = 42
INITIAL_N  = 96
BATCH_SIZE = 96
N_EST      = 100
TS_K       = 20

rng = np.random.default_rng(SEED)
labeled_idx   = rng.choice(len(fitness), size=INITIAL_N, replace=False).tolist()
remaining_idx = list(set(range(len(fitness))) - set(labeled_idx))

X_train = emb[labeled_idx]
y_train = fitness[labeled_idx]
X_pool  = emb[remaining_idx]

print(f"Pool size: {len(remaining_idx)}  batch_size: {BATCH_SIZE}  ts_k: {TS_K}", flush=True)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_pool_s  = scaler.transform(X_pool)

rf = RandomForestRegressor(n_estimators=N_EST, criterion="friedman_mse",
                            random_state=SEED, n_jobs=-1)
rf.fit(X_train_s, y_train)

print("Precomputing all_preds...", flush=True)
t0 = time.time()
all_preds = np.vstack(
    Parallel(n_jobs=-1, prefer="threads")(
        delayed(tree.predict)(X_pool_s) for tree in rf.estimators_
    )
)
print(f"  all_preds shape: {all_preds.shape}  ({time.time()-t0:.2f}s)\n", flush=True)


# ── Correctness check ─────────────────────────────────────────────────────────

rng_a = np.random.default_rng(SEED)
rng_b = np.random.default_rng(SEED)

sel_orig = select_ts_original(all_preds, None, BATCH_SIZE, TS_K, rng_a)
sel_vec  = select_ts_vectorized(all_preds, None, BATCH_SIZE, TS_K, rng_b)

assert np.array_equal(sel_orig, sel_vec), (
    f"MISMATCH!\n  orig: {sel_orig}\n  vec:  {sel_vec}"
)
print("✓ Correctness check passed — selections are identical\n")


# ── Timing ────────────────────────────────────────────────────────────────────

N_RUNS = 5

def time_fn(fn, n_runs):
    times = []
    for _ in range(n_runs):
        rng_t = np.random.default_rng(SEED)
        t0 = time.time()
        fn(all_preds, None, BATCH_SIZE, TS_K, rng_t)
        times.append(time.time() - t0)
    return times

print(f"Timing {N_RUNS} runs each (batch_size={BATCH_SIZE}, ts_k={TS_K})...")
t_orig = time_fn(select_ts_original, N_RUNS)
t_vec  = time_fn(select_ts_vectorized, N_RUNS)

print(f"  original:    {np.mean(t_orig)*1000:.1f}ms  (±{np.std(t_orig)*1000:.1f}ms)")
print(f"  vectorized:  {np.mean(t_vec)*1000:.1f}ms  (±{np.std(t_vec)*1000:.1f}ms)")
print(f"  speedup:     {np.mean(t_orig)/np.mean(t_vec):.2f}x")
