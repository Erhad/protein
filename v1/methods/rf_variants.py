"""
Random Forest surrogates with configurable acquisition functions.

Three variants, all sharing the same RF backbone (identical to EVOLVEpro):
  RandomForestOptimizer(acquisition="greedy")       — deterministic top-k (= EVOLVEpro)
  RandomForestOptimizer(acquisition="ucb")          — mean + beta * std_across_trees
  RandomForestOptimizer(acquisition="ts", ts_k=k)  — Thompson Sampling via sub-ensemble draw

Acquisition details
-------------------
Greedy
  scores = RF.predict(X_pool)
  select top-k by score

UCB
  mean   = RF.predict(X_pool)           # mean of all trees
  std    = std([t.predict(X_pool) for t in RF.estimators_], axis=0)
  scores = mean + beta * std
  select top-k by scores

TS (sub-ensemble)
  ts_k controls how many trees are averaged per draw:
    ts_k=1   — single tree (original, highest variance / most exploratory)
    ts_k=20  — 20-tree mean (matches ALDE's 1/5 ensemble ratio: 20% of 100 trees)
    ts_k=100 — full ensemble mean (collapses to greedy)

  For each position i in batch:
    draw ts_k trees without replacement from RF.estimators_
    scores_i = mean prediction of those ts_k trees on remaining pool
    select argmax(scores_i); remove from pool

  Per-step resampling gives diversity across the batch.
  Larger ts_k = smoother function draw = less noisy = closer to greedy.
"""

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from methods.base import Optimizer


class RandomForestOptimizer(Optimizer):
    """
    Parameters
    ----------
    seed         : RNG seed (passed to RF and numpy)
    acquisition  : 'greedy' | 'ucb' | 'ts'
    beta         : UCB exploration coefficient (default 2.0)
    n_estimators : number of trees in the forest (default 100)
    ts_k         : trees per sub-ensemble draw for TS (default 1; 20 matches ALDE 1/5 ratio)
    """

    def __init__(
        self,
        seed: int,
        acquisition: str = "greedy",
        beta: float = 2.0,
        n_estimators: int = 100,
        ts_k: int = 1,
    ):
        super().__init__(seed)
        assert acquisition in ("greedy", "ucb", "ts"), (
            f"acquisition must be 'greedy', 'ucb', or 'ts'; got '{acquisition}'"
        )
        self.acquisition = acquisition
        self.beta = beta
        self.ts_k = ts_k
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            criterion="friedman_mse",
            random_state=seed,
            n_jobs=-1,
        )
        self._fitted = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._fitted = True

    # ── Selection ─────────────────────────────────────────────────────────────

    def select(self, X_pool: np.ndarray, batch_size: int) -> np.ndarray:
        assert self._fitted, "Call train() before select()"
        X_scaled = self.scaler.transform(X_pool)

        if self.acquisition == "greedy":
            return self._select_greedy(X_scaled, batch_size)
        elif self.acquisition == "ucb":
            return self._select_ucb(X_scaled, batch_size)
        else:  # ts
            return self._select_ts(X_scaled, batch_size)

    def _select_greedy(self, X_scaled: np.ndarray, batch_size: int) -> np.ndarray:
        """Deterministic top-k by predicted mean. Identical to EVOLVEpro."""
        scores = self.model.predict(X_scaled)
        return np.argsort(scores)[-batch_size:][::-1]

    def _select_ucb(self, X_scaled: np.ndarray, batch_size: int) -> np.ndarray:
        """
        Upper Confidence Bound.
          score(x) = μ(x) + β · σ(x)
        where μ is the mean and σ is the std across all tree predictions.
        With β=0 this collapses to greedy; larger β trades exploitation for exploration.
        """
        tree_preds = np.vstack(
            Parallel(n_jobs=-1, prefer="threads")(
                delayed(tree.predict)(X_scaled) for tree in self.model.estimators_
            )
        )  # shape: (n_estimators, n_pool)
        mean = tree_preds.mean(axis=0)
        std  = tree_preds.std(axis=0)
        scores = mean + self.beta * std
        return np.argsort(scores)[-batch_size:][::-1]

    def _select_ts(self, X_scaled: np.ndarray, batch_size: int) -> np.ndarray:
        """
        Thompson Sampling via per-step sub-ensemble draw.

        At each step i in the batch:
          1. Sample ts_k trees without replacement from the ensemble.
          2. Average their predictions on the remaining pool (smoother function draw).
          3. Select the argmax; remove from the remaining pool.

        ts_k=1   → single tree (highest noise / most exploratory)
        ts_k=20  → matches ALDE's 1/5 ensemble ratio (20 of 100 trees)
        ts_k=100 → collapses to greedy (full ensemble mean)

        Per-step resampling gives diversity across the batch.
        """
        estimators = self.model.estimators_
        k = min(self.ts_k, len(estimators))

        # Precompute all tree predictions once — shape (n_estimators, n_pool).
        # Uses threads (not processes) since sklearn tree.predict releases the GIL.
        all_preds = np.vstack(
            Parallel(n_jobs=-1, prefer="threads")(
                delayed(tree.predict)(X_scaled) for tree in estimators
            )
        )  # (n_estimators, n_pool)

        mask = np.ones(len(X_scaled), dtype=bool)
        remaining = np.arange(len(X_scaled))
        selected  = []

        for _ in range(batch_size):
            if len(remaining) == 0:
                break
            idx    = self.rng.choice(len(estimators), size=k, replace=False)
            scores = all_preds[idx][:, remaining].mean(axis=0)
            best_local = int(np.argmax(scores))
            best_global = int(remaining[best_local])
            selected.append(best_global)
            mask[best_global] = False
            remaining = np.where(mask)[0]

        return np.array(selected)
