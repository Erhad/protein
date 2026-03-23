"""
Random Forest surrogates with configurable acquisition functions.

Three variants, all sharing the same RF backbone (identical to EVOLVEpro):
  RandomForestOptimizer(acquisition="greedy")  — deterministic top-k (= EVOLVEpro)
  RandomForestOptimizer(acquisition="ucb")     — mean + beta * std_across_trees
  RandomForestOptimizer(acquisition="ts")      — Thompson Sampling via random tree draw

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

TS
  For each position i in batch:
    tree_i ~ Uniform(RF.estimators_)    # sample a random tree
    scores_i = tree_i.predict(X_pool[remaining])
    select argmax(scores_i); remove from pool
  This is the RF analogue of Bayesian TS: each "function draw" is one tree from
  the ensemble, which is a leaf-constant interpolation of the training data.
  Per-step resampling gives O(batch_size) diversity.
"""

import numpy as np
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
    """

    def __init__(
        self,
        seed: int,
        acquisition: str = "greedy",
        beta: float = 2.0,
        n_estimators: int = 100,
    ):
        super().__init__(seed)
        assert acquisition in ("greedy", "ucb", "ts"), (
            f"acquisition must be 'greedy', 'ucb', or 'ts'; got '{acquisition}'"
        )
        self.acquisition = acquisition
        self.beta = beta
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
        tree_preds = np.array(
            [tree.predict(X_scaled) for tree in self.model.estimators_]
        )  # shape: (n_estimators, n_pool)
        mean = tree_preds.mean(axis=0)
        std  = tree_preds.std(axis=0)
        scores = mean + self.beta * std
        return np.argsort(scores)[-batch_size:][::-1]

    def _select_ts(self, X_scaled: np.ndarray, batch_size: int) -> np.ndarray:
        """
        Thompson Sampling via per-step random tree draw.

        At each step i in the batch:
          1. Sample one tree uniformly from the ensemble.
          2. Evaluate that tree on the remaining pool candidates.
          3. Select the argmax; remove from the remaining pool.

        This gives batch_size independent "function draws" with natural diversity
        — different trees can disagree, so the batch explores multiple modes.
        Unlike greedy, the same high-scoring cluster cannot dominate the entire batch.
        """
        remaining = np.arange(len(X_scaled))
        selected  = []
        estimators = self.model.estimators_

        for _ in range(batch_size):
            if len(remaining) == 0:
                break
            # Draw one tree — independent per step
            tree = self.rng.choice(estimators)
            scores = tree.predict(X_scaled[remaining])
            best_local = int(np.argmax(scores))
            selected.append(int(remaining[best_local]))
            remaining = np.delete(remaining, best_local)

        return np.array(selected)
