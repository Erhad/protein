"""
Mutation Statistics baseline (archive method, adapted).

Scores each candidate as the sum of per-position mean fitness contributions,
with a UCB exploration bonus. No surrogate model or embeddings — operates
purely on which amino acids appear at the variable positions.

This is an additive (ProSAR-like) model: it assumes each mutation contributes
independently to fitness. Useful as an interpretable non-parametric baseline.

Only meaningful for fixed-position combinatorial landscapes (GB1, TrpB).
Not used on GFP (variable mutation positions, high-order epistasis expected).

Encoding expected: one-hot or integer index vectors of shape (n, n_sites).
Each value is an integer in [0, 20) identifying the amino acid at that site.
"""

import numpy as np
from collections import defaultdict
from methods.base import Optimizer


class MutationStats(Optimizer):
    """
    Parameters
    ----------
    beta : float
        UCB exploration weight. Higher = more exploration.
        Default 2.0 (standard UCB heuristic).
    """

    def __init__(self, seed: int, beta: float = 2.0):
        super().__init__(seed)
        self.beta = beta
        # stats[site][aa] = {'mean': float, 'M2': float, 'count': int}
        # Uses Welford's online algorithm for numerically stable running stats.
        self.stats: dict = defaultdict(lambda: defaultdict(
            lambda: {"mean": 0.0, "M2": 0.0, "count": 0}
        ))
        self._global_mean = 0.0
        self._global_M2 = 0.0
        self._global_count = 0

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        X : (n, n_sites) integer array — amino acid index at each variable site
        y : (n,) fitness values
        """
        # Full refit from scratch (no warm-starting, consistent with other methods)
        self.stats = defaultdict(lambda: defaultdict(
            lambda: {"mean": 0.0, "M2": 0.0, "count": 0}
        ))
        self._global_mean = 0.0
        self._global_M2 = 0.0
        self._global_count = 0

        for xi, yi in zip(X, y):
            self._update(xi, float(yi))

    def select(self, X_pool: np.ndarray, batch_size: int) -> np.ndarray:
        scores = np.array([self._score(x) for x in X_pool])
        return np.argsort(scores)[-batch_size:][::-1]

    # ── internals ────────────────────────────────────────────────────────────

    def _update(self, x: np.ndarray, y: float) -> None:
        """Welford online update for per-(site, aa) statistics."""
        for site, aa in enumerate(x):
            s = self.stats[site][int(aa)]
            s["count"] += 1
            delta = y - s["mean"]
            s["mean"] += delta / s["count"]
            s["M2"] += delta * (y - s["mean"])

        # Global stats (fallback for unseen mutations)
        self._global_count += 1
        delta = y - self._global_mean
        self._global_mean += delta / self._global_count
        self._global_M2 += delta * (y - self._global_mean)

    def _score(self, x: np.ndarray) -> float:
        """UCB score: sum over sites of (mean + beta * std)."""
        total = 0.0
        global_std = (
            (self._global_M2 / self._global_count) ** 0.5
            if self._global_count > 1 else 1.0
        )
        for site, aa in enumerate(x):
            s = self.stats[site][int(aa)]
            if s["count"] > 1:
                std = (s["M2"] / s["count"]) ** 0.5
                total += s["mean"] + self.beta * std
            elif s["count"] == 1:
                total += s["mean"] + self.beta * global_std
            else:
                # Unseen mutation: use global mean + full exploration bonus
                total += self._global_mean + self.beta * global_std
        return total
