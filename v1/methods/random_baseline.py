"""
Random sampling baseline.

Selects variants uniformly at random from the unlabeled pool.
No model, no embeddings. Useful as lower bound on performance.
"""

import numpy as np
from methods.base import Optimizer


class RandomBaseline(Optimizer):

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        pass  # no model

    def select(self, X_pool: np.ndarray, batch_size: int) -> np.ndarray:
        n = len(X_pool)
        return self.rng.choice(n, size=min(batch_size, n), replace=False)
