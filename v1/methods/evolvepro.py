"""
EVOLVEpro — Jiang et al., Science 2024.

Surrogate : RandomForestRegressor (n_estimators=100)
Encoding  : ESM2-650M mean-pooled (1280-dim) — passed in as X
Acquisition: Greedy top-k by predicted mean. No uncertainty used.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from methods.base import Optimizer


class EVOLVEpro(Optimizer):

    def __init__(self, seed: int, n_estimators: int = 100):
        super().__init__(seed)
        self.n_estimators = n_estimators
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            criterion="friedman_mse",
            random_state=seed,
            n_jobs=-1,
        )
        self._fitted = False

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self._fitted = True

    def select(self, X_pool: np.ndarray, batch_size: int) -> np.ndarray:
        assert self._fitted, "Call train() before select()"
        X_scaled = self.scaler.transform(X_pool)
        y_pred = self.model.predict(X_scaled)
        # Greedy top-k — no diversity, no uncertainty
        top_indices = np.argsort(y_pred)[-batch_size:][::-1]
        return top_indices
