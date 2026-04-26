"""
DNN Ensemble optimizer for active learning.

Supports two configs used in experiments:
  - ALDE-style (onehot):  hidden=[30, 30],  acquisition='ts'  (Li 2024 exact)
  - ESMc-L:               hidden=[500, 150, 50], acquisition='ts'

Thompson Sampling (TS): pick one random ensemble member, rank pool by its predictions.
UCB: rank by mean + sqrt(beta) * std across ensemble members.

Architecture is inferred at train() time from X_train.shape[1]:
  full_arch = [input_dim] + hidden + [1]
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class _DNN(nn.Module):
    """Feedforward DNN with LeakyReLU activations, float64 weights."""

    def __init__(self, architecture: list[int]):
        super().__init__()
        layers = []
        for i in range(len(architecture) - 1):
            layers.append(nn.Linear(architecture[i], architecture[i + 1]).double())
            if i < len(architecture) - 2:
                layers.append(nn.LeakyReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class DNNEnsembleOptimizer:
    """
    DNN ensemble active learning optimizer.

    Parameters
    ----------
    seed        : random seed
    hidden      : hidden layer widths, e.g. [500, 150, 50] or [30, 30]
    n_ensemble  : number of independent models (default 5, matching ALDE)
    acquisition : 'ts' (Thompson Sampling) | 'greedy' | 'ucb'
    beta        : UCB exploration weight (only used if acquisition='ucb')
    lr          : Adam learning rate
    max_iter    : max training iterations per model
    early_stop_w: early stopping window (stop if recent min >= prior min)
    """

    def __init__(
        self,
        seed: int,
        hidden: list[int],
        n_ensemble: int = 5,
        acquisition: str = "ts",
        beta: float = 2.0,
        lr: float = 1e-3,
        max_iter: int = 300,
        early_stop_w: int = 30,
        bootstrap_size: float = 0.9,  # fraction of training data per member (ALDE default)
    ):
        self.seed = seed
        self.hidden = hidden
        self.n_ensemble = n_ensemble
        self.acquisition = acquisition.lower()
        self.beta = beta
        self.lr = lr
        self.max_iter = max_iter
        self.early_stop_w = early_stop_w
        self.bootstrap_size = bootstrap_size
        self._models: list[_DNN] = []
        self._rng = np.random.default_rng(seed)
        self._round = 0

    # ── Training ────────────────────────────────────────────────────────────

    def _train_one(self, model: _DNN, X_t: torch.Tensor, y_t: torch.Tensor) -> _DNN:
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        mse = nn.MSELoss()
        losses: list[float] = []
        model.train()
        for i in range(self.max_iter):
            opt.zero_grad()
            loss = mse(model(X_t), y_t)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if i > self.early_stop_w:
                w = self.early_stop_w
                if min(losses[i - w + 1:]) >= min(losses[:i - w + 1]):
                    break
        model.eval()
        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        arch = [X_train.shape[1]] + self.hidden + [1]
        n = len(X_train)
        k = max(1, int(n * self.bootstrap_size))
        self._models = []
        for i in range(self.n_ensemble):
            if self.bootstrap_size < 1.0:
                idx = self._rng.choice(n, size=k, replace=False)
                X_t = torch.tensor(X_train[idx]).double()
                y_t = torch.tensor(y_train[idx]).double()
            else:
                X_t = torch.tensor(X_train).double()
                y_t = torch.tensor(y_train).double()
            self._models.append(self._train_one(_DNN(arch), X_t, y_t))
        self._round += 1

    # ── Selection ───────────────────────────────────────────────────────────

    def _pool_preds(self, X_pool: np.ndarray) -> np.ndarray:
        """Returns shape (n_pool, n_ensemble) — raw predictions from each model."""
        X_t = torch.tensor(X_pool).double()
        with torch.no_grad():
            preds = np.stack([m(X_t).numpy() for m in self._models], axis=1)
        return preds  # (n_pool, n_ensemble)

    def select(self, X_pool: np.ndarray, batch_size: int) -> list[int]:
        preds = self._pool_preds(X_pool)   # (n_pool, n_ensemble)
        self._last_all_preds = preds.T.astype(np.float32)  # (n_ensemble, n_pool) — calibration reuse

        if self.acquisition == "ts":
            # Proper batch TS: resample ensemble member independently for each item.
            # For item i: pick random member j, take its argmax over remaining pool.
            selected = []
            available = np.ones(len(X_pool), dtype=bool)
            for _ in range(batch_size):
                col = np.random.randint(self.n_ensemble)
                scores = preds[:, col].copy()
                scores[~available] = -np.inf
                idx = int(np.argmax(scores))
                selected.append(idx)
                available[idx] = False
            return selected
        elif self.acquisition == "alde_ts":
            # ALDE exact: pick ONE ensemble member for the entire batch (Li 2024)
            col = np.random.randint(self.n_ensemble)
            scores = preds[:, col]
        elif self.acquisition == "greedy":
            scores = preds.mean(axis=1)
        elif self.acquisition == "ucb":
            mu = preds.mean(axis=1)
            sigma = preds.std(axis=1)
            scores = mu + np.sqrt(self.beta) * sigma
        elif self.acquisition == "ei":
            # Pseudo-EI: average improvement over current best across ensemble members.
            # Targets are normalized so best-in-training = 1.0.
            scores = np.maximum(preds - 1.0, 0).mean(axis=1)
        elif self.acquisition == "pi":
            # Probability of Improvement: fraction of ensemble members predicting above best.
            scores = (preds > 1.0).mean(axis=1)
        else:
            raise ValueError(f"Unknown acquisition: {self.acquisition}")

        top = np.argpartition(scores, -batch_size)[-batch_size:]
        return top[np.argsort(scores[top])[::-1]].tolist()

    # ── Uncertainty readout (for analysis) ──────────────────────────────────

    def predict_with_uncertainty(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (mean, std) across ensemble for each point in X."""
        preds = self._pool_preds(X)
        return preds.mean(axis=1), preds.std(axis=1)
