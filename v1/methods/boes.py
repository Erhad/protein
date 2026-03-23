"""
BOES — Bayesian Optimisation in Embedding Space.
Shen et al., BIBM 2024.

Surrogate  : Exact GP with Matérn 3/2 kernel (GPyTorch)
Acquisition: EI (paper baseline) or TS via Matheron's rule + RFF (thesis addition)

See docs/theory_boes.md for full derivations.
"""

import numpy as np
import torch
import gpytorch
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

from methods.base import Optimizer

# ── GP model definition ──────────────────────────────────────────────────────

class _ExactGP(gpytorch.models.ExactGP):
    def __init__(self, X, y, likelihood):
        super().__init__(X, y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=1.5)
        )

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x),
            self.covar_module(x),
        )


# ── Main class ───────────────────────────────────────────────────────────────

class BOES(Optimizer):
    """
    acquisition : 'ei'  — Expected Improvement (paper default)
                  'ts'  — Thompson Sampling via Matheron + RFF (thesis addition)
    n_restarts  : GP hyperparameter optimisation restarts (default 5, per thesis plan)
    n_rff       : Random Fourier Features for TS prior sampling (default 512)
    """

    def __init__(
        self,
        seed: int,
        acquisition: str = "ei",
        n_restarts: int = 5,
        n_rff: int = 512,
    ):
        super().__init__(seed)
        assert acquisition in ("ei", "ts"), "acquisition must be 'ei' or 'ts'"
        self.acquisition = acquisition
        self.n_restarts = n_restarts
        self.n_rff = n_rff

        self.x_scaler = StandardScaler()
        self.y_mean = 0.0
        self.y_std = 1.0

        # Set after train()
        self._gp = None
        self._likelihood = None
        self._X_train_t = None   # torch tensor, scaled
        self._y_train_np = None  # numpy, standardised
        self._chol_K = None      # cached Cholesky of (K + σ²I) for TS

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        # Standardise inputs
        X_scaled = self.x_scaler.fit_transform(X)
        self.y_mean, self.y_std = y.mean(), y.std() + 1e-8
        y_std = (y - self.y_mean) / self.y_std

        self._y_train_np = y_std
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        y_t = torch.tensor(y_std,    dtype=torch.float32)
        self._X_train_t = X_t

        best_loss, best_state = float("inf"), None

        for restart in range(self.n_restarts):
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            gp = _ExactGP(X_t, y_t, likelihood)

            # Random initialisation for restarts
            if restart > 0:
                torch.manual_seed(self.seed * 100 + restart)
                gp.covar_module.base_kernel.lengthscale = (
                    torch.rand(1) * 2.0 + 0.1
                )
                gp.covar_module.outputscale = torch.rand(1) * 2.0 + 0.1

            gp.train(); likelihood.train()
            optimiser = torch.optim.Adam(gp.parameters(), lr=0.1)
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp)

            for _ in range(100):
                optimiser.zero_grad()
                loss = -mll(gp(X_t), y_t)
                loss.backward()
                optimiser.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = (
                    {k: v.clone() for k, v in gp.state_dict().items()},
                    {k: v.clone() for k, v in likelihood.state_dict().items()},
                )

        # Load best hyperparameters
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        gp = _ExactGP(X_t, y_t, likelihood)
        gp.load_state_dict(best_state[0])
        likelihood.load_state_dict(best_state[1])
        gp.eval(); likelihood.eval()

        self._gp = gp
        self._likelihood = likelihood

    # ── Selection ─────────────────────────────────────────────────────────────

    def select(self, X_pool: np.ndarray, batch_size: int) -> np.ndarray:
        assert self._gp is not None, "Call train() before select()"
        X_scaled = self.x_scaler.transform(X_pool)

        if self.acquisition == "ei":
            return self._select_ei(X_scaled, batch_size)
        else:
            return self._select_ts(X_scaled, batch_size)

    def _select_ei(self, X_scaled: np.ndarray, batch_size: int) -> np.ndarray:
        """Closed-form EI. See docs/theory_boes.md §2."""
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self._likelihood(self._gp(X_t))
            mu  = pred.mean.numpy()
            std = pred.variance.sqrt().clamp(min=1e-6).numpy()

        f_star = self._y_train_np.max()  # incumbent in standardised space
        Z = (mu - f_star) / std
        ei = (mu - f_star) * norm.cdf(Z) + std * norm.pdf(Z)
        ei = np.where(std < 1e-6, np.maximum(mu - f_star, 0.0), ei)

        return np.argsort(ei)[-batch_size:][::-1]

    def _select_ts(self, X_scaled: np.ndarray, batch_size: int) -> np.ndarray:
        """
        Thompson Sampling via Matheron's rule + RFF.
        See docs/theory_boes.md §3.

        Each of the batch_size selections draws an independent function sample
        from the GP posterior and takes its argmax over the pool.
        """
        # Extract fitted GP hyperparameters
        ell   = self._gp.covar_module.base_kernel.lengthscale.item()
        alpha = self._gp.covar_module.outputscale.item() ** 0.5
        sigma2 = self._likelihood.noise.item()

        X_train = self._X_train_t.numpy()   # (n, d)
        y_train = self._y_train_np          # (n,)
        d = X_train.shape[1]

        # Precompute: K(X,X) + σ²I  Cholesky (reused across all TS draws)
        K_train = _matern32(X_train, X_train, ell, alpha)
        K_train += (sigma2 + 1e-6) * np.eye(len(X_train))
        L = np.linalg.cholesky(K_train)           # lower triangular

        selected = []
        remaining = np.arange(len(X_scaled))

        for _ in range(batch_size):
            if len(remaining) == 0:
                break

            X_rem = X_scaled[remaining]

            # --- Draw one prior sample via RFF (Matérn 3/2 spectral density) ---
            # Sample frequencies: ω = z * sqrt(3/u) / ℓ  (Student-t trick)
            # See theory_boes.md §3d
            rng = self.rng
            z = rng.standard_normal((self.n_rff, d))     # (D, d)
            u = rng.chisquare(df=3, size=self.n_rff)     # (D,)  df = 2ν = 3
            W = z * np.sqrt(3.0 / u[:, None]) / ell      # (D, d)
            b = rng.uniform(0, 2 * np.pi, self.n_rff)    # (D,)
            theta = rng.standard_normal(self.n_rff)       # (D,)

            def rff(Xb):
                return np.sqrt(2.0 / self.n_rff) * np.cos(Xb @ W.T + b)  # (N, D)

            phi_train = rff(X_train)    # (n_train, D)
            phi_rem   = rff(X_rem)      # (N_rem, D)

            f_prior_train = alpha * phi_train @ theta   # (n_train,)
            f_prior_rem   = alpha * phi_rem   @ theta   # (N_rem,)

            # --- Matheron update: deterministic shift to match observations ---
            residual = y_train - f_prior_train           # (n_train,)
            v = np.linalg.solve(L.T, np.linalg.solve(L, residual))  # K⁻¹ r

            K_rem_train = _matern32(X_rem, X_train, ell, alpha)  # (N_rem, n_train)
            f_star = f_prior_rem + K_rem_train @ v       # posterior sample scores

            best = remaining[np.argmax(f_star)]
            selected.append(best)
            remaining = remaining[remaining != best]

        return np.array(selected)


# ── Kernel helper ────────────────────────────────────────────────────────────

def _matern32(X1: np.ndarray, X2: np.ndarray, ell: float, alpha: float) -> np.ndarray:
    """Matérn 3/2 kernel matrix. k(x,x') = α²(1 + √3·d/ℓ)exp(−√3·d/ℓ)"""
    from sklearn.metrics import pairwise_distances
    D = pairwise_distances(X1, X2)
    s = np.sqrt(3.0) * D / ell
    return alpha ** 2 * (1.0 + s) * np.exp(-s)
