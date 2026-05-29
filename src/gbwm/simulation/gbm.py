"""Vectorized multi-asset Geometric Brownian Motion (GBM).

Given a *regime label* for every (path, step), draw correlated lognormal asset
returns. The per-regime parameters (annualized drift vector and covariance
matrix) come from :class:`gbwm.config.MarketConfig`; here we only do the math.

Discretization
--------------
Over a step of length ``dt`` years, an asset's log-return is Gaussian::

    log(1 + R) ~ Normal( (mu - 0.5 * diag(Sigma)) * dt ,  Sigma * dt )

with assets correlated through the regime covariance ``Sigma``. Simple returns
are ``expm1`` of the log-returns, which keeps wealth strictly positive.
"""

from __future__ import annotations

import numpy as np


class GBMSimulator:
    """Sample correlated GBM returns conditional on a regime path.

    Parameters
    ----------
    drift : array (K, A)
        Annualized drift per regime ``K`` and risky asset ``A``.
    cov : array (K, A, A)
        Annualized covariance matrix per regime.
    dt : float
        Step length in years (e.g. 1/12 for monthly).
    """

    def __init__(self, drift: np.ndarray, cov: np.ndarray, dt: float) -> None:
        self.drift = np.asarray(drift, dtype=float)
        self.cov = np.asarray(cov, dtype=float)
        self.dt = float(dt)
        if self.drift.ndim != 2:
            raise ValueError("drift must be (n_regimes, n_assets)")
        self.n_regimes, self.n_assets = self.drift.shape
        if self.cov.shape != (self.n_regimes, self.n_assets, self.n_assets):
            raise ValueError("cov must be (n_regimes, n_assets, n_assets)")
        # Per-regime log-return mean and Cholesky factor of the per-step cov.
        diag = np.diagonal(self.cov, axis1=1, axis2=2)  # (K, A)
        self.mean_log = (self.drift - 0.5 * diag) * self.dt  # (K, A)
        self.chol = np.linalg.cholesky(self.cov * self.dt)  # (K, A, A)

    # ------------------------------------------------------------------ #
    def sample_log_returns(
        self,
        regimes: np.ndarray,
        rng: np.random.Generator | None = None,
        z: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return log-returns of shape ``(n_paths, n_steps, n_assets)``.

        ``z`` lets the caller supply standard-normal shocks (used for antithetic
        variates); otherwise they are drawn from ``rng``.
        """
        regimes = np.asarray(regimes)
        if regimes.ndim != 2:
            raise ValueError("regimes must be (n_paths, n_steps)")
        n_paths, n_steps = regimes.shape
        if z is None:
            if rng is None:
                raise ValueError("provide either rng or z")
            z = rng.standard_normal((n_paths, n_steps, self.n_assets))
        elif z.shape != (n_paths, n_steps, self.n_assets):
            raise ValueError("z has the wrong shape")
        mean = self.mean_log[regimes]  # (P, S, A)
        chol = self.chol[regimes]  # (P, S, A, A)
        shocks = np.einsum("psij,psj->psi", chol, z)
        return mean + shocks

    def sample_simple_returns(
        self,
        regimes: np.ndarray,
        rng: np.random.Generator | None = None,
        z: np.ndarray | None = None,
    ) -> np.ndarray:
        """Simple (arithmetic) returns ``exp(log_return) - 1``."""
        return np.expm1(self.sample_log_returns(regimes, rng=rng, z=z))
