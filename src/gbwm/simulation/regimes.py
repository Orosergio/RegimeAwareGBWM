"""Markov-switching market regimes and the combined market model.

The *true* data-generating process (ADR-003) is a hidden Markov chain over
regimes (bull / stable / high-vol / bear); each regime drives a GBM with its own
drift and covariance. The agent never observes the regime directly — it sees an
HMM belief estimated from returns (see ``gbwm.detection.hmm``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gbwm.config import MarketConfig
from gbwm.simulation.gbm import GBMSimulator


# --------------------------------------------------------------------------- #
@dataclass
class MarketPaths:
    """Container for a batch of simulated market trajectories."""

    regimes: np.ndarray  # (n_paths, n_steps) int — hidden regime active each step
    risky_returns: np.ndarray  # (n_paths, n_steps, n_assets) simple returns
    cash_return: float  # per-step simple risk-free return
    dt: float
    regime_names: list[str]
    asset_names: list[str]

    @property
    def n_paths(self) -> int:
        return self.regimes.shape[0]

    @property
    def n_steps(self) -> int:
        return self.regimes.shape[1]

    @property
    def n_assets(self) -> int:
        return self.risky_returns.shape[2]

    def portfolio_returns(self, risky_weights: np.ndarray) -> np.ndarray:
        """Per-step portfolio simple returns for given risky weights.

        ``risky_weights`` may be shape ``(n_assets,)``, ``(n_steps, n_assets)`` or
        ``(n_paths, n_steps, n_assets)``. The residual ``1 - sum(weights)`` is held
        in cash earning the risk-free rate.
        """
        w = np.asarray(risky_weights, dtype=float)
        if w.ndim == 1:
            w = np.broadcast_to(w, (self.n_paths, self.n_steps, self.n_assets))
        elif w.ndim == 2:
            w = np.broadcast_to(w, (self.n_paths, self.n_steps, self.n_assets))
        cash_w = 1.0 - w.sum(axis=-1)  # (P, S)
        risky_part = np.einsum("psa,psa->ps", w, self.risky_returns)
        return risky_part + cash_w * self.cash_return


# --------------------------------------------------------------------------- #
class RegimeSimulator:
    """Sample regime-index paths from a Markov transition matrix."""

    def __init__(self, transition: np.ndarray) -> None:
        self.P = np.asarray(transition, dtype=float)
        if self.P.ndim != 2 or self.P.shape[0] != self.P.shape[1]:
            raise ValueError("transition must be a square matrix")
        self.n_regimes = self.P.shape[0]
        self._cdf = np.cumsum(self.P, axis=1)

    def stationary_distribution(self) -> np.ndarray:
        """Left eigenvector of P for eigenvalue 1 (the long-run regime mix)."""
        vals, vecs = np.linalg.eig(self.P.T)
        idx = int(np.argmin(np.abs(vals - 1.0)))
        pi = np.real(vecs[:, idx])
        pi = np.clip(pi, 0.0, None)
        total = pi.sum()
        if total <= 0:  # pragma: no cover - degenerate fallback
            return np.full(self.n_regimes, 1.0 / self.n_regimes)
        return pi / total

    def simulate(
        self,
        n_paths: int,
        n_steps: int,
        rng: np.random.Generator,
        start: int | None = None,
    ) -> np.ndarray:
        """Return an int array ``(n_paths, n_steps)`` of regime indices.

        ``start`` fixes the initial regime; if ``None`` the first regime is drawn
        from the stationary distribution.
        """
        regimes = np.empty((n_paths, n_steps), dtype=np.int64)
        if start is None:
            pi = self.stationary_distribution()
            regimes[:, 0] = rng.choice(self.n_regimes, size=n_paths, p=pi)
        else:
            regimes[:, 0] = int(start)
        for t in range(1, n_steps):
            u = rng.random(n_paths)
            cur = regimes[:, t - 1]
            # vectorized categorical draw via per-row CDF
            regimes[:, t] = (u[:, None] < self._cdf[cur]).argmax(axis=1)
        return regimes


# --------------------------------------------------------------------------- #
class MarketModel:
    """Combines the regime process and the GBM return model.

    This is the simulation backbone used by both the fast Monte-Carlo evaluation
    harness and the Gymnasium environment.
    """

    def __init__(
        self,
        drift: np.ndarray,
        cov: np.ndarray,
        transition: np.ndarray,
        dt: float,
        risk_free_rate: float,
        regime_names: list[str],
        asset_names: list[str],
    ) -> None:
        self.gbm = GBMSimulator(drift, cov, dt)
        self.regime_sim = RegimeSimulator(transition)
        self.dt = float(dt)
        self.risk_free_rate = float(risk_free_rate)
        self.regime_names = list(regime_names)
        self.asset_names = list(asset_names)
        if len(self.regime_names) != self.gbm.n_regimes:
            raise ValueError("regime_names length mismatch")
        if len(self.asset_names) != self.gbm.n_assets:
            raise ValueError("asset_names length mismatch")

    @property
    def n_regimes(self) -> int:
        return self.gbm.n_regimes

    @property
    def n_assets(self) -> int:
        return self.gbm.n_assets

    @property
    def cash_return(self) -> float:
        """Per-step simple risk-free return."""
        return float(np.expm1(self.risk_free_rate * self.dt))

    @classmethod
    def from_config(cls, market: MarketConfig) -> MarketModel:
        return cls(
            drift=market.drift_matrix(),
            cov=market.covariances(),
            transition=market.transition_matrix(),
            dt=1.0 / market.steps_per_year,
            risk_free_rate=market.risk_free_rate,
            regime_names=market.regime_names,
            asset_names=market.risky_assets,
        )

    def regime_index(self, name: str) -> int:
        return self.regime_names.index(name)

    def simulate(
        self,
        n_paths: int,
        n_steps: int,
        rng: np.random.Generator,
        antithetic: bool = False,
        start_regime: int | str | None = None,
    ) -> MarketPaths:
        """Simulate ``n_paths`` trajectories of ``n_steps`` steps each."""
        if isinstance(start_regime, str):
            start_regime = self.regime_index(start_regime)

        if antithetic:
            n_pairs = (n_paths + 1) // 2
            reg_half = self.regime_sim.simulate(n_pairs, n_steps, rng, start=start_regime)
            regimes = np.concatenate([reg_half, reg_half], axis=0)[:n_paths]
            z_half = rng.standard_normal((n_pairs, n_steps, self.n_assets))
            z = np.concatenate([z_half, -z_half], axis=0)[:n_paths]
            risky = self.gbm.sample_simple_returns(regimes, z=z)
        else:
            regimes = self.regime_sim.simulate(n_paths, n_steps, rng, start=start_regime)
            risky = self.gbm.sample_simple_returns(regimes, rng=rng)

        return MarketPaths(
            regimes=regimes,
            risky_returns=risky,
            cash_return=self.cash_return,
            dt=self.dt,
            regime_names=self.regime_names,
            asset_names=self.asset_names,
        )
