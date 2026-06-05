"""True multi-asset allocators: split capital across S&P / international / bonds
/ gold *at once*, in real time, according to the market regime.

The tabular G-Learner only controls a single **risky fraction** and splits it
equally across assets (a risk-level controller — see :mod:`gbwm.policies.g_learner`).
That can never rotate *into* bonds/gold when stocks fall. This module adds the
missing brain:

:class:`RegimeAwareMultiAsset`
    Two decisions, composed:

    1. **How much risk overall** — the regime-aware G-Learner risk-budget DP
       ``f_t`` (a function of wealth, time-left and the regime belief). This is
       the goal-based part: take more risk when behind, protect gains when ahead.
    2. **How to split that risk across assets** — a *regime-conditioned,
       long-only mean-variance* tilt ``c_k`` per regime (maximize
       ``mu_k·w - (lambda/2) w'Sigma_k w`` s.t. ``w>=0, sum w=1, w<=cap``),
       blended at decision time by the belief: ``c_t = belief · c_k``.

    Final weights ``= f_t · c_t`` (cash ``= 1 - f_t``). In a bull regime the tilt
    concentrates in equities; in bear/high-vol it rotates into bonds and gold —
    exactly the behavior nobody hand-coded.

:class:`StaticMultiAsset`
    A fixed multi-asset mix (e.g. a classic balanced portfolio, or all-in one
    asset) — the honest non-learning baselines to compare against.

Both are analytic and solve in milliseconds, so they run live in the web app and
re-solve cheaply inside the walk-forward causal recalibration.
"""

from __future__ import annotations

import numpy as np

from gbwm.config import Config
from gbwm.policies.base import DecisionContext, Policy, policy_registry
from gbwm.policies.g_learner import RegimeAwareGLearner
from gbwm.simulation.regimes import MarketModel


# --------------------------------------------------------------------------- #
# Long-only capped-simplex mean-variance (tiny QP, pure NumPy — no scipy dep)
# --------------------------------------------------------------------------- #
def _project_capped_simplex(v: np.ndarray, cap: float, total: float = 1.0,
                            iters: int = 60) -> np.ndarray:
    """Euclidean projection of ``v`` onto ``{0 <= w <= cap, sum w = total}``.

    Solved by bisection on the dual offset ``theta`` (the map
    ``theta -> sum(clip(v-theta,0,cap))`` is monotone decreasing). Requires
    ``cap * len(v) >= total`` for feasibility.
    """
    lo, hi = float(v.min() - cap), float(v.max())
    for _ in range(iters):
        theta = 0.5 * (lo + hi)
        if np.clip(v - theta, 0.0, cap).sum() > total:
            lo = theta
        else:
            hi = theta
    return np.clip(v - 0.5 * (lo + hi), 0.0, cap)


def mean_variance_weights(mu: np.ndarray, sigma: np.ndarray, lam: float,
                          cap: float = 1.0, iters: int = 500) -> np.ndarray:
    """Long-only max-utility weights: argmax ``mu·w - (lam/2) w'Sigma w``.

    Constraints ``w >= 0``, ``sum w = 1``, ``w <= cap``. Projected-gradient
    ascent on a concave quadratic, so it converges to the global optimum.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    A = mu.shape[0]
    cap = max(cap, 1.0 / A)  # keep the feasible set non-empty
    w = _project_capped_simplex(np.full(A, 1.0 / A), cap)
    lmax = float(np.linalg.eigvalsh(sigma).max())
    step = 1.0 / (lam * lmax + 1e-9)
    for _ in range(iters):
        grad = mu - lam * (sigma @ w)
        w = _project_capped_simplex(w + step * grad, cap)
    return w


# --------------------------------------------------------------------------- #
# Fixed multi-asset mixes (baselines)
# --------------------------------------------------------------------------- #
class StaticMultiAsset(Policy):
    """Hold a constant weight vector across the risky assets (cash = 1 - sum)."""

    name = "Static Multi-Asset"

    def __init__(self, weights, name: str | None = None) -> None:
        self.w = np.asarray(weights, dtype=float)
        if name:
            self.name = name

    def weights(self, ctx: DecisionContext) -> np.ndarray:
        if self.w.shape[0] != ctx.n_assets:
            raise ValueError(f"{self.name}: {self.w.shape[0]} weights for {ctx.n_assets} assets")
        return np.tile(self.w, (len(ctx.wealth), 1))


# --------------------------------------------------------------------------- #
# The regime-aware multi-asset allocator
# --------------------------------------------------------------------------- #
@policy_registry.register("regime_aware_multi_asset")
class RegimeAwareMultiAsset(RegimeAwareGLearner):
    """Goal-based risk budget x regime-conditioned cross-asset mean-variance tilt."""

    name = "Regime-Aware Multi-Asset"
    requires_belief = True

    def __init__(
        self,
        config: Config,
        market_model: MarketModel | None = None,
        *,
        cross_asset_risk_aversion: float = 8.0,
        max_weight: float = 0.7,
        **base_kwargs,
    ) -> None:
        self.cross_lambda = float(cross_asset_risk_aversion)
        self.max_weight = float(max_weight)
        # Build the DP machinery but defer the backward induction: we first need
        # the per-regime cross-asset weights so the risk-budget DP sizes the
        # *actual* basket it will hold (not an equal-weight one).
        base_kwargs["solve"] = False
        super().__init__(config, market_model, **base_kwargs)

        self.cross_weights = self._solve_cross_asset_weights()  # (K, A)
        cov_log = self.mm.gbm.cov * self.mm.dt  # (K, A, A) per-step log cov
        self.reg_mean_log = np.einsum("ka,ka->k", self.mm.gbm.mean_log, self.cross_weights)
        self.reg_sd_log = np.sqrt(np.clip(
            np.einsum("ka,kab,kb->k", self.cross_weights, cov_log, self.cross_weights),
            1e-16, None))
        self.solve()  # regime-conditioned risk-budget DP (RegimeAwareGLearner.solve)

    def _solve_cross_asset_weights(self) -> np.ndarray:
        """Per-regime long-only mean-variance weights over the risky assets."""
        mu = self.mm.gbm.drift          # (K, A) annualized drift
        cov = self.mm.gbm.cov           # (K, A, A) annualized covariance
        K, A = mu.shape
        out = np.empty((K, A))
        for k in range(K):
            out[k] = mean_variance_weights(mu[k], cov[k], self.cross_lambda, self.max_weight)
        return out

    def cross_asset_weights(self, belief: np.ndarray) -> np.ndarray:
        """Belief-blended cross-asset split, shape ``(N, A)`` summing to 1."""
        c = np.asarray(belief) @ self.cross_weights
        return c / np.clip(c.sum(axis=1, keepdims=True), 1e-12, None)

    def weights(self, ctx: DecisionContext) -> np.ndarray:
        t = min(ctx.step, self.T - 1)
        per_regime_frac = self.mean_action[t, self._bin(ctx.wealth)]  # (N, K)
        risk_budget = np.einsum("nk,nk->n", ctx.belief, per_regime_frac)  # (N,) f_t in [0,1]
        split = self.cross_asset_weights(ctx.belief)  # (N, A)
        return risk_budget[:, None] * split

    @classmethod
    def _kwargs_from_config(cls, config: Config) -> dict:
        kw = super()._kwargs_from_config(config)
        ma = config.agents.extra.get("multi_asset", {}) if config.agents.extra else {}
        kw["cross_asset_risk_aversion"] = float(ma.get("cross_asset_risk_aversion", 8.0))
        kw["max_weight"] = float(ma.get("max_weight_per_asset", 0.7))
        return kw

    @classmethod
    def from_config(cls, config: Config, market_model: MarketModel | None = None
                    ) -> RegimeAwareMultiAsset:
        return cls(config, market_model, **cls._kwargs_from_config(config))
