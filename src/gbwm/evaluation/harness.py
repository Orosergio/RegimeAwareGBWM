"""Vectorized Monte-Carlo evaluation harness.

Rolls any :class:`~gbwm.policies.base.Policy` through the market over many paths
at once, threading the online regime belief (true-parameter Bayes filter) so
regime-aware policies get a realistic posterior. :func:`compare_policies` reuses
the *same* simulated paths for every policy, so differences reflect the policy,
not the draw.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gbwm.config import Config
from gbwm.detection.filter import GaussianRegimeFilter
from gbwm.evaluation.metrics import PolicyResult, evaluate
from gbwm.policies.base import DecisionContext, Policy
from gbwm.simulation.regimes import MarketModel, MarketPaths
from gbwm.utils.seeding import make_rng


def run_policy(
    policy: Policy,
    config: Config,
    n_episodes: int | None = None,
    rng: np.random.Generator | None = None,
    market_model: MarketModel | None = None,
    market_paths: MarketPaths | None = None,
) -> PolicyResult:
    mm = market_model or MarketModel.from_config(config.market)
    T, A = config.total_steps, mm.n_assets
    G, c, W0 = config.goal.target_wealth, config.goal.contribution, config.goal.initial_wealth
    rng = rng or make_rng(config.seed)
    if market_paths is None:
        n = n_episodes or config.simulation.n_episodes
        market_paths = mm.simulate(n, T, rng, antithetic=config.simulation.antithetic)
    N = market_paths.n_paths

    filt = GaussianRegimeFilter(mm.gbm.mean_log, mm.gbm.cov * mm.dt, mm.regime_sim.P)
    alpha = np.tile(filt.prior, (N, 1))
    wealth = np.full(N, float(W0))

    wealth_hist = np.empty((N, T + 1))
    wealth_hist[:, 0] = wealth
    weights_hist = np.empty((N, T, A))
    belief_hist = np.empty((N, T, mm.n_regimes))

    policy.reset(N)
    for t in range(T):
        belief = filt.predict_batch(alpha)  # decision-time belief
        ctx = DecisionContext(
            step=t,
            n_steps=T,
            wealth=wealth,
            target=G,
            belief=belief,
            n_assets=A,
            regime_names=mm.regime_names,
        )
        w = policy.weights(ctx)
        r = market_paths.risky_returns[:, t, :]
        port_ret = (w * r).sum(axis=1) + (1.0 - w.sum(axis=1)) * mm.cash_return
        wealth = (wealth + c) * (1.0 + port_ret)

        weights_hist[:, t, :] = w
        belief_hist[:, t, :] = belief
        wealth_hist[:, t + 1] = wealth
        alpha = filt.update_batch(alpha, np.log1p(r))

    histories = {
        "wealth": wealth_hist,
        "weights": weights_hist,
        "belief": belief_hist,
        "regime": market_paths.regimes,
        "asset_names": mm.asset_names,
        "regime_names": mm.regime_names,
    }
    return evaluate(policy.name, wealth, G, histories, mm.regime_names)


def compare_policies(
    policies: dict[str, Policy],
    config: Config,
    n_episodes: int | None = None,
    rng: np.random.Generator | None = None,
    market_model: MarketModel | None = None,
) -> dict[str, PolicyResult]:
    """Evaluate several policies on a shared set of Monte-Carlo paths."""
    mm = market_model or MarketModel.from_config(config.market)
    rng = rng or make_rng(config.seed)
    n = n_episodes or config.simulation.n_episodes
    paths = mm.simulate(n, config.total_steps, rng, antithetic=config.simulation.antithetic)
    return {
        name: run_policy(pol, config, market_model=mm, market_paths=paths)
        for name, pol in policies.items()
    }


def results_table(results: dict[str, PolicyResult]) -> pd.DataFrame:
    df = pd.DataFrame([r.summary_row() for r in results.values()])
    return df.sort_values("P(goal)", ascending=False).reset_index(drop=True)


def run_on_returns(
    policy: Policy,
    config: Config,
    risky_returns: np.ndarray,
    market_model: MarketModel | None = None,
) -> PolicyResult:
    """Roll a policy over a *given* return path (e.g. real history), not a
    simulated one. ``risky_returns`` is ``(T,)`` or ``(T, A)`` simple returns and
    must have ``T == config.total_steps``. The regime belief is filtered online
    from the realized returns (no look-ahead); the "regime" recorded for display
    is the filtered most-likely state.
    """
    mm = market_model or MarketModel.from_config(config.market)
    rr = np.asarray(risky_returns, dtype=float)
    if rr.ndim == 1:
        rr = rr[:, None]
    T, A = rr.shape
    if A != mm.n_assets:
        raise ValueError(f"returns have {A} asset(s) but model has {mm.n_assets}")
    if T != config.total_steps:
        raise ValueError(f"returns length {T} != config.total_steps {config.total_steps}")

    G, c = config.goal.target_wealth, config.goal.contribution
    filt = GaussianRegimeFilter(mm.gbm.mean_log, mm.gbm.cov * mm.dt, mm.regime_sim.P)
    alpha = filt.reset()
    wealth = float(config.goal.initial_wealth)

    wealth_hist = np.empty((1, T + 1)); wealth_hist[0, 0] = wealth
    weights_hist = np.empty((1, T, A))
    belief_hist = np.empty((1, T, mm.n_regimes))
    regime_hist = np.empty((1, T), dtype=int)

    policy.reset(1)
    for t in range(T):
        belief = filt.predict(alpha)
        ctx = DecisionContext(
            step=t, n_steps=T, wealth=np.array([wealth]), target=G,
            belief=belief[None, :], n_assets=A, regime_names=mm.regime_names,
        )
        w = policy.weights(ctx)[0]
        r = rr[t]
        port_ret = float(w @ r + (1.0 - w.sum()) * mm.cash_return)
        wealth = (wealth + c) * (1.0 + port_ret)
        weights_hist[0, t, :] = w
        belief_hist[0, t, :] = belief
        regime_hist[0, t] = int(np.argmax(belief))
        wealth_hist[0, t + 1] = wealth
        alpha = filt.update(alpha, np.log1p(r))

    histories = {
        "wealth": wealth_hist, "weights": weights_hist, "belief": belief_hist,
        "regime": regime_hist, "asset_names": mm.asset_names, "regime_names": mm.regime_names,
    }
    return evaluate(policy.name, np.array([wealth]), G, histories, mm.regime_names)
