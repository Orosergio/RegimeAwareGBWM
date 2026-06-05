"""Honest historical backtest: registry, data plumbing, scorecard, and — most
importantly — the **no-look-ahead** invariants that make the demonstration valid.

All tests are network-free: they use ``offline=True`` (deterministic synthetic
data) or hand-built return arrays, so they run in CI without yfinance.
"""

import numpy as np
import pandas as pd

from gbwm.backtesting import (
    LONG_HISTORY_MARKETS,
    diary_sentences,
    list_markets,
    run_deployment,
    scorecard,
    sequence_risk,
)
from gbwm.backtesting.historical import (
    Market,
    _cagr,
    _max_drawdown,
    _portfolio_monthly_log,
    _worst_window_return,
)
from gbwm.config import default_config
from gbwm.evaluation.harness import run_on_returns
from gbwm.policies import RegimeAwareGLearner
from gbwm.simulation.regimes import MarketModel


def _fast_cfg(years=5, cache_dir=None):
    cfg = default_config()
    cfg.goal.horizon_years = years
    cfg.agents.g_learner.update(dict(n_wealth_bins=121, n_actions=15))
    if cache_dir is not None:  # hermetic: a fresh cache dir => offline truly synthetic
        cfg.data.cache_dir = str(cache_dir)
    return cfg


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_market_registry_has_named_markets():
    keys = {m.key for m in list_markets()}
    assert {"sp500", "nasdaq", "kospi", "nikkei", "mag7"} <= keys
    assert LONG_HISTORY_MARKETS["mag7"].is_basket
    assert not LONG_HISTORY_MARKETS["sp500"].is_basket
    assert len(LONG_HISTORY_MARKETS["mag7"].tickers) == 7


# --------------------------------------------------------------------------- #
# Data plumbing: index vs equal-weight basket
# --------------------------------------------------------------------------- #
def test_index_monthly_log_from_prices():
    idx = pd.bdate_range("2020-01-01", "2020-12-31")
    growth = 100.0 * (1.0005 ** np.arange(len(idx)))
    prices = pd.DataFrame({"^GSPC": growth}, index=idx)
    m = Market("sp500", "S&P 500", ("^GSPC",), "")
    s = _portfolio_monthly_log(prices, m)
    assert len(s) >= 10                       # ~12 month-ends
    assert (s > 0).all()                      # rising series -> positive log-returns
    # month-end compounded log-returns reconstruct the price ratio
    assert np.isclose(np.exp(s.sum()), growth[-1] / growth[0], rtol=2e-3)


def test_basket_is_equal_weight_of_constituents():
    idx = pd.bdate_range("2021-01-01", "2021-06-30")
    a = 100.0 * (1.001 ** np.arange(len(idx)))     # +0.1%/day
    b = 100.0 * (1.000 ** np.arange(len(idx)))     # flat
    prices = pd.DataFrame({"A": a, "B": b}, index=idx)
    m = Market("x", "X", ("A", "B"), "", kind="basket")
    basket = _portfolio_monthly_log(prices, m)
    only_a = _portfolio_monthly_log(prices, Market("a", "A", ("A",), ""))
    only_b = _portfolio_monthly_log(prices, Market("b", "B", ("B",), ""))
    # the equal-weight basket return sits strictly between its two constituents
    assert (only_b.values < basket.values).all()
    assert (basket.values < only_a.values).all()


# --------------------------------------------------------------------------- #
# Deployment (offline / synthetic — no network)
# --------------------------------------------------------------------------- #
def test_run_deployment_offline_shapes(tmp_path):
    cfg = _fast_cfg(cache_dir=tmp_path)
    dep = run_deployment(cfg, "sp500", start="2000-01-01", horizon_years=5,
                         initial=100_000, contribution=500, target=300_000, offline=True)
    assert dep.source == "synthetic"
    assert len(dep.dates) == dep.config.total_steps == 60
    for res in dep.results.values():
        assert res.histories["wealth"].shape == (1, 61)
        assert res.histories["weights"].shape == (1, 60, 1)
    # price index has one extra anchor point
    assert dep.price_index.shape == (61,)


def test_prior_mode_does_not_calibrate_on_the_future(tmp_path):
    """The headline honesty property: with regime_mode='prior' the market model
    parameters are the config's economic priors — never fitted to the path."""
    cfg = _fast_cfg(cache_dir=tmp_path)
    dep = run_deployment(cfg, "sp500", start="2000-01-01", horizon_years=5,
                         offline=True, regime_mode="prior")
    assert dep.regime_mode == "prior"
    prior = MarketModel.from_config(cfg.market)
    used = MarketModel.from_config(dep.config.market)
    assert np.allclose(used.gbm.drift, prior.gbm.drift)
    assert np.allclose(used.gbm.cov, prior.gbm.cov)


# --------------------------------------------------------------------------- #
# No look-ahead: the belief is causal (reacts AFTER a shock, never before)
# --------------------------------------------------------------------------- #
def _risk_belief(histories, regime_names):
    idx = [i for i, n in enumerate(regime_names) if n in ("bear", "high_vol")]
    return histories["belief"][0][:, idx].sum(axis=1)


def test_belief_is_causal_no_anticipation_of_a_crash():
    cfg = _fast_cfg(years=5)
    T = cfg.total_steps
    rng = np.random.default_rng(0)
    simple = 0.01 + rng.normal(0, 0.005, size=T)   # calm, gently positive
    crash_t = 40
    simple[crash_t] = -0.25                          # a sudden one-month crash
    pol = RegimeAwareGLearner.from_config(cfg)
    mm = MarketModel.from_config(cfg.market)
    res = run_on_returns(pol, cfg, simple, market_model=mm)
    rb = _risk_belief(res.histories, mm.regime_names)
    # The decision-time belief just BEFORE the crash month is still calm: the
    # agent cannot see the crash coming (no look-ahead).
    assert rb[crash_t] < 0.5
    # After the crash is realized, the bad-weather belief jumps.
    assert rb[crash_t + 1] > rb[crash_t] + 0.1
    # And it had not pre-emptively spiked in the calm run-up.
    assert rb[crash_t - 1] < 0.5


# --------------------------------------------------------------------------- #
# Scorecard metrics
# --------------------------------------------------------------------------- #
def test_scorecard_metric_helpers():
    wealth = np.array([100.0, 120.0, 60.0, 90.0, 130.0])  # peak 120 -> trough 60
    assert np.isclose(_max_drawdown(wealth), 0.5)
    assert np.isclose(_worst_window_return(wealth, 1), -0.5)   # 120 -> 60
    # 4 steps at 12/yr -> 1/3 year; (130/100)^3 - 1
    assert np.isclose(_cagr(wealth, 12), (130 / 100) ** (12 / 4) - 1, rtol=1e-9)


def test_scorecard_reached_goal_and_sorting(tmp_path):
    cfg = _fast_cfg(cache_dir=tmp_path)
    dep = run_deployment(cfg, "sp500", start="2005-01-01", horizon_years=5,
                         initial=100_000, contribution=500, target=120_000, offline=True)
    sc = scorecard(dep)
    assert set(sc.columns) >= {"Strategy", "Final balance", "Reached goal",
                               "Max drawdown", "Worst 12-month"}
    # reached-goal flag matches the raw final balance vs target
    for _, row in sc.iterrows():
        fin = dep.results[row["Strategy"]].terminal_wealth[0]
        assert bool(fin >= dep.target) == bool(row["Reached goal"])
    # safest-first ordering among goal reachers (drawdown non-decreasing)
    reached = sc[sc["Reached goal"]]
    assert list(reached["Max drawdown"]) == sorted(reached["Max drawdown"])


# --------------------------------------------------------------------------- #
# Sequence-of-returns risk + decision diary
# --------------------------------------------------------------------------- #
def test_sequence_risk_runs_multiple_starts_offline(tmp_path):
    cfg = _fast_cfg(cache_dir=tmp_path)
    seq = sequence_risk(cfg, "sp500", starts=["2000-01-01", "2005-01-01"],
                        horizon_years=5, initial=100_000, contribution=500,
                        target=200_000, offline=True)
    assert set(seq["Start"]) == {"2000", "2005"}
    assert {"Buy & Hold", "Regime-Aware G-Learner"} <= set(seq["Strategy"])
    assert np.isfinite(seq["Final balance"]).all()


def test_decision_diary_emits_sentences_for_in_window_crises(tmp_path):
    cfg = _fast_cfg(years=10, cache_dir=tmp_path)
    dep = run_deployment(cfg, "sp500", start="2000-01-01", horizon_years=10,
                         initial=100_000, contribution=500, target=300_000, offline=True)
    lines = diary_sentences(dep)
    # 2000-2010 window contains the dot-com and GFC crisis annotations
    assert any("Dot-com" in s for s in lines)
    assert any("Global Financial Crisis" in s for s in lines)
    assert all(isinstance(s, str) and len(s) > 20 for s in lines)
