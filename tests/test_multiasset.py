"""True multi-asset backtest: the data panel, the regime-aware cross-asset
allocator, and — most importantly — the **no-look-ahead** invariants of the
walk-forward causal recalibration.

All tests are network-free (``offline=True`` synthetic data or hand-built return
arrays), so they run in CI without yfinance.
"""

import numpy as np
import pytest

from gbwm.backtesting.historical import scorecard
from gbwm.backtesting.multiasset import (
    MULTI_ASSET_UNIVERSE,
    STRATEGY_LABELS,
    build_multi_asset_policies,
    default_multi_asset_config,
    effective_step_contribution,
    fetch_asset_panel,
    multi_asset_diary,
    run_multi_asset_deployment,
)
from gbwm.evaluation.harness import run_on_returns, run_on_returns_walk_forward
from gbwm.policies.multi_asset import (
    RegimeAwareMultiAsset,
    StaticMultiAsset,
    mean_variance_weights,
)
from gbwm.simulation.regimes import MarketModel


def _fast_cfg(years=6, cache_dir=None):
    cfg = default_multi_asset_config()
    cfg.goal.horizon_years = years
    cfg.agents.g_learner.update(dict(n_wealth_bins=81, n_actions=11))
    if cache_dir is not None:
        cfg.data.cache_dir = str(cache_dir)
    return cfg


# --------------------------------------------------------------------------- #
# Data panel
# --------------------------------------------------------------------------- #
def test_universe_has_four_assets_in_config_order():
    keys = [a.key for a in MULTI_ASSET_UNIVERSE]
    assert keys == ["us_equity", "intl_equity", "bonds", "gold"]
    assert set(STRATEGY_LABELS) == {"Regime-Aware Multi-Asset", "Balanced Mix", "All-in S&P 500"}


def test_asset_panel_offline_is_aligned_and_distinct(tmp_path):
    panel = fetch_asset_panel(start="2005-01-01", end="2015-01-01",
                              offline=True, cache_dir=tmp_path)
    assert panel.source == "synthetic"
    assert panel.r_log.shape[1] == 4
    assert panel.keys == ["us_equity", "intl_equity", "bonds", "gold"]
    assert len(panel.dates) == panel.r_log.shape[0]
    # the four synthetic legs must NOT be the same series (else "multi-asset" is a lie)
    corr = np.corrcoef(panel.r_log.T)
    assert np.all(np.abs(corr[np.triu_indices(4, k=1)]) < 0.95)


def test_effective_step_contribution_converts_cadence():
    assert effective_step_contribution(1500, "quarterly", 12) == 500.0
    assert effective_step_contribution(6000, "annual", 12) == 500.0
    assert effective_step_contribution(500, "monthly", 12) == 500.0


# --------------------------------------------------------------------------- #
# Cross-asset mean-variance + regime tilts
# --------------------------------------------------------------------------- #
def test_mean_variance_weights_on_simplex_and_prefers_better_asset():
    mu = np.array([0.12, 0.02])
    sigma = np.diag([0.04, 0.04])
    w = mean_variance_weights(mu, sigma, lam=4.0, cap=1.0)
    assert np.isclose(w.sum(), 1.0)
    assert np.all(w >= -1e-9)
    assert w[0] > w[1]  # the higher-return, equal-risk asset gets more
    # the per-asset cap is respected
    wc = mean_variance_weights(mu, sigma, lam=1.0, cap=0.6)
    assert wc.max() <= 0.6 + 1e-9 and np.isclose(wc.sum(), 1.0)


def test_allocator_rotates_into_safe_havens_in_bear():
    cfg = _fast_cfg(years=6)
    pol = RegimeAwareMultiAsset.from_config(cfg)
    keys = cfg.market.risky_assets
    bull = cfg.market.regime_names.index("bull")
    bear = cfg.market.regime_names.index("bear")
    safe = [keys.index("bonds"), keys.index("gold")]
    stocks = [keys.index("us_equity"), keys.index("intl_equity")]
    cw = pol.cross_weights
    # bear tilts to bonds+gold; bull tilts to stocks
    assert cw[bear][safe].sum() > cw[bull][safe].sum() + 0.2
    assert cw[bull][stocks].sum() > cw[bear][stocks].sum() + 0.2
    # a pure-bear belief decision is majority safe-haven
    from gbwm.policies.base import DecisionContext
    K = cfg.market.n_regimes
    bel = np.zeros((1, K)); bel[0, bear] = 1.0
    ctx = DecisionContext(step=cfg.total_steps // 2, n_steps=cfg.total_steps,
                          wealth=np.array([120_000.0]), target=cfg.goal.target_wealth,
                          belief=bel, n_assets=len(keys), regime_names=cfg.market.regime_names)
    w = pol.weights(ctx)[0]
    assert w[safe].sum() > w[stocks].sum()


def test_static_multi_asset_holds_constant_weights():
    pol = StaticMultiAsset([0.4, 0.1, 0.4, 0.1], name="Mix")
    from gbwm.policies.base import DecisionContext
    ctx = DecisionContext(step=0, n_steps=12, wealth=np.array([1.0, 2.0]), target=1.0,
                          belief=np.zeros((2, 4)), n_assets=4, regime_names=["a"])
    w = pol.weights(ctx)
    assert w.shape == (2, 4)
    assert np.allclose(w, [0.4, 0.1, 0.4, 0.1])


# --------------------------------------------------------------------------- #
# The headline honesty property: the belief is causal even multi-asset, because
# the regime is read off the equity sleeve (bonds can't mask an equity crash).
# --------------------------------------------------------------------------- #
def test_equity_crash_is_detected_not_masked_by_calm_bonds():
    cfg = _fast_cfg(years=5)
    mm = MarketModel.from_config(cfg.market)
    assert mm.obs_idx == [0]  # belief inferred from us_equity only
    T = cfg.total_steps
    rng = np.random.default_rng(0)
    simple = np.zeros((T, 4))
    simple[:, 0] = 0.01 + rng.normal(0, 0.005, T)   # S&P: calm, gently up
    simple[:, 1] = 0.009 + rng.normal(0, 0.006, T)  # intl
    simple[:, 2] = 0.002 + rng.normal(0, 0.002, T)  # bonds: calm throughout
    simple[:, 3] = 0.003 + rng.normal(0, 0.01, T)   # gold
    crash = 30
    simple[crash, 0] = -0.25   # an equity-only crash; bonds stay calm
    simple[crash, 1] = -0.22
    pol = RegimeAwareMultiAsset.from_config(cfg, mm)
    res = run_on_returns(pol, cfg, simple, market_model=mm)
    bel = res.histories["belief"][0]
    idx = [i for i, n in enumerate(mm.regime_names) if n in ("bear", "high_vol")]
    rb = bel[:, idx].sum(axis=1)
    assert rb[crash] < 0.5                      # cannot see the crash coming
    assert rb[crash + 1] > rb[crash] + 0.1      # reacts AFTER it is realized
    assert rb[crash - 1] < 0.5                  # no pre-emptive spike


# --------------------------------------------------------------------------- #
# Walk-forward causal recalibration: NO LOOK-AHEAD
# --------------------------------------------------------------------------- #
def test_walk_forward_decisions_do_not_depend_on_future_returns():
    """Two paths identical up to step k but different afterwards must produce the
    *same* decisions (weights) and wealth up to k — the regimes at each yearly
    boundary are re-learned only from data seen so far."""
    cfg = _fast_cfg(years=4)
    T = cfg.total_steps  # 48
    k = 30
    rng = np.random.default_rng(1)
    base = 0.005 + rng.normal(0, 0.02, (T, 4))
    alt = base.copy()
    alt[k:] += rng.normal(0, 0.05, (T - k, 4))  # diverge only after step k

    from gbwm.backtesting.multiasset import _smart_walk_forward_factory
    factory = _smart_walk_forward_factory(cfg, min_calib_months=24)

    res_a = run_on_returns_walk_forward(cfg, base, factory, recalibrate_every=12)
    res_b = run_on_returns_walk_forward(cfg, alt, factory, recalibrate_every=12)
    wa = res_a.histories["weights"][0]
    wb = res_b.histories["weights"][0]
    assert np.allclose(wa[:k], wb[:k])
    assert np.allclose(res_a.histories["wealth"][0][:k + 1],
                       res_b.histories["wealth"][0][:k + 1])


# --------------------------------------------------------------------------- #
# Deployment end-to-end (offline / synthetic — no network)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["prior", "causal_calibrate", "causal_walk_forward"])
def test_run_multi_asset_deployment_offline_shapes(tmp_path, mode):
    cfg = _fast_cfg(years=5, cache_dir=tmp_path)
    dep = run_multi_asset_deployment(
        cfg, start="2010-01-01", horizon_years=5, initial=100_000,
        contribution=600, contribution_freq="monthly", target=150_000,
        regime_mode=mode, offline=True,
    )
    assert dep.source == "synthetic"
    assert dep.n_assets == 4
    assert len(dep.dates) == dep.config.total_steps == 60
    assert set(dep.results) == {"Regime-Aware Multi-Asset", "Balanced Mix", "All-in S&P 500"}
    for name in dep.results:
        assert dep.weights(name).shape == (60, 4)
        # weights are long-only and never exceed fully-invested
        w = dep.weights(name)
        assert np.all(w >= -1e-9) and np.all(w.sum(axis=1) <= 1.0 + 1e-9)


def test_static_baselines_match_their_fixed_mix_on_history(tmp_path):
    cfg = _fast_cfg(years=5, cache_dir=tmp_path)
    dep = run_multi_asset_deployment(cfg, start="2010-01-01", horizon_years=5,
                                     regime_mode="prior", offline=True)
    # all-in S&P holds 100% us_equity every month
    allsp = dep.weights("All-in S&P 500")
    assert np.allclose(allsp[:, 0], 1.0) and np.allclose(allsp[:, 1:], 0.0)


def test_scorecard_and_diary_on_multiasset_deployment(tmp_path):
    cfg = _fast_cfg(years=10, cache_dir=tmp_path)
    dep = run_multi_asset_deployment(cfg, start="2005-01-01", horizon_years=10,
                                     initial=100_000, contribution=500, target=120_000,
                                     regime_mode="prior", offline=True)
    sc = scorecard(dep)
    assert set(sc.columns) >= {"Strategy", "Final balance", "Reached goal", "Max drawdown"}
    for _, row in sc.iterrows():
        fin = dep.results[row["Strategy"]].terminal_wealth[0]
        assert bool(fin >= dep.target) == bool(row["Reached goal"])
    # the diary returns plain-language lines (synthetic data has no crisis-window
    # crashes, so it may be empty — but must never error and must be strings)
    assert all(isinstance(s, str) for s in multi_asset_diary(dep))


def test_build_multi_asset_policies_names(tmp_path):
    cfg = _fast_cfg(cache_dir=tmp_path)
    pols = build_multi_asset_policies(cfg)
    assert set(pols) == {"Regime-Aware Multi-Asset", "Balanced Mix", "All-in S&P 500"}
