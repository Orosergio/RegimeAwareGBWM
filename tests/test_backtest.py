"""Historical backtest (run a policy over a real/given return path)."""
import numpy as np

from gbwm.config import default_config
from gbwm.evaluation.harness import run_on_returns
from gbwm import experiments as X
from gbwm.policies import SixtyForty


def _fast_cfg(years=6):
    cfg = default_config()
    cfg.goal.horizon_years = years
    cfg.agents.g_learner.update(dict(n_wealth_bins=121, n_actions=15))
    return cfg


def test_run_on_returns_zero_returns_is_deterministic():
    cfg = _fast_cfg(years=2)
    pol = SixtyForty.from_config(cfg)
    T = cfg.total_steps
    res = run_on_returns(pol, cfg, np.zeros(T))
    # zero risky return -> port return = 0.4 * cash each step
    w = cfg.goal.initial_wealth
    cash = res  # noqa: just for clarity
    from gbwm.simulation.regimes import MarketModel
    cr = MarketModel.from_config(cfg.market).cash_return
    for _ in range(T):
        w = (w + cfg.goal.contribution) * (1.0 + 0.4 * cr)
    assert np.isclose(res.terminal_wealth[0], w, rtol=1e-9)
    assert res.histories["weights"].shape == (1, T, cfg.market.n_risky)


def test_run_on_returns_length_validation():
    cfg = _fast_cfg(years=2)
    pol = SixtyForty.from_config(cfg)
    try:
        run_on_returns(pol, cfg, np.zeros(cfg.total_steps - 1))
        assert False, "expected length error"
    except ValueError:
        pass


def test_backtest_history_calibrates_and_uses_whole_years():
    cfg = _fast_cfg()
    rng = np.random.default_rng(0)
    returns = rng.normal(0.006, 0.04, size=100)  # ~8.3 years monthly
    res, used, calib = X.backtest_history(cfg, "regime_aware_g_learner", returns, calibrate=True)
    assert used.total_steps == (100 // 12) * 12      # whole years
    assert res.terminal_wealth.shape == (1,)
    assert calib is not None and used.market.n_regimes == cfg.hmm.n_states


def test_backtest_history_without_calibration():
    cfg = _fast_cfg()
    returns = np.random.default_rng(1).normal(0.005, 0.03, size=72)
    res, used, calib = X.backtest_history(cfg, "glide_path", returns, calibrate=False)
    assert calib is None and used.total_steps == 72
    assert np.isfinite(res.terminal_wealth[0])


def test_backtest_all_on_history_runs_every_strategy():
    from gbwm import experiments as X
    cfg = _fast_cfg()
    returns = np.random.default_rng(2).normal(0.006, 0.04, size=96)  # 8 years
    results, used, calib = X.backtest_all_on_history(cfg, returns, calibrate=True)
    assert set(results) >= {"Buy & Hold", "60/40", "Glide Path", "G-Learner", "Regime-Aware G-Learner"}
    assert used.total_steps == 96
    for r in results.values():
        assert r.terminal_wealth.shape == (1,) and np.isfinite(r.terminal_wealth[0])
    assert len(X.MARKETS) >= 8
