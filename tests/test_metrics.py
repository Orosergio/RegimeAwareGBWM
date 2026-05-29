"""Metrics, Monte-Carlo harness, and plot smoke tests."""
import numpy as np

from gbwm.config import default_config
from gbwm.evaluation import metrics
from gbwm.evaluation.harness import compare_policies, results_table, run_policy
from gbwm.evaluation import plots
from gbwm.policies import GlidePath, SixtyForty


def test_prob_goal_and_shortfall():
    tw = np.array([100.0, 200.0, 300.0, 250.0])
    assert metrics.prob_goal(tw, 250.0) == 0.5
    assert np.allclose(metrics.shortfall(tw, 250.0), [150, 50, 0, 0])


def test_max_drawdown_known_path():
    wh = np.array([[100.0, 120.0, 90.0, 130.0]])
    assert np.isclose(metrics.max_drawdown(wh)[0], 0.25)


def test_turnover_known_weights():
    wts = np.array([[[0.5], [0.5], [1.0]]])  # (N=1, T=3, A=1)
    # deltas vs prev(0): 0.5, 0.0, 0.5 -> mean = 1.0/3
    assert np.isclose(metrics.turnover(wts)[0], 1.0 / 3.0)


def test_harness_runs_baseline():
    cfg = default_config()
    res = run_policy(SixtyForty.from_config(cfg), cfg, n_episodes=200)
    assert res.terminal_wealth.shape == (200,)
    assert 0.0 <= res.p_goal <= 1.0
    assert res.histories["weights"].shape == (200, cfg.total_steps, cfg.market.n_risky)
    assert res.histories["belief"].shape == (200, cfg.total_steps, cfg.market.n_regimes)


def test_compare_uses_shared_paths_and_table():
    cfg = default_config()
    policies = {"60/40": SixtyForty.from_config(cfg), "Glide Path": GlidePath.from_config(cfg)}
    results = compare_policies(policies, cfg, n_episodes=300)
    assert set(results) == {"60/40", "Glide Path"}
    df = results_table(results)
    assert "P(goal)" in df.columns and len(df) == 2


def test_plots_return_figures():
    cfg = default_config()
    res = run_policy(SixtyForty.from_config(cfg), cfg, n_episodes=80)
    h = res.histories
    assert plots.plot_wealth_paths(h, cfg.goal.target_wealth) is not None
    assert plots.plot_allocation_over_time(h["weights"][0], h["asset_names"]) is not None
    assert plots.plot_regime_beliefs(h["belief"][0], h["regime_names"]) is not None
    assert plots.plot_terminal_distribution(res.terminal_wealth, cfg.goal.target_wealth) is not None
    assert plots.plot_strategy_comparison({"60/40": res}) is not None
