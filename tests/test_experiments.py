"""High-level experiments + checkpoint registry (no SB3 required)."""
import tempfile

import numpy as np

from gbwm.checkpoints import ModelRegistry
from gbwm.config import default_config
from gbwm import experiments as X
from gbwm.cli import build_parser


def _fast_cfg(years=3):
    cfg = default_config()
    cfg.goal.horizon_years = years
    cfg.simulation.n_episodes = 300
    cfg.agents.g_learner.update(dict(n_wealth_bins=121, n_actions=15))
    return cfg


def test_glearner_checkpoint_roundtrip():
    cfg = _fast_cfg()
    with tempfile.TemporaryDirectory() as d:
        reg = ModelRegistry(d)
        meta = X.train(cfg, "g_learner", reg)
        assert meta.kind == "g_learner" and reg.exists("g_learner")
        loaded = reg.load("g_learner", cfg)
        assert loaded.mean_action.shape == (cfg.total_steps, loaded.n_wbins)


def test_build_policies_counts_and_skips_untrained_rl():
    cfg = _fast_cfg()
    assert len(X.build_policies(cfg, "baselines")) == 3
    # all = 3 baselines + 2 G-learners (+ ppo/sac skipped without checkpoints)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pols = X.build_policies(cfg, "all")
    assert len(pols) == 5


def test_evaluate_returns_ranked_table():
    cfg = _fast_cfg()
    _, table = X.evaluate(cfg, n_episodes=300, which="baselines")
    assert len(table) == 3 and "P(goal)" in table.columns
    # table is sorted by P(goal) descending
    assert list(table["P(goal)"]) == sorted(table["P(goal)"], reverse=True)


def test_backtest_returns_result_and_explanation():
    cfg = _fast_cfg()
    res, explanation = X.backtest(cfg, "regime_aware_g_learner", seed=3)
    assert res.terminal_wealth.shape == (1,)
    assert isinstance(explanation, str) and "goal" in explanation.lower()


def test_cli_parser_wires_subcommands():
    p = build_parser()
    args = p.parse_args(["evaluate", "--agents", "baselines", "--episodes", "100"])
    assert args.command == "evaluate" and hasattr(args, "func")
    args2 = p.parse_args(["train", "--agent", "g_learner"])
    assert args2.agent == "g_learner"
