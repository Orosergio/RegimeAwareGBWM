"""G-Learner: solver shapes + goal-based behavioral properties.

Uses a short horizon for speed; behavior is qualitatively horizon-independent.
"""
import numpy as np

from gbwm.config import default_config
from gbwm.policies.base import DecisionContext
from gbwm.policies.g_learner import GLearner, RegimeAwareGLearner


def _cfg(years=4, **gl):
    cfg = default_config()
    cfg.goal.horizon_years = years
    cfg.agents.g_learner.update(dict(n_wealth_bins=151, n_actions=21, beta=15.0, greedy=True))
    cfg.agents.g_learner.update(gl)
    return cfg


def test_solver_shapes():
    cfg = _cfg()
    gl = GLearner.from_config(cfg)
    ra = RegimeAwareGLearner.from_config(cfg)
    assert gl.mean_action.shape == (cfg.total_steps, gl.n_wbins)
    assert ra.mean_action.shape == (cfg.total_steps, ra.n_wbins, cfg.market.n_regimes)


def test_actions_are_valid_fractions():
    gl = GLearner.from_config(_cfg())
    assert gl.mean_action.min() >= 0.0 and gl.mean_action.max() <= 1.0


def test_gamble_near_goal_protect_above_goal_last_step():
    """Last step: just below goal -> gamble to cross; at/above goal -> de-risk."""
    cfg = _cfg()
    gl = GLearner.from_config(cfg)
    G, T = cfg.goal.target_wealth, cfg.total_steps
    eq = lambda w: gl.mean_action[T - 1, gl._bin(np.array([w]))[0]]
    assert eq(0.90 * G) > eq(1.02 * G)


def test_gamble_when_behind_midhorizon():
    """Mid-horizon: far below goal -> much more equity than comfortably ahead."""
    cfg = _cfg()
    gl = GLearner.from_config(cfg)
    G, T = cfg.goal.target_wealth, cfg.total_steps
    eq = lambda w: gl.mean_action[T // 2, gl._bin(np.array([w]))[0]]
    assert eq(0.45 * G) > eq(1.30 * G)


def test_regime_ordering_bull_ge_bear():
    """For the same state, equity in bull >= equity in bear (de-risk in bad weather)."""
    cfg = _cfg()
    ra = RegimeAwareGLearner.from_config(cfg)
    G, T = cfg.goal.target_wealth, cfg.total_steps
    names = cfg.market.regime_names
    wb = ra._bin(np.array([0.85 * G]))[0]
    by = {nm: ra.mean_action[T // 2, wb, k] for k, nm in enumerate(names)}
    assert by["bull"] >= by["bear"]
    assert by["bull"] >= by["high_vol"] - 1e-9


def test_weights_shape_and_belief_mixing():
    cfg = _cfg()
    ra = RegimeAwareGLearner.from_config(cfg)
    n = 6
    ctx = DecisionContext(
        step=10,
        n_steps=cfg.total_steps,
        wealth=np.full(n, 0.8 * cfg.goal.target_wealth),
        target=cfg.goal.target_wealth,
        belief=np.tile([0.7, 0.1, 0.1, 0.1], (n, 1)),
        n_assets=cfg.market.n_risky,
        regime_names=cfg.market.regime_names,
    )
    w = ra.weights(ctx)
    assert w.shape == (n, 1)
    assert np.all(w >= 0) and np.all(w.sum(axis=1) <= 1.0 + 1e-9)
