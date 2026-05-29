"""WealthEnv: spaces, dynamics, reward, belief."""
import numpy as np

from gbwm.config import default_config
from gbwm.envs import WealthEnv


def _env(**overrides):
    cfg = default_config()
    for k, v in overrides.items():
        setattr(cfg.env, k, v)
    return WealthEnv(cfg)


def test_spaces_shapes():
    env = _env()
    assert env.action_space.shape == (1,)            # single risky asset
    assert env.observation_space.shape == (3 + 4,)   # 3 head + 4 regimes
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    for key in ("wealth", "weights", "belief", "true_regime", "regime_name"):
        assert key in info


def test_regime_obs_modes_change_dim():
    assert _env(observe_regime="none").observation_space.shape == (3,)
    assert _env(observe_regime="onehot").observation_space.shape == (7,)


def test_episode_terminates_after_T_steps():
    env = _env()
    env.reset(seed=1)
    steps = 0
    done = False
    while not done:
        _, r, done, trunc, _ = env.step(env.action_space.sample(np.random.default_rng(steps)))
        assert np.isfinite(r)
        steps += 1
    assert steps == env.T and not trunc


def test_all_cash_path_is_deterministic_risk_free():
    env = _env()
    env.reset(seed=2)
    # closed-form: W_{t+1} = (W_t + c)(1 + cash_return)
    w = env.W0
    for _ in range(env.T):
        w = (w + env.contribution) * (1.0 + env.mm.cash_return)
    done = False
    while not done:
        _, _, done, _, info = env.step(np.zeros(1, dtype=np.float32))
    assert np.isclose(info["wealth"], w, rtol=1e-9)


def test_action_projection_caps_leverage():
    env = _env()
    env.reset(seed=3)
    _, _, _, _, info = env.step(np.array([5.0], dtype=np.float32))  # over-allocated
    assert np.isclose(info["weights"].sum(), 1.0)
    assert np.isclose(info["cash_weight"], 0.0)


def test_belief_is_a_distribution():
    env = _env()
    obs, info = env.reset(seed=4)
    b = info["belief"]
    assert b.shape == (4,)
    assert abs(b.sum() - 1.0) < 1e-9 and np.all(b >= 0)


def test_terminal_reward_threshold_semantics():
    env = _env()
    env.reset(seed=5)
    env.wealth = env.G                     # exactly at goal
    assert np.isclose(env._terminal_reward(), env.rew.goal_bonus)
    env.wealth = 0.5 * env.G               # 50% short
    assert np.isclose(env._terminal_reward(), -env.rew.shortfall_penalty * 0.5)


def test_reset_is_reproducible():
    e1, e2 = _env(), _env()
    e1.reset(seed=99)
    e2.reset(seed=99)
    assert np.array_equal(e1._regimes, e2._regimes)
    assert np.allclose(e1._risky, e2._risky)
