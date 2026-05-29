"""SB3 wrapper logic that does NOT require stable-baselines3 to be installed.

(Full PPO/SAC training is exercised on a machine with the 'rl' extra.)
"""
import numpy as np

from gbwm.config import default_config
from gbwm.policies.base import DecisionContext
from gbwm.policies.rl_agents import SB3Policy, _project_long_only, _tail_from_belief


class _StubModel:
    """Stands in for an SB3 model: returns a fixed raw action per row."""

    def __init__(self, raw):
        self.raw = np.asarray(raw, dtype=float)

    def predict(self, obs, deterministic=True):
        n = obs.shape[0] if obs.ndim == 2 else 1
        return np.tile(self.raw, (n, 1)), None


def test_projection_caps_and_normalizes():
    w = _project_long_only(np.array([[5.0], [0.3]]), 0.0, 1.0, leverage=False)
    assert np.allclose(w[0], 1.0)      # over-allocation scaled to 1
    assert np.allclose(w[1], 0.3)


def test_tail_modes():
    b = np.array([[0.1, 0.2, 0.3, 0.4]])
    assert _tail_from_belief(b, "probs").shape == (1, 4)
    assert np.allclose(_tail_from_belief(b, "onehot"), [[0, 0, 0, 1]])
    assert _tail_from_belief(b, "none").shape == (1, 0)


def test_sb3policy_weights_with_stub_model():
    cfg = default_config()
    pol = SB3Policy(_StubModel([5.0]), cfg, algo="ppo")  # raw action over-allocates
    n = 4
    ctx = DecisionContext(
        step=12,
        n_steps=cfg.total_steps,
        wealth=np.full(n, 1.2e5),
        target=cfg.goal.target_wealth,
        belief=np.tile([0.25, 0.25, 0.25, 0.25], (n, 1)),
        n_assets=cfg.market.n_risky,
        regime_names=cfg.market.regime_names,
    )
    w = pol.weights(ctx)
    assert w.shape == (n, 1)
    assert np.allclose(w, 1.0)         # projected to fully-invested


def test_require_sb3_raises_clear_error_when_missing():
    import importlib.util

    from gbwm.policies import rl_agents

    if importlib.util.find_spec("stable_baselines3") is None:
        try:
            rl_agents._require_sb3()
            assert False, "expected ImportError"
        except ImportError as e:
            assert "rl" in str(e)
