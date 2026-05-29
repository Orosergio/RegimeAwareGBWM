"""Tabular Q-Learning agent learns by interaction (RL)."""
import numpy as np

from gbwm.config import default_config
from gbwm.evaluation.harness import run_policy
from gbwm.policies import QLearner


def _cfg():
    cfg = default_config()
    cfg.goal.horizon_years = 20
    cfg.goal.target_wealth = 400_000.0  # moderate: room to visibly learn
    return cfg


def test_qlearner_shapes_and_valid_actions():
    cfg = _cfg()
    ql = QLearner(cfg, episodes=20000, batch_size=2000, n_wealth_bins=101, n_actions=15, seed=0)
    assert ql.Q.shape == (cfg.total_steps, 101, 15)
    assert ql.mean_action.shape == (cfg.total_steps, 101)
    assert ql.mean_action.min() >= 0.0 and ql.mean_action.max() <= 1.0


def test_qlearner_learns_over_training():
    cfg = _cfg()
    ql = QLearner(cfg, episodes=40000, batch_size=2000, seed=0)
    assert len(ql.learning_curve) >= 2
    first, last = ql.learning_curve[0][1], ql.learning_curve[-1][1]
    assert last > first + 0.05            # clearly improved by trial-and-error
    pgoal = run_policy(ql, cfg, n_episodes=2000, rng=np.random.default_rng(1)).p_goal
    assert pgoal > 0.15                   # learned a non-trivial policy


def test_qlearner_registered():
    from gbwm.policies.base import policy_registry
    assert "q_learner" in policy_registry
