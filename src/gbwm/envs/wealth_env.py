"""The goal-based wealth-management MDP as a Gymnasium environment.

State (observation)
    [ time_fraction, wealth_ratio, gap_to_goal, *regime_belief ]
      - time_fraction : t / T in [0, 1]
      - wealth_ratio  : W_t / G (clipped)
      - gap_to_goal   : (G - W_t) / G (clipped)
      - regime_belief : online HMM posterior over the K regimes (ADR-003);
                        configurable: 'probs' | 'onehot'(true regime) | 'none'

Action
    risky-asset weights in [floor, cap]^A, projected to a feasible long-only
    portfolio (the residual 1 - sum(w) is held in cash). Single-risky default
    ⇒ a scalar equity weight.

Reward
    Dense path penalties each step (turnover ≈ transaction cost, optional
    drawdown) plus a terminal goal utility. The headline objective is
    P(W_T ≥ G); the terminal term is configurable ('threshold' | 'power' |
    'linear'). See :class:`gbwm.config.RewardConfig`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gbwm.config import Config
from gbwm.detection.filter import GaussianRegimeFilter
from gbwm.objective import terminal_utility
from gbwm.envs._compat import Box, Env
from gbwm.simulation.regimes import MarketModel


def assemble_obs(time_frac, wealth, target: float, tail) -> np.ndarray:
    """Build the observation [time_frac, wealth_ratio, gap, *tail].

    Shared by :class:`WealthEnv` and the SB3 deploy-time policy so training and
    inference observations are guaranteed identical. Works on scalars or batches.
    """
    wealth = np.asarray(wealth, dtype=float)
    ratio = np.clip(wealth / target, 0.0, 10.0)
    gap = np.clip((target - wealth) / target, -10.0, 2.0)
    tf = np.broadcast_to(np.asarray(time_frac, dtype=float), ratio.shape)
    head = np.stack([tf, ratio, gap], axis=-1)
    tail = np.asarray(tail, dtype=float)
    if tail.shape[-1] == 0:
        return head.astype(np.float32)
    return np.concatenate([head, tail], axis=-1).astype(np.float32)


class WealthEnv(Env):
    metadata = {"render_modes": []}

    def __init__(self, config: Config, market_model: MarketModel | None = None) -> None:
        super().__init__()
        self.cfg = config
        self.mm = market_model or MarketModel.from_config(config.market)
        self.T = config.total_steps
        self.A = self.mm.n_assets
        self.K = self.mm.n_regimes
        self.G = config.goal.target_wealth
        self.W0 = config.goal.initial_wealth
        self.contribution = config.goal.contribution
        self.rew = config.env.reward
        self.regime_obs = config.env.observe_regime

        # Online belief filter using the (true/calibrated) market parameters.
        self.filter = GaussianRegimeFilter(
            mean_log=self.mm.gbm.mean_log,
            cov_log=self.mm.gbm.cov * self.mm.dt,
            transition=self.mm.regime_sim.P,
        )

        # Spaces.
        floor = config.env.action_floor
        cap = config.env.action_cap
        self.action_space = Box(low=floor, high=cap, shape=(self.A,), dtype=np.float32)
        regime_dim = self.K if self.regime_obs in ("probs", "onehot") else 0
        obs_dim = 3 + regime_dim
        high = np.concatenate([[1.0, 10.0, 2.0], np.ones(regime_dim)]).astype(np.float32)
        low = np.concatenate([[0.0, 0.0, -10.0], np.zeros(regime_dim)]).astype(np.float32)
        self.observation_space = Box(low=low, high=high, shape=(obs_dim,), dtype=np.float32)

        self._rng = np.random.default_rng(config.seed)
        self._regimes: np.ndarray | None = None
        self._risky: np.ndarray | None = None
        self.reset(seed=config.seed)

    # ------------------------------------------------------------------ #
    def _project_action(self, action: np.ndarray) -> np.ndarray:
        """Map any Box action to a feasible long-only portfolio (sum ≤ 1)."""
        w = np.clip(np.asarray(action, dtype=float), self.cfg.env.action_floor, self.cfg.env.action_cap)
        if not self.cfg.env.allow_leverage:
            s = w.sum()
            if s > 1.0:
                w = w / s
        return w

    def _belief(self) -> np.ndarray:
        """Predicted regime belief at the current decision point."""
        return self.filter.predict(self._alpha)

    def _get_obs(self) -> np.ndarray:
        if self.regime_obs == "probs":
            tail = self._belief()
        elif self.regime_obs == "onehot":
            tail = np.zeros(self.K)
            tail[int(self._regimes[min(self.t, self.T - 1)])] = 1.0
        else:
            tail = np.zeros(0)
        return assemble_obs(self.t / self.T, self.wealth, self.G, tail)

    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        start = (options or {}).get("start_regime")
        paths = self.mm.simulate(1, self.T, self._rng, antithetic=False, start_regime=start)
        self._regimes = paths.regimes[0]
        self._risky = paths.risky_returns[0]
        self.t = 0
        self.wealth = float(self.W0)
        self.peak = float(self.W0)
        self._alpha = self.filter.reset()
        self._prev_w = np.zeros(self.A)
        return self._get_obs(), self._info(reward=0.0, weights=self._prev_w)

    def step(self, action: np.ndarray):
        if self._regimes is None:
            raise RuntimeError("call reset() before step()")
        w = self._project_action(action)

        # invest the post-contribution balance for one step
        invested = self.wealth + self.contribution
        r = self._risky[self.t]  # (A,) simple returns this step
        cash_w = 1.0 - w.sum()
        port_ret = float(w @ r + cash_w * self.mm.cash_return)
        new_wealth = invested * (1.0 + port_ret)

        # path penalties
        turnover = float(np.abs(w - self._prev_w).sum())
        reward = -self.rew.turnover_penalty * turnover
        self.peak = max(self.peak, new_wealth)
        if self.rew.drawdown_penalty:
            reward -= self.rew.drawdown_penalty * max(0.0, (self.peak - new_wealth) / self.peak)
        if self.rew.step_shaping:
            prev_prog = min(self.wealth / self.G, 1.0)
            new_prog = min(new_wealth / self.G, 1.0)
            reward += self.rew.step_shaping * (new_prog - prev_prog)

        # belief update from realized log-return
        x = np.log1p(r)
        self._alpha = self.filter.update(self._alpha, x)

        self.wealth = new_wealth
        self._prev_w = w
        self.t += 1
        terminated = self.t >= self.T
        if terminated:
            reward += self._terminal_reward()
        return self._get_obs(), float(reward), bool(terminated), False, self._info(reward, w, port_ret)

    # ------------------------------------------------------------------ #
    def _terminal_reward(self) -> float:
        return terminal_utility(self.wealth, self.G, self.rew)

    def _info(self, reward: float, weights: np.ndarray, port_ret: float = 0.0) -> dict[str, Any]:
        true_regime = int(self._regimes[min(self.t, self.T - 1)]) if self._regimes is not None else 0
        return {
            "t": self.t,
            "wealth": self.wealth,
            "weights": np.asarray(weights, dtype=float),
            "cash_weight": float(1.0 - np.sum(weights)),
            "portfolio_return": port_ret,
            "true_regime": true_regime,
            "regime_name": self.mm.regime_names[true_regime],
            "belief": self._belief(),
            "reward": reward,
        }


def make_wealth_env(config: Config, market_model: MarketModel | None = None) -> WealthEnv:
    """Factory used by the CLI / SB3 training (kept thin per ADR-001)."""
    return WealthEnv(config, market_model=market_model)
