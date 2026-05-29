"""Tabular Q-Learning agent — learns the allocation by trial and error.

This is the textbook **reinforcement-learning** counterpart to the G-Learner: it
does not solve the Bellman equations exactly, it *learns* Q(state, action) from
sampled experience with temporal-difference updates and an ε-greedy policy::

    Q(s, a) ← Q(s, a) + α · [ r + γ · max_a' Q(s', a') − Q(s, a) ]

G-learning is the entropy-regularized generalization of exactly this update, and
greedy Q-learning is its zero-temperature (β→∞) limit — so as the Q-Learner
trains, its policy converges toward the G-Learner's exact solution. We record a
learning curve (chance of reaching the goal vs. training episodes) to show the
agent improving. Training is vectorized over a batch of parallel episodes for
speed.

State = (time step, wealth bin); action = risky-fraction bin. Regime-agnostic
(like the vanilla G-Learner), which makes the convergence comparison clean.
"""

from __future__ import annotations

import numpy as np

from gbwm.config import Config
from gbwm.objective import terminal_utility
from gbwm.policies.base import DecisionContext, policy_registry
from gbwm.policies.g_learner import _BaseGLearner
from gbwm.simulation.regimes import MarketModel


@policy_registry.register("q_learner")
class QLearner(_BaseGLearner):
    """ε-greedy tabular Q-learning over (time, wealth) → risky fraction."""

    name = "Q-Learner"
    requires_belief = False

    def __init__(
        self,
        config: Config,
        market_model: MarketModel | None = None,
        *,
        n_wealth_bins: int = 121,
        n_actions: int = 15,
        gamma: float = 0.999,
        episodes: int = 60000,
        batch_size: int = 2000,
        lr: float = 0.5,
        lr_min: float = 0.05,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eval_every: int = 2,
        eval_episodes: int = 1500,
        reward_scale: float = 1.0,
        seed: int = 0,
        solve: bool = True,
    ) -> None:
        self.episodes = episodes
        self.batch_size = batch_size
        self.lr0 = lr
        self.lr_min = lr_min
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eval_every = eval_every
        self.eval_episodes = eval_episodes
        self._seed = seed
        self.Q: np.ndarray | None = None
        self.learning_curve: list[tuple[int, float]] = []
        super().__init__(
            config, market_model, n_wealth_bins=n_wealth_bins, n_actions=n_actions,
            gamma=gamma, reward_scale=reward_scale, solve=solve,
        )

    # ------------------------------------------------------------------ #
    def _rollout_pgoal(self, regimes, risky, Q) -> float:
        """Greedy-policy P(goal) on a fixed evaluation set (vectorized)."""
        B = regimes.shape[0]
        wealth = np.full(B, float(self.cfg.goal.initial_wealth))
        for t in range(self.T):
            wbin = self._bin(wealth)
            a = Q[t, wbin].argmax(axis=1)
            frac = self.action_grid[a]
            r = risky[:, t, 0]
            wealth = (wealth + self.c) * (1.0 + frac * r + (1.0 - frac) * self.cash_return)
        return float(np.mean(wealth >= self.G))

    def solve(self) -> None:
        rng = np.random.default_rng(self._seed)
        na = len(self.action_grid)
        Q = np.zeros((self.T, self.n_wbins, na))
        # fixed evaluation set for a stable learning curve
        eval_paths = self.mm.simulate(self.eval_episodes, self.T, np.random.default_rng(self._seed + 999))
        eval_reg, eval_ret = eval_paths.regimes, eval_paths.risky_returns

        rounds = max(1, self.episodes // self.batch_size)
        self.learning_curve = []
        for rd in range(rounds):
            frac_done = rd / max(1, rounds - 1)
            eps = self.eps_start + (self.eps_end - self.eps_start) * frac_done
            lr = max(self.lr_min, self.lr0 * (1.0 - 0.9 * frac_done))
            paths = self.mm.simulate(self.batch_size, self.T, rng)
            risky = paths.risky_returns
            wealth = np.full(self.batch_size, float(self.cfg.goal.initial_wealth))
            for t in range(self.T):
                wbin = self._bin(wealth)
                qsa = Q[t, wbin]  # (B, na)
                greedy = qsa.argmax(axis=1)
                rand = rng.integers(na, size=self.batch_size)
                explore = rng.random(self.batch_size) < eps
                a = np.where(explore, rand, greedy)
                frac = self.action_grid[a]
                r = risky[:, t, 0]
                new_wealth = (wealth + self.c) * (1.0 + frac * r + (1.0 - frac) * self.cash_return)
                if t < self.T - 1:
                    nb = self._bin(new_wealth)
                    target = self.gamma * Q[t + 1, nb].max(axis=1)
                else:
                    target = self.reward_scale * terminal_utility(new_wealth, self.G, self.reward)
                # batched TD update with per-cell averaging to avoid collisions
                tgt_sum = np.zeros((self.n_wbins, na))
                cnt = np.zeros((self.n_wbins, na))
                np.add.at(tgt_sum, (wbin, a), target)
                np.add.at(cnt, (wbin, a), 1.0)
                visited = cnt > 0
                mean_tgt = np.where(visited, tgt_sum / np.maximum(cnt, 1), 0.0)
                Q[t][visited] += lr * (mean_tgt[visited] - Q[t][visited])
                wealth = new_wealth
            if rd % self.eval_every == 0 or rd == rounds - 1:
                pg = self._rollout_pgoal(eval_reg, eval_ret, Q)
                self.learning_curve.append(((rd + 1) * self.batch_size, pg))

        self.Q = Q
        self.mean_action = self.action_grid[Q.argmax(axis=2)]  # (T, n_wbins) greedy fractions

    def weights(self, ctx: DecisionContext) -> np.ndarray:
        t = min(ctx.step, self.T - 1)
        return self._split(self.mean_action[t, self._bin(ctx.wealth)])

    @classmethod
    def from_config(cls, config: Config, market_model: MarketModel | None = None) -> "QLearner":
        p = dict(config.agents.extra.get("q_learner", {}))
        return cls(
            config, market_model,
            n_wealth_bins=int(p.get("n_wealth_bins", 121)),
            n_actions=int(p.get("n_actions", 15)),
            gamma=float(p.get("gamma", 0.999)),
            episodes=int(p.get("episodes", 60000)),
            batch_size=int(p.get("batch_size", 2000)),
            seed=int(p.get("seed", config.seed)),
        )
