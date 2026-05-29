"""G-Learning for goal-based wealth management (Halperin; Dixon & Halperin).

G-learning is entropy-regularized RL: with a reference policy ``pi0`` and inverse
temperature ``beta`` the optimal policy is a Gibbs (softmax) distribution over
actions and the value is a *free energy*::

    pi(a|s) ∝ pi0(a|s) · exp(beta · G(s, a))
    F(s)    = (1/beta) · log Σ_a pi0(a|s) · exp(beta · G(s, a))
    G(s, a) = r(s, a) + gamma · E_{s'|s,a}[ F(s') ]

We solve this finite-horizon goal MDP exactly by **backward induction** over a
discretized (time, wealth[, regime]) state and a discretized risky-weight action,
integrating the one-step return with Gauss-Hermite quadrature. The terminal value
is the shared goal utility (:func:`gbwm.objective.terminal_utility`), so the
learned policy optimizes the same objective the harness evaluates.

Decision read-out. ``greedy=True`` (default) deploys the greedy action and uses
the hard-max value backup — the ``beta -> ∞`` limit of G-learning, i.e. standard
value iteration, which gives a crisp, interpretable allocator. ``greedy=False``
deploys the finite-temperature **Gibbs** policy (the mean risky weight under
``pi(a|s)``) with the free-energy backup — the canonical stochastic G-Learner.

Two variants:

* :class:`GLearner` — **regime-agnostic**: return model is the stationary mixture
  over regimes (the agent does not track the regime).
* :class:`RegimeAwareGLearner` — **regime-conditioned**: solves a policy per regime
  using the true regime dynamics and, at decision time, mixes those policies by the
  HMM belief (a QMDP-style belief approximation for the POMDP).

The tabular DP controls a single *risky fraction*; for multi-asset configs that
fraction is split equally across risky assets (a risk-level controller). The deep
RL agents handle full multi-asset allocation.
"""

from __future__ import annotations

import numpy as np

from gbwm.config import Config
from gbwm.objective import terminal_utility
from gbwm.policies.base import DecisionContext, Policy, policy_registry
from gbwm.simulation.regimes import MarketModel


def _hermgauss_standard_normal(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes/weights for E[g(Z)], Z ~ N(0,1) via probabilists' Hermite quadrature."""
    nodes, weights = np.polynomial.hermite_e.hermegauss(n)
    probs = weights / np.sqrt(2.0 * np.pi)
    return nodes, probs / probs.sum()


class _BaseGLearner(Policy):
    """Shared machinery: wealth grid, quadrature, composite-risky reduction, backup."""

    def __init__(
        self,
        config: Config,
        market_model: MarketModel | None = None,
        *,
        n_wealth_bins: int = 201,
        n_actions: int = 21,
        gamma: float = 0.999,
        beta: float = 15.0,
        greedy: bool = True,
        reward_scale: float = 1.0,
        n_quad: int = 15,
        wealth_min_ratio: float = 0.02,
        wealth_max_ratio: float = 6.0,
        solve: bool = True,
    ) -> None:
        self.cfg = config
        self.mm = market_model or MarketModel.from_config(config.market)
        self.T = config.total_steps
        self.A = self.mm.n_assets
        self.G = config.goal.target_wealth
        self.c = config.goal.contribution
        self.reward = config.env.reward
        self.gamma = gamma
        self.beta = beta
        self.greedy = greedy
        self.reward_scale = reward_scale

        # wealth grid (log-spaced)
        self.n_wbins = n_wealth_bins
        self.w_min = max(1.0, wealth_min_ratio * self.G)
        self.w_max = wealth_max_ratio * self.G
        self.log_min = np.log(self.w_min)
        self.log_max = np.log(self.w_max)
        self.dlog = (self.log_max - self.log_min) / (self.n_wbins - 1)
        self.w_grid = np.exp(self.log_min + self.dlog * np.arange(self.n_wbins))

        # action grid = risky fraction in [0, 1]
        self.action_grid = np.linspace(0.0, 1.0, n_actions)

        # composite risky return per regime (equal-weight basket; exact for A=1)
        u = np.full(self.A, 1.0 / self.A)
        cov_log = self.mm.gbm.cov * self.mm.dt  # (K,A,A)
        self.reg_mean_log = self.mm.gbm.mean_log @ u  # (K,)
        self.reg_sd_log = np.sqrt(np.clip(np.einsum("a,kab,b->k", u, cov_log, u), 1e-16, None))
        self.cash_return = self.mm.cash_return
        self.P = self.mm.regime_sim.P
        self.pi_stationary = self.mm.regime_sim.stationary_distribution()

        self._z, self._p = _hermgauss_standard_normal(n_quad)
        self.mean_action: np.ndarray | None = None
        if solve:
            self.solve()

    # ------------------------------------------------------------------ #
    def _bin(self, wealth: np.ndarray) -> np.ndarray:
        logw = np.log(np.clip(wealth, self.w_min, self.w_max))
        idx = np.rint((logw - self.log_min) / self.dlog).astype(np.int64)
        return np.clip(idx, 0, self.n_wbins - 1)

    def _terminal_value(self) -> np.ndarray:
        return self.reward_scale * terminal_utility(self.w_grid, self.G, self.reward)

    def _returns_for_regime(self, k: int) -> np.ndarray:
        return np.expm1(self.reg_mean_log[k] + self.reg_sd_log[k] * self._z)

    def _backup(self, G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (deployed risky fraction per state, value per state) from G(s,·).

        ``G`` has shape ``(n_states, n_actions)``.
        """
        if self.greedy:
            action = self.action_grid[np.argmax(G, axis=1)]
            value = G.max(axis=1)
            return action, value
        logits = self.beta * G
        m = logits.max(axis=1, keepdims=True)
        ex = np.exp(logits - m)
        policy = ex / ex.sum(axis=1, keepdims=True)
        action = policy @ self.action_grid
        value = (np.log(ex.sum(axis=1)) + m[:, 0] - np.log(len(self.action_grid))) / self.beta
        return action, value

    def _split(self, frac: np.ndarray) -> np.ndarray:
        return np.repeat((frac / self.A)[:, None], self.A, axis=1)

    def surface(self, regime: int | str | None = None) -> np.ndarray:
        """Learned risky-fraction policy over (time, wealth) → (T, n_wbins).

        For regime-conditioned agents, pass a regime index/name; otherwise the
        regime axis is ignored.
        """
        ma = self.mean_action
        if ma is None:
            raise RuntimeError("policy not solved/trained yet")
        if ma.ndim == 3:
            if regime is None:
                k = 0
            elif isinstance(regime, str):
                k = self.mm.regime_names.index(regime)
            else:
                k = int(regime)
            return ma[:, :, k]
        return ma

    # --- persistence ---------------------------------------------------- #
    _SAVE_KEYS = ("n_wbins", "gamma", "beta", "greedy", "reward_scale")

    def save(self, path) -> None:
        import numpy as _np
        meta = dict(
            n_wealth_bins=self.n_wbins,
            n_actions=len(self.action_grid),
            gamma=self.gamma,
            beta=self.beta,
            greedy=int(self.greedy),
            reward_scale=self.reward_scale,
            kind=type(self).__name__,
        )
        _np.savez(path, mean_action=self.mean_action, meta=_np.array([str(meta)], dtype=object))

    @classmethod
    def load(cls, path, config):
        import ast

        import numpy as _np
        data = _np.load(path, allow_pickle=True)
        meta = ast.literal_eval(str(data["meta"][0]))
        obj = cls(
            config,
            n_wealth_bins=int(meta["n_wealth_bins"]),
            n_actions=int(meta["n_actions"]),
            gamma=float(meta["gamma"]),
            beta=float(meta["beta"]),
            greedy=bool(meta["greedy"]),
            reward_scale=float(meta["reward_scale"]),
            solve=False,
        )
        obj.mean_action = data["mean_action"]
        return obj

    def solve(self) -> None:  # pragma: no cover
        raise NotImplementedError

    @classmethod
    def _kwargs_from_config(cls, config: Config) -> dict:
        p = config.agents.g_learner
        return dict(
            n_wealth_bins=int(p.get("n_wealth_bins", 201)),
            n_actions=int(p.get("n_actions", 21)),
            gamma=float(p.get("gamma", 0.999)),
            beta=float(p.get("beta", 15.0)),
            greedy=bool(p.get("greedy", True)),
            reward_scale=float(p.get("reward_scale", 1.0)),
        )


@policy_registry.register("g_learner")
class GLearner(_BaseGLearner):
    """Regime-agnostic G-Learner (stationary-mixture return model)."""

    name = "G-Learner"
    requires_belief = False

    def solve(self) -> None:
        R_nodes, probs = [], []
        for k in range(self.mm.n_regimes):
            R_nodes.append(self._returns_for_regime(k))
            probs.append(self._p * self.pi_stationary[k])
        R_nodes = np.concatenate(R_nodes)
        probs = np.concatenate(probs)
        probs /= probs.sum()

        port = self.action_grid[:, None] * R_nodes[None, :] + (
            1.0 - self.action_grid
        )[:, None] * self.cash_return  # (A_act, M)
        factor = self.w_grid + self.c
        bin_idx = self._bin(factor[:, None, None] * (1.0 + port[None, :, :]))  # (W, A_act, M)

        mean_action = np.empty((self.T, self.n_wbins))
        V_next = self._terminal_value()
        for t in range(self.T - 1, -1, -1):
            EV = (V_next[bin_idx] * probs[None, None, :]).sum(axis=2)
            action, V_next = self._backup(self.gamma * EV)
            mean_action[t] = action
        self.mean_action = mean_action

    @classmethod
    def from_config(cls, config: Config, market_model: MarketModel | None = None) -> "GLearner":
        return cls(config, market_model, **cls._kwargs_from_config(config))

    def weights(self, ctx: DecisionContext) -> np.ndarray:
        t = min(ctx.step, self.T - 1)
        return self._split(self.mean_action[t, self._bin(ctx.wealth)])


@policy_registry.register("regime_aware_g_learner")
class RegimeAwareGLearner(_BaseGLearner):
    """Regime-conditioned G-Learner; mixes per-regime policies by the belief."""

    name = "Regime-Aware G-Learner"
    requires_belief = True

    def solve(self) -> None:
        K = self.mm.n_regimes
        factor = self.w_grid + self.c
        bin_idx = np.empty(
            (K, self.n_wbins, len(self.action_grid), len(self._z)), dtype=np.int64
        )
        for k in range(K):
            Rk = self._returns_for_regime(k)
            port = self.action_grid[:, None] * Rk[None, :] + (
                1.0 - self.action_grid
            )[:, None] * self.cash_return
            bin_idx[k] = self._bin(factor[:, None, None] * (1.0 + port[None, :, :]))

        mean_action = np.empty((self.T, self.n_wbins, K))
        term = self._terminal_value()
        V_next = np.repeat(term[:, None], K, axis=1)  # (W, K)
        for t in range(self.T - 1, -1, -1):
            V_cur = np.empty((self.n_wbins, K))
            for k in range(K):
                ev_next = V_next[bin_idx[k]] @ self.P[k]  # (W, A_act, n_quad)
                EV = (ev_next * self._p[None, None, :]).sum(axis=2)
                action, V_cur[:, k] = self._backup(self.gamma * EV)
                mean_action[t, :, k] = action
            V_next = V_cur
        self.mean_action = mean_action

    @classmethod
    def from_config(
        cls, config: Config, market_model: MarketModel | None = None
    ) -> "RegimeAwareGLearner":
        return cls(config, market_model, **cls._kwargs_from_config(config))

    def weights(self, ctx: DecisionContext) -> np.ndarray:
        t = min(ctx.step, self.T - 1)
        per_regime = self.mean_action[t, self._bin(ctx.wealth)]  # (N, K)
        frac = np.einsum("nk,nk->n", ctx.belief, per_regime)
        return self._split(frac)
