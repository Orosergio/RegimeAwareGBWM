"""Deep-RL allocators (PPO / SAC) via Stable-Baselines3, behind the Policy API.

These are strong *learned* comparators for the G-Learner and handle full
multi-asset continuous allocation. SB3/torch/gymnasium are the optional ``rl``
extra; everything here imports them lazily so the core stays dependency-light
(ADR-008). Training is offline (CLI / scripts); the demo loads checkpoints
(ADR-002).

The deploy-time observation is built with the *same* assembler the env uses
(:func:`gbwm.envs.wealth_env.assemble_obs`), so training and inference can't
drift. RL agents are trained and deployed with ``observe_regime='probs'`` (the
HMM belief) so they are regime-aware comparators.
"""

from __future__ import annotations

import numpy as np

from gbwm.config import Config
from gbwm.envs.wealth_env import WealthEnv, assemble_obs
from gbwm.policies.base import DecisionContext, Policy, policy_registry
from gbwm.simulation.regimes import MarketModel

_INSTALL_HINT = (
    "Deep-RL agents need the optional 'rl' extra. Install with:\n"
    '    pip install -e ".[rl]"   (gymnasium + stable-baselines3 + torch)'
)


def _require_sb3():
    try:
        import stable_baselines3 as sb3  # noqa: F401
    except Exception as exc:  # pragma: no cover - exercised only without rl extra
        raise ImportError(_INSTALL_HINT) from exc
    return sb3


def _project_long_only(action: np.ndarray, floor: float, cap: float, leverage: bool) -> np.ndarray:
    w = np.clip(np.asarray(action, dtype=float), floor, cap)
    if not leverage:
        s = w.sum(axis=-1, keepdims=True)
        scale = np.where(s > 1.0, s, 1.0)
        w = w / scale
    return w


def _tail_from_belief(belief: np.ndarray, regime_obs: str) -> np.ndarray:
    belief = np.asarray(belief, dtype=float)
    if regime_obs == "probs":
        return belief
    if regime_obs == "onehot":
        oh = np.zeros_like(belief)
        oh[np.arange(len(belief)), belief.argmax(axis=1)] = 1.0
        return oh
    return np.zeros((belief.shape[0], 0))


class SB3Policy(Policy):
    """Wrap a trained SB3 model (or any object exposing ``predict``)."""

    def __init__(self, model, config: Config, algo: str = "ppo") -> None:
        self.model = model
        self.cfg = config
        self.algo = algo
        self.name = algo.upper()
        self.requires_belief = config.env.observe_regime == "probs"
        self.regime_obs = config.env.observe_regime
        self.target = config.goal.target_wealth
        self.n_assets = config.market.n_risky

    def weights(self, ctx: DecisionContext) -> np.ndarray:
        tail = _tail_from_belief(ctx.belief, self.regime_obs)
        obs = assemble_obs(ctx.time_frac, ctx.wealth, ctx.target, tail)
        action, _ = self.model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=float).reshape(len(ctx.wealth), self.n_assets)
        return _project_long_only(
            action, self.cfg.env.action_floor, self.cfg.env.action_cap, self.cfg.env.allow_leverage
        )

    # --- persistence ---------------------------------------------------- #
    def save(self, path: str) -> None:
        self.model.save(path)

    @classmethod
    def load(cls, path: str, config: Config, algo: str) -> "SB3Policy":
        sb3 = _require_sb3()
        Algo = {"ppo": sb3.PPO, "sac": sb3.SAC}[algo]
        return cls(Algo.load(path), config, algo)


@policy_registry.register("ppo")
class PPOPolicy(SB3Policy):
    name = "PPO"

    @classmethod
    def from_checkpoint(cls, path: str, config: Config) -> "PPOPolicy":
        return cls.load(path, config, "ppo")  # type: ignore[return-value]


@policy_registry.register("sac")
class SACPolicy(SB3Policy):
    name = "SAC"

    @classmethod
    def from_checkpoint(cls, path: str, config: Config) -> "SACPolicy":
        return cls.load(path, config, "sac")  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
def train_agent(
    algo: str,
    config: Config,
    market_model: MarketModel | None = None,
    total_timesteps: int | None = None,
    seed: int | None = None,
    verbose: int = 0,
):
    """Train a PPO/SAC agent on :class:`WealthEnv` and return the SB3 model.

    Offline training entrypoint (ADR-002). RL agents should observe the regime
    belief, so we force ``observe_regime='probs'`` for training.
    """
    sb3 = _require_sb3()
    from stable_baselines3.common.monitor import Monitor

    if config.env.observe_regime == "none":
        config.env.observe_regime = "probs"

    mm = market_model or MarketModel.from_config(config.market)
    env = Monitor(WealthEnv(config, mm))
    params = config.agents.ppo if algo == "ppo" else config.agents.sac
    steps = total_timesteps or int(params.get("total_timesteps", 200_000))
    common = dict(
        seed=seed if seed is not None else config.seed,
        verbose=verbose,
        gamma=float(params.get("gamma", 0.999)),
        learning_rate=float(params.get("learning_rate", 3e-4)),
    )
    if algo == "ppo":
        model = sb3.PPO(
            "MlpPolicy",
            env,
            n_steps=int(params.get("n_steps", 2048)),
            batch_size=int(params.get("batch_size", 256)),
            **common,
        )
    elif algo == "sac":
        model = sb3.SAC(
            "MlpPolicy", env, batch_size=int(params.get("batch_size", 256)), **common
        )
    else:
        raise ValueError(f"unknown algo '{algo}' (expected 'ppo' or 'sac')")
    model.learn(total_timesteps=steps, progress_bar=False)
    return model


def train_ppo_with_curve(
    config: Config,
    total_timesteps: int = 60_000,
    eval_freq: int = 10_000,
    eval_episodes: int = 600,
    market_model: MarketModel | None = None,
    seed: int | None = None,
    progress_cb=None,
):
    """Train PPO on WealthEnv and record a *learning curve* of P(goal) vs steps.

    Returns (PPO SB3Policy, curve=[(timesteps, p_goal), ...]). Lets the demo show
    a deep-RL agent visibly improving as it trains. Requires the ``rl`` extra.
    """
    sb3 = _require_sb3()
    import numpy as _np
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor

    from gbwm.evaluation.harness import run_policy

    if config.env.observe_regime == "none":
        config.env.observe_regime = "probs"
    mm = market_model or MarketModel.from_config(config.market)
    params = config.agents.ppo
    model = sb3.PPO(
        "MlpPolicy", Monitor(WealthEnv(config, mm)),
        seed=seed if seed is not None else config.seed, verbose=0,
        gamma=float(params.get("gamma", 0.999)),
        learning_rate=float(params.get("learning_rate", 3e-4)),
        n_steps=int(params.get("n_steps", 2048)),
        batch_size=int(params.get("batch_size", 256)),
    )
    curve: list[tuple[int, float]] = []

    def _evaluate(model_) -> float:
        pol = SB3Policy(model_, config, "ppo")
        return run_policy(pol, config, n_episodes=eval_episodes, rng=_np.random.default_rng(0)).p_goal

    class _Curve(BaseCallback):
        def __init__(self):
            super().__init__()
            self.last = 0

        def _on_step(self) -> bool:
            if self.num_timesteps - self.last >= eval_freq:
                self.last = self.num_timesteps
                pg = _evaluate(self.model)
                curve.append((int(self.num_timesteps), float(pg)))
                if progress_cb:
                    progress_cb(self.num_timesteps, total_timesteps, pg)
            return True

    model.learn(total_timesteps=total_timesteps, callback=_Curve(), progress_bar=False)
    pol = SB3Policy(model, config, "ppo")
    curve.append((int(total_timesteps), _evaluate(model)))
    return pol, curve
