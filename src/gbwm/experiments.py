"""High-level operations (train / evaluate / backtest / calibrate).

These are plain functions so they are unit-testable and reusable by the CLI, the
Streamlit app, notebooks, or a future API (ADR-001). The CLI is a thin argparse
shell over this module.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from gbwm.checkpoints import ModelRegistry
from gbwm.config import Config
from gbwm.evaluation.harness import compare_policies, results_table, run_policy
from gbwm.explain import EpisodeContext, RuleBasedAdvisor
from gbwm.policies import GLearner, RegimeAwareGLearner
from gbwm.policies.base import policy_registry
from gbwm.utils.logging import get_logger
from gbwm.utils.seeding import make_rng

log = get_logger()

BASELINE_KEYS = ["buy_and_hold", "sixty_forty", "glide_path"]
GLEARNER_KEYS = ["g_learner", "regime_aware_g_learner"]
RL_KEYS = ["ppo", "sac"]
ALL_KEYS = BASELINE_KEYS + GLEARNER_KEYS + RL_KEYS


def _expand(which: str | list[str]) -> list[str]:
    if isinstance(which, list):
        return which
    if which == "all":
        return ALL_KEYS
    if which == "baselines":
        return BASELINE_KEYS
    return [w.strip() for w in which.split(",") if w.strip()]


def build_policies(config: Config, which: str | list[str] = "all", registry: ModelRegistry | None = None) -> dict:
    """Construct policies by key. Baselines and G-Learners are built fresh
    (G-Learners solve quickly); RL agents load from the registry if present."""
    pols: dict = {}
    for k in _expand(which):
        if k in BASELINE_KEYS:
            pol = policy_registry.get(k).from_config(config)
        elif k == "g_learner":
            pol = GLearner.from_config(config)
        elif k == "regime_aware_g_learner":
            pol = RegimeAwareGLearner.from_config(config)
        elif k in RL_KEYS:
            if registry is not None and registry.exists(k):
                pol = registry.load(k, config)
            else:
                warnings.warn(f"no checkpoint for '{k}' — train it first; skipping.")
                continue
        else:
            raise ValueError(f"unknown agent key '{k}'")
        pols[pol.name] = pol
    return pols


def train(config: Config, agent: str, registry: ModelRegistry, timesteps: int | None = None, name: str | None = None):
    """Train/solve one agent and store it in the registry. Returns CheckpointMeta."""
    name = name or agent
    if agent == "g_learner":
        pol = GLearner.from_config(config)
    elif agent == "regime_aware_g_learner":
        pol = RegimeAwareGLearner.from_config(config)
    elif agent in RL_KEYS:
        from gbwm.policies.rl_agents import SB3Policy, train_agent

        model = train_agent(agent, config, total_timesteps=timesteps)
        pol = SB3Policy(model, config, algo=agent)
    else:
        raise ValueError(f"agent '{agent}' is not trainable (baselines need no training)")
    # quick eval for metadata
    res = run_policy(pol, config, n_episodes=min(1000, config.simulation.n_episodes))
    meta = registry.save(name, pol, config, metrics={"p_goal": res.p_goal, "avg_shortfall": res.avg_shortfall})
    log.info("saved '%s' (%s): P(goal)=%.3f", name, agent, res.p_goal)
    return meta


def evaluate(config: Config, n_episodes: int | None = None, which: str | list[str] = "all", registry: ModelRegistry | None = None):
    """Evaluate policies on shared Monte-Carlo paths; returns (results, table)."""
    pols = build_policies(config, which, registry)
    results = compare_policies(pols, config, n_episodes=n_episodes)
    return results, results_table(results)


def backtest(config: Config, agent: str, seed: int | None = None, registry: ModelRegistry | None = None):
    """Roll a single path for one agent; returns (PolicyResult, episode explanation)."""
    pols = build_policies(config, [agent], registry)
    if not pols:
        raise ValueError(f"could not build agent '{agent}'")
    pol = next(iter(pols.values()))
    rng = make_rng(seed if seed is not None else config.seed)
    res = run_policy(pol, config, n_episodes=1, rng=rng)
    ep = EpisodeContext.from_histories(res.histories, config.goal.target_wealth, config.steps_per_year)
    return res, RuleBasedAdvisor().explain_episode(ep)


def calibrate(config: Config, offline: bool = False, save_path: str | None = None):
    """Fit regimes from (real or synthetic) equity returns; optional save of a
    calibrated config. Returns the RegimeCalibration."""
    from gbwm.data.calibration import apply_calibration_to_config, calibrate_regimes
    from gbwm.data.providers import load_equity_returns

    returns = load_equity_returns(config, offline=offline).to_numpy()
    calib = calibrate_regimes(
        returns, steps_per_year=config.steps_per_year, n_states=config.hmm.n_states,
        names=config.market.regime_names, seed=config.seed,
    )
    if save_path:
        apply_calibration_to_config(config, calib).to_yaml(save_path)
        log.info("wrote calibrated config -> %s", save_path)
    return calib


def clone_config(config: Config) -> Config:
    """Deep-ish clone via the dict round-trip (configs are plain data)."""
    return Config.from_dict(config.to_dict())


def calibrated_config(config: Config, returns_log: np.ndarray):
    """Return (config_with_data_driven_regimes, RegimeCalibration)."""
    from gbwm.data.calibration import apply_calibration_to_config, calibrate_regimes

    calib = calibrate_regimes(
        np.asarray(returns_log, dtype=float),
        steps_per_year=config.steps_per_year,
        n_states=config.hmm.n_states,
        names=config.market.regime_names,
        seed=config.seed,
    )
    return apply_calibration_to_config(config, calib), calib


def backtest_history(
    config: Config,
    agent: str,
    returns_log_monthly: np.ndarray,
    calibrate: bool = True,
    registry: ModelRegistry | None = None,
):
    """Backtest one strategy over a *real* monthly log-return series.

    Uses whole years of the most recent history, (optionally) calibrates the
    regimes from that window, solves the policy for that horizon, then rolls it
    over the actual returns. Returns (PolicyResult, used_config, calibration|None).
    """
    from gbwm.evaluation.harness import run_on_returns

    r = np.asarray(returns_log_monthly, dtype=float)
    spy = config.steps_per_year
    nyears = max(1, len(r) // spy)
    r = r[-nyears * spy:]  # whole years, most recent

    cfg = clone_config(config)
    cfg.goal.horizon_years = nyears
    calib = None
    if calibrate:
        cfg, calib = calibrated_config(cfg, r)

    pols = build_policies(cfg, [agent], registry)
    if not pols:
        raise ValueError(f"could not build agent '{agent}'")
    pol = next(iter(pols.values()))
    mm = __import__("gbwm.simulation.regimes", fromlist=["MarketModel"]).MarketModel.from_config(cfg.market)
    simple = np.expm1(r)
    res = run_on_returns(pol, cfg, simple, market_model=mm)
    return res, cfg, calib


# Friendly market name -> ETF/index proxy ticker (used by the demo).
MARKETS = {
    "S&P 500 — US large caps (SPY)": "SPY",
    "NASDAQ-100 — US tech (QQQ)": "QQQ",
    "Dow Jones (DIA)": "DIA",
    "Total US market (VTI)": "VTI",
    "Developed ex-US (EFA)": "EFA",
    "Emerging markets (EEM)": "EEM",
    "Japan (EWJ)": "EWJ",
    "China large caps (FXI)": "FXI",
    "Hong Kong (EWH)": "EWH",
    "Asia-Pacific (VPL)": "VPL",
    "Gold (GLD)": "GLD",
    "US bonds (AGG)": "AGG",
}


def backtest_all_on_history(
    config: Config,
    returns_log_monthly: np.ndarray,
    calibrate: bool = True,
    agents: list[str] | None = None,
    registry: ModelRegistry | None = None,
):
    """Roll *every* strategy over the same real monthly log-return series.

    Returns (results: name->PolicyResult, used_config, calibration|None). Each
    result is a single real path, so ``terminal_wealth`` has length 1.
    """
    from gbwm.evaluation.harness import run_on_returns
    from gbwm.simulation.regimes import MarketModel

    agents = agents or ["buy_and_hold", "sixty_forty", "glide_path",
                        "g_learner", "regime_aware_g_learner"]
    r = np.asarray(returns_log_monthly, dtype=float)
    spy = config.steps_per_year
    nyears = max(1, len(r) // spy)
    r = r[-nyears * spy:]

    cfg = clone_config(config)
    cfg.goal.horizon_years = nyears
    calib = None
    if calibrate:
        cfg, calib = calibrated_config(cfg, r)

    mm = MarketModel.from_config(cfg.market)
    simple = np.expm1(r)
    pols = build_policies(cfg, agents, registry)
    results = {name: run_on_returns(pol, cfg, simple, market_model=mm) for name, pol in pols.items()}
    return results, cfg, calib
