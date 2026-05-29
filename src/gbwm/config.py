"""Typed, validated configuration for the Regime-Aware GBWM lab.

The whole system is configuration-driven: simulators, the environment, policies
and the demo all read from a single validated :class:`Config` tree. This keeps
the core library free of hard-coded numbers and makes experiments reproducible.

Implementation note
-------------------
We use stdlib ``dataclasses`` (not a third-party validation lib) so the core has
**zero hard dependencies beyond NumPy/PyYAML** and runs anywhere — including
minimal CI sandboxes. Validation lives in ``__post_init__`` and raises
``ValueError`` with actionable messages. Regime drift/vol may be written as
scalars (single risky asset) or lists (multi-asset); they are coerced to lists
and length-checked against the number of risky assets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

Number = float | int


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_float_list(v: Any) -> list[float]:
    if isinstance(v, (int, float)):
        return [float(v)]
    return [float(x) for x in v]


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


# --------------------------------------------------------------------------- #
# Market / simulation
# --------------------------------------------------------------------------- #
@dataclass
class AssetConfig:
    name: str
    risky: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> AssetConfig:
        return cls(name=str(d["name"]), risky=bool(d.get("risky", True)))


@dataclass
class RegimeParams:
    """Annualized drift/vol per risky asset for one market regime."""

    name: str
    mu: list[float]
    sigma: list[float]

    def __post_init__(self) -> None:
        self.mu = _as_float_list(self.mu)
        self.sigma = _as_float_list(self.sigma)
        _require(
            len(self.mu) == len(self.sigma),
            f"regime '{self.name}': mu has {len(self.mu)} entries but sigma has "
            f"{len(self.sigma)}",
        )

    @classmethod
    def from_dict(cls, d: dict) -> RegimeParams:
        return cls(name=str(d["name"]), mu=d["mu"], sigma=d["sigma"])


@dataclass
class TransitionConfig:
    persistence: float = 0.85
    matrix: list[list[float]] | None = None

    def __post_init__(self) -> None:
        _require(0.0 <= self.persistence <= 1.0, "transition.persistence must be in [0,1]")

    @classmethod
    def from_dict(cls, d: dict | None) -> TransitionConfig:
        d = d or {}
        return cls(persistence=float(d.get("persistence", 0.85)), matrix=d.get("matrix"))


@dataclass
class MarketConfig:
    assets: list[AssetConfig]
    regimes: list[RegimeParams]
    steps_per_year: int = 12
    risk_free_rate: float = 0.03
    transition: TransitionConfig = field(default_factory=TransitionConfig)
    correlation: list[list[float]] | None = None
    stress_correlation_scale: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require(self.steps_per_year > 0, "steps_per_year must be > 0")
        n = self.n_risky
        _require(n >= 1, "market must contain at least one risky asset")
        for r in self.regimes:
            _require(
                len(r.mu) == n,
                f"regime '{r.name}' defines {len(r.mu)} drift(s) but there are {n} "
                f"risky asset(s)",
            )
        if self.correlation is not None:
            corr = np.asarray(self.correlation, dtype=float)
            _require(corr.shape == (n, n), f"correlation must be {n}x{n}, got {corr.shape}")
            _require(np.allclose(corr, corr.T, atol=1e-8), "correlation must be symmetric")
        if self.transition.matrix is not None:
            m = np.asarray(self.transition.matrix, dtype=float)
            k = self.n_regimes
            _require(m.shape == (k, k), f"transition matrix must be {k}x{k}")

    # --- derived -------------------------------------------------------- #
    @property
    def risky_assets(self) -> list[str]:
        return [a.name for a in self.assets if a.risky]

    @property
    def n_risky(self) -> int:
        return len(self.risky_assets)

    @property
    def regime_names(self) -> list[str]:
        return [r.name for r in self.regimes]

    @property
    def n_regimes(self) -> int:
        return len(self.regimes)

    # --- numpy views used by the simulator ------------------------------ #
    def drift_matrix(self) -> np.ndarray:
        """(n_regimes, n_risky) annualized drift."""
        return np.array([r.mu for r in self.regimes], dtype=float)

    def vol_matrix(self) -> np.ndarray:
        """(n_regimes, n_risky) annualized volatility."""
        return np.array([r.sigma for r in self.regimes], dtype=float)

    def correlation_matrix(self) -> np.ndarray:
        if self.correlation is None:
            return np.eye(self.n_risky)
        return np.asarray(self.correlation, dtype=float)

    def transition_matrix(self) -> np.ndarray:
        """Row-stochastic regime transition matrix."""
        k = self.n_regimes
        if self.transition.matrix is not None:
            m = np.asarray(self.transition.matrix, dtype=float)
        else:
            p = self.transition.persistence
            off = (1.0 - p) / (k - 1) if k > 1 else 0.0
            m = np.full((k, k), off)
            np.fill_diagonal(m, p)
        return m / m.sum(axis=1, keepdims=True)

    def covariances(self) -> np.ndarray:
        """(n_regimes, n_risky, n_risky) annualized covariance per regime,
        applying contagion scaling for stressed regimes."""
        vols = self.vol_matrix()
        base_corr = self.correlation_matrix()
        covs = np.empty((self.n_regimes, self.n_risky, self.n_risky))
        for k, reg in enumerate(self.regimes):
            corr = base_corr.copy()
            scale = self.stress_correlation_scale.get(reg.name)
            if scale is not None and self.n_risky > 1:
                off = corr.copy()
                np.fill_diagonal(off, 0.0)
                corr = np.eye(self.n_risky) + off * scale
                corr = np.clip(corr, -0.999, 0.999)
                np.fill_diagonal(corr, 1.0)
            s = vols[k]
            covs[k] = _nearest_psd(corr * np.outer(s, s))
        return covs

    @classmethod
    def from_dict(cls, d: dict) -> MarketConfig:
        return cls(
            assets=[AssetConfig.from_dict(a) for a in d["assets"]],
            regimes=[RegimeParams.from_dict(r) for r in d["regimes"]],
            steps_per_year=int(d.get("steps_per_year", 12)),
            risk_free_rate=float(d.get("risk_free_rate", 0.03)),
            transition=TransitionConfig.from_dict(d.get("transition")),
            correlation=d.get("correlation"),
            stress_correlation_scale={k: float(v) for k, v in (d.get("stress_correlation_scale") or {}).items()},
        )


def _nearest_psd(mat: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix to the nearest positive semi-definite one."""
    sym = (mat + mat.T) / 2.0
    vals, vecs = np.linalg.eigh(sym)
    vals = np.clip(vals, 1e-12, None)
    return (vecs * vals) @ vecs.T


# --------------------------------------------------------------------------- #
# Goal / env / reward
# --------------------------------------------------------------------------- #
@dataclass
class GoalConfig:
    initial_wealth: float
    target_wealth: float
    horizon_years: int
    contribution: float = 0.0
    contribution_freq: str = "monthly"
    risk_free_rate: float = 0.03

    def __post_init__(self) -> None:
        _require(self.initial_wealth > 0, "initial_wealth must be > 0")
        _require(self.target_wealth > 0, "target_wealth must be > 0")
        _require(self.horizon_years > 0, "horizon_years must be > 0")

    @classmethod
    def from_dict(cls, d: dict) -> GoalConfig:
        return cls(
            initial_wealth=float(d["initial_wealth"]),
            target_wealth=float(d["target_wealth"]),
            horizon_years=int(d["horizon_years"]),
            contribution=float(d.get("contribution", 0.0)),
            contribution_freq=str(d.get("contribution_freq", "monthly")),
            risk_free_rate=float(d.get("risk_free_rate", 0.03)),
        )


@dataclass
class RewardConfig:
    goal_bonus: float = 1.0
    shortfall_penalty: float = 1.0
    overshoot_credit: float = 0.05
    turnover_penalty: float = 0.02
    drawdown_penalty: float = 0.0
    step_shaping: float = 0.0
    utility: str = "power"
    risk_aversion: float = 3.0

    def __post_init__(self) -> None:
        _require(self.utility in {"power", "linear", "threshold"}, f"unknown utility '{self.utility}'")
        _require(self.risk_aversion > 0, "risk_aversion must be > 0")

    @classmethod
    def from_dict(cls, d: dict | None) -> RewardConfig:
        d = d or {}
        return cls(**{k: d[k] for k in d if k in RewardConfig.__dataclass_fields__})


@dataclass
class EnvConfig:
    rebalance_every: int = 1
    action_floor: float = 0.0
    action_cap: float = 1.0
    allow_leverage: bool = False
    observe_regime: str = "probs"
    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        _require(self.rebalance_every >= 1, "rebalance_every must be >= 1")
        _require(self.observe_regime in {"probs", "onehot", "none"}, "observe_regime invalid")

    @classmethod
    def from_dict(cls, d: dict | None) -> EnvConfig:
        d = d or {}
        return cls(
            rebalance_every=int(d.get("rebalance_every", 1)),
            action_floor=float(d.get("action_floor", 0.0)),
            action_cap=float(d.get("action_cap", 1.0)),
            allow_leverage=bool(d.get("allow_leverage", False)),
            observe_regime=str(d.get("observe_regime", "probs")),
            reward=RewardConfig.from_dict(d.get("reward")),
        )


@dataclass
class SimulationConfig:
    n_episodes: int = 1000
    antithetic: bool = True

    @classmethod
    def from_dict(cls, d: dict | None) -> SimulationConfig:
        d = d or {}
        return cls(n_episodes=int(d.get("n_episodes", 1000)), antithetic=bool(d.get("antithetic", True)))


@dataclass
class HMMConfig:
    n_states: int = 4
    covariance_type: str = "full"
    n_iter: int = 200
    lookback_steps: int = 12
    min_history: int = 6

    @classmethod
    def from_dict(cls, d: dict | None) -> HMMConfig:
        d = d or {}
        return cls(**{k: d[k] for k in d if k in HMMConfig.__dataclass_fields__})


@dataclass
class AgentsConfig:
    glide_path: dict[str, Any] = field(default_factory=dict)
    g_learner: dict[str, Any] = field(default_factory=dict)
    ppo: dict[str, Any] = field(default_factory=dict)
    sac: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict | None) -> AgentsConfig:
        d = dict(d or {})
        known = {k: d.pop(k, {}) or {} for k in ("glide_path", "g_learner", "ppo", "sac")}
        return cls(**known, extra=d)


@dataclass
class DataConfig:
    cache_dir: str = "data/cache"
    equity_ticker: str = "SPY"
    benchmark_tickers: list[str] = field(default_factory=lambda: ["SPY", "AGG", "GLD", "EFA"])
    fred_series: list[str] = field(default_factory=list)
    start: str = "2005-01-01"

    @classmethod
    def from_dict(cls, d: dict | None) -> DataConfig:
        d = d or {}
        return cls(**{k: d[k] for k in d if k in DataConfig.__dataclass_fields__})


@dataclass
class PathsConfig:
    artifacts: str = "artifacts"
    checkpoints: str = "artifacts/checkpoints"

    @classmethod
    def from_dict(cls, d: dict | None) -> PathsConfig:
        d = d or {}
        return cls(**{k: d[k] for k in d if k in PathsConfig.__dataclass_fields__})


@dataclass
class Config:
    goal: GoalConfig
    market: MarketConfig
    seed: int = 42
    env: EnvConfig = field(default_factory=EnvConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    hmm: HMMConfig = field(default_factory=HMMConfig)
    agents: AgentsConfig = field(default_factory=AgentsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    def __post_init__(self) -> None:
        # Keep the two risk-free declarations consistent; goal wins if they differ.
        if abs(self.goal.risk_free_rate - self.market.risk_free_rate) > 1e-12:
            self.market.risk_free_rate = self.goal.risk_free_rate

    # --- derived convenience -------------------------------------------- #
    @property
    def steps_per_year(self) -> int:
        return self.market.steps_per_year

    @property
    def total_steps(self) -> int:
        return self.goal.horizon_years * self.market.steps_per_year

    @property
    def dt(self) -> float:
        return 1.0 / self.market.steps_per_year

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    @classmethod
    def from_dict(cls, d: dict) -> Config:
        return cls(
            goal=GoalConfig.from_dict(d["goal"]),
            market=MarketConfig.from_dict(d["market"]),
            seed=int(d.get("seed", 42)),
            env=EnvConfig.from_dict(d.get("env")),
            simulation=SimulationConfig.from_dict(d.get("simulation")),
            hmm=HMMConfig.from_dict(d.get("hmm")),
            agents=AgentsConfig.from_dict(d.get("agents")),
            data=DataConfig.from_dict(d.get("data")),
            paths=PathsConfig.from_dict(d.get("paths")),
        )


# --------------------------------------------------------------------------- #
# Loading + merging
# --------------------------------------------------------------------------- #
def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_yaml(path: str | Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def load_config(path: str | Path, base: str | Path | None = None) -> Config:
    """Load and validate a config, optionally merged on top of a base file."""
    data = _read_yaml(path)
    if base is not None:
        data = _deep_merge(_read_yaml(base), data)
    return Config.from_dict(data)


def default_config() -> Config:
    """Load the repository default config (configs/default.yaml)."""
    root = Path(__file__).resolve().parents[2]
    return load_config(root / "configs" / "default.yaml")
