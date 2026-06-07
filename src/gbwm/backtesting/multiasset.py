"""True multi-asset historical backtest — *S&P 500 + International + Bonds + Gold,
allocated all at once, live, with no look-ahead.*

Where :mod:`gbwm.backtesting.historical` rolls every strategy over **one** risky
market (a single index/basket collapsed to one return series), this module rolls
them over a **panel of several risky assets at once** and lets a regime-aware
allocator split capital across them in real time — loading up on stocks in good
weather and rotating into bonds/gold when storms hit.

Two things make it honest:

* **Real, aligned data, no peeking.** Each asset's monthly (log) returns are
  fetched from its longest-history proxy and aligned on the common month-ends
  (so the walk-forward starts only when *all* assets exist). Offline, a
  deterministic synthetic panel stands in (clearly flagged).
* **Regimes learned only from the past.** Three honesty levels are offered:
  ``prior`` (hand-set economic regimes), ``causal_calibrate`` (fit the HMM once
  on data strictly *before* the start), and ``causal_walk_forward`` (re-learn the
  regimes every year using only data seen so far — the strictest test, and the
  default for the app).

Everything reuses the validated core: the multi-asset :class:`MarketModel`, the
:class:`~gbwm.detection.filter.GaussianRegimeFilter`, the HMM calibrator and the
single-path roller in :mod:`gbwm.evaluation.harness`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gbwm.config import Config


# --------------------------------------------------------------------------- #
# The investable universe: one AssetSpec per risky leg (order matches the
# configs/sp_intl_bonds_gold.yaml asset order exactly).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AssetSpec:
    """A risky asset leg and the real proxy used to fetch its history.

    ``ticker`` is the preferred (longest-history, total-return where possible)
    proxy; ``fallback`` is a shorter-history stand-in used if the preferred one
    is unavailable. ``key`` must match the config asset name.
    """

    key: str
    label: str          # human-friendly (shown in the app)
    ticker: str         # preferred proxy
    fallback: str       # shorter-history fallback proxy
    note: str
    inception: str      # earliest date the preferred proxy has data


# Preferred proxies chosen for the *longest* clean history (verified on Yahoo):
#   SPY 1993 · VGTSX 1996 · VBMFX 1990 · GC=F 2000-08  -> common start ~2000-09.
MULTI_ASSET_UNIVERSE: list[AssetSpec] = [
    AssetSpec("us_equity", "S&P 500 (US)", "SPY", "^GSPC",
              "The 500 biggest US companies — the growth engine.",
              "1993-01-29"),
    AssetSpec("intl_equity", "International stocks", "VGTSX", "EFA",
              "Developed markets outside the US (Europe, Japan…) — diversifies across countries.",
              "1996-04-29"),
    AssetSpec("bonds", "Bonds (fixed income)", "VBMFX", "AGG",
              "US bonds — the stabilizing ballast; they often rise when stocks fall.",
              "1990-01-02"),
    AssetSpec("gold", "Gold", "GC=F", "GLD",
              "Gold — the classic safe haven: it shines in panics and inflation.",
              "2000-08-30"),
]


def list_universe() -> list[AssetSpec]:
    """The risky-asset universe, in the fixed config order."""
    return list(MULTI_ASSET_UNIVERSE)


# --------------------------------------------------------------------------- #
# Data: a panel of real monthly (log) returns, one column per asset, aligned.
# --------------------------------------------------------------------------- #
@dataclass
class AssetPanel:
    """Aligned monthly log-returns for the whole universe."""

    keys: list[str]                 # asset keys, in column order
    labels: list[str]               # human labels, in column order
    dates: pd.DatetimeIndex         # (T,) common month-end dates
    r_log: np.ndarray               # (T, A) monthly log-returns
    source: str                     # cache | yfinance | synthetic
    tickers_used: dict[str, str] = field(default_factory=dict)

    @property
    def n_assets(self) -> int:
        return self.r_log.shape[1]

    @property
    def n_months(self) -> int:
        return self.r_log.shape[0]


def _monthly_log_from_prices(prices: pd.Series) -> pd.Series:
    """Daily prices -> month-end compounded **log**-returns (single asset)."""
    s = prices.dropna().sort_index()
    daily_simple = s.pct_change().dropna()
    monthly_simple = (1.0 + daily_simple).resample("ME").prod() - 1.0
    return np.log1p(monthly_simple).dropna()


def fetch_asset_panel(
    universe: list[AssetSpec] | None = None,
    start: str | None = None,
    end: str | None = None,
    *,
    offline: bool = False,
    cache_dir: str = "data/cache",
    min_months: int = 24,
) -> AssetPanel:
    """Fetch and align a real monthly-return panel for the whole universe.

    All assets are fetched in one call (so the offline synthetic fallback yields
    distinct series per asset), each converted to month-end log-returns, then
    aligned on the **intersection** of month-ends — i.e. the walk-forward only
    spans dates on which *every* asset already exists (no forward fill, no
    survivorship hindsight).
    """
    from gbwm.data.providers import MarketDataProvider

    universe = universe or MULTI_ASSET_UNIVERSE
    # Fetch from well before any inception; the alignment below trims to the
    # common window where every asset actually exists.
    start = start or "1990-01-01"
    prov = MarketDataProvider(cache_dir, offline=offline)
    prices = prov.fetch_prices([a.ticker for a in universe], start, end)
    source = prov.last_source

    monthly: dict[str, pd.Series] = {}
    tickers_used: dict[str, str] = {}
    for a in universe:
        col, m = a.ticker, pd.Series(dtype=float)
        if a.ticker in prices.columns:
            m = _monthly_log_from_prices(prices[a.ticker])
        # Live data missing/too short -> try the shorter-history fallback proxy.
        if len(m) < min_months and a.fallback and not offline:
            fb = prov.fetch_prices([a.fallback], start, end)
            if a.fallback in fb.columns:
                mfb = _monthly_log_from_prices(fb[a.fallback])
                if len(mfb) > len(m):
                    m, col = mfb, a.fallback
        monthly[a.key] = m
        tickers_used[a.key] = col

    df = pd.DataFrame(monthly).dropna(how="any")
    if df.empty:
        raise ValueError("no overlapping monthly history across the asset universe")
    return AssetPanel(
        keys=[a.key for a in universe],
        labels=[a.label for a in universe],
        dates=pd.DatetimeIndex(df.index),
        r_log=df.to_numpy(dtype=float),
        source=source,
        tickers_used=tickers_used,
    )


# --------------------------------------------------------------------------- #
# Strategies compared in the multi-asset demo
# --------------------------------------------------------------------------- #
# Display labels for the three portfolios (policy.name -> label).
STRATEGY_LABELS = {
    "Regime-Aware Multi-Asset": "Smart plan (multi-asset)",
    "Balanced Mix": "Classic balanced portfolio",
    "All-in S&P 500": "All-in S&P 500",
}

# A classic balanced multi-asset mix over [us_equity, intl_equity, bonds, gold].
_BALANCED_MIX = np.array([0.45, 0.15, 0.35, 0.05])


def default_multi_asset_config() -> Config:
    """Load the bundled 4-asset (S&P + intl + bonds + gold) config."""
    from pathlib import Path

    from gbwm.config import load_config

    root = Path(__file__).resolve().parents[3]
    return load_config(root / "configs" / "sp_intl_bonds_gold.yaml")


def build_multi_asset_policies(config: Config, market_model=None) -> dict:
    """The three compared portfolios: the smart regime-aware allocator, a classic
    balanced mix, and all-in S&P 500. Keyed by ``policy.name``."""
    from gbwm.policies.multi_asset import RegimeAwareMultiAsset, StaticMultiAsset
    from gbwm.simulation.regimes import MarketModel

    mm = market_model or MarketModel.from_config(config.market)
    A = config.market.n_risky
    balanced = _BALANCED_MIX[:A] if A == 4 else np.full(A, 0.8 / A)
    all_sp = np.eye(A)[0]
    pols = [
        RegimeAwareMultiAsset.from_config(config, mm),
        StaticMultiAsset(balanced, name="Balanced Mix"),
        StaticMultiAsset(all_sp, name="All-in S&P 500"),
    ]
    return {p.name: p for p in pols}


# --------------------------------------------------------------------------- #
# The multi-asset deployment result
# --------------------------------------------------------------------------- #
@dataclass
class MultiAssetDeployment:
    """One honest walk-forward of every strategy over a real multi-asset panel.

    Duck-compatible with :func:`gbwm.backtesting.historical.scorecard` (exposes
    ``results``, ``config`` and ``target``).
    """

    dates: pd.DatetimeIndex          # (T,) month-end decision dates
    r_log: np.ndarray                # (T, A) realized monthly log-returns
    results: dict                    # strategy name -> single-path PolicyResult
    config: Config
    asset_keys: list[str]
    asset_labels: list[str]
    regime_names: list[str]
    source: str                      # cache | yfinance | synthetic
    regime_mode: str                 # prior | causal_calibrate | causal_walk_forward
    tickers_used: dict[str, str] = field(default_factory=dict)

    @property
    def target(self) -> float:
        return self.config.goal.target_wealth

    @property
    def horizon_years(self) -> int:
        return self.config.goal.horizon_years

    @property
    def n_assets(self) -> int:
        return self.r_log.shape[1]

    def asset_price_index(self) -> np.ndarray:
        """Growth of $1 in each asset (buy-and-hold), shape (T+1, A)."""
        gross = np.cumprod(np.expm1(self.r_log) + 1.0, axis=0)
        return np.vstack([np.ones((1, self.n_assets)), gross])

    # --- per-strategy views ------------------------------------------------- #
    def wealth(self, strategy: str) -> np.ndarray:
        return self.results[strategy].histories["wealth"][0]

    def weights(self, strategy: str) -> np.ndarray:
        """Per-asset risky weights over time, shape (T, A)."""
        return self.results[strategy].histories["weights"][0]

    def cash(self, strategy: str) -> np.ndarray:
        return 1.0 - self.weights(strategy).sum(axis=1)

    def equity(self, strategy: str) -> np.ndarray:
        """Total fraction invested in risky assets (1 - cash)."""
        return self.weights(strategy).sum(axis=1)

    def stock_equity(self, strategy: str) -> np.ndarray:
        """Fraction in *stocks* (us + intl equity) — the crash-exposed sleeve."""
        idx = [i for i, k in enumerate(self.asset_keys) if k in ("us_equity", "intl_equity")]
        return self.weights(strategy)[:, idx].sum(axis=1)

    def safe_haven(self, strategy: str) -> np.ndarray:
        """Fraction in bonds + gold — the defensive sleeve it rotates into."""
        idx = [i for i, k in enumerate(self.asset_keys) if k in ("bonds", "gold")]
        return self.weights(strategy)[:, idx].sum(axis=1)

    def belief(self, strategy: str) -> np.ndarray:
        return self.results[strategy].histories["belief"][0]

    def risk_belief(self, strategy: str) -> np.ndarray:
        """P(bear) + P(high-vol) over time — the 'bad weather' gauge."""
        bel = self.belief(strategy)
        idx = [i for i, n in enumerate(self.regime_names) if n in ("bear", "high_vol")]
        return bel[:, idx].sum(axis=1)


# --------------------------------------------------------------------------- #
# Honesty: re-learn regimes only from data prior to each date
# --------------------------------------------------------------------------- #
_FREQ_STEPS = {"monthly": 1, "quarterly": 3, "annual": 12}


def effective_step_contribution(amount: float, freq: str, steps_per_year: int = 12) -> float:
    """Convert a per-period contribution into the per-step (monthly) equivalent.

    e.g. $1,500 'quarterly' -> $500 each month. Keeps the total contributed and
    the engine's per-step cashflow simple while letting the user pick a cadence.
    """
    per = _FREQ_STEPS.get(freq, 1)
    return float(amount) / per


def _calibrate_config(base_cfg: Config, history_log: np.ndarray, *, fast: bool):
    """Return a config whose regimes are re-learned from ``history_log`` only."""
    from gbwm.data.calibration import apply_calibration_to_config, calibrate_regimes
    from gbwm.experiments import clone_config

    cfg = clone_config(base_cfg)
    calib = calibrate_regimes(
        np.asarray(history_log, dtype=float),
        steps_per_year=base_cfg.steps_per_year,
        n_states=base_cfg.hmm.n_states,
        names=base_cfg.market.regime_names,
        seed=base_cfg.seed,
        n_restarts=3 if fast else 6,
        n_iter=120 if fast else 200,
    )
    return apply_calibration_to_config(cfg, calib), calib


def _smart_walk_forward_factory(base_cfg: Config, min_calib_months: int):
    """Factory for the walk-forward roller: calibrate regimes on the supplied
    history (only past data) and re-solve the smart allocator."""
    from gbwm.policies.multi_asset import RegimeAwareMultiAsset
    from gbwm.simulation.regimes import MarketModel

    def make(history_log):
        cfg = base_cfg
        if history_log is not None and len(history_log) >= min_calib_months:
            cfg, _ = _calibrate_config(base_cfg, history_log, fast=True)
        mm = MarketModel.from_config(cfg.market)
        return mm, RegimeAwareMultiAsset.from_config(cfg, mm)

    return make


# --------------------------------------------------------------------------- #
# The honest multi-asset walk-forward
# --------------------------------------------------------------------------- #
def run_multi_asset_deployment(
    config: Config | None = None,
    *,
    start: str | None = None,
    end: str | None = None,
    horizon_years: int | None = None,
    initial: float | None = None,
    contribution: float | None = None,
    contribution_freq: str = "monthly",
    target: float | None = None,
    regime_mode: str = "causal_walk_forward",
    recalibrate_every: int | None = None,
    universe: list[AssetSpec] | None = None,
    offline: bool = False,
    min_calib_months: int | None = None,
) -> MultiAssetDeployment:
    """Roll the three portfolios over the real S&P/intl/bonds/gold panel.

    ``regime_mode``:

    * ``"prior"`` — hand-set economic regimes (robust, never fitted to data).
    * ``"causal_calibrate"`` — fit the HMM **once** on data strictly before
      ``start`` (the pre-deployment history).
    * ``"causal_walk_forward"`` — re-fit the HMM **every year** from data seen so
      far (default; the strictest honesty).
    """
    from gbwm.evaluation.harness import run_on_returns, run_on_returns_walk_forward
    from gbwm.experiments import clone_config
    from gbwm.simulation.regimes import MarketModel

    base = config or default_multi_asset_config()
    universe = universe or MULTI_ASSET_UNIVERSE
    spy = base.steps_per_year
    recalibrate_every = recalibrate_every or spy
    min_calib_months = min_calib_months or max(60, 5 * spy)

    panel = fetch_asset_panel(universe, end=end, offline=offline, cache_dir=base.data.cache_dir)

    # Split the aligned panel into pre-start "prior" history and the deployment
    # window anchored at `start` (defaults to the first common month).
    if start is not None:
        start_ts = pd.Timestamp(start)
        start_idx = int(np.searchsorted(panel.dates.values, np.datetime64(start_ts)))
    else:
        start_idx = 0
    prior_log = panel.r_log[:start_idx]
    realized = panel.r_log[start_idx:]
    realized_dates = panel.dates[start_idx:]
    if len(realized) < spy:
        where = start or panel.dates[0].date()
        raise ValueError(f"only {len(realized)} months of overlapping history from {where}")

    n_years = horizon_years or max(1, len(realized) // spy)
    keep = n_years * spy
    if keep > len(realized):
        raise ValueError(
            f"requested {n_years}y ({keep} months) but only {len(realized)} months available "
            f"from {start or panel.dates[0].date()}."
        )
    realized = realized[:keep]
    realized_dates = realized_dates[:keep]

    cfg = clone_config(base)
    cfg.goal.horizon_years = int(n_years)
    if initial is not None:
        cfg.goal.initial_wealth = float(initial)
    if contribution is not None:
        cfg.goal.contribution = effective_step_contribution(contribution, contribution_freq, spy)
    if target is not None:
        cfg.goal.target_wealth = float(target)

    simple = np.expm1(realized)  # (T, A)
    have_prior = len(prior_log) >= min_calib_months
    mode_used = regime_mode

    # The market model used for the STATIC baselines (they ignore regimes; this
    # only sets the cash rate + the recorded belief for display).
    if regime_mode == "causal_calibrate" and have_prior:
        cfg_static, _ = _calibrate_config(cfg, prior_log, fast=False)
    elif regime_mode == "causal_calibrate":
        warnings.warn("not enough pre-start history to calibrate; using economic-prior regimes.",
                      stacklevel=2)
        cfg_static, mode_used = cfg, "prior"
    else:
        cfg_static = cfg
    mm_static = MarketModel.from_config(cfg_static.market)

    results: dict = {}
    static_pols = build_multi_asset_policies(cfg_static, mm_static)
    for name, pol in static_pols.items():
        is_smart = name == "Regime-Aware Multi-Asset"
        if is_smart and mode_used == "causal_walk_forward":
            factory = _smart_walk_forward_factory(cfg, min_calib_months)
            results[name] = run_on_returns_walk_forward(
                cfg, simple, factory, prior_returns_log=prior_log,
                recalibrate_every=recalibrate_every,
            )
        else:
            results[name] = run_on_returns(pol, cfg_static, simple, market_model=mm_static)

    return MultiAssetDeployment(
        dates=pd.DatetimeIndex(realized_dates),
        r_log=realized,
        results=results,
        config=cfg_static,
        asset_keys=panel.keys,
        asset_labels=panel.labels,
        regime_names=mm_static.regime_names,
        source=panel.source,
        regime_mode=mode_used,
        tickers_used=panel.tickers_used,
    )


# --------------------------------------------------------------------------- #
# Decision diary — what the smart allocator *did* at the historical turning points
# --------------------------------------------------------------------------- #
def multi_asset_diary(dep: MultiAssetDeployment,
                      smart: str = "Regime-Aware Multi-Asset") -> list[str]:
    """Plain-English lines on how the smart plan rotated into bonds/gold at each
    in-window crisis (display only — the agent discovered each crash online)."""
    from gbwm.backtesting.historical import CRISES

    if smart not in dep.results:
        return []
    dates = dep.dates
    sp_price = dep.asset_price_index()[:, 0]  # growth of $1 in the S&P (col 0)
    stock = dep.stock_equity(smart)
    safe = dep.safe_haven(smart)
    rb = dep.risk_belief(smart)
    out: list[str] = []
    for cr in CRISES:
        lo = pd.Timestamp(cr.start + "-01")
        hi = pd.Timestamp(cr.end + "-28")
        idx = np.where((dates >= lo) & (dates <= hi))[0]
        if idx.size == 0:
            continue
        a, b = int(idx[0]), int(idx[-1])
        seg = sp_price[a:b + 2]
        mkt = float(seg.min() / seg.max() - 1.0) if seg.size else 0.0
        out.append(
            f"**{cr.name}** ({dates[a]:%b %Y}–{dates[b]:%b %Y}): the S&P fell "
            f"{abs(mkt) * 100:.0f}%. The smart plan cut stocks from "
            f"{stock[a] * 100:.0f}% to {stock[a:b + 1].min() * 100:.0f}% and lifted "
            f"bonds+gold up to {safe[a:b + 1].max() * 100:.0f}% (bad-weather belief "
            f"peaked at {rb[a:b + 1].max() * 100:.0f}%)."
        )
    return out
