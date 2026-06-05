"""Honest, out-of-sample historical backtest — *"if you had deployed in 1999."*

This is the real-world stress test the simulation lab was built for. Unlike the
in-sample calibration demo (which fits the regimes to the *whole* realized path,
including the future), this engine **never uses future data**:

* **Economic-prior regimes.** The market model uses the four hand-set, common-sense
  regimes (bull / stable / high-vol / bear) from the config — assumptions an
  investor could write down on day one — *not* parameters fitted to what later
  happened. (Optionally, ``regime_mode="causal_calibrate"`` fits the HMM only on
  data **before** the deployment date — still no look-ahead.)
* **Online belief, no peeking.** Each month the agent classifies the realized
  return into a regime with the online Bayes filter, using only the past.
* **Walk forward through the *actual* returns.** We then roll every strategy over
  the real month-by-month history and record every allocation decision.

The headline finding is not "beat buy-and-hold on a single rising path" (you
cannot — max equity wins when stocks happen to go up). It is **risk-adjusted**:
the goal-based RL agents reach the goal with a *fraction* of the drawdown, and
de-risk in real time as the dot-com crash, the GFC and COVID unfold.

Everything here reuses the validated core (:func:`gbwm.evaluation.harness.run_on_returns`,
the :class:`~gbwm.detection.filter.GaussianRegimeFilter`, the G-Learner); this
module only adds the *data plumbing*, the *honest framing*, and the *scorecard /
decision-diary* read-outs.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from gbwm.config import Config
from gbwm.evaluation.metrics import PolicyResult
from gbwm.simulation.regimes import MarketModel


# --------------------------------------------------------------------------- #
# Market registry — long-history indices/baskets (back to 1999 where possible)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Market:
    """A real, investable market the backtest can run on.

    ``tickers`` is a single index (``kind="index"``) or several constituents that
    are combined into an equal-weight, daily-rebalanced basket (``kind="basket"``).
    """

    key: str
    label: str
    tickers: tuple[str, ...]
    note: str
    kind: str = "index"  # "index" | "basket"
    inception: str = "1999-01-01"  # earliest sensible start

    @property
    def is_basket(self) -> bool:
        return self.kind == "basket"


_MAG7 = ("AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA")

LONG_HISTORY_MARKETS: dict[str, Market] = {
    m.key: m
    for m in [
        Market("sp500", "S&P 500 — US large caps", ("^GSPC",),
               "The US blue-chip benchmark. Lived through dot-com, the GFC and COVID."),
        Market("nasdaq", "NASDAQ Composite — US tech", ("^IXIC",),
               "Tech-heavy: the biggest dot-com boom-and-bust and the AI-era melt-up."),
        Market("dowjones", "Dow Jones Industrial Average", ("^DJI",),
               "30 US industrial blue chips — the oldest US equity gauge."),
        Market("kospi", "KOSPI — South Korea", ("^KS11",),
               "Korea's main index — exposed to the 2000 tech bust and 2008 in full."),
        Market("nikkei", "Nikkei 225 — Japan", ("^N225",),
               "Japan's benchmark — a very different, range-bound regime history."),
        Market("hangseng", "Hang Seng — Hong Kong", ("^HSI",),
               "Hong Kong large caps — China cycle, 2008 and 2015 swings."),
        Market("dax", "DAX — Germany", ("^GDAXI",),
               "Germany's blue-chip index — a core European market."),
        Market("mag7", "Magnificent 7 — US mega-cap tech", _MAG7,
               "Equal-weight AAPL/MSFT/GOOGL/AMZN/NVDA/META/TSLA. Only complete from 2013 "
               "(TSLA listed 2010, META 2012) — a modern, extreme boom case.",
               kind="basket", inception="2013-01-01"),
    ]
}


def list_markets() -> list[Market]:
    """All registered long-history markets (in display order)."""
    return list(LONG_HISTORY_MARKETS.values())


# --------------------------------------------------------------------------- #
# Crisis windows used to *annotate* the realized path (display only — the agent
# never sees these; it discovers stress online from returns).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Crisis:
    name: str
    start: str  # "YYYY-MM"
    end: str  # "YYYY-MM"
    blurb: str


CRISES: list[Crisis] = [
    Crisis("Dot-com crash", "2000-03", "2002-10",
           "Tech bubble bursts; the S&P roughly halves over 2.5 years."),
    Crisis("Global Financial Crisis", "2007-10", "2009-03",
           "Housing/credit collapse; global equities fall ~50–57%."),
    Crisis("COVID-19 crash", "2020-02", "2020-04",
           "Fastest-ever ~34% drop, then a V-shaped rebound."),
    Crisis("2022 inflation bear", "2022-01", "2022-10",
           "Rate-shock selloff; stocks and bonds fall together (~25%)."),
]


# --------------------------------------------------------------------------- #
# Data: fetch a real monthly (log) return series for a market
# --------------------------------------------------------------------------- #
def _portfolio_monthly_log(prices: pd.DataFrame, market: Market) -> pd.Series:
    """Daily prices → monthly **log**-returns of the market (index or basket).

    For a basket we form an equal-weight, daily-rebalanced portfolio over the
    days on which *all* constituents trade (so the weights are well defined),
    then compound to month-end. For a single index it is just the month-end
    compounded log-return.
    """
    prices = prices.sort_index()
    if market.is_basket:
        cols = [c for c in market.tickers if c in prices.columns]
        if len(cols) < 2:
            raise ValueError(f"basket '{market.key}' needs >=2 constituents, found {cols}")
        panel = prices[cols].dropna(how="any")  # all names available → clean equal weight
        daily_simple = panel.pct_change().dropna(how="any")
        port_daily = daily_simple.mean(axis=1)  # equal-weight, daily-rebalanced
    else:
        col = market.tickers[0]
        s = prices[col] if col in prices.columns else prices.iloc[:, 0]
        port_daily = s.dropna().pct_change().dropna()
    monthly_simple = (1.0 + port_daily).resample("ME").prod() - 1.0
    return np.log1p(monthly_simple).dropna()


def fetch_market_returns(
    market: Market | str,
    start: str | None = None,
    end: str | None = None,
    *,
    offline: bool = False,
    cache_dir: str = "data/cache",
) -> tuple[pd.DatetimeIndex, np.ndarray, str]:
    """Fetch real monthly log-returns for ``market``.

    Returns ``(dates, log_returns, source)`` where ``source`` is one of
    ``cache`` / ``yfinance`` / ``synthetic`` (the offline fallback).
    """
    from gbwm.data.providers import MarketDataProvider

    if isinstance(market, str):
        market = LONG_HISTORY_MARKETS[market]
    start = start or market.inception
    prov = MarketDataProvider(cache_dir, offline=offline)
    # Fetch daily prices (one column per ticker). Provider caches + falls back.
    prices = prov.fetch_prices(list(market.tickers), start, end)
    monthly = _portfolio_monthly_log(prices, market)
    return monthly.index, monthly.to_numpy(dtype=float), prov.last_source


# --------------------------------------------------------------------------- #
# The deployment result
# --------------------------------------------------------------------------- #
@dataclass
class Deployment:
    """One honest walk-forward of every strategy over a real market history."""

    market: Market
    dates: pd.DatetimeIndex  # (T,) month-end dates of each decision
    log_returns: np.ndarray  # (T,) realized monthly log-returns
    results: dict[str, PolicyResult]  # strategy name -> single-path result (len-1)
    config: Config  # the config actually used (horizon set from data)
    regime_names: list[str]
    source: str  # data provenance: cache / yfinance / synthetic
    regime_mode: str  # "prior" | "causal_calibrate"

    # --- convenience views -------------------------------------------------- #
    @property
    def target(self) -> float:
        return self.config.goal.target_wealth

    @property
    def horizon_years(self) -> int:
        return self.config.goal.horizon_years

    @property
    def price_index(self) -> np.ndarray:
        """Growth of $1 in the market itself (buy-and-hold the index), length T+1."""
        return np.concatenate([[1.0], np.cumprod(np.expm1(self.log_returns) + 1.0)])

    def wealth(self, strategy: str) -> np.ndarray:
        return self.results[strategy].histories["wealth"][0]

    def equity(self, strategy: str) -> np.ndarray:
        return self.results[strategy].histories["weights"][0].sum(axis=1)

    def belief(self, strategy: str) -> np.ndarray:
        return self.results[strategy].histories["belief"][0]

    def risk_belief(self, strategy: str) -> np.ndarray:
        """P(bear) + P(high-vol) over time — the agent's 'bad weather' gauge."""
        bel = self.belief(strategy)
        idx = [i for i, n in enumerate(self.regime_names) if n in ("bear", "high_vol")]
        return bel[:, idx].sum(axis=1)


# --------------------------------------------------------------------------- #
# The honest walk-forward
# --------------------------------------------------------------------------- #
DEFAULT_AGENTS = ["buy_and_hold", "sixty_forty", "glide_path",
                  "g_learner", "regime_aware_g_learner"]


def run_deployment(
    config: Config,
    market: Market | str,
    *,
    start: str | None = None,
    end: str | None = None,
    target: float | None = None,
    initial: float | None = None,
    contribution: float | None = None,
    horizon_years: int | None = None,
    agents: list[str] | None = None,
    regime_mode: str = "prior",
    prior_returns: np.ndarray | None = None,
    offline: bool = False,
    registry=None,
) -> Deployment:
    """Roll every strategy over a real market history, with **no look-ahead**.

    Parameters
    ----------
    config : Config
        Supplies the goal economics (initial/contribution/target unless overridden),
        the reward, the agent hyper-parameters and the **economic-prior regimes**.
    market : Market | str
        A registered market key (e.g. ``"sp500"``) or a :class:`Market`.
    start, end : str | None
        ISO dates bounding the real history. ``start`` defaults to the market's
        inception; the horizon is taken from the available data (whole years)
        unless ``horizon_years`` is given.
    regime_mode : str
        ``"prior"`` (default, fully honest, robust) or ``"causal_calibrate"``
        (fit the HMM on ``prior_returns`` only — data strictly before ``start``).
    prior_returns : np.ndarray | None
        Optional pre-``start`` monthly log-returns used when
        ``regime_mode="causal_calibrate"``.
    """
    from gbwm.evaluation.harness import run_on_returns
    from gbwm.experiments import build_policies, clone_config

    if isinstance(market, str):
        market = LONG_HISTORY_MARKETS[market]
    agents = agents or DEFAULT_AGENTS

    dates, r_log, source = fetch_market_returns(
        market, start, end, offline=offline, cache_dir=config.data.cache_dir
    )
    spy = config.steps_per_year
    n_years = horizon_years or max(1, len(r_log) // spy)
    keep = n_years * spy
    if keep > len(r_log):
        raise ValueError(
            f"requested {n_years}y ({keep} months) but only {len(r_log)} months "
            f"of {market.label} are available from {start or market.inception}."
        )
    # Anchor at the start date and run forward (the "deployed in 1999" framing),
    # dropping only the trailing partial year.
    r_log = r_log[:keep]
    dates = dates[:keep]

    cfg = clone_config(config)
    cfg.goal.horizon_years = int(n_years)
    if initial is not None:
        cfg.goal.initial_wealth = float(initial)
    if contribution is not None:
        cfg.goal.contribution = float(contribution)
    if target is not None:
        cfg.goal.target_wealth = float(target)

    have_prior = prior_returns is not None and len(prior_returns) >= 2 * spy
    if regime_mode == "causal_calibrate" and have_prior:
        from gbwm.experiments import calibrated_config

        cfg, _calib = calibrated_config(cfg, np.asarray(prior_returns, dtype=float))
        mode_used = "causal_calibrate"
    else:
        if regime_mode == "causal_calibrate":
            warnings.warn("not enough pre-start history to calibrate; using economic-prior "
                          "regimes.", stacklevel=2)
        mode_used = "prior"

    mm = MarketModel.from_config(cfg.market)
    simple = np.expm1(r_log)
    pols = build_policies(cfg, agents, registry)
    if not pols:
        raise ValueError(f"could not build any of agents={agents}")
    results = {name: run_on_returns(pol, cfg, simple, market_model=mm)
               for name, pol in pols.items()}
    return Deployment(
        market=market, dates=dates, log_returns=r_log, results=results, config=cfg,
        regime_names=mm.regime_names, source=source, regime_mode=mode_used,
    )


# --------------------------------------------------------------------------- #
# Scorecard — the metrics that actually matter on a single real path
# --------------------------------------------------------------------------- #
def _max_drawdown(wealth: np.ndarray) -> float:
    peaks = np.maximum.accumulate(wealth)
    return float(((peaks - wealth) / peaks).max())


def _worst_window_return(wealth: np.ndarray, window: int) -> float:
    """Worst peak-to-future change of the *balance* over any ``window`` steps."""
    if len(wealth) <= window:
        return 0.0
    ratios = wealth[window:] / wealth[:-window] - 1.0
    return float(ratios.min())


def _cagr(wealth: np.ndarray, steps_per_year: int) -> float:
    n = len(wealth) - 1
    if n <= 0 or wealth[0] <= 0:
        return 0.0
    years = n / steps_per_year
    return float((wealth[-1] / wealth[0]) ** (1.0 / years) - 1.0)


def scorecard(dep: Deployment) -> pd.DataFrame:
    """One row per strategy: outcome and (crucially) **path-risk** metrics.

    Sorted so the safest goal-reachers surface — the honest way to read a single
    historical path, where raw terminal wealth simply rewards whoever took the
    most equity in a market that happened to rise.
    """
    spy = dep.config.steps_per_year
    rows = []
    for name, res in dep.results.items():
        w = res.histories["wealth"][0]
        eq = res.histories["weights"][0].sum(axis=1)
        final = float(w[-1])
        rows.append({
            "Strategy": name,
            "Final balance": final,
            "Reached goal": bool(final >= dep.target),
            "Max drawdown": _max_drawdown(w),
            "Worst 12-month": _worst_window_return(w, spy),
            "Growth rate (CAGR)": _cagr(w, spy),
            "Avg equity": float(eq.mean()),
        })
    df = pd.DataFrame(rows)
    # Safest-first among goal reachers; then by final balance.
    df = df.sort_values(["Reached goal", "Max drawdown"], ascending=[False, True])
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Decision diary — what each strategy *did* at the historical turning points
# --------------------------------------------------------------------------- #
def _crisis_window(dates: pd.DatetimeIndex, crisis: Crisis) -> tuple[int, int] | None:
    lo = pd.Timestamp(crisis.start + "-01")
    hi = pd.Timestamp(crisis.end + "-28")
    mask = (dates >= lo) & (dates <= hi)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return None
    return int(idx[0]), int(idx[-1])


def decision_diary(
    dep: Deployment,
    smart: str = "Regime-Aware G-Learner",
    naive: str = "Buy & Hold",
) -> list[dict]:
    """For each crisis inside the window, summarize the smart vs. naive response.

    Returns a list of dicts (one per crisis) with the market fall, the naive
    strategy's drawdown, and how the smart agent moved equity as its risk-belief
    rose — the "how would it have decided in real time" story.
    """
    out = []
    price = dep.price_index  # length T+1
    smart_eq = dep.equity(smart) if smart in dep.results else None
    smart_rb = dep.risk_belief(smart) if smart in dep.results else None
    naive_w = dep.wealth(naive) if naive in dep.results else None
    for cr in CRISES:
        win = _crisis_window(dep.dates, cr)
        if win is None:
            continue
        a, b = win
        # market peak-to-trough within (a slightly padded) window
        seg = price[a:b + 2]
        mkt_fall = float(seg.min() / seg.max() - 1.0) if seg.size else 0.0
        entry = {"crisis": cr.name, "blurb": cr.blurb,
                 "from": dep.dates[a].strftime("%b %Y"),
                 "to": dep.dates[b].strftime("%b %Y"),
                 "market_fall": mkt_fall}
        if naive_w is not None:
            seg_w = naive_w[a:b + 2]
            entry["naive_drawdown"] = float(seg_w.min() / seg_w.max() - 1.0) if seg_w.size else 0.0
            entry["naive"] = naive
        if smart_eq is not None:
            entry["smart"] = smart
            entry["smart_equity_start"] = float(smart_eq[a])
            entry["smart_equity_min"] = float(smart_eq[a:b + 1].min())
            entry["smart_riskbelief_max"] = float(smart_rb[a:b + 1].max())
        out.append(entry)
    return out


def diary_sentences(dep: Deployment,
                    smart: str = "Regime-Aware G-Learner",
                    naive: str = "Buy & Hold") -> list[str]:
    """Render :func:`decision_diary` as plain-language lines."""
    lines = []
    for e in decision_diary(dep, smart, naive):
        s = (f"**{e['crisis']}** ({e['from']}–{e['to']}): the market fell "
             f"{e['market_fall'] * 100:.0f}%.")
        if "naive" in e:
            s += f" {e['naive']} rode it down (balance -{abs(e['naive_drawdown']) * 100:.0f}%)."
        if "smart" in e:
            s += (f" The regime-aware agent's bad-weather belief peaked at "
                  f"{e['smart_riskbelief_max'] * 100:.0f}% and it cut equity from "
                  f"{e['smart_equity_start'] * 100:.0f}% to "
                  f"{e['smart_equity_min'] * 100:.0f}%.")
        lines.append(s)
    return lines


# --------------------------------------------------------------------------- #
# Sequence-of-returns risk — *when* you start matters
# --------------------------------------------------------------------------- #
def sequence_risk(
    config: Config,
    market: Market | str,
    starts: list[str],
    horizon_years: int,
    *,
    target: float | None = None,
    initial: float | None = None,
    contribution: float | None = None,
    agents: list[str] | None = None,
    offline: bool = False,
) -> pd.DataFrame:
    """Run the same fixed-horizon plan from several start dates.

    Starting right before a crash (2000, 2007) is the real test of a strategy:
    it exposes *sequence-of-returns risk*. Returns a tidy DataFrame with one row
    per (start, strategy): final balance, reached-goal, max drawdown.
    """
    if isinstance(market, str):
        market = LONG_HISTORY_MARKETS[market]
    agents = agents or DEFAULT_AGENTS
    rows = []
    for start in starts:
        end = (pd.Timestamp(start) + pd.DateOffset(years=horizon_years)
               + pd.DateOffset(months=1)).strftime("%Y-%m-%d")
        try:
            dep = run_deployment(
                config, market, start=start, end=end, horizon_years=horizon_years,
                target=target, initial=initial, contribution=contribution,
                agents=agents, offline=offline,
            )
        except ValueError as exc:  # not enough data from this start
            warnings.warn(f"skipping start {start}: {exc}", stacklevel=2)
            continue
        sc = scorecard(dep)
        for _, r in sc.iterrows():
            rows.append({
                "Start": pd.Timestamp(start).strftime("%Y"),
                "Strategy": r["Strategy"],
                "Final balance": r["Final balance"],
                "Reached goal": r["Reached goal"],
                "Max drawdown": r["Max drawdown"],
            })
    return pd.DataFrame(rows)
