"""Real market & macro data adapters (Adapter pattern, ADR-008).

``yfinance`` (ETF prices) and ``pandas-datareader`` (FRED macro) are the optional
``data`` extra and imported lazily. Results are cached to CSV under
``data/cache`` so re-runs and the demo are fast and offline-friendly. If a
provider is unavailable or a fetch fails, we fall back to a **deterministic
synthetic** series (clearly flagged) so the full pipeline always runs without
network access — important for CI and the sandboxed demo.
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def _cache_key(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


def _source_path(cache: Path) -> Path:
    """Sidecar file recording the true provenance of a cached series."""
    return cache.with_suffix(".source")


def _cached_source(cache: Path) -> str:
    """Honest provenance of a cache hit: ``"synthetic"`` if the file holds the
    synthetic fallback (so the app never presents it as real), else ``"cache"``
    (a previously-fetched real series)."""
    p = _source_path(cache)
    origin = p.read_text().strip() if p.exists() else ""
    return "synthetic" if origin == "synthetic" else "cache"


def _record_source(cache: Path, source: str) -> None:
    try:
        _source_path(cache).write_text(source)
    except OSError:  # pragma: no cover - best-effort provenance marker
        pass


def _synthetic_prices(tickers: list[str], index: pd.DatetimeIndex, seed: int = 0) -> pd.DataFrame:
    """Deterministic GBM-ish price panel used when no provider is available."""
    rng = np.random.default_rng(seed)
    out = {}
    for i, tk in enumerate(tickers):
        mu, sig = 0.07, 0.17 + 0.02 * i
        dt = 1 / 252
        shocks = rng.normal((mu - 0.5 * sig**2) * dt, sig * np.sqrt(dt), size=len(index))
        out[tk] = 100.0 * np.exp(np.cumsum(shocks))
    return pd.DataFrame(out, index=index)


class MarketDataProvider:
    def __init__(self, cache_dir: str = "data/cache", offline: bool = False) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self.last_source: str = "unknown"

    def fetch_prices(
        self, tickers: list[str], start: str, end: str | None = None
    ) -> pd.DataFrame:
        end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
        parts = ["prices", ",".join(tickers), start, end]
        if self.offline:
            parts.append("offline")  # never collide with (or be served as) live data
        key = _cache_key(*parts)
        cache = self.cache_dir / f"{key}.csv"
        if cache.exists():
            self.last_source = _cached_source(cache)
            return pd.read_csv(cache, index_col=0, parse_dates=True)

        df: pd.DataFrame | None = None
        if not self.offline:
            try:
                import yfinance as yf  # lazy (data extra)

                raw = yf.download(
                    tickers, start=start, end=end, auto_adjust=True, progress=False
                )
                if isinstance(raw.columns, pd.MultiIndex):
                    # Multi-ticker: columns under "Close" are the ticker NAMES (yfinance
                    # returns them sorted, not in request order). Reindex BY NAME to the
                    # requested order — never blindly relabel, which would mis-assign one
                    # ticker's prices to another's column.
                    close = raw["Close"]
                    present = [t for t in tickers if t in close.columns]
                    df = close.reindex(columns=present).dropna(how="all")
                else:
                    # Single ticker: a flat OHLCV frame; take Close and name it.
                    df = raw[["Close"]].dropna(how="all")
                    df.columns = tickers
                self.last_source = "yfinance"
            except Exception as exc:  # network or missing dep
                warnings.warn(f"yfinance unavailable ({exc}); using synthetic prices.")
                df = None

        if df is None or df.empty:
            idx = pd.bdate_range(start=start, end=end)
            df = _synthetic_prices(tickers, idx)
            self.last_source = "synthetic"

        df.to_csv(cache)
        _record_source(cache, self.last_source)
        return df

    def get_returns(
        self, tickers: list[str], start: str, end: str | None = None, kind: str = "log"
    ) -> pd.DataFrame:
        prices = self.fetch_prices(tickers, start, end)
        if kind == "log":
            r = np.log(prices / prices.shift(1))
        else:
            r = prices.pct_change()
        return r.dropna(how="all")

    def resample_returns(
        self, returns: pd.DataFrame, steps_per_year: int = 12
    ) -> pd.DataFrame:
        """Aggregate (e.g. daily→monthly) log-returns by summation."""
        rule = {12: "ME", 4: "QE", 1: "YE", 252: "B"}.get(steps_per_year, "ME")
        return returns.resample(rule).sum().dropna(how="all")


class FredProvider:
    def __init__(self, cache_dir: str = "data/cache", offline: bool = False) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self.last_source = "unknown"

    def fetch_series(
        self, series_ids: list[str], start: str, end: str | None = None
    ) -> pd.DataFrame:
        end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
        parts = ["fred", ",".join(series_ids), start, end]
        if self.offline:
            parts.append("offline")
        key = _cache_key(*parts)
        cache = self.cache_dir / f"{key}.csv"
        if cache.exists():
            self.last_source = _cached_source(cache)
            return pd.read_csv(cache, index_col=0, parse_dates=True)

        df: pd.DataFrame | None = None
        if not self.offline:
            try:
                from pandas_datareader import data as pdr  # lazy (data extra)

                df = pdr.DataReader(series_ids, "fred", start, end)
                self.last_source = "fred"
            except Exception as exc:
                warnings.warn(f"FRED unavailable ({exc}); using synthetic macro.")
                df = None

        if df is None or df.empty:
            idx = pd.bdate_range(start=start, end=end, freq="ME")
            rng = np.random.default_rng(7)
            df = pd.DataFrame(
                {s: 2.0 + np.cumsum(rng.normal(0, 0.1, len(idx))) for s in series_ids},
                index=idx,
            )
            self.last_source = "synthetic"

        df.to_csv(cache)
        _record_source(cache, self.last_source)
        return df


def load_equity_returns(config, offline: bool = False) -> pd.Series:
    """Convenience: monthly log-returns of the config's equity ticker."""
    prov = MarketDataProvider(config.data.cache_dir, offline=offline)
    daily = prov.get_returns([config.data.equity_ticker], config.data.start, kind="log")
    monthly = prov.resample_returns(daily, config.market.steps_per_year)
    return monthly.iloc[:, 0].dropna()
