"""Data providers (offline fallback + cache) and regime calibration."""
import tempfile

import numpy as np
import pandas as pd

from gbwm.config import default_config
from gbwm.data.calibration import apply_calibration_to_config, calibrate_regimes
from gbwm.data.providers import MarketDataProvider


def test_offline_synthetic_prices_and_cache():
    with tempfile.TemporaryDirectory() as d:
        prov = MarketDataProvider(cache_dir=d, offline=True)
        px = prov.fetch_prices(["SPY", "AGG"], "2018-01-01", "2020-01-01")
        assert list(px.columns) == ["SPY", "AGG"]
        assert px.shape[0] > 100 and px.notna().all().all()
        assert prov.last_source == "synthetic"
        # second call hits the cache
        prov2 = MarketDataProvider(cache_dir=d, offline=True)
        px2 = prov2.fetch_prices(["SPY", "AGG"], "2018-01-01", "2020-01-01")
        assert prov2.last_source == "cache"
        assert np.allclose(px.values, px2.values)


def test_get_and_resample_returns():
    with tempfile.TemporaryDirectory() as d:
        prov = MarketDataProvider(cache_dir=d, offline=True)
        daily = prov.get_returns(["SPY"], "2015-01-01", "2020-01-01", kind="log")
        assert daily.notna().all().all()
        monthly = prov.resample_returns(daily, 12)
        assert len(monthly) < len(daily)


def _two_regime_logreturns(T=600, seed=0):
    rng = np.random.default_rng(seed)
    P = np.array([[0.95, 0.05], [0.05, 0.95]])
    mean, sd = [0.012, -0.02], [0.02, 0.05]
    st = np.zeros(T, dtype=int)
    for t in range(1, T):
        st[t] = rng.choice(2, p=P[st[t - 1]])
    return rng.normal([mean[s] for s in st], [sd[s] for s in st])


def test_calibration_annualizes_and_orders():
    x = _two_regime_logreturns()
    calib = calibrate_regimes(x, steps_per_year=12, n_states=2, seed=1)
    assert calib.mu_annual.shape == (2,)
    # bull-like state (index 0, highest mean) has higher annual drift
    assert calib.mu_annual[0] > calib.mu_annual[1]
    # annualized vol is positive and on a sensible scale
    assert np.all(calib.sigma_annual > 0) and np.all(calib.sigma_annual < 1.0)


def test_apply_calibration_produces_valid_config():
    x = _two_regime_logreturns()
    calib = calibrate_regimes(x, steps_per_year=12, n_states=4, seed=2)
    cfg = apply_calibration_to_config(default_config(), calib)
    assert cfg.market.n_regimes == 4
    tm = cfg.market.transition_matrix()
    assert np.allclose(tm.sum(axis=1), 1.0)
    # config still drives the simulator
    from gbwm.simulation.regimes import MarketModel
    mm = MarketModel.from_config(cfg.market)
    assert mm.n_regimes == 4
