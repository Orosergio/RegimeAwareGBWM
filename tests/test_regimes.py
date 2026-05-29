"""Regime Markov chain + combined market model."""
import numpy as np

from gbwm.config import default_config
from gbwm.simulation.regimes import MarketModel, RegimeSimulator


def _transition(p=0.85, k=4):
    off = (1 - p) / (k - 1)
    m = np.full((k, k), off)
    np.fill_diagonal(m, p)
    return m


def test_empirical_transitions_match_matrix():
    P = _transition()
    sim = RegimeSimulator(P)
    rng = np.random.default_rng(0)
    paths = sim.simulate(2000, 1500, rng, start=0)
    k = P.shape[0]
    counts = np.zeros((k, k))
    cur = paths[:, :-1].ravel()
    nxt = paths[:, 1:].ravel()
    np.add.at(counts, (cur, nxt), 1)
    emp = counts / counts.sum(axis=1, keepdims=True)
    assert np.allclose(emp, P, atol=0.01)


def test_stationary_distribution():
    P = _transition()
    sim = RegimeSimulator(P)
    pi = sim.stationary_distribution()
    assert abs(pi.sum() - 1.0) < 1e-9
    assert np.allclose(pi @ P, pi, atol=1e-9)
    # symmetric transition -> uniform stationary
    assert np.allclose(pi, 0.25, atol=1e-9)


def test_simulate_bounds_and_dtype():
    sim = RegimeSimulator(_transition(0.9, 3))
    paths = sim.simulate(50, 30, np.random.default_rng(1))
    assert paths.shape == (50, 30)
    assert paths.dtype.kind == "i"
    assert paths.min() >= 0 and paths.max() < 3


def test_market_model_from_config_and_cashflow_identities():
    cfg = default_config()
    mm = MarketModel.from_config(cfg.market)
    rng = np.random.default_rng(7)
    mp = mm.simulate(64, cfg.total_steps, rng, antithetic=False)
    assert mp.regimes.shape == (64, cfg.total_steps)
    assert mp.risky_returns.shape == (64, cfg.total_steps, 1)
    # all-cash portfolio earns exactly the risk-free per-step return
    r_cash = mp.portfolio_returns(np.zeros(1))
    assert np.allclose(r_cash, mm.cash_return)
    # fully-invested single risky asset equals that asset's return
    r_full = mp.portfolio_returns(np.ones(1))
    assert np.allclose(r_full, mp.risky_returns[:, :, 0])


def test_cash_return_formula():
    cfg = default_config()
    mm = MarketModel.from_config(cfg.market)
    assert np.isclose(mm.cash_return, np.expm1(cfg.market.risk_free_rate * cfg.dt))


def test_antithetic_pairs_share_regime_paths_and_reproducible():
    cfg = default_config()
    mm = MarketModel.from_config(cfg.market)
    mp = mm.simulate(8, 12, np.random.default_rng(5), antithetic=True)
    half = 4
    assert np.array_equal(mp.regimes[:half], mp.regimes[half:])
    # reproducibility with same seed
    a = mm.simulate(8, 12, np.random.default_rng(5), antithetic=True)
    assert np.array_equal(a.risky_returns, mp.risky_returns)
