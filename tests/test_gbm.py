"""GBM simulator: distributional correctness and shapes."""
import numpy as np

from gbwm.simulation.gbm import GBMSimulator


def test_log_return_moments_single_regime():
    drift = np.array([[0.10, 0.06]])
    cov = np.array([[[0.04, 0.012], [0.012, 0.09]]])
    dt = 1.0 / 12
    sim = GBMSimulator(drift, cov, dt)
    rng = np.random.default_rng(0)
    regimes = np.zeros((4000, 60), dtype=int)
    logr = sim.sample_log_returns(regimes, rng=rng)
    flat = logr.reshape(-1, 2)
    emp_mean = flat.mean(axis=0)
    emp_cov = np.cov(flat.T)
    assert np.allclose(emp_mean, sim.mean_log[0], atol=2e-3)
    assert np.allclose(emp_cov, cov[0] * dt, atol=2e-3)


def test_simple_returns_are_expm1_of_log():
    drift = np.array([[0.08]])
    cov = np.array([[[0.04]]])
    sim = GBMSimulator(drift, cov, 1 / 12)
    regimes = np.zeros((10, 5), dtype=int)
    z = np.random.default_rng(1).standard_normal((10, 5, 1))
    logr = sim.sample_log_returns(regimes, z=z)
    simple = sim.sample_simple_returns(regimes, z=z)
    assert np.allclose(simple, np.expm1(logr))


def test_antithetic_shocks_are_symmetric():
    drift = np.array([[0.08, 0.05]])
    cov = np.array([[[0.04, 0.0], [0.0, 0.05]]])
    sim = GBMSimulator(drift, cov, 1 / 12)
    regimes = np.zeros((3, 4), dtype=int)
    z = np.random.default_rng(2).standard_normal((3, 4, 2))
    a = sim.sample_log_returns(regimes, z=z)
    b = sim.sample_log_returns(regimes, z=-z)
    # both halves should be mirror images around the deterministic mean
    assert np.allclose((a + b) / 2.0, sim.mean_log[0], atol=1e-12)


def test_shape_and_regime_selection():
    drift = np.array([[0.2], [-0.1]])
    cov = np.array([[[0.01]], [[0.09]]])
    sim = GBMSimulator(drift, cov, 1 / 12)
    regimes = np.array([[0, 1, 0], [1, 1, 0]])
    out = sim.sample_simple_returns(regimes, rng=np.random.default_rng(3))
    assert out.shape == (2, 3, 1)
