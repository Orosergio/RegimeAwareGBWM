"""Estimate regime parameters from real returns via the Gaussian HMM.

Turns an observed (log-)return series into annualized per-regime drift/vol plus a
transition matrix — i.e. a data-driven replacement for the hand-set regime
parameters in the config. Aligns regimes to the canonical bull→bear order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gbwm.config import Config, MarketConfig, RegimeParams, TransitionConfig
from gbwm.detection.hmm import HMMRegimeDetector


@dataclass
class RegimeCalibration:
    names: list[str]
    mu_annual: np.ndarray  # (K,) or (K, A)
    sigma_annual: np.ndarray  # (K,) or (K, A)
    transition: np.ndarray  # (K, K) per-step
    detector: HMMRegimeDetector

    def as_regime_params(self) -> list[RegimeParams]:
        mu = np.atleast_2d(self.mu_annual.T).T if self.mu_annual.ndim == 1 else self.mu_annual
        sg = np.atleast_2d(self.sigma_annual.T).T if self.sigma_annual.ndim == 1 else self.sigma_annual
        out = []
        for k, name in enumerate(self.names):
            mu_k = [float(self.mu_annual[k])] if self.mu_annual.ndim == 1 else list(map(float, self.mu_annual[k]))
            sg_k = [float(self.sigma_annual[k])] if self.sigma_annual.ndim == 1 else list(map(float, self.sigma_annual[k]))
            out.append(RegimeParams(name=name, mu=mu_k, sigma=sg_k))
        return out


def calibrate_regimes(
    returns_log_step: np.ndarray,
    steps_per_year: int,
    n_states: int = 4,
    names: list[str] | None = None,
    seed: int = 0,
    n_restarts: int = 6,
) -> RegimeCalibration:
    """Fit an HMM to per-step log-returns and annualize the regime parameters.

    GBM mapping: if per-step log-returns have mean ``m`` and variance ``v`` then
    ``sigma_annual = sqrt(v * steps_per_year)`` and
    ``mu_annual = m * steps_per_year + 0.5 * sigma_annual**2``.
    """
    x = np.asarray(returns_log_step, dtype=float)
    det = HMMRegimeDetector(n_states=n_states, n_restarts=n_restarts, seed=seed).fit(x)
    mean_log = det.means_  # (K, A)
    var_log = np.array([np.diag(c) for c in det.covs_])  # (K, A)
    sigma_annual = np.sqrt(var_log * steps_per_year)
    mu_annual = mean_log * steps_per_year + 0.5 * sigma_annual**2
    if mean_log.shape[1] == 1:
        mu_annual = mu_annual[:, 0]
        sigma_annual = sigma_annual[:, 0]
    if names is None:
        default = ["bull", "stable", "high_vol", "bear"]
        names = default[:n_states] if n_states <= 4 else [f"regime_{i}" for i in range(n_states)]
    return RegimeCalibration(names, mu_annual, sigma_annual, det.transmat_, det)


def apply_calibration_to_config(config: Config, calib: RegimeCalibration) -> Config:
    """Return a new Config whose market regimes/transition come from calibration."""
    d = config.to_dict()
    d["market"]["regimes"] = [
        {"name": rp.name, "mu": rp.mu, "sigma": rp.sigma} for rp in calib.as_regime_params()
    ]
    d["market"]["transition"] = {"matrix": calib.transition.tolist()}
    return Config.from_dict(d)
