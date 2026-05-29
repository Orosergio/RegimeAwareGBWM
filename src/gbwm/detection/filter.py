"""Online Gaussian regime-belief filter (HMM forward recursion).

This is the *belief* the agent observes (ADR-003): a posterior over the hidden
regime, updated each step from realized log-returns. It is the shared primitive
behind both the environment's online belief and the standalone HMM detector.

Given log-return ``x_t`` and a known/estimated Gaussian emission per regime plus
a transition matrix ``P``::

    predict:  pred_k = sum_j alpha_{t-1}(j) P[j, k]
    update:   alpha_t(k) ∝ pred_k * N(x_t; mu_k, Sigma_k)

All updates are done in log-space for numerical stability.
"""

from __future__ import annotations

import numpy as np


class GaussianRegimeFilter:
    def __init__(
        self,
        mean_log: np.ndarray,
        cov_log: np.ndarray,
        transition: np.ndarray,
        prior: np.ndarray | None = None,
    ) -> None:
        self.mean_log = np.asarray(mean_log, dtype=float)  # (K, A)
        self.cov_log = np.asarray(cov_log, dtype=float)  # (K, A, A)
        self.P = np.asarray(transition, dtype=float)  # (K, K)
        self.K, self.A = self.mean_log.shape
        # Precompute inverse covariances and Gaussian normalizers.
        self._inv = np.linalg.inv(self.cov_log)  # (K, A, A)
        signs, logdets = np.linalg.slogdet(self.cov_log)
        self._const = -0.5 * (self.A * np.log(2.0 * np.pi) + logdets)  # (K,)
        self.prior = self.stationary() if prior is None else np.asarray(prior, float)

    def stationary(self) -> np.ndarray:
        vals, vecs = np.linalg.eig(self.P.T)
        pi = np.real(vecs[:, int(np.argmin(np.abs(vals - 1.0)))])
        pi = np.clip(pi, 0.0, None)
        s = pi.sum()
        return pi / s if s > 0 else np.full(self.K, 1.0 / self.K)

    def reset(self, prior: np.ndarray | None = None) -> np.ndarray:
        return (self.prior if prior is None else np.asarray(prior, float)).copy()

    def predict(self, alpha: np.ndarray) -> np.ndarray:
        return alpha @ self.P

    def emission_loglik(self, x: np.ndarray) -> np.ndarray:
        d = x[None, :] - self.mean_log  # (K, A)
        quad = np.einsum("ka,kab,kb->k", d, self._inv, d)
        return self._const - 0.5 * quad

    def update(self, alpha: np.ndarray, x: np.ndarray) -> np.ndarray:
        """One filtering step: posterior over the regime that emitted ``x``."""
        pred = self.predict(alpha)
        logpost = np.log(np.clip(pred, 1e-300, None)) + self.emission_loglik(np.asarray(x, float))
        logpost -= logpost.max()
        post = np.exp(logpost)
        return post / post.sum()

    # --- vectorized over N paths (used by the Monte-Carlo harness) -------- #
    def predict_batch(self, alpha: np.ndarray) -> np.ndarray:
        return alpha @ self.P

    def emission_loglik_batch(self, x: np.ndarray) -> np.ndarray:
        d = x[:, None, :] - self.mean_log[None, :, :]  # (N, K, A)
        quad = np.einsum("nka,kab,nkb->nk", d, self._inv, d)
        return self._const[None, :] - 0.5 * quad

    def update_batch(self, alpha: np.ndarray, x: np.ndarray) -> np.ndarray:
        pred = self.predict_batch(alpha)
        logpost = np.log(np.clip(pred, 1e-300, None)) + self.emission_loglik_batch(np.asarray(x, float))
        logpost -= logpost.max(axis=1, keepdims=True)
        post = np.exp(logpost)
        return post / post.sum(axis=1, keepdims=True)
