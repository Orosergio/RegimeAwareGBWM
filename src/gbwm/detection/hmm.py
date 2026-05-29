"""Gaussian Hidden Markov Model for market-regime detection.

Implemented from scratch in NumPy (Baum-Welch EM, log-space forward-backward,
Viterbi) so the method is transparent and has no hard dependency (ADR-008). If
the optional ``hmmlearn`` backend is installed it can be used instead.

The HMM is fit on **log-returns** so its emission parameters live in the same
space as :class:`gbwm.detection.filter.GaussianRegimeFilter`, letting a fitted
model drive the env's online belief. Fitted states are aligned to the canonical
regime order (bull → stable → high-vol → bear) by descending mean return.
"""

from __future__ import annotations

import numpy as np

from gbwm.detection.filter import GaussianRegimeFilter


def _logsumexp(a: np.ndarray, axis=None, keepdims=False) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    return out if keepdims else np.squeeze(out, axis=axis)


def _kmeans(x: np.ndarray, k: int, rng: np.random.Generator, iters: int = 25):
    """Tiny k-means for HMM mean initialization."""
    idx = rng.choice(len(x), size=k, replace=False)
    centers = x[idx].copy()
    for _ in range(iters):
        d = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        lab = d.argmin(axis=1)
        new = np.array([x[lab == j].mean(axis=0) if np.any(lab == j) else centers[j] for j in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    return centers, lab


class HMMRegimeDetector:
    def __init__(
        self,
        n_states: int = 4,
        covariance_type: str = "full",
        n_iter: int = 200,
        tol: float = 1e-4,
        reg: float = 1e-6,
        n_restarts: int = 6,
        init_persistence: float = 0.92,
        seed: int = 0,
    ) -> None:
        self.K = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.reg = reg
        self.n_restarts = n_restarts
        self.init_persistence = init_persistence
        self.seed = seed
        self.means_: np.ndarray | None = None
        self.covs_: np.ndarray | None = None
        self.transmat_: np.ndarray | None = None
        self.startprob_: np.ndarray | None = None
        self.loglik_: list[float] = []

    # ------------------------------------------------------------------ #
    @staticmethod
    def _as_2d(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return x[:, None] if x.ndim == 1 else x

    def _emission_logprob(self, x: np.ndarray) -> np.ndarray:
        T, A = x.shape
        out = np.empty((T, self.K))
        for k in range(self.K):
            inv = np.linalg.inv(self.covs_[k])
            sign, logdet = np.linalg.slogdet(self.covs_[k])
            d = x - self.means_[k]
            quad = np.einsum("ta,ab,tb->t", d, inv, d)
            out[:, k] = -0.5 * (A * np.log(2 * np.pi) + logdet + quad)
        return out

    def _forward_backward(self, framelogprob: np.ndarray):
        T = framelogprob.shape[0]
        log_t = np.log(self.transmat_ + 1e-300)
        log_start = np.log(self.startprob_ + 1e-300)
        log_alpha = np.empty((T, self.K))
        log_alpha[0] = log_start + framelogprob[0]
        for t in range(1, T):
            log_alpha[t] = _logsumexp(log_alpha[t - 1][:, None] + log_t, axis=0) + framelogprob[t]
        loglik = _logsumexp(log_alpha[-1])
        log_beta = np.zeros((T, self.K))
        for t in range(T - 2, -1, -1):
            log_beta[t] = _logsumexp(
                log_t + framelogprob[t + 1][None, :] + log_beta[t + 1][None, :], axis=1
            )
        log_gamma = log_alpha + log_beta
        log_gamma -= _logsumexp(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)
        return gamma, log_alpha, log_beta, loglik, log_t

    def _init_params(self, x, rng):
        T, A = x.shape
        centers, lab = _kmeans(x, self.K, rng)
        covs = np.stack([
            (np.cov(x[lab == k].T).reshape(A, A) if np.sum(lab == k) > 1 else np.cov(x.T).reshape(A, A))
            + self.reg * np.eye(A)
            for k in range(self.K)
        ])
        # spread the initial covariances across a range of volatilities so EM can
        # discover volatility regimes (the dominant signal in market returns)
        scales = np.linspace(0.5, 2.5, self.K)
        base = np.cov(x.T).reshape(A, A) + self.reg * np.eye(A)
        covs = np.stack([base * scales[k] for k in range(self.K)])
        p = self.init_persistence
        transmat = np.full((self.K, self.K), (1 - p) / (self.K - 1))
        np.fill_diagonal(transmat, p)
        startprob = np.full(self.K, 1.0 / self.K)
        return centers, covs, transmat, startprob

    def _em(self, x, means, covs, transmat, startprob):
        T, A = x.shape
        self.means_, self.covs_, self.transmat_, self.startprob_ = means, covs, transmat, startprob
        loglik_hist = []
        prev = -np.inf
        for _ in range(self.n_iter):
            framelogprob = self._emission_logprob(x)
            gamma, log_alpha, log_beta, loglik, log_t = self._forward_backward(framelogprob)
            loglik_hist.append(float(loglik))
            log_xi = (
                log_alpha[:-1, :, None]
                + log_t[None, :, :]
                + framelogprob[1:, None, :]
                + log_beta[1:, None, :]
                - loglik
            )
            xi = np.exp(_logsumexp(log_xi, axis=0))
            self.startprob_ = gamma[0] / gamma[0].sum()
            self.transmat_ = xi / xi.sum(axis=1, keepdims=True)
            Nk = gamma.sum(axis=0)
            self.means_ = (gamma.T @ x) / Nk[:, None]
            for k in range(self.K):
                d = x - self.means_[k]
                self.covs_[k] = (gamma[:, k, None, None] * np.einsum("ta,tb->tab", d, d)).sum(0) / Nk[k]
                self.covs_[k] += self.reg * np.eye(A)
            if abs(loglik - prev) < self.tol:
                break
            prev = loglik
        return loglik_hist

    def fit(self, returns: np.ndarray) -> "HMMRegimeDetector":
        x = self._as_2d(returns)
        best = None
        for r in range(self.n_restarts):
            rng = np.random.default_rng(self.seed + r)
            means, covs, transmat, startprob = self._init_params(x, rng)
            hist = self._em(x, means.copy(), covs.copy(), transmat.copy(), startprob.copy())
            cand = (hist[-1], self.means_.copy(), self.covs_.copy(), self.transmat_.copy(),
                    self.startprob_.copy(), hist)
            if best is None or cand[0] > best[0]:
                best = cand
        _, self.means_, self.covs_, self.transmat_, self.startprob_, self.loglik_ = best
        self._align_to_mean_descending()
        return self

    def _align_to_mean_descending(self) -> None:
        """Relabel states so state 0 has the highest mean return (bull) ... bear last."""
        order = np.argsort(self.means_[:, 0])[::-1]
        self.means_ = self.means_[order]
        self.covs_ = self.covs_[order]
        self.startprob_ = self.startprob_[order]
        self.transmat_ = self.transmat_[order][:, order]

    # ------------------------------------------------------------------ #
    def predict_proba(self, returns: np.ndarray) -> np.ndarray:
        """Smoothed posterior P(state_t | all returns), shape (T, K)."""
        x = self._as_2d(returns)
        gamma, *_ = self._forward_backward(self._emission_logprob(x))
        return gamma

    def predict(self, returns: np.ndarray) -> np.ndarray:
        """Most-likely state path (Viterbi)."""
        x = self._as_2d(returns)
        framelogprob = self._emission_logprob(x)
        T = x.shape[0]
        log_t = np.log(self.transmat_ + 1e-300)
        delta = np.log(self.startprob_ + 1e-300) + framelogprob[0]
        psi = np.zeros((T, self.K), dtype=int)
        for t in range(1, T):
            m = delta[:, None] + log_t
            psi[t] = m.argmax(axis=0)
            delta = m.max(axis=0) + framelogprob[t]
        states = np.empty(T, dtype=int)
        states[-1] = int(delta.argmax())
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states

    def to_filter(self) -> GaussianRegimeFilter:
        """Build an online belief filter from the fitted parameters."""
        return GaussianRegimeFilter(self.means_, self.covs_, self.transmat_)

    @property
    def annualized_summary(self) -> dict:
        """Human-readable per-state drift/vol assuming the fit was on log-returns."""
        return {
            "mean_log": self.means_[:, 0].tolist(),
            "vol_log": np.sqrt(self.covs_[:, 0, 0]).tolist(),
        }
