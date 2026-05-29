"""Gaussian HMM: algorithmic correctness + recovery when regimes are separable.

We do NOT assert strong recovery on realistic overlapping monthly returns (a
genuinely hard problem); the agent's in-env belief uses the true-parameter Bayes
filter, so it is not bottlenecked by HMM estimation error.
"""
import numpy as np

from gbwm.detection.hmm import HMMRegimeDetector


def _two_regime_series(T=600, seed=0):
    rng = np.random.default_rng(seed)
    P = np.array([[0.97, 0.03], [0.03, 0.97]])
    mus, sd = [0.04, -0.04], 0.01
    st = np.zeros(T, dtype=int)
    for t in range(1, T):
        st[t] = rng.choice(2, p=P[st[t - 1]])
    x = rng.normal([mus[s] for s in st], sd)
    return x, st


def test_em_loglikelihood_is_monotone():
    x, _ = _two_regime_series()
    det = HMMRegimeDetector(n_states=2, n_iter=50, n_restarts=2, seed=1).fit(x)
    ll = det.loglik_
    assert all(b >= a - 1e-6 for a, b in zip(ll, ll[1:]))


def test_recovers_well_separated_regimes():
    x, st = _two_regime_series()
    det = HMMRegimeDetector(n_states=2, n_iter=60, n_restarts=3, seed=1).fit(x)
    pred = det.predict(x)
    acc = max((pred == st).mean(), (pred != st).mean())  # label-permutation invariant
    assert acc > 0.9
    assert np.min(np.diag(det.transmat_)) > 0.85


def test_posterior_and_viterbi_valid():
    x, _ = _two_regime_series(T=300)
    det = HMMRegimeDetector(n_states=2, n_iter=40, n_restarts=2, seed=2).fit(x)
    g = det.predict_proba(x)
    assert g.shape == (300, 2)
    assert np.allclose(g.sum(axis=1), 1.0)
    states = det.predict(x)
    assert states.shape == (300,) and set(np.unique(states)).issubset({0, 1})


def test_states_aligned_mean_descending():
    x, _ = _two_regime_series()
    det = HMMRegimeDetector(n_states=2, n_iter=60, n_restarts=3, seed=1).fit(x)
    assert det.means_[0, 0] >= det.means_[1, 0]  # state 0 = highest mean (bull-like)


def test_to_filter_produces_valid_beliefs():
    x, _ = _two_regime_series(T=300)
    det = HMMRegimeDetector(n_states=2, n_iter=40, n_restarts=2, seed=3).fit(x)
    filt = det.to_filter()
    b = filt.reset()
    for val in x[:20]:
        b = filt.update(b, np.array([val]))
        assert abs(b.sum() - 1.0) < 1e-9 and np.all(b >= 0)
