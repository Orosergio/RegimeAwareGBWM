"""Baseline policies behave as specified."""
import numpy as np

from gbwm.config import default_config
from gbwm.policies import BuyAndHold, GlidePath, SixtyForty
from gbwm.policies.base import DecisionContext


def _ctx(step, n_steps=240, n_paths=5, n_assets=1):
    return DecisionContext(
        step=step,
        n_steps=n_steps,
        wealth=np.full(n_paths, 100_000.0),
        target=250_000.0,
        belief=np.tile([0.25, 0.25, 0.25, 0.25], (n_paths, 1)),
        n_assets=n_assets,
        regime_names=["bull", "stable", "high_vol", "bear"],
    )


def test_buy_and_hold_fully_invested():
    w = BuyAndHold(1).weights(_ctx(0))
    assert w.shape == (5, 1)
    assert np.allclose(w.sum(axis=1), 1.0)


def test_sixty_forty_fraction():
    w = SixtyForty(1).weights(_ctx(10))
    assert np.allclose(w.sum(axis=1), 0.6)


def test_glide_path_endpoints_and_monotonic():
    gp = GlidePath(1, start_equity=0.9, end_equity=0.3)
    w0 = gp.weights(_ctx(0)).sum(axis=1)[0]
    wT = gp.weights(_ctx(240)).sum(axis=1)[0]
    wmid = gp.weights(_ctx(120)).sum(axis=1)[0]
    assert np.isclose(w0, 0.9)
    assert np.isclose(wT, 0.3)
    assert w0 > wmid > wT


def test_from_config_builds():
    cfg = default_config()
    gp = GlidePath.from_config(cfg)
    assert gp.n_assets == cfg.market.n_risky
    assert 0 < gp.end_equity < gp.start_equity <= 1.0


def test_multi_asset_equal_split():
    w = BuyAndHold(4).weights(_ctx(0, n_assets=4))
    assert w.shape == (5, 4)
    assert np.allclose(w, 0.25)
