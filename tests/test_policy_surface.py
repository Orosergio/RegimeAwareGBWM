"""Learned-policy surface + heatmap."""
import numpy as np

from gbwm.config import default_config
from gbwm.evaluation import plots
from gbwm.policies import GLearner, RegimeAwareGLearner


def _cfg():
    cfg = default_config(); cfg.goal.horizon_years = 4
    cfg.agents.g_learner.update(dict(n_wealth_bins=121, n_actions=15))
    return cfg


def test_surface_shapes():
    cfg = _cfg()
    gl = GLearner.from_config(cfg)
    ra = RegimeAwareGLearner.from_config(cfg)
    assert gl.surface().shape == (cfg.total_steps, gl.n_wbins)
    assert ra.surface(regime="bear").shape == (cfg.total_steps, ra.n_wbins)
    # bull holds >= bear equity on average across the surface
    assert ra.surface("bull").mean() >= ra.surface("bear").mean()


def test_heatmap_returns_figure():
    cfg = _cfg()
    gl = GLearner.from_config(cfg)
    fig = plots.plot_policy_heatmap(gl, cfg.goal.target_wealth, cfg.steps_per_year)
    assert fig is not None
    plots.plt.close("all")
