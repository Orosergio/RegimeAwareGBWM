"""Rule-based + pluggable LLM advisor."""
import numpy as np

from gbwm.explain import EpisodeContext, LLMAdvisor, RuleBasedAdvisor, StepContext, make_advisor

REGIMES = ["bull", "stable", "high_vol", "bear"]


def _step(prev_eq, eq, prev_belief, belief, wealth=200_000.0, target=250_000.0):
    return StepContext(
        weights=np.array([eq]),
        prev_weights=np.array([prev_eq]),
        belief=np.array(belief),
        prev_belief=np.array(prev_belief),
        wealth=wealth,
        target=target,
        step=120,
        n_steps=240,
        steps_per_year=12,
        regime_names=REGIMES,
        asset_names=["equity"],
    )


def test_explains_derisk_on_rising_bear():
    ctx = _step(0.7, 0.4, [0.5, 0.3, 0.1, 0.1], [0.1, 0.2, 0.05, 0.65])
    msg = RuleBasedAdvisor().explain_step(ctx)
    assert "bear" in msg.lower()
    assert "40%" in msg and "70%" in msg


def test_explains_protect_when_above_goal():
    ctx = _step(0.5, 0.3, [0.25] * 4, [0.25] * 4, wealth=300_000.0, target=250_000.0)
    msg = RuleBasedAdvisor().explain_step(ctx)
    assert "above the goal" in msg.lower() and "protect" in msg.lower()


def test_episode_explanation_reports_outcome():
    T = 24
    ep = EpisodeContext(
        wealth=np.linspace(100_000, 260_000, T + 1),
        weights=np.full((T, 1), 0.5),
        belief=np.tile([0.25] * 4, (T, 1)),
        regime=np.array([0] * 12 + [3] * 12),
        target=250_000.0,
        regime_names=REGIMES,
        asset_names=["equity"],
        steps_per_year=12,
    )
    msg = RuleBasedAdvisor().explain_episode(ep)
    assert "goal reached" in msg.lower()


def test_llm_advisor_falls_back_without_fn_and_uses_fn_when_given():
    ctx = _step(0.7, 0.4, [0.5, 0.3, 0.1, 0.1], [0.1, 0.2, 0.05, 0.65])
    assert LLMAdvisor().explain_step(ctx) == RuleBasedAdvisor().explain_step(ctx)
    spy = LLMAdvisor(complete_fn=lambda p: "LLM:" + p[:5])
    assert spy.explain_step(ctx).startswith("LLM:")
    assert isinstance(make_advisor("rule_based"), RuleBasedAdvisor)
    assert isinstance(make_advisor("llm"), LLMAdvisor)
