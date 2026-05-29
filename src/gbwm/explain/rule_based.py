"""Deterministic, template-based explanations of agent decisions."""

from __future__ import annotations

import numpy as np

from gbwm.explain.base import RISKY_REGIMES, Advisor, EpisodeContext, StepContext


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


class RuleBasedAdvisor(Advisor):
    name = "rule_based"

    def __init__(self, change_threshold: float = 0.05) -> None:
        self.change_threshold = change_threshold

    def explain_step(self, ctx: StepContext) -> str:
        parts = [f"The agent holds {_pct(ctx.equity)} in risky assets and "
                 f"{_pct(1 - ctx.equity)} in cash."]

        d_eq = ctx.equity - ctx.prev_equity
        d_belief = ctx.belief - ctx.prev_belief
        rise_idx = int(np.argmax(d_belief))
        fall_idx = int(np.argmin(d_belief))
        rise_name = ctx.regime_names[rise_idx]
        fall_name = ctx.regime_names[fall_idx]

        if d_eq <= -self.change_threshold:
            if rise_name in RISKY_REGIMES and d_belief[rise_idx] > 0.05:
                parts.append(
                    f"It cut equity (from {_pct(ctx.prev_equity)} to {_pct(ctx.equity)}) because the "
                    f"{rise_name.replace('_', '-')} probability rose from "
                    f"{_pct(ctx.prev_belief[rise_idx])} to {_pct(ctx.belief[rise_idx])}."
                )
            else:
                parts.append(f"It reduced risk from {_pct(ctx.prev_equity)} to {_pct(ctx.equity)}.")
        elif d_eq >= self.change_threshold:
            if fall_name in RISKY_REGIMES and d_belief[fall_idx] < -0.05:
                parts.append(
                    f"It raised equity (from {_pct(ctx.prev_equity)} to {_pct(ctx.equity)}) as the "
                    f"{fall_name.replace('_', '-')} probability fell from "
                    f"{_pct(ctx.prev_belief[fall_idx])} to {_pct(ctx.belief[fall_idx])}."
                )
            else:
                parts.append(f"It increased risk from {_pct(ctx.prev_equity)} to {_pct(ctx.equity)}.")

        if ctx.gap > 0.05:
            parts.append(
                f"You are {_pct(ctx.gap)} below the goal with {ctx.years_left:.0f} years left, "
                f"so it leans into risk to try to catch up."
            )
        elif ctx.gap < -0.05:
            parts.append(
                f"Wealth is {_pct(-ctx.gap)} above the goal, so it protects gains by holding less equity."
            )
        else:
            parts.append("Wealth is close to the goal, so it balances growth against protection.")
        return " ".join(parts)

    def explain_episode(self, ctx: EpisodeContext) -> str:
        final = float(ctx.wealth[-1])
        reached = final >= ctx.target
        equity = ctx.weights.sum(axis=1)
        is_risky = np.isin(
            ctx.regime, [i for i, n in enumerate(ctx.regime_names) if n in RISKY_REGIMES]
        )
        out = [
            f"Outcome: ended at ${final:,.0f} versus a ${ctx.target:,.0f} goal — "
            f"{'goal reached.' if reached else f'short by ${ctx.target - final:,.0f}.'}"
        ]
        if is_risky.any() and (~is_risky).any():
            eq_risky = equity[is_risky].mean()
            eq_calm = equity[~is_risky].mean()
            if eq_risky < eq_calm - 0.03:
                out.append(
                    f"During bear / high-volatility periods it held {_pct(eq_risky)} equity on average, "
                    f"versus {_pct(eq_calm)} in calmer regimes — it de-risked in bad weather."
                )
            else:
                out.append(
                    f"It held {_pct(eq_risky)} equity in turbulent regimes vs {_pct(eq_calm)} in calm ones."
                )
        out.append(
            f"Equity ranged from {_pct(equity.min())} to {_pct(equity.max())} over the horizon, "
            f"adapting to wealth, time and the detected regime."
        )
        return " ".join(out)
