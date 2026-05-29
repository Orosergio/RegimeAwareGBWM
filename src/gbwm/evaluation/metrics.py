"""Goal-based evaluation metrics.

The headline metric is ``P(goal) = P(W_T >= G)``. We also report expected
shortfall (and its conditional / CVaR form), the terminal-wealth distribution,
average turnover (a transaction-cost / stability proxy), average max drawdown
(path risk), and regime-conditional P(goal).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PolicyResult:
    name: str
    terminal_wealth: np.ndarray
    target: float
    p_goal: float = 0.0
    avg_shortfall: float = 0.0
    cvar_shortfall: float = 0.0
    mean_terminal: float = 0.0
    median_terminal: float = 0.0
    p10_terminal: float = 0.0
    p90_terminal: float = 0.0
    avg_turnover: float = 0.0
    avg_max_drawdown: float = 0.0
    regime_p_goal: dict[str, float] = field(default_factory=dict)
    histories: dict | None = None

    def summary_row(self) -> dict:
        best = max(self.regime_p_goal, key=self.regime_p_goal.get) if self.regime_p_goal else "-"
        return {
            "Strategy": self.name,
            "P(goal)": round(self.p_goal, 4),
            "Avg shortfall": round(self.avg_shortfall, 1),
            "Expected shortfall (fails)": round(self.cvar_shortfall, 1),
            "Median terminal": round(self.median_terminal, 1),
            "Avg turnover": round(self.avg_turnover, 4),
            "Avg max drawdown": round(self.avg_max_drawdown, 4),
            "Best in regime": best,
        }


def prob_goal(terminal_wealth: np.ndarray, target: float) -> float:
    return float(np.mean(np.asarray(terminal_wealth) >= target))


def shortfall(terminal_wealth: np.ndarray, target: float) -> np.ndarray:
    return np.clip(target - np.asarray(terminal_wealth), 0.0, None)


def max_drawdown(wealth_history: np.ndarray) -> np.ndarray:
    """Per-path maximum drawdown from running peak. ``wealth_history`` (N, T+1)."""
    peaks = np.maximum.accumulate(wealth_history, axis=1)
    dd = (peaks - wealth_history) / peaks
    return dd.max(axis=1)


def turnover(weights_history: np.ndarray) -> np.ndarray:
    """Per-path average per-step turnover. ``weights_history`` (N, T, A)."""
    prev = np.concatenate([np.zeros_like(weights_history[:, :1, :]), weights_history[:, :-1, :]], axis=1)
    return np.abs(weights_history - prev).sum(axis=2).mean(axis=1)


def evaluate(
    name: str,
    terminal_wealth: np.ndarray,
    target: float,
    histories: dict | None = None,
    regime_names: list[str] | None = None,
) -> PolicyResult:
    tw = np.asarray(terminal_wealth, dtype=float)
    sf = shortfall(tw, target)
    failing = sf[sf > 0]
    res = PolicyResult(
        name=name,
        terminal_wealth=tw,
        target=target,
        p_goal=prob_goal(tw, target),
        avg_shortfall=float(sf.mean()),
        cvar_shortfall=float(failing.mean()) if failing.size else 0.0,
        mean_terminal=float(tw.mean()),
        median_terminal=float(np.median(tw)),
        p10_terminal=float(np.percentile(tw, 10)),
        p90_terminal=float(np.percentile(tw, 90)),
        histories=histories,
    )
    if histories is not None:
        if "wealth" in histories:
            res.avg_max_drawdown = float(max_drawdown(histories["wealth"]).mean())
        if "weights" in histories:
            res.avg_turnover = float(turnover(histories["weights"]).mean())
        if "regime" in histories and regime_names is not None:
            dom = _dominant_regime(histories["regime"])
            for k, nm in enumerate(regime_names):
                mask = dom == k
                if mask.any():
                    res.regime_p_goal[nm] = prob_goal(tw[mask], target)
    return res


def _dominant_regime(regime_history: np.ndarray) -> np.ndarray:
    """Most-frequent regime per path. ``regime_history`` (N, T) int."""
    N = regime_history.shape[0]
    K = int(regime_history.max()) + 1
    counts = np.zeros((N, K), dtype=int)
    for k in range(K):
        counts[:, k] = (regime_history == k).sum(axis=1)
    return counts.argmax(axis=1)
