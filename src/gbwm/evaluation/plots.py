"""Matplotlib figures for the report and the demo (Agg backend, headless-safe).

Each function returns a Figure so callers (Streamlit, notebooks, CLI) decide how
to render or save. Matplotlib is a core dependency; the app may also use Plotly.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REGIME_COLORS = {
    "bull": "#2e9e5b",
    "stable": "#4c78a8",
    "high_vol": "#f58518",
    "bear": "#e45756",
}


def plot_wealth_paths(histories: dict, target: float, n_show: int = 60, ax=None):
    wealth = histories["wealth"]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    t = np.arange(wealth.shape[1])
    idx = np.random.default_rng(0).choice(wealth.shape[0], size=min(n_show, wealth.shape[0]), replace=False)
    ax.plot(t, wealth[idx].T, color="#4c78a8", alpha=0.12, lw=0.8)
    ax.plot(t, np.median(wealth, axis=0), color="#1f4e79", lw=2.2, label="median")
    ax.plot(t, np.percentile(wealth, 10, axis=0), color="#1f4e79", lw=1, ls="--", label="10th pct")
    ax.plot(t, np.percentile(wealth, 90, axis=0), color="#1f4e79", lw=1, ls=":", label="90th pct")
    ax.axhline(target, color="#e45756", lw=1.8, label="goal")
    ax.set(xlabel="step", ylabel="wealth", title="Wealth trajectories")
    ax.legend(loc="upper left", fontsize=8)
    return ax.figure


def plot_allocation_over_time(weights_path: np.ndarray, asset_names: list[str], ax=None):
    """Stacked allocation for a single path; weights_path (T, A)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.2))
    T, A = weights_path.shape
    cash = 1.0 - weights_path.sum(axis=1)
    stack = np.column_stack([weights_path, cash])
    labels = list(asset_names) + ["cash"]
    ax.stackplot(np.arange(T), stack.T, labels=labels, alpha=0.85)
    ax.set(xlabel="step", ylabel="weight", ylim=(0, 1), title="Allocation over time")
    ax.legend(loc="upper right", fontsize=8, ncol=len(labels))
    return ax.figure


def plot_regime_beliefs(belief_path: np.ndarray, regime_names: list[str], ax=None):
    """Stacked regime posterior for a single path; belief_path (T, K)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.2))
    T = belief_path.shape[0]
    colors = [REGIME_COLORS.get(n, None) for n in regime_names]
    ax.stackplot(np.arange(T), belief_path.T, labels=regime_names, colors=colors, alpha=0.85)
    ax.set(xlabel="step", ylabel="P(regime)", ylim=(0, 1), title="Detected regime probabilities")
    ax.legend(loc="upper right", fontsize=8, ncol=len(regime_names))
    return ax.figure


def plot_terminal_distribution(terminal_wealth: np.ndarray, target: float, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    ax.hist(terminal_wealth, bins=60, color="#4c78a8", alpha=0.8)
    ax.axvline(target, color="#e45756", lw=2, label="goal")
    ax.set(xlabel="terminal wealth", ylabel="count", title="Terminal-wealth distribution")
    ax.legend(fontsize=8)
    return ax.figure


def plot_strategy_comparison(results: dict, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    names = list(results.keys())
    pgoal = [results[n].p_goal for n in names]
    bars = ax.bar(range(len(names)), pgoal, color="#4c78a8")
    if names:
        bars[int(np.argmax(pgoal))].set_color("#2e9e5b")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set(ylabel="P(goal)", ylim=(0, 1), title="Probability of reaching the goal")
    return ax.figure


def plot_policy_heatmap(policy, target: float, steps_per_year: int = 12,
                        regime=None, max_wealth_mult: float = 2.5, ax=None):
    """Heatmap of the *learned* policy: % in stocks over wealth (y) × time (x).

    Works for any tabular agent exposing ``surface()`` and ``w_grid`` (G-Learner,
    Regime-Aware G-Learner, Q-Learner). This visualizes the actual RL artifact —
    the policy the agent learned.
    """
    surf = policy.surface(regime)            # (T, n_wbins)
    w = policy.w_grid
    keep = w <= max_wealth_mult * target
    T = surf.shape[0]
    years = np.arange(T + 1) / steps_per_year
    yw = np.concatenate([w[keep], [w[keep][-1] * 1.001]]) / target
    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 4))
    mesh = ax.pcolormesh(years, yw, surf[:, keep].T, cmap="RdYlGn_r", vmin=0, vmax=1, shading="flat")
    ax.axhline(1.0, color="black", lw=1.2, ls="--")
    ax.text(years[-1] * 0.99, 1.02, "goal", ha="right", va="bottom", fontsize=8)
    ax.set(xlabel="years from now", ylabel="wealth (× goal)",
           title="Learned policy — share held in stocks")
    cb = ax.figure.colorbar(mesh, ax=ax)
    cb.set_label("% in stocks")
    return ax.figure
