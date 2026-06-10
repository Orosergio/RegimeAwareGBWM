"""Matplotlib figures for the report and the demo (Agg backend, headless-safe).

Each function returns a Figure so callers (Streamlit, notebooks, CLI) decide how
to render or save. Matplotlib is a core dependency; the app may also use Plotly.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ---- Deck palette (mirrors the app CSS tokens so charts blend into the page) ----------
INK = "#15120d"    # == app --ink (warm near-black); was #0a0b0d (cooler, off by a hair)
PAPER = "#f7f4ee"  # == app --paper; was #fafaf7, which painted a visibly lighter rectangle on the page
MUTED = "#5f574c"  # darker than #6b6b66 so chart ticks/labels clear WCAG AA on paper
GOAL = "#2e9e5b"   # green  — safety / goal
RISK = "#cf8a2c"   # amber  — risk / regime
RL = "#4c6ef5"     # blue   — RL / data
LOSS = "#e0553a"   # red    — shortfall

REGIME_COLORS = {
    "bull": GOAL,
    "stable": RL,
    "high_vol": RISK,
    "bear": LOSS,
}


def apply_deck_style() -> None:
    """Match figures to the proposal deck: warm paper, ink text, clean spines."""
    plt.rcParams.update({
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#d8d3c8",
        "grid.linewidth": 0.8,
        "font.size": 12,
        "axes.titlesize": 13.5,
        "axes.titleweight": "600",
        "legend.frameon": False,
        "figure.autolayout": True,
    })


apply_deck_style()


def plot_wealth_paths(histories: dict, target: float, n_show: int = 60, ax=None):
    wealth = histories["wealth"]
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    t = np.arange(wealth.shape[1])
    idx = np.random.default_rng(0).choice(wealth.shape[0], size=min(n_show, wealth.shape[0]), replace=False)
    ax.plot(t, wealth[idx].T, color=RL, alpha=0.12, lw=0.8)
    ax.plot(t, np.median(wealth, axis=0), color=INK, lw=2.2, label="median")
    ax.plot(t, np.percentile(wealth, 10, axis=0), color=INK, lw=1, ls="--", label="10th pct")
    ax.plot(t, np.percentile(wealth, 90, axis=0), color=INK, lw=1, ls=":", label="90th pct")
    ax.axhline(target, color=LOSS, lw=1.8, label="goal")
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
    ax.hist(terminal_wealth, bins=60, color=RL, alpha=0.8)
    ax.axvline(target, color=LOSS, lw=2, label="goal")
    ax.set(xlabel="terminal wealth", ylabel="count", title="Terminal-wealth distribution")
    ax.legend(fontsize=8)
    return ax.figure


def plot_strategy_comparison(results: dict, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    names = list(results.keys())
    pgoal = [results[n].p_goal for n in names]
    bars = ax.bar(range(len(names)), pgoal, color=RL)
    if names:
        bars[int(np.argmax(pgoal))].set_color(GOAL)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set(ylabel="P(goal)", ylim=(0, 1), title="Probability of reaching the goal")
    return ax.figure


# Risk colormap: red = risk-on, blue = safe. RdYlBu is colorblind-safe (red–blue
# axis survives deuteranopia, unlike the previous RdYlGn) and the contour labels
# below make the surface readable with no color at all.
RISK_CMAP = "RdYlBu_r"


def _heatmap_panel(ax, surf, w, keep, target, steps_per_year, *, frontier=True):
    """Draw one policy surface on ``ax``; returns the mesh (for a shared colorbar)."""
    T = surf.shape[0]
    years = np.arange(T + 1) / steps_per_year
    yw = np.concatenate([w[keep], [w[keep][-1] * 1.001]]) / target
    ax.grid(False)  # rcParams grid would draw white lines over the surface
    mesh = ax.pcolormesh(years, yw, surf[:, keep].T, cmap=RISK_CMAP, vmin=0, vmax=1, shading="flat")
    ax.axhline(1.0, color="black", lw=1.1, ls="--")
    if frontier:
        # One labeled contour at the 50/50 risk frontier — readable without color.
        # Tiny-wealth rows (< 0.2× goal) are skipped: the log wealth grid stripes
        # there and would repeat the label over a bin artifact.
        try:
            yc = (years[:-1] + years[1:]) / 2
            wc = (yw[:-1] + yw[1:]) / 2
            sub = wc >= 0.2
            cs = ax.contour(yc, wc[sub], surf[:, keep].T[sub], levels=[0.5],
                            colors=[INK], linewidths=0.8, alpha=0.6)
            ax.clabel(cs, fmt=lambda v: "50/50 frontier", fontsize=6.5, inline=True)
        except Exception:  # degenerate (flat) surfaces have no contours — fine
            pass
    return mesh


def _risk_colorbar(fig, mesh, ax):
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.set_ticklabels(["0% — all safe", "50 / 50", "100% — all stocks"])
    cb.ax.tick_params(labelsize=8)
    return cb


def plot_policy_heatmap(policy, target: float, steps_per_year: int = 12,
                        regime=None, max_wealth_mult: float = 2.5, ax=None):
    """Heatmap of the *learned* policy: % in stocks over wealth (y) × time (x).

    Works for any tabular agent exposing ``surface()`` and ``w_grid`` (G-Learner,
    Regime-Aware G-Learner, Q-Learner). This visualizes the actual RL artifact —
    the policy the agent learned. Self-teaching: dashed goal line, labeled
    contours ("50% stocks"), and plain-words hints for the two halves.
    """
    surf = policy.surface(regime)            # (T, n_wbins)
    w = policy.w_grid
    keep = w <= max_wealth_mult * target
    if ax is None:
        _, ax = plt.subplots(figsize=(7.2, 4))
    mesh = _heatmap_panel(ax, surf, w, keep, target, steps_per_year)
    years_end = surf.shape[0] / steps_per_year
    ax.text(years_end * 0.99, 1.02, "goal", ha="right", va="bottom", fontsize=8)
    ax.text(0.02, 0.97, "ahead of the goal → it locks in safety", transform=ax.transAxes,
            fontsize=7.5, color=INK, alpha=0.75, va="top")
    ax.text(0.02, 0.12, "behind the goal → it pushes harder", transform=ax.transAxes,
            fontsize=7.5, color=INK, alpha=0.75, va="bottom")
    ax.set(xlabel="years from now", ylabel="wealth (× goal)")
    ax.set_title("Learned policy — share held in stocks", fontsize=12, loc="left")
    _risk_colorbar(ax.figure, mesh, ax)
    return ax.figure


def plot_policy_regime_grid(policy, target: float, steps_per_year: int = 12,
                            regime_names: list[str] | None = None,
                            regime_labels: dict[str, str] | None = None,
                            max_wealth_mult: float = 2.5):
    """The thesis figure: the SAME learned brain, one policy map per market regime.

    Side by side, the regime-dependence is visible at a glance (no recall needed):
    in bad weather the red risk-on region shrinks and the safe blue region grows.
    """
    names = list(regime_names or ["bull", "stable", "high_vol", "bear"])
    labels = regime_labels or {}
    w = policy.w_grid
    keep = w <= max_wealth_mult * target
    fig, axes = plt.subplots(1, len(names), figsize=(2.75 * len(names) + 1.6, 3.4), sharey=True)
    fig.set_layout_engine("constrained")     # tight_layout fights a shared colorbar
    axes = np.atleast_1d(axes)
    mesh = None
    for ax, rn in zip(axes, names):
        mesh = _heatmap_panel(ax, policy.surface(rn), w, keep, target, steps_per_year,
                              frontier=False)
        ax.set_title(labels.get(rn, rn), fontsize=10.5)
        ax.set_xlabel("years from now", fontsize=8.5)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("wealth (× goal)")
    axes[-1].text(0.97, 1.03, "goal", transform=axes[-1].get_yaxis_transform(),
                  ha="right", va="bottom", fontsize=7.5)
    _risk_colorbar(fig, mesh, list(axes))
    fig.suptitle("One brain, four weathers — the same situation gets a different answer in each regime",
                 fontsize=11.5, fontweight="600")
    return fig
