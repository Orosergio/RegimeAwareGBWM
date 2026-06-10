"""Figures for the historical backtest (the report's "money shots").

Kept separate from :mod:`gbwm.evaluation.plots` because these know about a
:class:`~gbwm.backtesting.historical.Deployment`; they reuse that module's deck
palette so everything stays visually consistent.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.dates import date2num

from gbwm.backtesting.historical import CRISES, Deployment
from gbwm.evaluation import plots
from gbwm.evaluation.plots import INK, LOSS, MUTED, REGIME_COLORS, RISK, RL

plt = plots.plt

_STRAT_COLOR = {
    "Buy & Hold": LOSS,
    "60/40": "#9aa0a6",
    "Glide Path": RISK,
    "G-Learner": "#7048e8",
    "Regime-Aware G-Learner": RL,
    # multi-asset strategies
    "Regime-Aware Multi-Asset": RL,
    "Balanced Mix": "#9aa0a6",
    "All-in S&P 500": LOSS,
}

# Per-asset colors for the multi-asset stacked-allocation charts (by asset key).
_ASSET_COLOR = {
    "us_equity": "#4c6ef5",    # blue   — S&P 500
    "intl_equity": "#7048e8",  # purple — international stocks
    "bonds": "#2e9e5b",        # green  — bonds (the safe leg)
    "gold": "#e0a100",         # gold   — gold
}
_CASH_COLOR = "#cdcdc4"

# Friendly plan names — keep chart legends and the app's tables saying the same
# thing (mirrors the app's FRIENDLY_NAME map; multi-asset names come from
# STRATEGY_LABELS via :func:`_multi_label`).
_FRIENDLY = {
    "Buy & Hold": "All-in stocks",
    "60/40": "Classic 60/40 mix",
    "Glide Path": "Target-date glide path",
    "G-Learner": "Goal-based plan",
    "Regime-Aware G-Learner": "Smart adaptive plan",
    "Q-Learner": "Self-taught (Q-learning)",
}


def _label(name: str) -> str:
    return _FRIENDLY.get(name) or _multi_label(name)


def _shade_crises(ax, dates: pd.DatetimeIndex, label: bool = False) -> None:
    lo, hi = dates[0], dates[-1]
    for cr in CRISES:
        a = pd.Timestamp(cr.start + "-01")
        b = pd.Timestamp(cr.end + "-28")
        if b < lo or a > hi:
            continue
        ax.axvspan(max(a, lo), min(b, hi), color=LOSS, alpha=0.07, lw=0)
        if label:
            mid = max(a, lo) + (min(b, hi) - max(a, lo)) / 2
            ax.text(mid, 0.98, cr.name, rotation=90, va="top", ha="center",
                    fontsize=7, color=MUTED, transform=ax.get_xaxis_transform())


def plot_journey(
    dep: Deployment,
    smart: str = "Regime-Aware G-Learner",
    naive: str = "Buy & Hold",
    figsize: tuple[float, float] = (10, 9),
):
    """Three stacked panels sharing a time axis:

    1. **Balance over time** for every strategy, goal line, crisis bands.
    2. **The smart agent's stock allocation** vs the naive one — it cuts risk.
    3. **The agent's 'bad-weather' belief** (P(bear)+P(high-vol)) — detected online.
    """
    dates = dep.dates
    t = np.concatenate([[dates[0] - pd.DateOffset(months=1)], dates])  # wealth has T+1 points

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, sharex=True,
                                        gridspec_kw={"height_ratios": [3, 2, 2]})

    # --- Panel 1: balances ------------------------------------------------- #
    for name, res in dep.results.items():
        w = res.histories["wealth"][0]
        c = _STRAT_COLOR.get(name, MUTED)
        lw = 2.6 if name in (smart, naive) else 1.3
        alpha = 1.0 if name in (smart, naive) else 0.55
        ax1.plot(t, w, color=c, lw=lw, alpha=alpha, label=_label(name))
    ax1.axhline(dep.target, color=INK, lw=1.4, ls="--")
    ax1.text(t[1], dep.target, "  goal", va="bottom", ha="left", fontsize=8, color=INK)
    _shade_crises(ax1, dates, label=True)
    ax1.set(ylabel="balance ($)",
            title=f"{dep.market.label} — deployed {dates[0]:%b %Y}: "
                  "every plan on the SAME real history")
    ax1.legend(loc="upper left", fontsize=8, ncol=2)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))

    # --- Panel 2: allocation ---------------------------------------------- #
    if smart in dep.results:
        eq = dep.equity(smart)
        ax2.fill_between(dates, 0, eq * 100, color=RL, alpha=0.20)
        ax2.plot(dates, eq * 100, color=RL, lw=2.0, label=f"{_label(smart)} (adapts)")
        # On-canvas teaching: label the deepest risk cut inside each crisis band,
        # so the reaction is read off the chart itself, not from a caption.
        for cr in CRISES:
            a = pd.Timestamp(cr.start + "-01")
            b = pd.Timestamp(cr.end + "-28")
            mask = (dates >= a) & (dates <= b)
            if not mask.any():
                continue
            sub = np.asarray(eq[mask], dtype=float) * 100
            i = int(np.argmin(sub))
            d0, v = dates[mask][i], float(sub[i])
            ax2.annotate(f"cut to {v:.0f}%", xy=(d0, v), xytext=(d0, min(v + 30, 86)),
                         fontsize=7.5, color=INK, ha="center",
                         arrowprops=dict(arrowstyle="-", color=INK, lw=0.7, alpha=0.6))
    if naive in dep.results:
        ax2.plot(dates, dep.equity(naive) * 100, color=LOSS, lw=1.6, ls="--",
                 label=f"{_label(naive)} (always 100%)")
    _shade_crises(ax2, dates)
    ax2.set(ylabel="% in stocks", ylim=(0, 105),
            title="The smart agent dials risk up and down; buy-and-hold never moves")
    ax2.legend(loc="upper right", fontsize=8)

    # --- Panel 3: bad-weather belief -------------------------------------- #
    if smart in dep.results:
        rb = dep.risk_belief(smart)
        ax3.fill_between(dates, 0, rb * 100, color=RISK, alpha=0.30)
        ax3.plot(dates, rb * 100, color=RISK, lw=1.8)
    _shade_crises(ax3, dates)
    ax3.set(ylabel="P(bad weather) %", xlabel="year", ylim=(0, 105),
            title="What the agent believed in real time: P(bear) + P(high-vol), "
                  "from past returns only")

    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Multi-asset money-shots: how three portfolios split S&P / intl / bonds / gold
# / cash over real history, and how the "smart" one rotates with the regime.
# --------------------------------------------------------------------------- #
def _multi_label(name: str) -> str:
    from gbwm.backtesting.multiasset import STRATEGY_LABELS
    return STRATEGY_LABELS.get(name, name)


def _mark_crises(ax, dates: pd.DatetimeIndex) -> None:
    """Thin dotted vertical lines at the *onset* of each in-window crisis."""
    lo, hi = dates[0], dates[-1]
    for cr in CRISES:
        a = pd.Timestamp(cr.start + "-01")
        if lo <= a <= hi:
            ax.axvline(a, color=INK, lw=0.8, ls=":", alpha=0.45)


def _regime_ribbon(ax, dates: pd.DatetimeIndex, regime_idx: np.ndarray,
                   regime_names: list[str]) -> None:
    """A 1-row colored strip of the detected market 'weather' over time."""
    edges = date2num(dates)
    step = (edges[-1] - edges[-2]) if len(edges) > 1 else 30.0
    edges = np.append(edges, edges[-1] + step)
    cmap = mcolors.ListedColormap([REGIME_COLORS.get(n, MUTED) for n in regime_names])
    ax.pcolormesh(edges, np.array([0.0, 1.0]), regime_idx[None, :].astype(float),
                  cmap=cmap, vmin=0, vmax=max(len(regime_names) - 1, 1), shading="flat")
    ax.set_yticks([])
    ax.margins(x=0)


def _stack_allocation(ax, dep, strategy: str) -> None:
    """Stacked allocation area (assets + cash) for one strategy over real dates."""
    W = dep.weights(strategy)                       # (T, A)
    cash = np.clip(1.0 - W.sum(axis=1), 0.0, 1.0)
    stack = np.column_stack([W, cash]).T            # (A+1, T)
    colors = [_ASSET_COLOR.get(k, MUTED) for k in dep.asset_keys] + [_CASH_COLOR]
    labels = list(dep.asset_labels) + ["Cash"]
    ax.stackplot(dep.dates, stack, colors=colors, labels=labels, alpha=0.92)
    ax.set_ylim(0, 1)
    ax.margins(x=0)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))


def plot_allocation_grid(dep, strategies: list[str] | None = None,
                         regime_strategy: str = "Regime-Aware Multi-Asset",
                         figsize: tuple[float, float] = (11, 9.5)):
    """The centerpiece: a market-weather ribbon on top, then one stacked
    allocation panel per portfolio — so you *see* the smart plan rotate into
    bonds/gold during crises while the static mixes never budge."""
    strategies = strategies or list(dep.results.keys())
    n = len(strategies)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(n + 1, 1, height_ratios=[0.5] + [3] * n, hspace=0.32)

    rax = fig.add_subplot(gs[0])
    reg_src = regime_strategy if regime_strategy in dep.results else strategies[0]
    reg = dep.results[reg_src].histories["regime"][0]
    _regime_ribbon(rax, dep.dates, reg, dep.regime_names)
    rax.set_title("Market weather the smart plan detected "
                  "(green=bull · blue=stable · amber=choppy · red=bear)",
                  fontsize=10, loc="left")
    plt.setp(rax.get_xticklabels(), visible=False)

    first_ax = None
    for i, s in enumerate(strategies):
        ax = fig.add_subplot(gs[i + 1], sharex=rax)
        first_ax = first_ax or ax
        _stack_allocation(ax, dep, s)
        _mark_crises(ax, dep.dates)
        ax.set_title(_multi_label(s), fontsize=11, loc="left", fontweight="600")
        if i < n - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
    handles, labels = first_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               fontsize=9, bbox_to_anchor=(0.5, 0.965), frameon=False)
    fig.subplots_adjust(top=0.9)
    return fig


def plot_multi_balances(dep, figsize: tuple[float, float] = (11, 4.6)):
    """Balances of every portfolio over the real history, with goal + crises."""
    dates = dep.dates
    t = np.concatenate([[dates[0] - pd.DateOffset(months=1)], dates])
    fig, ax = plt.subplots(figsize=figsize)
    for name in dep.results:
        ax.plot(t, dep.wealth(name), color=_STRAT_COLOR.get(name, MUTED), lw=2.3,
                label=_multi_label(name))
    ax.axhline(dep.target, color=INK, lw=1.3, ls="--")
    ax.text(t[1], dep.target, "  goal", va="bottom", ha="left", fontsize=8, color=INK)
    _shade_crises(ax, dates, label=True)
    ax.set(ylabel="balance ($)", xlabel="year",
           title=f"Your money over real history ({dates[0]:%b %Y} → {dates[-1]:%b %Y})")
    ax.legend(loc="upper left", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    ax.margins(x=0)
    return fig


def plot_allocation_snapshot(dep, strategy: str, step: int,
                             figsize: tuple[float, float] = (4.8, 4.8)):
    """A donut of one portfolio's split at a single month — the 'live' view."""
    step = int(np.clip(step, 0, dep.weights(strategy).shape[0] - 1))
    W = dep.weights(strategy)[step]
    cash = max(0.0, 1.0 - float(W.sum()))
    vals = list(W) + [cash]
    labels = list(dep.asset_labels) + ["Cash"]
    colors = [_ASSET_COLOR.get(k, MUTED) for k in dep.asset_keys] + [_CASH_COLOR]
    keep = [(v, l, c) for v, l, c in zip(vals, labels, colors) if v > 0.005]
    vv, ll, cc = zip(*keep)
    fig, ax = plt.subplots(figsize=figsize)
    ax.pie(vv, labels=[f"{l}\n{v*100:.0f}%" for v, l in zip(vv, ll)], colors=cc,
           startangle=90, textprops={"fontsize": 9},
           wedgeprops={"width": 0.46, "edgecolor": "white", "linewidth": 1.5})
    ax.set_title(f"{_multi_label(strategy)} — {dep.dates[step]:%b %Y}", fontsize=11)
    return fig


def plot_sequence_risk(seq_df: pd.DataFrame, metric: str = "Max drawdown",
                       figsize: tuple[float, float] = (9, 4.5)):
    """Grouped bars of ``metric`` by start year × strategy.

    The point: starting right before a crash (2000, 2007) is where a
    risk-managing agent earns its keep — its bars stay low.
    """
    starts = list(dict.fromkeys(seq_df["Start"]))
    strategies = list(dict.fromkeys(seq_df["Strategy"]))
    fig, ax = plt.subplots(figsize=figsize)
    n = len(strategies)
    width = 0.8 / max(n, 1)
    x = np.arange(len(starts))
    for i, strat in enumerate(strategies):
        sub = seq_df[seq_df["Strategy"] == strat].set_index("Start")
        vals = [sub.loc[s, metric] * 100 if s in sub.index else 0 for s in starts]
        ax.bar(x + i * width, vals, width, label=_label(strat), color=_STRAT_COLOR.get(strat, MUTED))
    ax.set_xticks(x + width * (n - 1) / 2)
    ax.set_xticklabels([f"started {s}" for s in starts])
    ax.set(ylabel=f"{metric} (%)",
           title=f"{metric} by when you started — sequence-of-returns risk")
    ax.legend(fontsize=8, ncol=2, loc="upper right")
    fig.tight_layout()
    return fig
