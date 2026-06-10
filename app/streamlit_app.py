"""Regime-Aware GBWM — interactive goal planner + reinforcement-learning lab.

Styled to match the CME 241 proposal deck (docs/): Geist + Instrument Serif,
warm paper / ink palette, goal-green / risk-amber / RL-blue / loss-red accents.

Navigation is a guided story in three phases — Learn (How it works · Inside the
AI) → Play (Your plan · Live bank NT$ · One journey) → Proof (Compare · Time
machine · Multi-asset · Proposal coverage). One page renders at a time (stateful
nav, not st.tabs) so each screen has a single focus, "Next" buttons really
navigate, and heavy pages only compute when visited. "How it works" opens with a
live demo that queries the actual learned policy. The Live Bank is a step-by-step
simulated bank (deposit/withdraw into wallets) driven live by the goal-based RL
allocator, with a fully transparent month-by-month decision & trade log.

Educational simulation — NOT financial advice. Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import inspect
import json
import re
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gbwm.backtesting import (  # noqa: E402
    STRATEGY_LABELS, default_multi_asset_config, diary_sentences, list_markets,
    list_universe, multi_asset_diary, run_deployment, run_multi_asset_deployment,
    scorecard, sequence_risk,
)
from gbwm.backtesting.plotting import (  # noqa: E402
    plot_allocation_grid, plot_allocation_snapshot, plot_journey,
    plot_multi_balances, plot_sequence_risk,
)
from gbwm.config import default_config  # noqa: E402
from gbwm.evaluation import plots  # noqa: E402
from gbwm.evaluation.harness import compare_policies, run_policy  # noqa: E402
from gbwm.explain import EpisodeContext, RuleBasedAdvisor, StepContext  # noqa: E402
from gbwm.policies import (  # noqa: E402
    BuyAndHold, GLearner, GlidePath, QLearner, RegimeAwareGLearner, SixtyForty,
)
from gbwm.sandbox import build_session, list_sandbox_markets, restore_session  # noqa: E402

INK, PAPER, GOAL, RISK, RL, LOSS = plots.INK, plots.PAPER, plots.GOAL, plots.RISK, plots.RL, plots.LOSS

st.set_page_config(page_title="Goal Planner — Regime-Aware GBWM", page_icon="🎯", layout="wide")
ADVISOR = RuleBasedAdvisor()
# Auto-play needs st.fragment(run_every=…) (Streamlit ≥1.37); degrade gracefully if absent.
_HAS_RUN_EVERY = "run_every" in inspect.signature(st.fragment).parameters

# ---- design system (warm editorial / quiet-luxury; hand-tuned, not "AI slop") #
# Preconnect to Google Fonts so the @import below resolves its connection sooner (faster first paint).
st.markdown(
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
    unsafe_allow_html=True,
)
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Geist+Mono:wght@400;500;600&family=Instrument+Serif:ital@0;1&display=swap');
:root{
  --paper:#f7f4ee; --card:#fffdf8; --ink:#15120d; --muted-strong:#6a6157;
  --line:rgba(21,18,13,.12); --hair:rgba(21,18,13,.07);
  --accent:#1c6b50; --accent-soft:rgba(28,107,80,.10);
  --shadow:0 1px 2px rgba(21,18,13,.03), 0 14px 34px -22px rgba(21,18,13,.28);
  --shadow-lift:0 1px 2px rgba(21,18,13,.04), 0 22px 48px -24px rgba(21,18,13,.36);
  --ease:cubic-bezier(.2,.7,.2,1);
}
html, body, [class*="css"], .stApp, .stMarkdown, p, div, span, label { font-family: 'Geist', system-ui, sans-serif; }
.stApp { background: var(--paper); color: var(--ink); }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.6rem; max-width: 1200px; }
::selection { background: var(--accent-soft); }
*::-webkit-scrollbar { width: 10px; height: 10px; }
*::-webkit-scrollbar-thumb { background: rgba(21,18,13,.30); border-radius: 8px; }
*::-webkit-scrollbar-thumb:hover { background: rgba(21,18,13,.48); }

/* ---- type scale ---- */
h1, h2, h3, h4 { font-family: 'Geist', sans-serif; color: var(--ink); font-weight: 600; letter-spacing: -.025em; }
h1 { font-size: 2.55rem; line-height: 1.05; letter-spacing: -.042em; margin: .15rem 0 .35rem; }
h2 { font-size: 1.5rem; letter-spacing: -.03em; }
h3 { font-size: 1.16rem; }
.stMarkdown p, .stMarkdown li { color: #423c33; line-height: 1.6; }
.gbwm-serif { font-family: 'Instrument Serif', serif; font-style: italic; font-weight: 400; letter-spacing: -.005em; }
.gbwm-eyebrow { font-family: 'Geist Mono', monospace; font-size: .72rem; letter-spacing: .24em;
  text-transform: uppercase; color: var(--muted-strong); }
.gbwm-sub { font-family: 'Geist Mono', monospace; font-size: .72rem; letter-spacing: .15em;
  text-transform: uppercase; color: var(--muted-strong); }

/* ---- metrics (tabular numerals, quiet card) ---- */
[data-testid="stMetric"] { background: var(--card); border: 1px solid var(--hair); border-radius: 12px; padding: 12px 15px; }
[data-testid="stMetricValue"] { font-family: 'Geist Mono', monospace; font-weight: 600; letter-spacing: -.02em;
  color: var(--ink); font-variant-numeric: tabular-nums; }
[data-testid="stMetricLabel"] p { color: var(--muted-strong); font-size: .82rem; }

/* ---- tabs: quiet underline, no chrome ---- */
.stTabs [data-baseweb="tab-list"] { gap: 1px; border-bottom: 1px solid var(--line); flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] { font-weight: 500; color: var(--muted-strong); padding: 9px 13px;
  border-radius: 9px 9px 0 0; transition: color .16s var(--ease), background .16s var(--ease); }
.stTabs [data-baseweb="tab"]:hover { color: var(--ink); background: rgba(21,18,13,.035); }
.stTabs [aria-selected="true"] { color: var(--ink) !important; }
.stTabs [data-baseweb="tab-highlight"] { background: var(--accent) !important; height: 2px; }
.stTabs [data-baseweb="tab-border"] { background: transparent; }

/* ---- cards ---- */
.gbwm-card { position: relative; border: 1px solid var(--hair); border-radius: 14px; padding: 16px 18px;
  background: var(--card); height: 100%; box-shadow: var(--shadow);
  transition: transform .2s var(--ease), box-shadow .2s var(--ease), border-color .2s var(--ease); }
.gbwm-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-lift); border-color: var(--line); }
.gbwm-card .ico { font-size: 1.22rem; opacity: .92; }
.gbwm-card .lbl { font-family: 'Geist Mono', monospace; font-size: .66rem; letter-spacing: .16em;
  text-transform: uppercase; color: var(--muted-strong); margin-top: 9px; }
.gbwm-card .val { font-size: 1.0rem; margin-top: 5px; line-height: 1.45; color: var(--ink); }
.gbwm-card .dot { position: absolute; top: 16px; right: 16px; width: 7px; height: 7px; border-radius: 50%; }

/* ---- intro callout: refined paper card, full hairline box (no side-stripe) ---- */
.gbwm-intro { position: relative; border: 1px solid var(--hair);
  border-radius: 12px; padding: 15px 18px; margin: 4px 0 22px; background: var(--card); box-shadow: var(--shadow); }
.gbwm-intro .t { font-weight: 600; font-size: 1.05rem; letter-spacing: -.012em; color: var(--ink); margin-bottom: 4px; }
.gbwm-intro .d { font-size: .95rem; color: #423c33; line-height: 1.58; }
.gbwm-intro ul { margin: 10px 0 0; padding-left: 1.05rem; }
.gbwm-intro li { font-size: .9rem; color: #534b41; margin: 4px 0; line-height: 1.5; }
.gbwm-intro strong { font-weight: 600; color: var(--ink); }

/* ---- RL badge (quiet, monospace) ---- */
.gbwm-rl { display: inline-block; background: transparent; color: var(--accent);
  border: 1px solid rgba(28,107,80,.34); border-radius: 6px; padding: 1px 7px;
  font-family: 'Geist Mono', monospace; font-weight: 600; font-size: .72rem; letter-spacing: .03em; }

/* ---- weather pill ---- */
.gbwm-pill { display: inline-block; padding: 5px 14px; border-radius: 999px; font-weight: 600;
  font-size: .92rem; border: 1px solid var(--hair); }

/* ---- balance odometer ---- */
.gbwm-odo { font-family: 'Geist Mono', monospace; font-weight: 600; letter-spacing: -.02em;
  font-size: 2.2rem; color: var(--ink); line-height: 1.04; font-variant-numeric: tabular-nums; }
.gbwm-odo .cur { font-size: 1.05rem; color: var(--muted-strong); margin-right: 6px; font-weight: 500; }

/* ---- buttons ---- */
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
  border-radius: 10px; border: 1px solid var(--line); font-weight: 500; box-shadow: none;
  transition: transform .15s var(--ease), border-color .15s var(--ease), background .15s var(--ease); }
.stButton > button:hover, .stDownloadButton > button:hover, .stFormSubmitButton > button:hover {
  border-color: var(--ink); transform: translateY(-1px); }
.stButton > button:active, .stDownloadButton > button:active { transform: translateY(0); }
button[kind="primary"], button[kind="primaryFormSubmit"] {
  background: var(--ink) !important; color: var(--paper) !important; border-color: var(--ink) !important; }
button[kind="primary"]:hover, button[kind="primaryFormSubmit"]:hover { background: #000 !important; }
/* ---- visible keyboard focus (native chrome was stripped above) ---- */
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible,
.stFormSubmitButton > button:focus-visible, .stTabs [data-baseweb="tab"]:focus-visible,
[role="switch"]:focus-visible, [data-testid="stSidebar"] a:focus-visible {
  outline: 2px solid var(--accent) !important; outline-offset: 2px; border-radius: 8px; }
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within {
  outline: 2px solid var(--accent); outline-offset: 1px; }

/* ---- inputs ---- */
.stNumberInput input, .stTextInput input, [data-baseweb="select"] > div { border-radius: 10px !important; }
[data-baseweb="input"]:focus-within, [data-baseweb="select"] > div:focus-within { border-color: var(--accent) !important; }

/* ---- containers ---- */
[data-testid="stDataFrame"] { border: 1px solid var(--hair); border-radius: 12px; overflow: hidden; }
[data-testid="stExpander"] { border: 1px solid var(--hair) !important; border-radius: 12px !important;
  background: var(--card); box-shadow: var(--shadow); }
[data-testid="stExpander"] summary { font-weight: 500; }
[data-testid="stExpander"] summary:hover { color: var(--accent); }
hr { border-color: var(--line) !important; margin: 1.1rem 0; }
[data-testid="stProgress"] div[role="progressbar"] > div { background: var(--accent) !important; }
[data-testid="stProgress"] div[role="progressbar"] { background: rgba(21,18,13,.08) !important; }

/* ---- sidebar ---- */
[data-testid="stSidebar"] { background: #efebe1; border-right: 1px solid var(--line); }
[data-testid="stSidebar"] .stMarkdown p { color: #534b41; }

/* ---- alerts: soften the default chrome ---- */
[data-testid="stAlert"] { border-radius: 12px; border: 1px solid var(--hair); }

/* ---- links: keep the warm accent, not default blue ---- */
.stMarkdown a, .gbwm-intro a { color: var(--accent); text-decoration: underline; text-underline-offset: 2px; }

/* ---- page navigation: the nav radios become quiet pills (radio dot collapsed,
       input kept zero-size so keyboard focus still works) ---- */
.st-key-nav_phase [role="radiogroup"], [class*="st-key-nav_page"] [role="radiogroup"] { gap: 6px; flex-wrap: wrap; }
.st-key-nav_phase label[data-baseweb="radio"], [class*="st-key-nav_page"] label[data-baseweb="radio"] {
  border: 1px solid transparent; border-radius: 999px; padding: 4px 14px; margin: 0 2px 0 0;
  transition: background .16s var(--ease), border-color .16s var(--ease); }
.st-key-nav_phase label[data-baseweb="radio"] > div:first-child,
[class*="st-key-nav_page"] label[data-baseweb="radio"] > div:first-child {
  width: 0 !important; height: 0 !important; opacity: 0; margin: 0; overflow: hidden; }
.st-key-nav_phase label[data-baseweb="radio"]:hover,
[class*="st-key-nav_page"] label[data-baseweb="radio"]:hover { background: rgba(21,18,13,.045); }
.st-key-nav_phase label[data-baseweb="radio"]:focus-within,
[class*="st-key-nav_page"] label[data-baseweb="radio"]:focus-within {
  outline: 2px solid var(--accent); outline-offset: 2px; }
.st-key-nav_phase label[data-baseweb="radio"]:has(input:checked) { background: var(--ink); border-color: var(--ink); }
.st-key-nav_phase label[data-baseweb="radio"]:has(input:checked) p { color: var(--paper) !important; }
.st-key-nav_phase p { font-weight: 600; }
[class*="st-key-nav_page"] label[data-baseweb="radio"]:has(input:checked) {
  background: var(--accent-soft); border-color: rgba(28,107,80,.35); }
[class*="st-key-nav_page"] label[data-baseweb="radio"]:has(input:checked) p {
  color: var(--accent) !important; font-weight: 600; }

/* ---- chart-reading keys: tiny swatch + label, sits right under its figure ---- */
.gbwm-keys { display: flex; flex-wrap: wrap; gap: 4px 16px; margin: 2px 0 4px; }
.gbwm-keys .k { display: inline-flex; align-items: center; gap: 6px; font-size: .8rem; color: #534b41; }
.gbwm-keys .sw { width: 14px; height: 4px; border-radius: 2px; flex: none; }
.gbwm-keys .sw.dash { height: 3px; border-radius: 0;
  background: repeating-linear-gradient(90deg, currentColor 0 4px, transparent 4px 7px); }

/* ---- the verdict block: the ONE loud element on "Your plan" ---- */
.gbwm-verdict { border: 1px solid rgba(28,107,80,.30); background: rgba(28,107,80,.07);
  border-radius: 14px; padding: 18px 22px; box-shadow: var(--shadow);
  display: flex; gap: 28px; align-items: center; flex-wrap: wrap; }
.gbwm-verdict .cap { font-family: 'Geist Mono', monospace; font-size: .68rem; letter-spacing: .18em;
  text-transform: uppercase; color: var(--muted-strong); margin-bottom: 6px; }
.gbwm-verdict .big { font-family: 'Geist Mono', monospace; font-weight: 600; font-size: 2.7rem;
  line-height: 1; color: var(--accent); letter-spacing: -.03em; font-variant-numeric: tabular-nums; }
.gbwm-verdict .body { flex: 1 1 320px; min-width: 260px; line-height: 1.55; }
.gbwm-verdict .deltas { font-family: 'Geist Mono', monospace; font-size: .78rem;
  color: var(--muted-strong); letter-spacing: .02em; }

/* ---- stocks/cash split bar for the live decision demo ---- */
.gbwm-bar { display: flex; height: 16px; border-radius: 8px; overflow: hidden;
  border: 1px solid var(--hair); background: #e9e4d8; margin: 8px 0 4px; }
.gbwm-bar .seg { height: 100%; }

/* ---- captions: the app's pedagogy lives in the small print, so keep it readable (AA) ---- */
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: #5f574c !important; }

/* ---- reduced motion: honor the OS preference ---- */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { transition-duration: .01ms !important; animation-duration: .01ms !important; }
  .gbwm-card:hover { transform: none; }
}

/* ---- small screens: tighten padding so the dense tabs breathe ---- */
@media (max-width: 640px) {
  .block-container { padding-top: 1.4rem; padding-left: .8rem; padding-right: .8rem; }
  h1 { font-size: 2rem; }
}
</style>
""",
    unsafe_allow_html=True,
)


def eyebrow(text: str):
    st.markdown(f'<div class="gbwm-eyebrow">{text}</div>', unsafe_allow_html=True)


def _emph(s: str) -> str:
    """Render **bold** / *italic* as HTML (Markdown isn't parsed inside raw HTML)."""
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", s)
    return s


def intro(title: str, desc: str, bullets: list[str] | None = None):
    """A simple, friendly description shown at the top of every tab — what the
    user is about to see, in plain English."""
    items = "".join(f"<li>{_emph(b)}</li>" for b in (bullets or []))
    ul = f"<ul>{items}</ul>" if items else ""
    st.markdown(
        f'<div class="gbwm-intro"><div class="t">{_emph(title)}</div>'
        f'<div class="d">{_emph(desc)}</div>{ul}</div>',
        unsafe_allow_html=True,
    )


def card(col, icon: str, label: str, value: str, accent: str):
    ico = f'<div class="ico">{icon}</div>' if icon else ""
    col.markdown(
        f'<div class="gbwm-card"><span class="dot" style="background:{accent}"></span>'
        f'{ico}<div class="lbl">{label}</div>'
        f'<div class="val">{value}</div></div>',
        unsafe_allow_html=True,
    )


def show_fig(fig, container=None):
    """Render a matplotlib figure and close it (the app reruns often; unclosed
    figures accumulate in matplotlib's registry and leak memory)."""
    (container or st).pyplot(fig)
    plots.plt.close(fig)


def chart_keys(*items, container=None):
    """A compact 'how to read this chart' key row, placed right under its figure.

    Each item is ``(color, text)`` or ``(color, text, "dash")``; color ``None``
    renders text only."""
    chips = []
    for it in items:
        color, text = it[0], it[1]
        dash = len(it) > 2 and it[2] == "dash"
        if color is None:
            sw = ""
        elif dash:
            sw = f'<span class="sw dash" style="color:{color}"></span>'
        else:
            sw = f'<span class="sw" style="background:{color}"></span>'
        chips.append(f'<span class="k">{sw}{_emph(text)}</span>')
    (container or st).markdown(f'<div class="gbwm-keys">{"".join(chips)}</div>',
                               unsafe_allow_html=True)


def goto(page_key: str):
    """Navigate to another page on the next rerun (handled before the nav widgets)."""
    st.session_state["_goto"] = page_key
    st.rerun()


def next_step(target: str, blurb: str, key: str):
    """The guided-story footer: one primary 'Next' action + why it's next."""
    st.divider()
    c = st.columns([1.15, 2.2])
    if c[0].button(f"Next: {PAGE_LABEL[target]}", type="primary", key=key, width="stretch"):
        goto(target)
    c[1].caption(blurb)


# --------------------------------------------------------------------------- #
def money(x): return f"${x:,.0f}"
def pct(x): return f"{x * 100:.0f}%"
def ntd(x): return f"NT$ {x:,.0f}"


# Regime → friendly "weather" with emoji + accent color (kid-simple).
WEATHER = {
    # text colors are darkened so each pill clears WCAG AA (4.5:1) on its own tint.
    "bull": ("Sunny", "☀️", "#1c6340", "rgba(47,133,90,0.12)"),
    "stable": ("Calm", "🌤️", "#1f5288", "rgba(43,108,176,0.12)"),
    "high_vol": ("Choppy", "🌧️", "#7e5210", "rgba(183,121,31,0.14)"),
    "bear": ("Stormy", "⛈️", "#922424", "rgba(197,48,48,0.12)"),
}


def weather_pill(regime: str) -> str:
    name, emo, ink, bg = WEATHER.get(regime, (regime, "•", INK, "#eee"))
    return (f'<span class="gbwm-pill" style="color:{ink};background:{bg};'
            f'border-color:{ink}33">{emo} {name}</span>')


def risk_word(equity: float) -> str:
    return "conservative" if equity < 0.33 else ("balanced" if equity < 0.66 else "aggressive")


# ---- sandbox figures ------------------------------------------------------- #
_ALLOC_COLORS = [RL, "#7048e8", GOAL, RISK, "#0d9488", "#be185d"]
_NEUTRAL = "#bcb6a8"  # one shared warm-neutral for cash / non-highlighted bars (replaces 3 different greys)


def _plain(label: str) -> str:
    """Drop emoji/flags so matplotlib's font (no emoji glyphs) renders cleanly."""
    drop = {0xFE0F, 0x200D, 0x20E3}
    return "".join(ch for ch in label if ord(ch) < 0x1F000 and ord(ch) not in drop).strip()


def _finish_fig(fig, ax, *, legend=True):
    """Hand-tuned chart polish: transparent canvas (blends on paper or card),
    despined axes, hairline grid and muted ink — an editorial, non-default look."""
    fig.patch.set_alpha(0.0)
    for a in (ax if isinstance(ax, (list, tuple)) else [ax]):
        a.patch.set_alpha(0.0)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color("#cbc6bb"); a.spines[side].set_linewidth(0.9)
        a.tick_params(colors="#5f574c", labelsize=8.5, length=3)
        a.yaxis.label.set_color("#5f574c"); a.xaxis.label.set_color("#5f574c")
        a.title.set_color("#15120d"); a.title.set_fontsize(11.5); a.title.set_fontweight("600")
        a.grid(axis="y", color="#15120d", alpha=0.06, lw=0.8)
        a.set_axisbelow(True)
        if legend and a.get_legend_handles_labels()[0]:
            leg = a.legend(loc="upper left", fontsize=8, frameon=False)
            for t in leg.get_texts():
                t.set_color("#534b41")
    return fig


def plot_balance(session) -> "plots.plt.Figure":
    """Wallet balance vs. a buy-&-hold benchmark, with goal line + cashflow marks."""
    w = session.wealth_curve()
    b = session.bench_curve()
    x = list(range(len(w)))
    fig = plots.plt.figure(figsize=(8, 3.1)); ax = fig.add_subplot(111)
    ax.plot(x, b, color="#9aa0a6", lw=1.5, ls=":", label="if you just bought the market", zorder=2)
    ax.plot(x, w, color=INK, lw=2.0, label="the AI plan", zorder=3)
    ax.axhline(session.target, color=LOSS, lw=1.4, ls="--", label="goal")
    ax.fill_between(x, w, session.target, where=w >= session.target,
                    color=GOAL, alpha=0.15, zorder=1)
    for step, amt in session.deposit_marks:
        if step <= len(w) - 1:
            ax.scatter([step], [w[step]], s=42, zorder=4,
                       color=GOAL if amt > 0 else LOSS,
                       marker="^" if amt > 0 else "v",
                       edgecolor="white", linewidth=0.8)
    ax.set(xlabel="months played", ylabel=f"balance ({session.currency})")
    ax.set_title("Your balance, month by month")
    _finish_fig(fig, ax); fig.tight_layout(); return fig


def plot_allocation_bar(session) -> "plots.plt.Figure":
    """Horizontal bar of where the model puts the money right now."""
    alloc = session.current_allocation()
    labels = list(alloc.keys()); vals = list(alloc.values())
    colors = [_ALLOC_COLORS[i % len(_ALLOC_COLORS)] for i in range(len(labels) - 1)] + [_NEUTRAL]
    fig = plots.plt.figure(figsize=(7, 0.55 * len(labels) + 0.9)); ax = fig.add_subplot(111)
    y = range(len(labels))
    ax.barh(list(y), vals, color=colors, edgecolor="#f7f4ee", linewidth=1.2, height=0.66)
    for i, v in enumerate(vals):
        ax.text(min(v + 0.012, 0.965), i, f"{v * 100:.0f}%", va="center", fontsize=9,
                color="#534b41", fontweight="500")
    ax.set_yticks(list(y)); ax.set_yticklabels([_plain(lb) for lb in labels], fontsize=9.5)
    ax.set_xlim(0, 1); ax.invert_yaxis(); ax.set_xlabel("share of your money")
    ax.set_title("Where the model invests right now")
    fig.patch.set_alpha(0.0); ax.patch.set_alpha(0.0)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors="#534b41", length=0); ax.set_xticks([])
    ax.xaxis.label.set_color("#5f574c"); ax.title.set_color("#15120d")
    ax.title.set_fontsize(11.5); ax.title.set_fontweight("600")
    fig.tight_layout(); return fig


def plot_asset_money(session) -> "plots.plt.Figure | None":
    """Stacked area of how much MONEY sits in each asset + cash over time."""
    M = session.asset_money_matrix()  # (t, A+1)
    if M.shape[0] < 1:
        return None
    x = range(1, M.shape[0] + 1)
    labels = [_plain(lb) for lb in session.asset_labels] + ["Cash"]
    colors = [_ALLOC_COLORS[i % len(_ALLOC_COLORS)] for i in range(M.shape[1] - 1)] + [_NEUTRAL]
    fig = plots.plt.figure(figsize=(8, 3.0)); ax = fig.add_subplot(111)
    ax.stackplot(list(x), M.T, labels=labels, colors=colors, alpha=0.9)
    ax.set(xlabel="months played", ylabel=f"money held ({session.currency})")
    ax.set_title("How your money is split across assets, over time")
    ax.margins(x=0)
    _finish_fig(fig, ax)
    leg = ax.legend(loc="upper left", fontsize=7.5, frameon=False,
                    ncol=max(1, (M.shape[1] + 1) // 2))
    for t in leg.get_texts():
        t.set_color("#534b41")
    fig.tight_layout(); return fig


# --- transparency helpers: explain each month's trades + flags -------------- #
_FLAG_EMOJI = {"storm": "🔔 storm", "clearing": "🌤️ clearing",
               "big_drop": "📉 big drop", "goal": "🎯 goal!"}
_FLAG_TOAST = {"storm": "🔔 Storm detected — the AI is shielding your balance.",
               "clearing": "🌤️ Skies clearing — the AI is adding risk again.",
               "big_drop": "📉 Rough month for the market.",
               "goal": "🎯 Goal reached!"}


def _trades_text(decision, asset_labels: list[str]) -> str:
    """'+NT$3,200 stocks, −NT$3,200 bonds' — the biggest money moves this month."""
    moves = list(zip(asset_labels + ["Cash"],
                     list(decision.trades_money) + [decision.cash_trade]))
    moves = [(lb, m) for lb, m in moves if abs(m) >= 1.0]
    moves.sort(key=lambda x: -abs(x[1]))
    if not moves:
        return "no change — kept the same split"
    parts = [f"{'+' if m > 0 else '−'}NT$ {abs(m):,.0f} {_plain(lb)}" for lb, m in moves[:3]]
    return ", ".join(parts)


def _flag_text(flags: list[str]) -> str:
    return " ".join(_FLAG_EMOJI.get(f, "") for f in flags).strip()


REGIME_LABEL = {"bull": "Bull · good times", "stable": "Stable · normal",
                "high_vol": "Choppy · high swings", "bear": "Bear · downturn"}
STRATEGY_BLURB = {
    "Buy & Hold": "All-in on stocks, never adjust.",
    "60/40": "A fixed 60% stocks / 40% safe-cash mix.",
    "Glide Path": "Start bold, automatically get safer as the deadline nears (like a target-date fund).",
    "G-Learner": "A goal-based AI: more risk when you're behind, protect gains when you're ahead.",
    "Regime-Aware G-Learner": "The goal-based AI that ALSO reads the market 'weather' and dials risk up/down.",
    "Q-Learner": "A from-scratch AI that learns the strategy purely by trial and error.",
}
FRIENDLY_NAME = {
    "Regime-Aware G-Learner": "Smart adaptive plan", "G-Learner": "Goal-based plan",
    "Glide Path": "Target-date glide path", "60/40": "Classic 60/40 mix",
    "Buy & Hold": "All-in stocks", "Q-Learner": "Self-taught (Q-learning)",
}
PRESETS = {
    "🏖️ Retirement nest egg": (50_000, 1_000_000, 30, 1_000),
    "🏠 House down-payment": (20_000, 100_000, 7, 900),
    "🎓 Child's college fund": (10_000, 200_000, 18, 500),
    "🛟 Emergency fund": (3_000, 30_000, 4, 500),
    "✏️ Custom (set your own)": None,
}


def make_config(initial, target, horizon, contribution, rf, persistence, n_episodes):
    cfg = default_config()
    cfg.goal.initial_wealth = float(initial); cfg.goal.target_wealth = float(target)
    cfg.goal.horizon_years = int(horizon); cfg.goal.contribution = float(contribution)
    cfg.goal.risk_free_rate = float(rf); cfg.market.risk_free_rate = float(rf)
    cfg.market.transition.persistence = float(persistence)
    cfg.simulation.n_episodes = int(n_episodes)
    return cfg


def cfg_key(cfg):
    g = cfg.goal
    return (g.initial_wealth, g.target_wealth, g.horizon_years, g.contribution,
            g.risk_free_rate, cfg.market.transition.persistence, cfg.simulation.n_episodes)


@st.cache_resource(show_spinner="Preparing the strategies…")
def build_policies(key):
    cfg = make_config(*key)
    return {"Buy & Hold": BuyAndHold.from_config(cfg), "60/40": SixtyForty.from_config(cfg),
            "Glide Path": GlidePath.from_config(cfg), "G-Learner": GLearner.from_config(cfg),
            "Regime-Aware G-Learner": RegimeAwareGLearner.from_config(cfg)}


@st.cache_resource(show_spinner="Q-learning is learning by trial and error…")
def trained_qlearner(key):
    return QLearner.from_config(make_config(*key))


@st.cache_data(show_spinner="Simulating thousands of possible futures…")
def run_comparison(key):
    cfg = make_config(*key)
    results = compare_policies(build_policies(key), cfg)
    out = {}
    for n, r in results.items():
        best_reg = max(r.regime_p_goal, key=r.regime_p_goal.get) if r.regime_p_goal else "—"
        out[n] = {"p_goal": r.p_goal, "median": r.median_terminal, "cvar": r.cvar_shortfall,
                  "drawdown": r.avg_max_drawdown, "p10": r.p10_terminal, "p90": r.p90_terminal,
                  "turnover": r.avg_turnover, "best_regime": best_reg,
                  "start_stock": float(r.histories["weights"][:, 0, :].sum(axis=1).mean())}
    return out


@st.cache_data(show_spinner="Rolling a sample journey…")
def single_path(key, agent_name, seed):
    cfg = make_config(*key)
    res = run_policy(build_policies(key)[agent_name], cfg, n_episodes=1, rng=np.random.default_rng(seed))
    return res.histories, float(res.terminal_wealth[0])


def goal_chance_chart(pgoals):
    order = sorted(pgoals, key=pgoals.get)
    fig = plots.plt.figure(figsize=(7, 3.2)); ax = fig.add_subplot(111)
    vals = [pgoals[n] for n in order]
    # winner = green; the deck's named baseline (glide path) = amber so it never
    # disappears into the neutral pack; everything else = quiet sand.
    colors = [GOAL if n == order[-1] else (RISK if n == "Glide Path" else _NEUTRAL) for n in order]
    ax.barh(range(len(order)), vals, color=colors, edgecolor="#f7f4ee", linewidth=1.2, height=0.66)
    for i, v in enumerate(vals):
        ax.text(min(v + 0.02, 0.985), i, pct(v), va="center", fontsize=9, color="#534b41", fontweight="500")
    ax.set_yticks(range(len(order))); ax.set_yticklabels([FRIENDLY_NAME.get(n, n) for n in order], fontsize=9.5)
    ax.set_xlim(0, 1); ax.set_xlabel("chance of reaching your goal")
    ax.set_title("Which plan gives you the best chance?")
    # editorial polish to match the rest of the app (transparent canvas, no spines, no x-axis chrome)
    fig.patch.set_alpha(0.0); ax.patch.set_alpha(0.0)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors="#534b41", length=0); ax.set_xticks([])
    ax.xaxis.label.set_color("#5f574c"); ax.title.set_color("#15120d")
    ax.title.set_fontsize(11.5); ax.title.set_fontweight("600")
    fig.tight_layout(); return fig


# --------------------------------------------------------------------------- #
st.sidebar.title("Goal Planner")
st.sidebar.caption("Maximize the *chance of reaching* a money goal — Regime-Aware GBWM with RL.")
st.sidebar.info("New here? Start at **Learn › How it works** and follow the **Next** button at the "
                "bottom of each page — it walks you through the whole story.")
for k, v in {"initial": 100_000, "target": 250_000, "horizon": 20, "contribution": 500}.items():
    st.session_state.setdefault(k, v)


def _apply_preset():
    p = PRESETS.get(st.session_state.preset)
    if p:
        (st.session_state.initial, st.session_state.target,
         st.session_state.horizon, st.session_state.contribution) = p


st.sidebar.selectbox("What are you saving for?", list(PRESETS), key="preset", on_change=_apply_preset)
st.sidebar.number_input("Money you have now ($)", 0, 50_000_000, key="initial", step=5_000)
st.sidebar.number_input("Goal amount ($)", 1_000, 100_000_000, key="target", step=10_000)
st.sidebar.slider("Years until you need it", 1, 40, key="horizon")
st.sidebar.number_input("Monthly savings ($)", 0, 100_000, key="contribution", step=100)
st.sidebar.caption("These numbers drive **Your plan**, **One journey** and **Compare plans**. "
                   "The Live bank's wallets (NT$) are set up on their own page.")
with st.sidebar.expander("Advanced assumptions"):
    rf = st.slider("Safe-cash yearly return", 0.0, 0.08, 0.03, step=0.005)
    persistence = st.slider("How 'sticky' market moods are", 0.5, 0.99, 0.85, step=0.01)
    n_episodes = st.select_slider("How many futures to simulate", [500, 1000, 2000, 4000], value=1000)
st.sidebar.divider()
st.sidebar.caption("Educational simulation — **not financial advice.** Based on the "
                   "[2-minute pitch deck](https://orosergio.github.io/RegimeAwareGBWM/).")

cfg = make_config(st.session_state.initial, st.session_state.target, st.session_state.horizon,
                  st.session_state.contribution, rf, persistence, n_episodes)
key = cfg_key(cfg)
target = float(st.session_state.target)

# The learned policy surfaces, cached as plain arrays: the "How it works" live
# demo reads them on every slider move, so lookups must be instant.
@st.cache_data(show_spinner="Waking up the learned brain…")
def policy_surfaces(key):
    pols = build_policies(key)
    ra = pols["Regime-Aware G-Learner"]
    cfg_ = make_config(*key)
    return ({r: ra.surface(r) for r in cfg_.market.regime_names},
            pols["G-Learner"].surface(None), ra.w_grid)


# Stateful page navigation — a guided story in three phases. One page renders at
# a time (st.tabs rendered all ten bodies eagerly), so each screen keeps a single
# focus, "Next" buttons can really navigate, and heavy pages only compute when
# visited. Page blocks below are `if page == …:` — order in the file is layout-free.
PHASES = [
    ("Learn", [("how", "How it works"), ("brain", "Inside the AI")]),
    ("Play", [("plan", "Your plan"), ("bank", "Live bank (NT$)"), ("journey", "One journey")]),
    ("Proof", [("compare", "Compare plans"), ("history", "Time machine"),
               ("multi", "Multi-asset"), ("coverage", "Proposal coverage")]),
]
PAGE_LABEL = {k: lbl for _, pages in PHASES for k, lbl in pages}
PAGE_PHASE = {k: ph for ph, pages in PHASES for k, _ in pages}
PHASE_PAGES = dict(PHASES)
PHASE_BLURB = {
    "Learn": "Start here — the idea, a live decision you control, and a look inside the brain.",
    "Play": "Do it yourself, with fake money.",
    "Proof": "The honest evidence — judged by calm (small crashes), not by luck.",
}

st.session_state.setdefault("page", "how")
if "_goto" in st.session_state:  # a "Next" / "Open" button asked to navigate
    _tgt = st.session_state.pop("_goto")
    if _tgt in PAGE_LABEL:
        st.session_state["page"] = _tgt
        st.session_state["nav_phase"] = PAGE_PHASE[_tgt]
        st.session_state[f"nav_page_{PAGE_PHASE[_tgt]}"] = PAGE_LABEL[_tgt]

st.session_state.setdefault("nav_phase", PAGE_PHASE[st.session_state.page])
phase = st.radio("Section", [ph for ph, _ in PHASES], key="nav_phase",
                 horizontal=True, label_visibility="collapsed")
_pp = PHASE_PAGES[phase]
_key_by_label = {lbl: k for k, lbl in _pp}
_default_lbl = (PAGE_LABEL[st.session_state.page]
                if PAGE_PHASE[st.session_state.page] == phase else _pp[0][1])
st.session_state.setdefault(f"nav_page_{phase}", _default_lbl)
_sel = st.radio("Page", [lbl for _, lbl in _pp], key=f"nav_page_{phase}",
                horizontal=True, label_visibility="collapsed")
page = _key_by_label[_sel]
st.session_state["page"] = page
st.caption(PHASE_BLURB[phase])

# --------------------------------------------------------------------------- 0
# 📖 How it works — the guide / reading order (ELI10).
# --------------------------------------------------------------------------- #
if page == "how":
    eyebrow("Start here · what this project is · plain language")
    st.markdown('<h1>Watch a computer learn to reach a <span class="gbwm-serif">money goal</span>.</h1>',
                unsafe_allow_html=True)
    intro("Read me first",
          "This app demonstrates one idea: **goal-based wealth management with reinforcement learning "
          "(RL)**. A computer plays the 'reach-your-goal' game thousands of times and learns *how much "
          "to risk each month* from three things: **time left**, the **gap to your goal**, and the "
          "**market 'weather'** (sunny, calm, choppy or stormy).",
          ["Everything here is a **simulation with fake money** — never real money, not financial advice.",
           "Wherever the learned brain makes a decision you'll see the <span class='gbwm-rl'>RL</span> badge.",
           "It's the working proof of <a href='https://orosergio.github.io/RegimeAwareGBWM/' "
           "target='_blank'>the 2-minute pitch deck</a>."])

    # ---- the centerpiece: poke the actual learned policy, live --------------- #
    st.subheader("Try it — make the brain decide, right now")
    st.caption("This queries the **actual learned policy** (not a script). Set a situation; see how much "
               "it would risk. Change **only the weather** and watch the answer move — that one number "
               "moving is the regime-aware part, the project's whole point.")
    surf_by_reg, surf_blind, w_grid = policy_surfaces(key)
    _wopts = list(surf_by_reg)
    demo_regime = st.radio("Market weather right now", _wopts, index=len(_wopts) - 1,
                           horizontal=True, key="demo_weather",
                           format_func=lambda r: f"{WEATHER.get(r, (r, '·'))[1]} {WEATHER.get(r, (r,))[0]}")
    dc = st.columns(2)
    demo_pos = dc[0].slider("Where you stand (% of the goal you already have)", 25, 175, 70,
                            step=5, key="demo_pos")
    _hmax = max(2, int(st.session_state.horizon))
    demo_years = dc[1].slider("Years left", 1, _hmax, min(10, _hmax), key="demo_years")
    _T = cfg.total_steps
    _ti = int(np.clip(_T - demo_years * cfg.steps_per_year, 0, _T - 1))
    _wi = int(np.abs(w_grid - demo_pos / 100 * target).argmin())
    eq_by_reg = {r: float(surf_by_reg[r][_ti, _wi]) for r in _wopts}
    d_eq = eq_by_reg[demo_regime]
    d_blind = float(surf_blind[_ti, _wi])
    d_lo, d_hi = min(eq_by_reg.values()), max(eq_by_reg.values())
    out = st.columns([1, 1.45])
    with out[0]:
        st.markdown(f'<div class="gbwm-sub">THE BRAIN\'S DECISION</div>'
                    f'<div class="gbwm-odo">{d_eq * 100:.0f}%<span class="cur"> in stocks</span></div>'
                    f'<div class="gbwm-bar"><div class="seg" style="width:{d_eq * 100:.0f}%;background:{RL}"></div></div>',
                    unsafe_allow_html=True)
        chart_keys((RL, "stocks (risky)"), ("#d6d0c2", "cash (safe)"))
        st.caption(f"A **{risk_word(d_eq)}** mix. <span class='gbwm-rl'>RL</span> decided — nobody "
                   "hand-coded the number.", unsafe_allow_html=True)
    with out[1]:
        _wname = WEATHER.get(demo_regime, (demo_regime,))[0]
        _behind = demo_pos < 100
        _bits = [
            (f"the weather is **{_wname}**, so it shields what you have first"
             if demo_regime in ("bear", "high_vol") else
             f"the weather is **{_wname}**, so taking risk is cheaper"),
            ("you're **behind the goal**, so it has to climb" if _behind
             else "you're **ahead of the goal**, so there's little to win by gambling"),
            (f"**{demo_years} yr left** gives time to recover from a dip" if demo_years >= 8
             else f"**{demo_years} yr left** leaves little room to recover, which argues for care"),
        ]
        st.markdown(f"**Why {d_eq * 100:.0f}%?** Because " + "; ".join(_bits) + ".")
        st.markdown(
            f"- A **weather-blind** goal-based plan (same money, same time) holds **{d_blind * 100:.0f}%**.\n"
            f"- Across the four weathers the brain's answer ranges **{d_lo * 100:.0f}–{d_hi * 100:.0f}%** — "
            f"the weather **alone** moves it by **{(d_hi - d_lo) * 100:.0f} points**.")
        if st.button("See the full strategy it learned", key="how_to_brain"):
            goto("brain")

    st.subheader("The big idea, like you're 10")
    st.markdown(
        "Imagine **climbing a mountain**:\n"
        "- 🏔️ The **summit** = your money **goal** (e.g. turn 30,000 into 100,000).\n"
        "- 🧗 **How high you are** = how much money you have right now.\n"
        "- 🌦️ The **weather** = the market's mood: ☀️ sunny (going up), 🌤️ calm, 🌧️ choppy, ⛈️ stormy (crashing).\n"
        "- 🎒 **How risky a path you take** = how much you put in **stocks** (risky, climbs fast) vs. "
        "**cash/bonds** (safe, slow).\n\n"
        "The smart rule: **if the weather is good and you're behind, climb harder (more stocks). "
        "If a storm hits and you're near the top, slow down and protect what you have (more cash/bonds).** "
        "Nobody writes that rule by hand — the RL brain *learns* it by playing. You just felt it in the "
        "demo above.")
    with st.expander("Hear one journey, told as a story"):
        pick5 = st.selectbox("Narrate one journey", ["Regime-Aware G-Learner", "G-Learner"],
                             format_func=lambda n: FRIENDLY_NAME[n], key="eli5")
        hist5, _ = single_path(key, pick5, 7)
        ep = EpisodeContext.from_histories(hist5, target, cfg.steps_per_year)
        st.success(ADVISOR.explain_episode(ep))

    st.subheader("Where the reinforcement learning actually happens")
    rlc = st.columns(2)
    card(rlc[0], "", "The decision the RL brain makes",
         "Every month it chooses <b>how much to risk</b> (stocks vs. safe) using three things: "
         "<b>time left</b>, the <b>gap to your goal</b>, and the <b>market weather</b>. "
         "This is <i>goal-based</i> RL — it optimizes the <b>chance of hitting your goal</b>, not raw profit.", RL)
    card(rlc[1], "", "The honesty rule (no peeking)",
         "Even on real history, the brain only ever sees the <b>past</b>. At each month it decides "
         "<b>blind to the future</b> — so when it cuts stocks <i>after</i> a crash begins (not before), "
         "that reaction is real, not hindsight.", GOAL)

    st.subheader("Your route through the app")
    st.caption("Each page is one step of the story; every page ends with a **Next** button. "
               "Or jump anywhere from here.")
    ROUTE_DESC = {
        "how": "This page — the idea and the live demo.",
        "brain": "The learned strategy drawn as four weather maps, and the learning curve.",
        "plan": "Your numbers in → the plan with the best chance out.",
        "bank": "A simulated bank the brain drives month by month, fully transparent.",
        "journey": "Follow one possible future, step by step, with explanations.",
        "compare": "Every plan raced on the same thousands of futures.",
        "history": "The honest test on real history (1999 → today), no peeking.",
        "multi": "Real history with four assets at once — stocks, bonds, gold.",
        "coverage": "How the proposal deck became this working code.",
    }
    rcols = st.columns(3)
    for rcol, (ph, ph_pages) in zip(rcols, PHASES):
        with rcol:
            st.markdown(f"**{ph}**")
            for k, lbl in ph_pages:
                if st.button(lbl, key=f"route_{k}", width="stretch", disabled=(k == "how")):
                    goto(k)
                st.caption(ROUTE_DESC[k])

    st.subheader("The honest takeaway")
    st.info("RL is great at **deciding risk** (when to be brave, when to hide) — and you'll *see* it "
            "reach goals with far smaller crashes than 'all-in stocks'. But it is **not** a crystal ball "
            "that predicts prices. The right way to judge it isn't *'did it make the most money?'* but "
            "*'did it reach the goal with less fear, without cheating (no peeking at the future)?'*")
    next_step("plan", "Type your own money, goal and years — the app picks the plan with the best chance.",
              key="next_how")

# --------------------------------------------------------------------------- 1
if page == "plan":
    eyebrow("Goal-based wealth management · reinforcement learning")
    st.markdown('<h1>Reach your goal — <span class="gbwm-serif">safely</span>.</h1>', unsafe_allow_html=True)
    intro("What you'll see here",
          "Enter your money, your goal and your time **in the left sidebar** (or pick a preset), and the "
          "app races every plan on the same simulated futures to find which gives you the **best chance** "
          "of reaching the goal. <span class='gbwm-rl'>RL</span> plans are the goal-based ones.",
          ["The green block below is the verdict: the winning plan and your success %.",
           "It also shows how many **points of success** the winner adds vs the classic baselines."])
    c = st.columns(4)
    c[0].metric("You have now", money(st.session_state.initial))
    c[1].metric("Goal", money(target))
    c[2].metric("Time", f"{st.session_state.horizon} years")
    c[3].metric("Saving", f"{money(st.session_state.contribution)}/mo")
    if target <= float(st.session_state.initial):
        st.warning("Your goal is **at or below** the money you already have, so every plan trivially "
                   "'succeeds'. Raise the goal (or lower the starting money) in the sidebar to make the "
                   "comparison meaningful.")
    res = run_comparison(key)
    best = max(res, key=lambda n: res[n]["p_goal"]); bp = res[best]
    gp = res.get("Glide Path", bp); bh = res.get("Buy & Hold", bp)
    st.subheader("Our recommendation")
    _badge = (" <span class='gbwm-rl'>RL</span>"
              if best in ("Regime-Aware G-Learner", "G-Learner", "Q-Learner") else "")
    _deltas = (f"{(bp['p_goal'] - gp['p_goal']) * 100:+.0f} pts vs glide path · "
               f"{(bp['p_goal'] - bh['p_goal']) * 100:+.0f} pts vs all-in stocks · "
               f"worst dip {pct(bp['drawdown'])} vs {pct(bh['drawdown'])} all-in")
    st.markdown(
        f'<div class="gbwm-verdict">'
        f'<div><div class="cap">Best chance of success</div>'
        f'<div class="big">{pct(bp["p_goal"])}</div>'
        f'<div style="font-size:.85rem;color:#534b41;margin-top:6px">of simulated futures reach {money(target)}</div></div>'
        f'<div class="body"><strong style="font-size:1.08rem">{FRIENDLY_NAME.get(best, best)}</strong>{_badge}<br>'
        f'<span style="color:#423c33">{STRATEGY_BLURB.get(best, "")}</span><br>'
        f'<span class="deltas">{_deltas}</span><br>'
        f'<span style="font-size:.85rem;color:#534b41">Start near <strong>{pct(bp["start_stock"])} stocks</strong>; '
        f'the plan re-decides every month as your balance, time and the weather change.</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )
    st.caption(f"If you fall short, the typical gap is about {money(bp['cvar'])}. Everything recomputes "
               "from your sidebar numbers — try a preset and watch the verdict change.")
    next_step("bank", "Watch this brain run a live wallet, month by month — every decision and trade "
              "explained.", key="next_plan")

# --------------------------------------------------------------------------- 1b
# 🏦 Live Bank — a simulated bank you play with (no real money).
# --------------------------------------------------------------------------- #
SANDBOX_MARKETS_LIST = list_sandbox_markets()
SANDBOX_MARKET_BY_LABEL = {m.label: m.key for m in SANDBOX_MARKETS_LIST}
DATA_MODES = {"📈 Real history (public data)": "real",
              "🎲 Simulated future (model's Monte-Carlo)": "synthetic"}
COSTS = {"None": 0, "Low (10 bps/trade)": 10, "Realistic (30 bps/trade)": 30, "High (60 bps/trade)": 60}
_SPEEDS = {"Slow": 1.3, "Normal": 0.7, "Fast": 0.35, "Turbo (3 mo/tick)": 0.3}
_STEPS_PER_TICK = {"Turbo (3 mo/tick)": 3}


def _build_wallet(params: dict):
    """Construct a LiveSession from stored params (used on create + reset)."""
    return build_session(**params)


def _bank_run_every():
    """Auto-play cadence for the *currently active* wallet (or None if paused).

    Read from session_state so the bank-wide summary and the ranking can share the
    same timer as the live panel and stay 100% live during auto-play."""
    if not _HAS_RUN_EVERY:
        return None
    order = st.session_state.get("bank_order", [])
    a = st.session_state.get("bank_active")
    if a not in order:
        return None
    s = st.session_state.bank.get(a)
    if s is None or s.done or not st.session_state.get(f"play_{a}", False):
        return None
    return _SPEEDS.get(st.session_state.get(f"speedlbl_{a}", "Normal"), 0.7)


def _render_fragment(fn, *args, run_every=None):
    """Run ``fn`` as a fragment, using run_every when the Streamlit build supports it."""
    if _HAS_RUN_EVERY:
        st.fragment(run_every=run_every)(fn)(*args)
    else:  # very old Streamlit: plain fragment, no auto-refresh
        st.fragment(fn)(*args)


def _bank_summary_panel():
    """Bank-wide totals across every wallet (rendered live during auto-play)."""
    bank = st.session_state.bank
    order = st.session_state.bank_order
    if not order:
        return
    total = sum(w.wealth for w in bank.values())
    dep = sum(w.total_deposited for w in bank.values())
    wd = sum(w.total_withdrawn for w in bank.values())
    pnl = total + wd - dep
    sc = st.columns(4)
    card(sc[0], "", "Total bank balance", ntd(total), RL)
    card(sc[1], "", "Capital you added", ntd(dep), INK)
    card(sc[2], "", "Market profit / loss",
         f"{ntd(pnl)}  ({pnl / dep * 100:+.1f}%)" if dep else ntd(pnl),
         GOAL if pnl >= 0 else LOSS)
    card(sc[3], "", "Open wallets", str(len(order)), RISK)


def _ledger_csv(s) -> bytes:
    """One wallet's transaction ledger as CSV bytes (Excel-friendly UTF-8 BOM)."""
    df = pd.DataFrame([{"Step": tx.step, "When": tx.label, "Type": tx.kind,
                        "Amount": tx.amount, "Balance_after": tx.balance_after,
                        "Currency": s.currency} for tx in s.transactions])
    return df.to_csv(index=False).encode("utf-8-sig")


def _bank_csv() -> bytes:
    """Every wallet's ledger stacked into one CSV (with a Wallet column)."""
    bank = st.session_state.bank
    frames = [pd.DataFrame([{"Wallet": w.name, "Step": tx.step, "When": tx.label,
                             "Type": tx.kind, "Amount": tx.amount,
                             "Balance_after": tx.balance_after, "Currency": w.currency}
                            for tx in w.transactions])
              for w in (bank[wid] for wid in st.session_state.bank_order)]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out.to_csv(index=False).encode("utf-8-sig")


def _bank_state_json() -> str:
    """Serialize the whole bank (params + action log per wallet) for save/load."""
    state = {wid: {"name": st.session_state.bank[wid].name,
                   "params": st.session_state.bank_params[wid],
                   "log": st.session_state.bank[wid].action_log}
             for wid in st.session_state.bank_order}
    return json.dumps(state, indent=2)


def _ranking_export_panel():
    """Leaderboard across wallets + a one-click 'export the whole bank' button
    (rendered live during auto-play)."""
    bank = st.session_state.bank
    order = st.session_state.bank_order
    if not order:
        return
    if len(order) >= 2:
        st.subheader("Wallet leaderboard")
        metric = st.selectbox(
            "Sort by", ["Worst drop (%)", "Sharpe (risk-adjusted)", "Market profit (%)",
                        "Current balance (NT$)", "Progress to goal (%)"], key="rank_metric",
            help="This app judges plans by calm (a small worst-drop) and risk-adjusted return — "
                 "not by who made the most money — so the ranking defaults to the smallest crash.")
        rows = []
        for wid in order:
            w = bank[wid]
            mtr = w.metrics()
            parts = w.mode_label.split(" · ")
            rows.append({
                "Wallet": w.name,
                "Current balance (NT$)": w.wealth,
                "Market profit (%)": w.net_pnl_pct * 100,
                "Progress to goal (%)": (w.wealth / w.target * 100) if w.target else 0.0,
                "Sharpe (risk-adjusted)": mtr["sharpe"],
                "Volatility (%)": mtr["ann_vol"] * 100,
                "Worst drop (%)": mtr["max_drawdown"] * 100,
                "Month": w.t,
                "Market": parts[1] if len(parts) > 1 else w.mode_label,
            })
        ascending = metric == "Worst drop (%)"  # smaller crash is better; every other metric: bigger is better
        df = pd.DataFrame(rows).sort_values(metric, ascending=ascending).reset_index(drop=True)
        medals = ["🥇", "🥈", "🥉"]
        df.insert(0, "#", [medals[i] if i < 3 else str(i + 1) for i in range(len(df))])
        df["Current balance (NT$)"] = df["Current balance (NT$)"].map(lambda x: f"NT$ {x:,.0f}")
        df["Market profit (%)"] = df["Market profit (%)"].map(lambda x: f"{x:+.1f}%")
        df["Progress to goal (%)"] = df["Progress to goal (%)"].map(lambda x: f"{x:.0f}%")
        df["Sharpe (risk-adjusted)"] = df["Sharpe (risk-adjusted)"].map(lambda x: f"{x:.2f}")
        df["Volatility (%)"] = df["Volatility (%)"].map(lambda x: f"{x:.0f}%")
        df["Worst drop (%)"] = df["Worst drop (%)"].map(lambda x: f"{x:.0f}%")
        # Phone-friendly: pin the priority columns first (rank · wallet · headline · worst-drop),
        # push the three secondary columns (volatility · month · market) to the end, and give every
        # column an explicit narrow width + a help tooltip so ~10 columns don't overflow on a phone.
        # All columns stay present: the user-chosen `metric` may be any of them (it must stay visible),
        # and the CSV export downstream reads from the wallet objects, not this DataFrame.
        st.dataframe(
            df, hide_index=True, width="stretch",
            column_order=["#", "Wallet", "Current balance (NT$)", "Worst drop (%)",
                          "Sharpe (risk-adjusted)", "Market profit (%)", "Progress to goal (%)",
                          "Volatility (%)", "Month", "Market"],
            column_config={
                "#": st.column_config.TextColumn("Rank", width="small", help="Rank by the metric chosen above."),
                "Wallet": st.column_config.TextColumn("Wallet", width="medium"),
                "Current balance (NT$)": st.column_config.TextColumn("Balance", width="small", help="Current wallet balance (NT$)."),
                "Worst drop (%)": st.column_config.TextColumn("Worst drop", width="small", help="Scariest peak-to-valley fall along the way (lower is better)."),
                "Sharpe (risk-adjusted)": st.column_config.TextColumn("Sharpe", width="small", help="Profit per unit of stress (higher is better)."),
                "Market profit (%)": st.column_config.TextColumn("Profit", width="small", help="Net market profit vs. the capital you added."),
                "Progress to goal (%)": st.column_config.TextColumn("To goal", width="small", help="How far the balance is toward the goal."),
                "Volatility (%)": st.column_config.TextColumn("Vol", width="small", help="Bumpiness of the ride (annualized)."),
                "Month": st.column_config.NumberColumn("Month", width="small", help="Months played so far."),
                "Market": st.column_config.TextColumn("Market", width="small"),
            },
        )
        st.caption("**Sharpe** = profit per unit of stress (higher is better). **Worst drop** = the "
                   "scariest peak-to-valley fall along the way (lower is better).")
    st.download_button("Export the whole bank (CSV)", data=_bank_csv(),
                       file_name="bank_statements.csv", mime="text/csv", key="dl_bank")


def _decision_log_expander(s):
    """The transparency centerpiece: every month's decision + trades + 'why'."""
    with st.expander("How it's investing — month-by-month decision & trade log", expanded=True):
        st.caption("**The prices are real history, but the model only ever sees the past.** Each month "
                   "it decides *blind to the future*, then we apply what actually happened. That's why "
                   "it reacts *after* a crash starts — never before. Below: what it saw, what it moved, "
                   "and **why**.")
        if not s.decisions:
            st.info("Press **Advance 1 month** to make the model take its first decision.")
            return
        rows = []
        for d in reversed(s.decisions):  # newest first
            w_name, w_emo = WEATHER.get(d.regime, (d.regime, ""))[:2]
            rows.append({
                "When": d.label.replace("Month ", "M"),
                "Weather": f"{w_emo} {w_name}",
                "In risky assets": f"{d.equity * 100:.0f}%",
                "Money moved this month": _trades_text(d, s.asset_labels),
                "Month P&L": f"{'+' if d.month_pnl >= 0 else '−'}{s.currency} {abs(d.month_pnl):,.0f}",
                "Why the model did it": d.rationale,
                "Event": _flag_text(d.flags),
            })
        # Phone-friendly widths: keep the two free-text columns (the trades + the 'why') wide and
        # readable, and constrain the short status columns so the 7-column log doesn't overflow.
        st.dataframe(
            pd.DataFrame(rows), hide_index=True, width="stretch",
            height=min(420, 60 + 36 * len(rows)),
            column_config={
                "When": st.column_config.TextColumn("When", width="small"),
                "Weather": st.column_config.TextColumn("Weather", width="small", help="Market mood the model detected that month."),
                "In risky assets": st.column_config.TextColumn("In risky", width="small", help="Share of your money in risky assets that month."),
                "Money moved this month": st.column_config.TextColumn("Money moved", width="medium", help="The biggest buys/sells the model made this month."),
                "Month P&L": st.column_config.TextColumn("Month P&L", width="small"),
                "Why the model did it": st.column_config.TextColumn("Why", width="large"),
                "Event": st.column_config.TextColumn("Event", width="small"),
            },
        )


def _live_panel(wid: str):
    """The interactive panel for one wallet — rendered inside a fragment so the
    auto-play timer only re-renders this block, not the whole app."""
    bank = st.session_state.bank
    s = bank.get(wid)
    if s is None:
        return
    cur = s.currency
    _flash = st.session_state.pop(f"flash_{wid}", None)  # one-shot message that survived a rerun
    if _flash:
        st.toast(_flash)
    playing = st.session_state.get(f"play_{wid}", False)
    speed_label = st.session_state.get(f"speedlbl_{wid}", "Normal")
    # Auto-advance per fragment tick while "playing" (Turbo jumps several months).
    n_before = len(s.decisions)
    stepped = False
    if playing and not s.done:
        s.advance(_STEPS_PER_TICK.get(speed_label, 1))
        stepped = True
        if s.done:
            st.rerun()  # full rerun → run_every becomes None → auto-play stops cleanly

    # provenance + honesty note
    if s.source == "synthetic" and "Simulated" not in s.mode_label:
        st.warning(f"Offline: using **synthetic stand-in data**. {s.mode_label}")
    else:
        st.caption(f"{s.mode_label} · data source: **{s.source}** · "
                   f"the model only ever sees the past (no look-ahead).")

    # headline metrics
    m = st.columns([1.5, 1, 1, 1.1])
    with m[0]:
        st.markdown(f'<div class="gbwm-sub">CURRENT BALANCE · {s.when}</div>'
                    f'<div class="gbwm-odo"><span class="cur">{cur}</span>{s.wealth:,.0f}</div>',
                    unsafe_allow_html=True)
        st.progress(s.progress, text=f"{s.progress * 100:.0f}% of your goal ({cur} {s.target:,.0f})")
    m[1].metric("Market profit", f"{cur} {s.net_pnl:,.0f}",
                delta=f"{(s.net_pnl - s.bench_pnl):+,.0f} vs buy & hold")
    m[2].metric("You added", f"{cur} {s.total_deposited:,.0f}",
                delta=(f"-{cur} {s.total_withdrawn:,.0f} withdrawn" if s.total_withdrawn else None),
                delta_color="off")
    with m[3]:
        st.markdown('<div class="gbwm-sub">WEATHER THE MODEL SEES</div>', unsafe_allow_html=True)
        st.markdown(weather_pill(s.current_regime()), unsafe_allow_html=True)
        st.caption(f"bad-weather odds: {s.current_risk_belief() * 100:.0f}%")

    # the RL decision, in plain words (this is the transparency the user asked for)
    eq = s.equity_now()
    wname = WEATHER.get(s.current_regime(), (s.current_regime(),))[0]
    msg = (f"<span class='gbwm-rl'>RL decision</span> &nbsp; The weather looks **{wname}**, so it "
           f"invests **{eq * 100:.0f}%** of your money in risky assets (a *{risk_word(eq)}* mix) and "
           f"keeps the rest in cash.")
    if s.A > 1:
        msg += (f" Of that, ~{s.stock_fraction() * 100:.0f}% is in stocks and "
                f"~{s.safe_haven_fraction() * 100:.0f}% in bonds + gold (the safe harbor).")
    st.markdown(f'<div class="gbwm-intro"><div class="d">{_emph(msg)}</div></div>',
                unsafe_allow_html=True)
    if s.decisions:
        last = s.decisions[-1]
        st.caption(f"Last month ({last.label.replace('Month ', 'M')}): **{_trades_text(last, s.asset_labels)}**. "
                   f"_{last.rationale}_")
        if stepped:  # ONE coalesced toast per tick (Turbo plays 3 months/tick — avoid a toast strobe)
            new_flags = []
            for d in s.decisions[n_before:]:
                for fl in d.flags:
                    if fl not in new_flags:
                        new_flags.append(fl)
            if new_flags:
                priority = ["goal", "storm", "big_drop", "clearing"]
                new_flags.sort(key=lambda f: priority.index(f) if f in priority else 99)
                extra = len(new_flags) - 1
                st.toast(_FLAG_TOAST.get(new_flags[0], new_flags[0])
                         + (f"  (+{extra} more this step)" if extra > 0 else ""))

    # While auto-playing, manual controls are disabled (only the timer advances time).
    if playing:
        st.caption("Pause **Auto-play** (above) to use the controls.")

    st.markdown("**Time controls**")
    tc = st.columns(4)
    # st.rerun() after a manual move so the bank summary & leaderboard fragments refresh too.
    if tc[0].button("Advance 1 month", key=f"adv1_{wid}", disabled=s.done or playing,
                    width="stretch"):
        s.advance(1); st.rerun()
    if tc[1].button("Advance 1 year", key=f"adv12_{wid}", disabled=s.done or playing,
                    width="stretch"):
        s.advance(12); st.rerun()
    if tc[2].button("Jump to the end", key=f"advall_{wid}", disabled=s.done or playing,
                    width="stretch"):
        s.advance(s.T); st.rerun()
    if tc[3].button("Restart wallet", key=f"reset_{wid}", disabled=playing, width="stretch"):
        st.session_state[f"confirm_reset_{wid}"] = True
    if st.session_state.get(f"confirm_reset_{wid}"):
        st.warning(f"Restart **{s.name}**? This erases its {s.t} played months and every deposit "
                   "(the wallet starts over from its opening settings).")
        rc = st.columns([1, 1, 2])
        if rc[0].button("Yes, restart it", key=f"reset_yes_{wid}", type="primary"):
            bank[wid] = _build_wallet(st.session_state.bank_params[wid])
            st.session_state.pop(f"confirm_reset_{wid}", None)
            st.rerun()
        if rc[1].button("Keep playing", key=f"reset_no_{wid}"):
            st.session_state.pop(f"confirm_reset_{wid}", None)
            st.rerun()
    if s.done:
        st.success("You reached the end of the available history. Press **Restart wallet**, open "
                   "another wallet, or create one with **Simulated future** to keep playing forever.")

    st.markdown("**Bank controls** — add or take out capital whenever you like")
    bc = st.columns([1.4, 1, 1])
    amt = bc[0].number_input(f"Amount ({cur})", min_value=0, max_value=100_000_000,
                             value=5_000, step=1_000, key=f"amt_{wid}", disabled=playing)
    if bc[1].button("Deposit", key=f"dep_{wid}", type="primary", disabled=playing,
                    width="stretch"):
        s.deposit(float(amt))
        st.session_state[f"flash_{wid}"] = f"Deposited {cur} {float(amt):,.0f}."
        st.rerun()
    if bc[2].button("Withdraw", key=f"wd_{wid}", disabled=playing or s.wealth <= 0,
                    width="stretch"):
        _before = s.wealth
        s.withdraw(float(amt))
        if (_before - s.wealth) + 1e-6 < float(amt):  # withdraw silently clamps to the balance
            st.session_state[f"flash_{wid}"] = (
                f"Only {cur} {_before - s.wealth:,.0f} was available, so that's what we took out.")
        st.rerun()

    # figures (closed after rendering — this panel re-runs on every autoplay tick);
    # each chart's reading key + takeaway sit in ITS OWN column, right under it.
    g = st.columns([1.45, 1])
    with g[0]:
        show_fig(plot_balance(s))
        chart_keys((INK, "the AI plan"), ("#9aa0a6", "just buying the market", "dash"),
                   (LOSS, "your goal", "dash"), (GOAL, "months at/above the goal"))
        st.caption("Takeaway: when the dark line holds up while the dotted one dives, the AI shielded "
                   "you through that crash. Triangles mark your deposits (up) and withdrawals (down).")
    with g[1]:
        show_fig(plot_allocation_bar(s))
        st.caption("Takeaway: where the money sits **right now**. A long stocks bar = being brave this "
                   "month; a long cash or bonds bar = playing it safe.")

    # --- transparency: the decision & trade log (collapsed by default) ------- #
    _decision_log_expander(s)

    with st.expander("Money in each asset, over time", expanded=False):
        fig_m = plot_asset_money(s)
        if fig_m is None:
            st.info("Advance a few months to see how your money gets split.")
        else:
            show_fig(fig_m)
            st.caption("Takeaway: each colored band is the money (NT$) sitting in one asset, stacked into "
                       "your total. When the weather turns bad, the AI drains the stock bands into bonds, "
                       "gold and cash to shield you — then refills them once things calm down.")

    with st.expander("Risk & return scorecard", expanded=False):
        mt = s.metrics()
        k = st.columns(4)
        k[0].metric("Yearly return", f"{mt['ann_return'] * 100:+.1f}%")
        k[1].metric("Volatility (bumpiness)", f"{mt['ann_vol'] * 100:.0f}%")
        k[2].metric("Sharpe (reward vs. stress)", f"{mt['sharpe']:.2f}")
        k[3].metric("Worst drop", f"{mt['max_drawdown'] * 100:.0f}%")
        st.caption("These ignore your deposits/withdrawals — they measure how the *investing* itself did. "
                   "A higher Sharpe and a smaller worst-drop mean a calmer ride to the goal.")

    with st.expander("Statement (every cash movement)", expanded=False):
        rows = [{"When": tx.label, "Type": tx.kind,
                 "Amount": f"{cur} {tx.amount:,.0f}",
                 "Balance after": f"{cur} {tx.balance_after:,.0f}"}
                for tx in reversed(s.transactions)]
        st.dataframe(
            pd.DataFrame(rows), hide_index=True, width="stretch",
            column_config={
                "When": st.column_config.TextColumn("When", width="medium"),
                "Type": st.column_config.TextColumn("Type", width="small"),
                "Amount": st.column_config.TextColumn("Amount", width="small"),
                "Balance after": st.column_config.TextColumn("Balance after", width="small"),
            },
        )
        st.download_button("Download statement (CSV)", data=_ledger_csv(s),
                           file_name=f"statement_{s.name}.csv", mime="text/csv",
                           key=f"dl_{wid}", width="stretch")


if page == "bank":
    eyebrow("Live sandbox · simulated bank · public data · fake money")
    st.markdown('<h1>Your <span class="gbwm-serif">simulated bank</span>: '
                'play with the real market.</h1>', unsafe_allow_html=True)
    intro("What you'll see here",
          "A make-believe bank to experiment with. Open one or more **wallets** (you start with "
          "30,000 NT$), pick a real world market, and the **goal-based reinforcement-learning brain** "
          "<span class='gbwm-rl'>RL</span> decides on its own how much to invest and where, month by "
          "month. You can **deposit or withdraw** capital anytime — but here **nothing is real money**.",
          ["Open a wallet, press **Advance 1 month** (or **Auto-play**) and watch the balance move.",
           "The **How it's investing** log (open by default) shows *exactly* what it bought/sold and **why**.",
           "Public market history, or a **simulated future** generated by the model itself."])
    st.warning("**Educational game.** No real money is used or moved. It's public market data and "
               "fake balances in **NT$** (New Taiwan Dollars). Not financial advice.")

    st.session_state.setdefault("bank", {})
    st.session_state.setdefault("bank_order", [])
    st.session_state.setdefault("bank_params", {})
    st.session_state.setdefault("bank_n", 0)
    bank = st.session_state.bank
    order = st.session_state.bank_order

    # --- bank-wide summary (its own fragment so it stays live during auto-play) #
    if order:
        _render_fragment(_bank_summary_panel, run_every=_bank_run_every())
        if max(w.t for w in bank.values()) >= 12:
            st.caption("Heads-up: the bank lives only in this browser tab — a refresh erases it. "
                       "**Save / load your bank** (below) downloads a file you can restore any time.")
        st.divider()

    # --- create a new wallet (staged: 3 visible decisions; the rest are good
    #     defaults behind a "customize" toggle, so the first play is one click) - #
    with st.expander("Open a new wallet", expanded=not order):
        customize = st.toggle("Customize the defaults (money, goal, years, costs)",
                              value=False, key="wallet_customize")
        with st.form("new_wallet", clear_on_submit=False):
            f1 = st.columns([1.2, 1.2, 1])
            wname = f1[0].text_input("Wallet name", value=f"Wallet {st.session_state.bank_n + 1}")
            mlabel = f1[1].selectbox("Market", list(SANDBOX_MARKET_BY_LABEL),
                                     help="Real history of a public world market, or the multi-asset "
                                          "portfolio (S&P + international + bonds + gold).")
            mode_label = f1[2].radio("Which data?", list(DATA_MODES), horizontal=False)
            # NB: distinct names — `target`/`initial` are module globals used by the planner pages.
            winitial, wtarget, recurring, whorizon = 30_000, 100_000, 0, 15
            start_year, cost_label, offline = 2008, list(COSTS)[0], False
            if customize:
                f2 = st.columns(4)
                winitial = f2[0].number_input("Starting capital (NT$)", 1_000, 100_000_000, 30_000, step=5_000)
                wtarget = f2[1].number_input("Goal (NT$)", 5_000, 1_000_000_000, 100_000, step=10_000)
                recurring = f2[2].number_input("Monthly top-up (NT$)", 0, 10_000_000, 0, step=500,
                                               help="Added automatically each month you advance.")
                whorizon = f2[3].slider("Years to play", 3, 30, 15)
                f3 = st.columns([1, 1, 1])
                start_year = f3[0].slider("Start year (real history only)", 2000, 2019, 2008)
                cost_label = f3[1].selectbox("Trading costs", list(COSTS), index=0,
                                             help="Fees + slippage charged on every rebalance — shows how "
                                                  "turnover eats returns in the real world.")
                offline = f3[2].checkbox("Offline", value=False,
                                         help="No internet? Uses synthetic stand-in data.")
            else:
                st.caption("Defaults: start **NT$ 30,000** toward a **NT$ 100,000** goal in **15 years**, "
                           "from **2008** (you'll live through the crash), no trading costs. "
                           "Flip the toggle above to change them.")
            submitted = st.form_submit_button("Open wallet", type="primary")
        if submitted:
            params = dict(
                name=wname or f"Wallet {st.session_state.bank_n + 1}",
                market_key=SANDBOX_MARKET_BY_LABEL[mlabel], data_mode=DATA_MODES[mode_label],
                start=f"{start_year}-01-01", horizon_years=int(whorizon),
                initial=float(winitial), target=float(wtarget), recurring=float(recurring),
                currency="NT$", fee_bps=float(COSTS[cost_label]), offline=bool(offline),
                seed=st.session_state.bank_n,
            )
            try:
                with st.spinner("Preparing the market and solving the model…"):
                    session = _build_wallet(params)
            except Exception as e:  # noqa: BLE001
                st.error("We couldn't open that wallet — usually the market data couldn't be reached. "
                         "Try **Offline** mode, a different market, or another start year.")
                with st.expander("Technical details"):
                    st.code(str(e))
            else:
                wid = uuid.uuid4().hex[:8]
                bank[wid] = session
                st.session_state.bank_params[wid] = params
                order.append(wid)
                st.session_state.bank_n += 1
                st.success(f"Opened **{session.name}** — press **Advance 1 month** below to make "
                           "its first move.")
                st.rerun()

    # --- save / load the whole bank ----------------------------------------- #
    with st.expander("Save / load your bank", expanded=False):
        sl = st.columns(2)
        if order:
            sl[0].download_button("Save bank (JSON)", data=_bank_state_json(),
                                  file_name="bank_save.json", mime="application/json",
                                  width="stretch")
        else:
            sl[0].caption("Open a wallet first to save.")
        up = sl[1].file_uploader("Load a saved bank (JSON)", type=["json"], key="bank_upload")
        if up is not None and sl[1].button("Load this save", width="stretch"):
            try:
                up.seek(0)  # allow re-reading the same uploaded file on repeat clicks
                data = json.loads(up.read())
                new_bank, new_params, new_order = {}, {}, []
                with st.spinner("Rebuilding your bank…"):
                    for wid, rec in data.items():
                        new_bank[wid] = restore_session(rec["params"], rec.get("log", []))
                        new_params[wid] = rec["params"]
                        new_order.append(wid)
            except Exception as e:  # noqa: BLE001
                st.error("We couldn't read that save file. Make sure it's a bank file you saved here "
                         "(a .json from **Save bank**) and wasn't edited by hand.")
                with st.expander("Technical details"):
                    st.code(str(e))
            else:
                # drop stale per-wallet widget state so loaded wallets start clean (paused, no phantom toggle)
                for _k in list(st.session_state.keys()):
                    if any(_k.startswith(p) for p in
                           ("play_", "speedlbl_", "amt_", "confirm_close_", "confirm_reset_", "flash_")):
                        st.session_state.pop(_k, None)
                st.session_state.pop("bank_active", None)
                st.session_state.pop("rank_metric", None)
                st.session_state.bank = new_bank
                st.session_state.bank_params = new_params
                st.session_state.bank_order = new_order
                st.session_state.bank_n = len(new_order)
                st.success(f"Loaded {len(new_order)} wallet(s).")
                st.rerun()

    if not order:
        st.info("Open your first wallet above to start. Tip: try the **S&P 500** from **2008** to live "
                "through the crash, or the **multi-asset portfolio**.")
    else:
        # --- pick the active wallet + play controls (outside the fragment) -- #
        if st.session_state.get("bank_active") not in order:  # stale after a close
            st.session_state.pop("bank_active", None)
        active = st.selectbox("Active wallet", order,
                              format_func=lambda w: bank[w].name, key="bank_active")
        if active:
            s = bank[active]
            if s.done:  # don't leave the toggle stuck visually ON when there's nothing to play
                st.session_state.pop(f"play_{active}", None)
            pc = st.columns([1, 1, 1])
            playing = pc[0].toggle("Auto-play", key=f"play_{active}", value=False,
                                   disabled=s.done,
                                   help="Advance months automatically. Pause to deposit/withdraw.")
            pc[1].selectbox("Speed", list(_SPEEDS), index=1, key=f"speedlbl_{active}")
            speed = _SPEEDS[st.session_state.get(f"speedlbl_{active}", "Normal")]
            if pc[2].button("Close wallet", key=f"close_{active}", disabled=playing):
                st.session_state[f"confirm_close_{active}"] = True
            if st.session_state.get(f"confirm_close_{active}"):
                st.warning(f"Close **{bank[active].name}**? This erases its months, deposits and "
                           "leaderboard entry, and can't be undone.")
                cca, ccb, _sp = st.columns([1, 1, 2])
                if cca.button("Yes, close it", key=f"close_yes_{active}", type="primary"):
                    bank.pop(active, None)
                    st.session_state.bank_params.pop(active, None)
                    order.remove(active)
                    st.session_state.pop(f"confirm_close_{active}", None)
                    st.rerun()
                if ccb.button("Keep it", key=f"close_no_{active}"):
                    st.session_state.pop(f"confirm_close_{active}", None)
                    st.rerun()
            else:
                run_every = speed if (playing and not s.done) else None
                _render_fragment(_live_panel, active, run_every=run_every)
                # leaderboard + whole-bank export, also live during auto-play
                st.divider()
                _render_fragment(_ranking_export_panel, run_every=_bank_run_every())

    # --- the honest answer: does RL work in real markets? ------------------- #
    with st.expander("Does this really work in real markets? (honest answer)", expanded=False):
        st.markdown(
            "**Short version:** reinforcement learning (RL) is great at *deciding risk* (how much to "
            "risk and when to protect), but it is **not a crystal ball** that predicts prices, nor a "
            "guaranteed money machine.\n\n"
            "**What it does well (and you can see here):**\n"
            "- It reacts to *context* (time, distance to the goal, market weather) far better than a "
            "fixed rule. In this app's honest tests it reaches the goal with **much smaller crashes** "
            "(less fear) than going 'all-in stocks'.\n"
            "- It learns the strategy on its own — nobody hand-codes it.\n\n"
            "**Why the real market is harder:**\n"
            "- **It doesn't repeat:** the market changes its rules (it's non-stationary); what worked "
            "yesterday can fail tomorrow.\n"
            "- **Real costs:** fees, taxes, *slippage* and price gaps eat much of the edge (toggle "
            "**Trading costs** when opening a wallet to feel this).\n"
            "- **Overfitting:** it's easy to memorize the past and then fail on new data.\n"
            "- **Rare events:** never-seen crises, panics and vanishing liquidity.\n\n"
            "**Honest conclusion:** in practice RL pays off more for **risk management and adaptive "
            "allocation** (exactly what this app does) than for *predicting and beating the market*. "
            "The right way to judge it isn't *'did it make more?'* but *'did it reach the goal with "
            "less risk and without cheating (no peeking at the future)?'* — which is what the "
            "**Proof › Time machine** page measures.")
    next_step("journey", "Slow it down: follow one single future month by month, with the brain "
              "explaining each move.", key="next_bank")

# --------------------------------------------------------------------------- 2
if page == "compare":
    eyebrow("Same futures · every plan · fair race")
    st.markdown('<h1>Compare the plans — <span class="gbwm-serif">fairly</span>.</h1>',
                unsafe_allow_html=True)
    intro("What you'll see here",
          "A fair fight: every plan is tested on the **same** thousands of simulated futures. You'll "
          "see which gives the best chance of reaching the goal and how scary the ride would be "
          "(the worst dip). <span class='gbwm-rl'>RL</span> plans are the goal-based / regime-aware ones.",
          ["The longest green bar = the plan with the best chance of success.",
           "Check the **worst dip** column: a high ending with huge crashes is not a good plan."])
    res = run_comparison(key)
    show_fig(goal_chance_chart({n: res[n]["p_goal"] for n in res}))
    chart_keys((GOAL, "the winning plan"), (RISK, "target-date glide path — the deck's baseline"),
               (_NEUTRAL, "the other plans"))
    st.caption("Takeaway: each bar is one plan's chance of reaching **your** goal on the same simulated "
               "futures. A high chance is only half the story — check the worst-dip column below before "
               "you pick.")
    _ra, _gp, _bh = (res.get("Regime-Aware G-Learner"), res.get("Glide Path"), res.get("Buy & Hold"))
    if _ra and _gp and _bh:
        st.success(f"**The deck's headline, measured live on your numbers:** the regime-aware agent "
                   f"reaches the goal in **{pct(_ra['p_goal'])}** of futures — "
                   f"**{(_ra['p_goal'] - _gp['p_goal']) * 100:+.0f} pts vs the glide path** and "
                   f"**{(_ra['p_goal'] - _bh['p_goal']) * 100:+.0f} pts vs buy & hold** — while riding a "
                   f"**{pct(_ra['drawdown'])}** worst dip vs **{pct(_bh['drawdown'])}** for all-in stocks. "
                   f"(The [pitch deck](https://orosergio.github.io/RegimeAwareGBWM/) sketched ~+13 / +21 pts "
                   f"as illustrative targets; here you watch the real number come out of the simulation.)")
    rows = [{"Plan": FRIENDLY_NAME.get(n, n), "Chance of reaching goal": pct(res[n]["p_goal"]),
             "Typical ending balance": money(res[n]["median"]),
             "If short, typical gap": money(res[n]["cvar"]),
             "Worst dip along the way": pct(res[n]["drawdown"])}
            for n in sorted(res, key=lambda x: res[x]["p_goal"], reverse=True)]
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch",
        column_order=["Plan", "Chance of reaching goal", "Worst dip along the way",
                      "Typical ending balance", "If short, typical gap"],
        column_config={
            "Plan": st.column_config.TextColumn("Plan", width="medium"),
            "Chance of reaching goal": st.column_config.TextColumn("Chance", width="small", help="Probability of reaching your goal across the simulated futures (higher is better)."),
            "Worst dip along the way": st.column_config.TextColumn("Worst dip", width="small", help="Average worst peak-to-valley fall along the way (lower is better)."),
            "Typical ending balance": st.column_config.TextColumn("Typical ending", width="small", help="Median final balance."),
            "If short, typical gap": st.column_config.TextColumn("If short, gap", width="small", help="Typical shortfall (CVaR) when the goal is missed."),
        },
    )
    with st.expander("Full evaluation — the proposal's six metrics"):
        full = [{"Plan": FRIENDLY_NAME[n],
                 "Goal attainment": pct(res[n]["p_goal"]),
                 "Shortfall if missed": money(res[n]["cvar"]),
                 "Terminal wealth (10th–90th)": f"{money(res[n]['p10'])} – {money(res[n]['p90'])}",
                 "Max drawdown": pct(res[n]["drawdown"]),
                 "Turnover": f"{res[n]['turnover']:.3f}",
                 "Best in regime": REGIME_LABEL.get(res[n]["best_regime"], res[n]["best_regime"])}
                for n in sorted(res, key=lambda x: res[x]["p_goal"], reverse=True)]
        st.dataframe(pd.DataFrame(full), hide_index=True, width="stretch")
        st.caption("Matches proposal slide 11: goal attainment · shortfall · terminal wealth · drawdown · "
                   "turnover · regime behavior (drawn as the four weather maps in *Inside the AI*).")
    with st.expander("What do these plans actually do?"):
        for n in ["Buy & Hold", "60/40", "Glide Path", "G-Learner", "Regime-Aware G-Learner"]:
            st.markdown(f"**{FRIENDLY_NAME[n]}** — {STRATEGY_BLURB[n]}")
    next_step("history", "Simulations are fair; history is honest. Run every plan over the real past — "
              "dot-com, 2008 and COVID — with no peeking.", key="next_compare")

# --------------------------------------------------------------------------- 3
if page == "journey":
    eyebrow("One simulated future · month by month · the brain explains itself")
    st.markdown('<h1>Walk through <span class="gbwm-serif">one journey</span>.</h1>',
                unsafe_allow_html=True)
    intro("What you'll see here",
          "We follow **one** possible future, month by month, for a plan you choose. You'll see your "
          "balance rise and fall, how much the model puts in stocks at each moment, and what it "
          "thought the market was doing.",
          ["Move *“Try a different future”* to see another possible story.",
           "Use *“Peek at a month”* and the model explains what it did that month "
           "(<span class='gbwm-rl'>RL</span> in action)."])
    names = ["Regime-Aware G-Learner", "G-Learner", "Glide Path", "60/40", "Buy & Hold"]
    pick = st.selectbox("Plan to follow", names, format_func=lambda n: FRIENDLY_NAME[n])
    seed = st.slider("Try a different future", 0, 200, 7)
    hist, terminal = single_path(key, pick, seed)
    reached = terminal >= target
    m1, m2 = st.columns(2)
    m1.metric("Ending balance on this path", money(terminal),
              delta="reached the goal" if reached else "-short of the goal")
    m2.metric("Your goal", money(target))
    wfig = plots.plt.figure(figsize=(8, 3.3)); wax = wfig.add_subplot(111)
    wax.plot(hist["wealth"][0], color=INK, lw=1.9)
    wax.axhline(target, color=LOSS, lw=1.6, label="goal")
    wax.fill_between(range(hist["wealth"].shape[1]), hist["wealth"][0], target,
                     where=hist["wealth"][0] >= target, color=GOAL, alpha=0.15)
    wax.set(title="Your balance over time", xlabel="month", ylabel="balance ($)")
    _finish_fig(wfig, wax)
    show_fig(wfig)
    chart_keys((INK, "your balance"), (LOSS, "your goal", "dash"),
               (GOAL, "stretches at/above the goal"))
    st.caption("Takeaway: one pretend future for the plan you picked. If the line ends above the red "
               "goal line, this journey made it. Drag the future slider to see how differently it can go.")
    show_fig(plots.plot_allocation_over_time(hist["weights"][0], ["stocks"]))
    st.caption("Takeaway: how much was in stocks each month of the journey above — high = chasing "
               "growth, low = protecting you. As a guide: under 33% is conservative, 33–66% balanced, "
               "over 66% aggressive.")
    step = st.slider("Peek at a month", 1, cfg.total_steps - 1, cfg.total_steps // 2)
    sc = StepContext(weights=hist["weights"][0, step], prev_weights=hist["weights"][0, step - 1],
                     belief=hist["belief"][0, step], prev_belief=hist["belief"][0, step - 1],
                     wealth=float(hist["wealth"][0, step]), target=target, step=step,
                     n_steps=cfg.total_steps, steps_per_year=cfg.steps_per_year,
                     regime_names=cfg.market.regime_names, asset_names=hist["asset_names"])
    st.info(ADVISOR.explain_step(sc))
    with st.expander("What did the model think the market was doing?"):
        show_fig(plots.plot_regime_beliefs(hist["belief"][0], [REGIME_LABEL[r] for r in hist["regime_names"]]))
        st.caption("Takeaway: the model's month-by-month guess about the weather — the tallest band is "
                   "the mood it believed most. It grows sure of a downturn only *after* the drop starts, "
                   "never before, because it cannot see the future.")
    next_step("compare", "One journey is luck; thousands are evidence. Race every plan on the same "
              "futures.", key="next_journey")

# --------------------------------------------------------------------------- 4
MARKET_BY_LABEL = {m.label: m.key for m in list_markets()}
LONG_NOTE = {m.key: m.note for m in list_markets()}


@st.cache_data(show_spinner="Time-travelling: rolling every plan over real history (no look-ahead)…")
def honest_deployment(market_key, start, initial, contribution, target, offline):
    return run_deployment(default_config(), market_key, start=start, initial=initial,
                          contribution=contribution, target=target, offline=offline)


@st.cache_data(show_spinner="Testing three start dates (1999 / 2000 / 2007)…")
def seq_risk_cached(market_key, initial, contribution, target, offline):
    return sequence_risk(default_config(), market_key, starts=["1999-01-01", "2000-01-01", "2007-01-01"],
                         horizon_years=15, initial=initial, contribution=contribution,
                         target=target, offline=offline)


# ---- multi-asset (S&P + intl + bonds + gold) live demo --------------------- #
WEATHER_EN = {"bull": "Sunny ☀️", "stable": "Calm 🌤️",
              "high_vol": "Choppy 🌧️", "bear": "Stormy ⛈️"}
HONESTY_MODES = {
    "Re-learn year by year (most honest)": "causal_walk_forward",
    "Learn once at the start": "causal_calibrate",
    "Common-sense regimes (robust)": "prior",
}
FREQ_EN = {"Monthly": "monthly", "Quarterly": "quarterly", "Annual": "annual"}


def ma_friendly(name):
    return STRATEGY_LABELS.get(name, name)


@st.cache_data(show_spinner=False)
def ma_deploy(start_year, initial, contribution, freq, target, mode, offline):
    cfg = default_multi_asset_config()
    return run_multi_asset_deployment(
        cfg, start=f"{start_year}-01-01", initial=float(initial),
        contribution=float(contribution), contribution_freq=freq, target=float(target),
        regime_mode=mode, offline=offline,
    )


@st.fragment
def ma_live_scrubber(ma_key):
    """Time scrubber for the smart plan. Isolated in a fragment so dragging the
    month slider only re-renders the donut + metrics, not the heavy figures."""
    dep = ma_deploy(*ma_key)  # cached -> instant; no re-solve
    smart = "Regime-Aware Multi-Asset"
    T = len(dep.dates)
    step = st.slider("Scrub to a month of the run", 0, T - 1, min(T - 1, T // 2), key="ma_step")
    mcol, icol = st.columns([1, 1])
    mcol.pyplot(plot_allocation_snapshot(dep, smart, step))
    mcol.caption("Takeaway: this snapshot shows how the smart plan split your money in the single month you "
                 "picked with the slider. Drag to a calm month and you'll see mostly stocks; drag into a crisis "
                 "and you'll see it shift heavily into bonds, gold and cash. The numbers beside it spell out "
                 "the exact split and the weather it detected.")
    reg = dep.regime_names[int(np.argmax(dep.belief(smart)[step]))]
    icol.metric("Month", f"{dep.dates[step]:%b %Y}")
    icol.metric("Detected weather", WEATHER_EN.get(reg, reg))
    icol.metric("In stocks (US + international)", pct(dep.stock_equity(smart)[step]),
                delta=f"{pct(dep.safe_haven(smart)[step])} in bonds+gold", delta_color="off")
    icol.metric("Bad-weather belief", pct(dep.risk_belief(smart)[step]))


if page == "history":
    eyebrow("Out-of-sample · real market history · no look-ahead")
    st.markdown('<h1>If you had switched this on in <span class="gbwm-serif">1999</span>…</h1>',
                unsafe_allow_html=True)
    intro("What you'll see here",
          "The honest test: we switch the plans on over the **real** month-by-month history of a world "
          "market and make them live through dot-com, 2008 and COVID — with no cheating (the model only "
          "sees the past each month, discovering each crash in real time). This is where you judge the "
          "<span class='gbwm-rl'>RL</span> brain for real.",
          ["Pick a market and a year, then press **Run the time machine**.",
           "Compare the final balance **and** the worst crash: the smart plan wins on calm, not luck."])
    cc = st.columns([2, 1, 1, 1])
    label = cc[0].selectbox("Market", list(MARKET_BY_LABEL))
    mkey = MARKET_BY_LABEL[label]
    start_year = cc[1].slider("Start year", 1999, 2018, 1999)
    goal_h = cc[2].number_input("Goal for this run ($)", 50_000, 50_000_000, value=600_000, step=50_000)
    offline = cc[3].checkbox("Offline", value=False, help="No internet? Uses a synthetic stand-in.")
    st.caption(f"Using your sidebar plan: start **{money(st.session_state.initial)}**, "
               f"add **{money(st.session_state.contribution)}/mo**. {LONG_NOTE.get(mkey, '')}")
    # The run is keyed to its exact inputs: changing any input un-runs the page
    # (with a nudge) instead of silently recomputing something the button no
    # longer represents.
    _run_args = (mkey, f"{start_year}-01-01", float(st.session_state.initial),
                 float(st.session_state.contribution), float(goal_h), offline)
    if st.button("Run the time machine", type="primary"):
        st.session_state["rd_args"] = _run_args
    if st.session_state.get("rd_args") and st.session_state["rd_args"] != _run_args:
        st.info("You changed the settings — press **Run the time machine** to rerun with them.")
    if st.session_state.get("rd_args") == _run_args:
        dep = None
        try:
            dep = honest_deployment(*_run_args)
        except Exception as e:  # noqa: BLE001
            st.error(f"We couldn't run **{label}** from {start_year} — the price history may be "
                     "unavailable for that market or year. Try **Offline** mode or a later start year.")
            with st.expander("Technical details"):
                st.code(str(e))
        if dep is not None:
            if dep.source == "synthetic":
                st.warning("Using **synthetic stand-in data** (no live connection). Illustrative only.")
            else:
                st.success(f"Loaded **real {dep.market.label}** — {dep.dates[0]:%b %Y} → "
                           f"{dep.dates[-1]:%b %Y} ({dep.horizon_years} years). "
                           "The agent saw only the past at every step.")
            sc = scorecard(dep).set_index("Strategy")

            def _val(name, col):
                return sc.loc[name, col] if name in sc.index else float("nan")

            cols = st.columns(3)
            card(cols[0], "", "All-in stocks — worst crash", pct(_val("Buy & Hold", "Max drawdown")), LOSS)
            card(cols[1], "", "Smart adaptive — worst crash",
                 pct(_val("Regime-Aware G-Learner", "Max drawdown")), GOAL)
            both = bool(_val("Buy & Hold", "Reached goal")) and bool(_val("Regime-Aware G-Learner", "Reached goal"))
            card(cols[2], "", "Both reached the goal?",
                 "yes — at a fraction of the risk" if both else "depends on the goal you set", RL)

            show_fig(plot_journey(dep))
            chart_keys((RL, "smart adaptive plan"), (LOSS, "all-in stocks"),
                       (RISK, "glide path"), ("#f4c7bd", "shaded band = a real crisis"),
                       (INK, "dashed = the goal", "dash"))
            st.caption("Takeaway: all-in stocks can end highest, but look how deep it plunges in each "
                       "crisis band. The smart plan usually still reaches the goal while falling far "
                       "less — a much calmer ride.")
            with st.expander("How to read the three panels"):
                st.markdown(
                    "1. **Balance** — every plan on the same real months; the shaded bands are real "
                    "crises (dot-com, 2008, COVID).\n"
                    "2. **% in stocks** — the smart plan's risk dial. The labeled arrows mark it cutting "
                    "risk *during* each storm: it reacts to what just happened, it never predicts.\n"
                    "3. **Bad-weather belief** — how sure the brain was that the weather had turned, "
                    "computed from past returns only. It rises *after* drops begin — the visible proof "
                    "of no peeking.")

            st.subheader("The scorecard — the outcome **and** the ride")
            show = scorecard(dep)
            show.insert(0, "Plan", show["Strategy"].map(lambda n: FRIENDLY_NAME.get(n, n)))
            show = show.drop(columns="Strategy")
            show["Final balance"] = show["Final balance"].map(money)
            show["Reached goal"] = show["Reached goal"].map(lambda b: "Yes" if b else "No")
            for c in ["Max drawdown", "Worst 12-month", "Growth rate (CAGR)", "Avg equity"]:
                show[c] = show[c].map(pct)
            st.dataframe(
                show, hide_index=True, width="stretch",
                column_order=["Plan", "Reached goal", "Max drawdown", "Final balance",
                              "Worst 12-month", "Growth rate (CAGR)", "Avg equity"],
                column_config={
                    "Plan": st.column_config.TextColumn("Plan", width="medium"),
                    "Reached goal": st.column_config.TextColumn("Reached?", width="small"),
                    "Max drawdown": st.column_config.TextColumn("Worst crash", width="small", help="Worst peak-to-valley fall lived through (lower is better)."),
                    "Final balance": st.column_config.TextColumn("Final balance", width="small"),
                    "Worst 12-month": st.column_config.TextColumn("Worst year", width="small", help="Worst rolling 12-month return."),
                    "Growth rate (CAGR)": st.column_config.TextColumn("Growth", width="small", help="Compound annual growth rate."),
                    "Avg equity": st.column_config.TextColumn("Avg invested", width="small", help="Average share kept in risky assets."),
                },
            )
            st.caption("Sorted safest-first. On one rising market the all-in plan wins on raw final balance — "
                       "but read the **Max drawdown** column: that's the 40–70% crash you'd have lived through. "
                       "The RL agents reach the goal with far less white-knuckle risk.")

            st.subheader("What the agent did at the turning points")
            for line in diary_sentences(dep):
                st.markdown("- " + line)

            with st.expander("Does *when* you start matter? — sequence-of-returns risk"):
                seq = seq_risk_cached(mkey, float(st.session_state.initial),
                                      float(st.session_state.contribution), 400_000.0, offline)
                if seq.empty:
                    st.info("Not enough history for this market to compare 1999 / 2000 / 2007 starts.")
                else:
                    show_fig(plot_sequence_risk(seq))
                    st.caption("Takeaway: 2000 and 2007 starts land right before a crash, which badly "
                               "hurts the all-in plan. The regime-aware plan's worst-drop bar stays low "
                               "even then, because it cut risk once the storm began — *when* you start "
                               "matters far less with the smart plan.")
            st.caption("Educational simulation — costs/taxes simplified, single risky asset, monthly "
                       "rebalancing. **Not financial advice.**")
    elif not st.session_state.get("rd_args"):
        st.info("Pick a market and press **Run the time machine**.")
    next_step("multi", "Same honest test, but the brain juggles four assets at once — stocks, bonds "
              "and gold.", key="next_history")

# --------------------------------------------------------------------------- 4b
if page == "multi":
    eyebrow("Multi-asset live · S&P 500 + international + bonds + gold · no look-ahead")
    st.markdown('<h1>A portfolio that <span class="gbwm-serif">splits itself</span> '
                'across stocks, bonds and gold.</h1>', unsafe_allow_html=True)
    intro("What you'll see here",
          "Like the time machine, but with **four assets at once** (S&P, international stocks, bonds "
          "and gold). The smart <span class='gbwm-rl'>RL</span> plan loads up on stocks when it's "
          "sunny and rotates into bonds + gold when it's stormy — all over real history, discovering "
          "each crisis in real time (never with the future).",
          ["Adjust your numbers and press **Split my money live**.",
           "Watch the smart plan empty out of stocks during crises and fill up on safe havens."])

    uni = list_universe()
    ACCENT = {"us_equity": RL, "intl_equity": "#7048e8", "bonds": GOAL, "gold": RISK}
    ucols = st.columns(len(uni) + 1)
    for col, a in zip(ucols, uni):
        card(col, "", a.label.split(" (")[0], a.note, ACCENT.get(a.key, INK))
    card(ucols[-1], "", "Cash", "The safe part, earning the risk-free rate.", INK)

    st.divider()
    cc = st.columns(4)
    ma_initial = cc[0].number_input("Starting money ($)", 0, 50_000_000, 100_000, step=5_000, key="ma_initial")
    ma_amount = cc[1].number_input("Top-up ($)", 0, 1_000_000, 800, step=100, key="ma_amount")
    ma_freq_label = cc[2].selectbox("How often do you add?", list(FREQ_EN), key="ma_freq")
    ma_start = cc[3].slider("Start year", 2001, 2018, 2005, key="ma_start")
    cc2 = st.columns([1, 2, 1])
    ma_target = cc2[0].number_input("Goal ($)", 50_000, 100_000_000, 600_000, step=50_000, key="ma_target")
    ma_mode_label = cc2[1].selectbox(
        "How honestly should it learn the weather?", list(HONESTY_MODES), index=0, key="ma_mode",
        help="‘Re-learn year by year’ re-learns the market's seasons every year using ONLY the past "
             "seen so far — the strictest test.")
    ma_offline = cc2[2].checkbox("Offline", value=False, key="ma_offline",
                                 help="No internet? Uses synthetic stand-in data.")
    freq = FREQ_EN[ma_freq_label]
    mode = HONESTY_MODES[ma_mode_label]
    per = {"monthly": 1, "quarterly": 3, "annual": 12}[freq]
    st.caption(f"You add **{money(ma_amount)}** ({ma_freq_label.lower()}) → equals "
               f"**{money(ma_amount / per)}/month**.  Universe: "
               f"{' · '.join(a.label.split(' (')[0] for a in uni)} + cash.")
    if mode == "causal_walk_forward":
        st.info("*Re-learn year by year* is the most honest and the slowest: it re-learns the regimes "
                "every year using only the past. The **first** run can take ~1–2 min; then it's cached.")

    ma_key = (ma_start, ma_initial, ma_amount, freq, ma_target, mode, ma_offline)
    if st.button("Split my money live", type="primary", key="ma_run"):
        st.session_state["ma_args"] = ma_key
    if st.session_state.get("ma_args") and st.session_state["ma_args"] != ma_key:
        st.info("You changed the settings — press **Split my money live** to rerun with them.")
    if st.session_state.get("ma_args") == ma_key:
        spinner_msg = ("Re-learning the regimes year by year over real history…"
                       if mode == "causal_walk_forward"
                       else "Splitting across S&P, international, bonds and gold…")
        dep = None
        try:
            with st.spinner(spinner_msg):
                dep = ma_deploy(*ma_key)
        except Exception as e:  # noqa: BLE001
            st.error("We couldn't run the multi-asset split — the data for one of the four assets "
                     "may be unavailable. Try **Offline** mode or a later start year.")
            with st.expander("Technical details"):
                st.code(str(e))
        if dep is not None:
            if dep.source == "synthetic":
                st.warning("Using **synthetic data** (offline). Illustrative only.")
            else:
                tk = " · ".join(f"{k}={v}" for k, v in dep.tickers_used.items())
                st.success(f"**Real** data {dep.dates[0]:%b %Y} → {dep.dates[-1]:%b %Y} "
                           f"({dep.horizon_years} years). Proxies: {tk}. The agent saw only the past at each step.")
            mode_txt = {"causal_walk_forward": "re-learning the regimes year by year (past only)",
                        "causal_calibrate": "with regimes learned once (data before the start)",
                        "prior": "with common-sense regimes"}[dep.regime_mode]
            st.caption(f"Honesty used: **{mode_txt}**.")

            sc = scorecard(dep).set_index("Strategy")

            def _vm(name, col):
                return sc.loc[name, col] if name in sc.index else float("nan")

            kc = st.columns(3)
            card(kc[0], "", "Smart plan — worst crash",
                 pct(_vm("Regime-Aware Multi-Asset", "Max drawdown")), GOAL)
            card(kc[1], "", "All-in S&P — worst crash",
                 pct(_vm("All-in S&P 500", "Max drawdown")), LOSS)
            card(kc[2], "", "Smart plan — final balance",
                 money(_vm("Regime-Aware Multi-Asset", "Final balance")), RL)

            st.subheader("How the 3 portfolios split over time")
            show_fig(plot_allocation_grid(dep))
            chart_keys((RL, "US stocks"), ("#7048e8", "international"), (GOAL, "bonds"),
                       ("#e0a100", "gold"), ("#cdcdc4", "cash"),
                       (None, "dotted vertical line = a crisis begins"))
            st.caption("Takeaway: the top strip is the market weather the smart plan detected. The "
                       "all-in and fixed-mix portfolios never change shape. The smart plan does: at each "
                       "crisis it empties the stock bands and fills up on bonds and gold, then rotates "
                       "back once the market recovers.")

            st.subheader("Your money over real history")
            show_fig(plot_multi_balances(dep))
            chart_keys((RL, "smart plan"), (LOSS, "all-in S&P"), ("#9aa0a6", "balanced mix"),
                       (INK, "dashed = the goal", "dash"), ("#f4c7bd", "shaded band = a real crisis"))
            st.caption("Takeaway: the all-in S&P line climbs highest in calm years but dives deepest in "
                       "crashes. The smart plan's line is smoother and still reaches the goal — a little "
                       "top-end growth traded for a much steadier path.")

            st.subheader("The scorecard — the outcome **and** the scare")
            show = scorecard(dep)
            show.insert(0, "Portfolio", show["Strategy"].map(ma_friendly))
            show = show.drop(columns="Strategy")
            show["Final balance"] = show["Final balance"].map(money)
            show["Reached goal"] = show["Reached goal"].map(lambda b: "Yes" if b else "No")
            for c in ["Max drawdown", "Worst 12-month", "Growth rate (CAGR)", "Avg equity"]:
                show[c] = show[c].map(pct)
            show.columns = ["Portfolio", "Final balance", "Reached goal?", "Worst crash",
                            "Worst 12 months", "Growth (CAGR)", "Avg invested %"]
            st.dataframe(
                show, hide_index=True, width="stretch",
                column_order=["Portfolio", "Reached goal?", "Worst crash", "Final balance",
                              "Worst 12 months", "Growth (CAGR)", "Avg invested %"],
                column_config={
                    "Portfolio": st.column_config.TextColumn("Portfolio", width="medium"),
                    "Reached goal?": st.column_config.TextColumn("Reached?", width="small"),
                    "Worst crash": st.column_config.TextColumn("Worst crash", width="small", help="Worst peak-to-valley fall lived through (lower is better)."),
                    "Final balance": st.column_config.TextColumn("Final balance", width="small"),
                    "Worst 12 months": st.column_config.TextColumn("Worst year", width="small", help="Worst rolling 12-month return."),
                    "Growth (CAGR)": st.column_config.TextColumn("Growth", width="small", help="Compound annual growth rate."),
                    "Avg invested %": st.column_config.TextColumn("Avg invested", width="small", help="Average share kept in risky assets."),
                },
            )
            st.caption("Sorted safest to riskiest. All-in S&P can end with more money in a bull market — "
                       "but read the **worst crash**: that's the scare you'd have lived. The smart plan "
                       "reaches the goal with a fraction of the risk.")

            st.subheader("Live: scrub through time and watch the portfolio change")
            ma_live_scrubber(ma_key)

            st.subheader("What the plan did at the key moments")
            for line in multi_asset_diary(dep):
                st.markdown("- " + line)

            with st.expander("Why is it honest? (no look-ahead)"):
                st.markdown(
                    "- **Real aligned data**: S&P, international, bonds and gold, month by month, since all four exist.\n"
                    "- **The regime is read off the stock market** (the S&P): so a low-volatility asset (bonds) "
                    "doesn't mask the crashes.\n"
                    "- **The belief is causal**: it rises *after* the fall happens, never before.\n"
                    f"- **Regimes {mode_txt}**.")
            st.caption("Educational simulation — costs/taxes simplified, monthly rebalancing. "
                       "**Not financial advice.**")
    elif not st.session_state.get("ma_args"):
        st.info("Adjust your numbers and press **Split my money live**.")
    next_step("coverage", "Last stop: how every slide of the proposal became the code you just used.",
              key="next_multi")

# --------------------------------------------------------------------------- 5
if page == "brain":
    eyebrow("Inside the policy · reinforcement learning · the learned artifact")
    st.markdown('<h1>The strategy the AI <span class="gbwm-serif">taught itself</span>.</h1>',
                unsafe_allow_html=True)
    intro("What you'll see here",
          "A look *inside* the RL brain. There's no fixed formula: the computer treats your goal as a "
          "game and learns the strategy by playing it thousands of times. That's **reinforcement "
          "learning (RL)** <span class='gbwm-rl'>RL</span>.",
          ["The weather maps show the **whole** learned strategy: red = hold more stocks, blue = play safe.",
           "The curve below shows the agent going from clueless to nearly the exact solution."])
    st.markdown("- **Q-Learning** — learn by *trial and error*, no model of the market.\n"
                "- **G-Learning** *(our main agent)* — a smarter, entropy-regularized version of Q-learning "
                "we can solve exactly; greedy Q-learning is its zero-temperature limit.\n"
                "- **Deep RL (PPO / SAC)** — neural-network agents for many assets at once.")
    with st.expander("Show the math (optional)"):
        st.latex(r"Q(s,a)\leftarrow Q(s,a)+\alpha\big[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\big]\quad\text{(Q-learning)}")
        st.latex(r"\pi(a\mid s)\propto\pi_0(a\mid s)\,e^{\beta G(s,a)},\quad F(s)=\tfrac1\beta\log\!\sum_a\pi_0(a\mid s)e^{\beta G(s,a)}\;\text{(G-learning)}")
        st.caption("As β→∞, G-learning becomes greedy value iteration (Q-learning). Dixon & Halperin, arXiv:2002.10990.")
    st.subheader("1) The strategy the AI learned")
    st.caption("Each square is one situation — your balance (vertical) at a moment in time (horizontal). "
               "The color is the brain's answer for that situation: **red = hold more stocks (take risk), "
               "blue = hold more cash (play safe)**. It learned to push when far **below** the goal and to "
               "protect gains **above** it — nobody coded that.")
    agent_label = st.selectbox("Show the learned policy of",
                               ["Smart adaptive plan", "Goal-based plan", "Self-taught (Q-learning)"])
    if agent_label == "Smart adaptive plan":
        pol = build_policies(key)["Regime-Aware G-Learner"]
        _rlabels = {r: f"{WEATHER.get(r, (r,))[0]} ({REGIME_LABEL.get(r, r).split(' · ')[0]})"
                    for r in cfg.market.regime_names}
        show_fig(plots.plot_policy_regime_grid(pol, target, cfg.steps_per_year,
                                               regime_names=cfg.market.regime_names,
                                               regime_labels=_rlabels))
        st.caption("Takeaway: **four maps, one brain** — this is the deck's central claim made visible. "
                   "Compare the same square across panels: in Stormy the red risk-on region shrinks and "
                   "the safe blue grows. The difference between the panels IS the regime-awareness.")
    else:
        pol = (build_policies(key)["G-Learner"] if agent_label == "Goal-based plan"
               else trained_qlearner(key))
        show_fig(plots.plot_policy_heatmap(pol, target, cfg.steps_per_year))
        st.caption("Takeaway: the whole strategy this agent taught itself, as one map. Follow any "
                   "horizontal line (one balance level) left to right to read what it does as the "
                   "deadline approaches. This agent is weather-blind — one map fits all regimes.")
    st.subheader("2) Watch Q-learning learn by trial and error")
    ql = trained_qlearner(key)
    exact = run_comparison(key)["G-Learner"]["p_goal"]
    curve = np.array(ql.learning_curve)
    cfig = plots.plt.figure(figsize=(8, 3.3)); cax = cfig.add_subplot(111)
    cax.plot(curve[:, 0], curve[:, 1], "-o", color=INK, lw=1.8, ms=3, label="Q-learner (learning)")
    cax.axhline(exact, color=GOAL, ls="--", lw=1.6, label="exact G-learning solution")
    cax.set(xlabel="training episodes (simulated lifetimes)", ylabel="chance of reaching goal",
            title="The agent starts clueless and learns", ylim=(0, 1))
    _finish_fig(cfig, cax)
    show_fig(cfig)
    chart_keys((INK, "Q-learner while practising"), (GOAL, "exact G-learning answer", "dash"))
    st.caption(f"Takeaway: the curve climbs from {pct(curve[0,1])} to {pct(curve[-1,1])} as the agent "
               f"practises — it starts clueless and gets good purely by trial and error, creeping up to "
               f"the exact answer ({pct(exact)}) with no formula handed to it.")
    with st.expander("Advanced: train a deep-RL agent (PPO) live (needs the 'rl' extra; ~1–2 min)"):
        if st.button("Train PPO now"):
            prog = st.progress(0.0, text="starting…")
            try:
                from gbwm.policies.rl_agents import train_ppo_with_curve
                def _cb(step, total, pg):
                    prog.progress(min(step / total, 1.0), text=f"{step:,}/{total:,} steps — success {pct(pg)}")
                _, ppo_curve = train_ppo_with_curve(make_config(*key), total_timesteps=40_000,
                                                    eval_freq=8_000, eval_episodes=400, progress_cb=_cb)
                arr = np.array(ppo_curve)
                pfig = plots.plt.figure(figsize=(8, 3.1)); pax = pfig.add_subplot(111)
                pax.plot(arr[:, 0], arr[:, 1], "-o", color="#8e44ad", lw=1.8, ms=3)
                pax.axhline(exact, color=GOAL, ls="--", lw=1.4, label="G-learning")
                pax.set(xlabel="training steps", ylabel="chance of reaching goal",
                        title="Deep RL (PPO) learning live", ylim=(0, 1))
                _finish_fig(pfig, pax)
                show_fig(pfig)
                st.caption("Takeaway: a neural-network agent (PPO) learning live — its chance of "
                           "reaching the goal rises toward the dashed G-Learner line, proving deep RL "
                           "can learn the allocation from scratch too.")
                st.success("A neural network learned the allocation from scratch.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Couldn't train PPO ({e}). Install:  pip install -e \".[rl]\"")
    next_step("plan", "Now put the brain to work on your own numbers.", key="next_brain")

# --------------------------------------------------------------------------- 7
if page == "coverage":
    eyebrow("CME 241 · midterm proposal → working build")
    st.markdown('<h1>Everything in the proposal, <span class="gbwm-serif">covered</span>.</h1>',
                unsafe_allow_html=True)
    intro("What you'll see here",
          "The technical part: how each slide of the course proposal became code that actually runs "
          "here (the MDP, the RL methods and the full coverage).")
    st.link_button("Open the pitch deck", "https://orosergio.github.io/RegimeAwareGBWM/")

    st.subheader("The MDP (proposal slide 6)")
    mdp = [("STATE", "wealth · time left · gap to goal · market-regime belief", RL),
           ("ACTION", "how much in stocks — conservative / balanced / aggressive", RISK),
           ("REWARD", "goal progress − shortfall − risk / turnover penalty", GOAL),
           ("MARKET", "returns + your monthly contribution → next state", INK)]
    cols = st.columns(4)
    for col, (lbl, val, acc) in zip(cols, mdp):
        card(col, "", lbl, val, acc)
    st.caption("The loop: observe **state** → take **action** → **market** moves & adds your contribution "
               "→ get **reward** → repeat each month. Solved as a Markov Decision Process.")

    st.subheader("Course alignment (proposal slide 5)")
    methods = pd.DataFrame(
        [["Markov Decision Process", "the goal as state → action → reward", "WealthEnv (Gymnasium)"],
         ["Dynamic programming", "exact backward induction", "G-Learner"],
         ["Temporal-difference (Q-learning)", "learn online from experience", "Q-Learner"],
         ["Monte Carlo", "thousands of simulated lifetimes", "evaluation harness"],
         ["Policy gradient", "neural-net agents", "PPO / SAC"]],
        columns=["CME 241 concept", "What it does", "In this build"])
    st.dataframe(methods, hide_index=True, width="stretch")

    st.subheader("Slide-by-slide coverage")
    cov_path = ROOT / "COVERAGE.md"
    if cov_path.exists():
        text = cov_path.read_text(encoding="utf-8")
        body = text.split("\n", 2)[2] if text.startswith("#") else text
        st.markdown(body)
    else:
        st.info("See COVERAGE.md in the repository.")

    st.subheader("References (proposal slide 15)")
    st.markdown(
        "- **Main paper** — Dixon, M. F. & Halperin, I. (2020). *G-Learner and GIRL: Goal Based Wealth "
        "Management with Reinforcement Learning.* arXiv:2002.10990.\n"
        "- **Regime-switching extension** — Bauman, Goluža, Gašperov & Kostanjčar (2024). *Deep RL for "
        "Goal-Based Investing Under Regime-Switching.* PMLR.\n"
        "- **Course** — Stanford CME 241: Foundations of Reinforcement Learning with Applications in Finance.\n"
        "- **Alternative considered** — Bühler et al. (2018). *Deep Hedging.* arXiv:1802.03042.\n"
        "- **Tools** — Farama Gymnasium; Stable-Baselines3 (PPO/SAC).")
    st.caption("Honest scope: GIRL (inverse RL) and the paper's analytic Gaussian policy are not "
               "implemented; the stochastic Gibbs G-learning policy is available via `greedy: false`.")
    st.divider()
    _end = st.columns([1.15, 2.2])
    if _end[0].button("Back to the live bank", key="cov_to_bank", width="stretch"):
        goto("bank")
    _end[1].caption("That's the whole story — the best way to end it is to go play with the brain.")
