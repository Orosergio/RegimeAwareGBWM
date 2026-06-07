"""Regime-Aware GBWM — interactive goal planner + reinforcement-learning lab.

Styled to match the CME 241 proposal deck (docs/): Geist + Instrument Serif,
warm paper / ink palette, goal-green / risk-amber / RL-blue / loss-red accents.

Tabs (in reading order): How it works · Your plan · Live Bank (NT$) · Compare ·
Your journey · Time machine · Multi-asset · How the AI learns · Simple analogy ·
Proposal coverage. Every tab opens with a plain-English "what you'll see here"
intro. The Live Bank is a step-by-step simulated bank (deposit/withdraw into
wallets) driven live by the goal-based RL allocator, with a fully transparent
month-by-month decision & trade log.

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

# ---- deck style (Geist / Instrument Serif / Geist Mono + palette) ---------- #
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@300;400;500;600;700&family=Geist+Mono:wght@400;500&family=Instrument+Serif:ital@0;1&display=swap');
html, body, [class*="css"], .stApp, .stMarkdown, p, div, span, label { font-family: 'Geist', system-ui, sans-serif; }
.stApp { background: #fafaf7; }
h1, h2, h3, h4 { font-family: 'Geist', sans-serif; letter-spacing: -0.025em; color: #0a0b0d; font-weight: 600; }
h1 { letter-spacing: -0.035em; }
[data-testid="stMetricValue"] { font-weight: 600; letter-spacing: -0.02em; color: #0a0b0d; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] { font-weight: 500; }
.gbwm-eyebrow { font-family: 'Geist Mono', monospace; font-size: 0.78rem; letter-spacing: 0.20em;
  text-transform: uppercase; color: #6b6b66; }
.gbwm-serif { font-family: 'Instrument Serif', serif; font-style: italic; letter-spacing: -0.01em; }
.gbwm-card { border: 1px solid rgba(10,11,13,0.12); border-radius: 16px; padding: 16px 18px;
  background: #ffffff; height: 100%; box-shadow: 0 10px 30px -20px rgba(10,11,13,0.25); }
.gbwm-card .lbl { font-family: 'Geist Mono', monospace; font-size: 0.70rem; letter-spacing: 0.16em;
  text-transform: uppercase; color: #6b6b66; }
.gbwm-card .val { font-size: 1.0rem; margin-top: 6px; line-height: 1.45; color: #0a0b0d; }
.gbwm-card .ico { font-size: 1.3rem; }
/* simple per-tab intro callout */
.gbwm-intro { border-radius: 14px; padding: 14px 18px; margin: 2px 0 18px 0;
  background: linear-gradient(135deg, rgba(43,108,176,0.08), rgba(43,108,176,0.02));
  border: 1px solid rgba(43,108,176,0.18); border-left: 4px solid #2b6cb0; }
.gbwm-intro .t { font-weight: 600; font-size: 1.02rem; color: #0a0b0d; margin-bottom: 3px; }
.gbwm-intro .d { font-size: 0.95rem; color: #3a3a36; line-height: 1.5; }
.gbwm-intro ul { margin: 8px 0 0 0; padding-left: 1.1rem; }
.gbwm-intro li { font-size: 0.92rem; color: #3a3a36; margin: 2px 0; }
/* an RL "where the brain works" badge */
.gbwm-rl { display:inline-block; background: rgba(43,108,176,0.12); color:#2b6cb0;
  border:1px solid rgba(43,108,176,0.30); border-radius:8px; padding:2px 8px; font-weight:600;
  font-size:0.80rem; }
/* weather / regime pill */
.gbwm-pill { display: inline-block; padding: 5px 14px; border-radius: 999px;
  font-weight: 600; font-size: 0.95rem; border: 1px solid rgba(10,11,13,0.12); }
/* big balance odometer */
.gbwm-odo { font-family: 'Geist Mono', monospace; font-weight: 600; letter-spacing: -0.02em;
  font-size: 2.1rem; color: #0a0b0d; line-height: 1.1; }
.gbwm-odo .cur { font-size: 1.1rem; color: #6b6b66; margin-right: 6px; }
.gbwm-sub { font-size: 0.82rem; color: #6b6b66; }
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
    col.markdown(
        f'<div class="gbwm-card" style="border-top:3px solid {accent}">'
        f'<div class="ico">{icon}</div><div class="lbl">{label}</div>'
        f'<div class="val">{value}</div></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
def money(x): return f"${x:,.0f}"
def pct(x): return f"{x * 100:.0f}%"
def ntd(x): return f"NT$ {x:,.0f}"


# Regime → friendly "weather" with emoji + accent color (kid-simple).
WEATHER = {
    "bull": ("Sunny", "☀️", "#2f855a", "rgba(47,133,90,0.12)"),
    "stable": ("Calm", "🌤️", "#2b6cb0", "rgba(43,108,176,0.12)"),
    "high_vol": ("Choppy", "🌧️", "#b7791f", "rgba(183,121,31,0.14)"),
    "bear": ("Stormy", "⛈️", "#c53030", "rgba(197,48,48,0.12)"),
}


def weather_pill(regime: str) -> str:
    name, emo, ink, bg = WEATHER.get(regime, (regime, "•", INK, "#eee"))
    return (f'<span class="gbwm-pill" style="color:{ink};background:{bg};'
            f'border-color:{ink}33">{emo} {name}</span>')


def risk_word(equity: float) -> str:
    return "conservative" if equity < 0.33 else ("balanced" if equity < 0.66 else "aggressive")


# ---- sandbox figures ------------------------------------------------------- #
_ALLOC_COLORS = [RL, "#7048e8", GOAL, RISK, "#0d9488", "#be185d"]


def _plain(label: str) -> str:
    """Drop emoji/flags so matplotlib's font (no emoji glyphs) renders cleanly."""
    drop = {0xFE0F, 0x200D, 0x20E3}
    return "".join(ch for ch in label if ord(ch) < 0x1F000 and ord(ch) not in drop).strip()


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
    ax.set_title("Your balance, month by month"); ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); return fig


def plot_allocation_bar(session) -> "plots.plt.Figure":
    """Horizontal bar of where the model puts the money right now."""
    alloc = session.current_allocation()
    labels = list(alloc.keys()); vals = list(alloc.values())
    colors = [_ALLOC_COLORS[i % len(_ALLOC_COLORS)] for i in range(len(labels) - 1)] + ["#a0aec0"]
    fig = plots.plt.figure(figsize=(7, 0.55 * len(labels) + 0.9)); ax = fig.add_subplot(111)
    y = range(len(labels))
    ax.barh(list(y), vals, color=colors, edgecolor="white")
    for i, v in enumerate(vals):
        ax.text(min(v + 0.01, 0.97), i, f"{v * 100:.0f}%", va="center", fontsize=9)
    ax.set_yticks(list(y)); ax.set_yticklabels([_plain(lb) for lb in labels], fontsize=9.5)
    ax.set_xlim(0, 1); ax.invert_yaxis(); ax.set_xlabel("share of your money")
    ax.set_title("Where the model invests right now"); fig.tight_layout(); return fig


def plot_asset_money(session) -> "plots.plt.Figure | None":
    """Stacked area of how much MONEY sits in each asset + cash over time."""
    M = session.asset_money_matrix()  # (t, A+1)
    if M.shape[0] < 1:
        return None
    x = range(1, M.shape[0] + 1)
    labels = [_plain(lb) for lb in session.asset_labels] + ["Cash"]
    colors = [_ALLOC_COLORS[i % len(_ALLOC_COLORS)] for i in range(M.shape[1] - 1)] + ["#cbd5e0"]
    fig = plots.plt.figure(figsize=(8, 3.0)); ax = fig.add_subplot(111)
    ax.stackplot(list(x), M.T, labels=labels, colors=colors, alpha=0.92)
    ax.set(xlabel="months played", ylabel=f"money held ({session.currency})")
    ax.set_title("How your money is split across assets, over time")
    ax.legend(loc="upper left", fontsize=7, ncol=max(1, (M.shape[1] + 1) // 2))
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
    colors = [GOAL if n == order[-1] else "#b9c6dd" for n in order]
    ax.barh(range(len(order)), vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(min(v + 0.02, 0.98), i, pct(v), va="center", fontsize=9)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([FRIENDLY_NAME.get(n, n) for n in order], fontsize=9)
    ax.set_xlim(0, 1); ax.set_xlabel("chance of reaching your goal")
    ax.set_title("Which plan gives you the best chance?"); fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
st.sidebar.title("🎯 Goal Planner")
st.sidebar.caption("Maximize the *chance of reaching* a money goal — Regime-Aware GBWM with RL.")
st.sidebar.info("New here? Open the **📖 How it works** tab first — it explains every section in "
                "plain language.")
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
with st.sidebar.expander("⚙️ Advanced assumptions"):
    rf = st.slider("Safe-cash yearly return", 0.0, 0.08, 0.03, step=0.005)
    persistence = st.slider("How 'sticky' market moods are", 0.5, 0.99, 0.85, step=0.01)
    n_episodes = st.select_slider("How many futures to simulate", [500, 1000, 2000, 4000], value=1000)
st.sidebar.divider()
st.sidebar.caption("⚠️ Educational simulation — **not financial advice.**")

cfg = make_config(st.session_state.initial, st.session_state.target, st.session_state.horizon,
                  st.session_state.contribution, rf, persistence, n_episodes)
key = cfg_key(cfg)
target = float(st.session_state.target)

(t_guide, t_plan, t_bank, t_cmp, t_journey, t_real, t_multi, t_ai, t_simple, t_cov) = st.tabs(
    ["📖 How it works", "🎯 Your plan", "🏦 Live Bank (NT$)", "📊 Compare plans", "🔍 Your journey",
     "🕰️ Time machine (1999→today)", "🌍 Multi-asset (live)", "🤖 How the AI learns",
     "🧒 Simple analogy", "📋 Proposal coverage"]
)

# --------------------------------------------------------------------------- 0
# 📖 How it works — the guide / reading order (ELI10).
# --------------------------------------------------------------------------- #
with t_guide:
    eyebrow("Start here · how this project works · plain language")
    st.markdown('<h1>How this <span class="gbwm-serif">whole thing</span> works.</h1>',
                unsafe_allow_html=True)
    intro("📖 Read me first",
          "This project answers one question: **what's the smartest way to reach a money goal — "
          "and how do you do it safely when the market gets scary?** A computer learns the answer "
          "by playing the 'reach-your-goal' game thousands of times. That kind of learning-by-playing "
          "is called **reinforcement learning (RL)**.",
          ["Everything here is a **simulation** with **fake money** — never real money.",
           "The fancy part (the **goal-based RL brain**) is marked with a blue "
           "<span class='gbwm-rl'>RL</span> badge wherever it makes a decision."])

    st.subheader("🧗 The big idea, like you're 10")
    st.markdown(
        "Imagine **climbing a mountain**:\n"
        "- 🏔️ The **summit** = your money **goal** (e.g. turn 30,000 into 100,000).\n"
        "- 🧗 **How high you are** = how much money you have right now.\n"
        "- 🌦️ The **weather** = the market's mood: ☀️ sunny (going up), 🌤️ calm, 🌧️ choppy, ⛈️ stormy (crashing).\n"
        "- 🎒 **How risky a path you take** = how much you put in **stocks** (risky, climbs fast) vs. "
        "**cash/bonds** (safe, slow).\n\n"
        "The smart rule: **if the weather is good and you're behind, climb harder (more stocks). "
        "If a storm hits and you're near the top, slow down and protect what you have (more cash/bonds).** "
        "Nobody writes that rule by hand — the RL brain *learns* it by playing.")

    st.subheader("🧠 Where the reinforcement learning actually happens")
    rlc = st.columns(2)
    card(rlc[0], "🎯", "The decision the RL brain makes",
         "Every month it chooses <b>how much to risk</b> (stocks vs. safe) using three things: "
         "<b>time left</b>, the <b>gap to your goal</b>, and the <b>market weather</b>. "
         "This is <i>goal-based</i> RL — it optimizes the <b>chance of hitting your goal</b>, not raw profit.", RL)
    card(rlc[1], "🔭", "The honesty rule (no peeking)",
         "Even on real history, the brain only ever sees the <b>past</b>. At each month it decides "
         "<b>blind to the future</b> — so when it cuts stocks <i>after</i> a crash begins (not before), "
         "that reaction is real, not hindsight.", GOAL)

    st.subheader("🗺️ What each tab is for — and how to read it")
    guide_rows = [
        ("🎯 Your plan", "Your starting point. Type your money, goal and years; it tells you which "
         "plan gives the best chance and how much stock to start with.",
         "Read the green **recommendation** and the success %."),
        ("🏦 Live Bank (NT$)", "The game. Open wallets (start with 30,000 NT$), press play, and watch "
         "the RL brain invest month by month. You can add/take out money like a real bank.",
         "Watch the balance grow, then open **How it's investing** to see *why* each month."),
        ("📊 Compare plans", "A fair race: every plan is tested on the *same* thousands of pretend "
         "futures.", "Longer green bar = better chance. Always check the **worst dip** column."),
        ("🔍 Your journey", "Follow ONE pretend future, month by month, for one plan.",
         "Drag the sliders; the brain explains what it did that month."),
        ("🕰️ Time machine", "The honest test on REAL history (1999→today) through real crashes.",
         "Compare the **final balance** AND the **worst crash** each plan lived through."),
        ("🌍 Multi-asset", "Like the time machine, but splitting across 4 assets at once "
         "(US stocks, world stocks, bonds, gold).",
         "Watch the smart plan dump stocks and load bonds/gold during crises."),
        ("🤖 How the AI learns", "A peek inside the RL brain: the strategy it learned, drawn as a map, "
         "and a curve of it getting smarter.",
         "Red = take more risk, green = play safe; see it climb from clueless to expert."),
        ("🧒 Simple analogy", "The mountain story, with one journey narrated in plain words.",
         "Just read it — it's the shortest explanation."),
        ("📋 Proposal coverage", "The technical map: how the school project turned into working code.",
         "For the curious / graders."),
    ]
    for tname, what, how in guide_rows:
        with st.container():
            st.markdown(f"**{tname}** — {what}")
            st.caption(f"👉 How to read it: {how}")

    st.subheader("⚠️ The honest takeaway")
    st.info("RL is great at **deciding risk** (when to be brave, when to hide) — and you'll *see* it "
            "reach goals with far smaller crashes than 'all-in stocks'. But it is **not** a crystal ball "
            "that predicts prices. The right way to judge it isn't *'did it make the most money?'* but "
            "*'did it reach the goal with less fear, without cheating (no peeking at the future)?'*")

# --------------------------------------------------------------------------- 1
with t_plan:
    eyebrow("Goal-based wealth management · reinforcement learning")
    st.markdown('<h1>Reach your goal — <span class="gbwm-serif">safely</span>.</h1>', unsafe_allow_html=True)
    intro("👋 What you'll see here",
          "This is the home screen. Enter your money, your goal and your time, and the app tells you "
          "which plan gives you the **best chance** of reaching the goal — and how much stock to hold "
          "at the start. <span class='gbwm-rl'>RL</span> picks the winning plan.",
          ["Change the numbers in the left sidebar (or pick a preset).",
           "Read the **recommendation** in green: the best plan and your success %.",
           "To *play* live with simulated money, open the **🏦 Live Bank (NT$)** tab."])
    c = st.columns(4)
    c[0].metric("You have now", money(st.session_state.initial))
    c[1].metric("Goal", money(target))
    c[2].metric("Time", f"{st.session_state.horizon} years")
    c[3].metric("Saving", f"{money(st.session_state.contribution)}/mo")
    res = run_comparison(key)
    best = max(res, key=lambda n: res[n]["p_goal"]); bp = res[best]
    simple = res.get("60/40", bp)
    st.subheader("✅ Our recommendation")
    r1, r2, r3 = st.columns([2, 1, 1])
    r1.success(f"**{FRIENDLY_NAME[best]}** gives you the best shot — about **{pct(bp['p_goal'])}** "
               f"chance of reaching {money(target)}.\n\n_{STRATEGY_BLURB[best]}_")
    r2.metric("Chance of success", pct(bp["p_goal"]),
              delta=f"{(bp['p_goal'] - simple['p_goal']) * 100:+.0f} pts vs 60/40")
    r3.metric("Suggested start mix", f"{pct(bp['start_stock'])} stocks",
              delta=f"{pct(1 - bp['start_stock'])} cash", delta_color="off")
    st.caption(f"If you fall short, the typical gap is about {money(bp['cvar'])}. The plan adjusts every "
               "month as your balance, time left, and market conditions change.")
    st.info("👈 Try a **preset** or change your numbers. The *smart adaptive plan* helps most when your "
            "goal is ambitious for the time you have.")

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
    card(sc[0], "🏦", "Total bank balance", ntd(total), RL)
    card(sc[1], "📥", "Capital you added", ntd(dep), INK)
    card(sc[2], "📈", "Market profit / loss",
         f"{ntd(pnl)}  ({pnl / dep * 100:+.1f}%)" if dep else ntd(pnl),
         GOAL if pnl >= 0 else LOSS)
    card(sc[3], "👛", "Open wallets", str(len(order)), RISK)


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
        st.subheader("🏆 Wallet leaderboard")
        metric = st.selectbox(
            "Sort by", ["Market profit (%)", "Sharpe (risk-adjusted)", "Current balance (NT$)",
                        "Progress to goal (%)"], key="rank_metric")
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
        df = pd.DataFrame(rows).sort_values(metric, ascending=False).reset_index(drop=True)
        medals = ["🥇", "🥈", "🥉"]
        df.insert(0, "#", [medals[i] if i < 3 else str(i + 1) for i in range(len(df))])
        df["Current balance (NT$)"] = df["Current balance (NT$)"].map(lambda x: f"NT$ {x:,.0f}")
        df["Market profit (%)"] = df["Market profit (%)"].map(lambda x: f"{x:+.1f}%")
        df["Progress to goal (%)"] = df["Progress to goal (%)"].map(lambda x: f"{x:.0f}%")
        df["Sharpe (risk-adjusted)"] = df["Sharpe (risk-adjusted)"].map(lambda x: f"{x:.2f}")
        df["Volatility (%)"] = df["Volatility (%)"].map(lambda x: f"{x:.0f}%")
        df["Worst drop (%)"] = df["Worst drop (%)"].map(lambda x: f"{x:.0f}%")
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.caption("**Sharpe** = profit per unit of stress (higher is better). **Worst drop** = the "
                   "scariest peak-to-valley fall along the way (lower is better).")
    st.download_button("📥 Export the WHOLE bank (CSV)", data=_bank_csv(),
                       file_name="bank_statements.csv", mime="text/csv", key="dl_bank")


def _decision_log_expander(s):
    """The transparency centerpiece: every month's decision + trades + 'why'."""
    with st.expander("🔍 How it's investing — month-by-month decision & trade log", expanded=False):
        st.caption("**The prices are real history, but the model only ever sees the past.** Each month "
                   "it decides *blind to the future*, then we apply what actually happened. That's why "
                   "it reacts *after* a crash starts — never before. Below: what it saw, what it moved, "
                   "and **why**.")
        if not s.decisions:
            st.info("Press **▶️ Advance** to make the model take its first decision.")
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
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                     height=min(420, 60 + 36 * len(rows)))


def _live_panel(wid: str):
    """The interactive panel for one wallet — rendered inside a fragment so the
    auto-play timer only re-renders this block, not the whole app."""
    bank = st.session_state.bank
    s = bank.get(wid)
    if s is None:
        return
    cur = s.currency
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
        st.warning(f"⚠️ Offline: using **synthetic stand-in data**. {s.mode_label}")
    else:
        st.caption(f"📡 {s.mode_label} · data source: **{s.source}** · "
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
    msg = (f"🧠 <span class='gbwm-rl'>RL decision</span> &nbsp; The weather looks **{wname}**, so it "
           f"invests **{eq * 100:.0f}%** of your money in risky assets (a *{risk_word(eq)}* mix) and "
           f"keeps the rest in cash.")
    if s.A > 1:
        msg += (f" Of that, ~{s.stock_fraction() * 100:.0f}% is in stocks and "
                f"~{s.safe_haven_fraction() * 100:.0f}% in bonds + gold (the safe harbor).")
    st.markdown(f'<div class="gbwm-intro"><div class="d">{_emph(msg)}</div></div>',
                unsafe_allow_html=True)
    if s.decisions:
        last = s.decisions[-1]
        st.caption(f"📋 Last month ({last.label.replace('Month ', 'M')}): **{_trades_text(last, s.asset_labels)}**. "
                   f"_{last.rationale}_")
        if stepped:  # surface milestone events as toasts for every month just played
            for d in s.decisions[n_before:]:
                for fl in d.flags:
                    st.toast(_FLAG_TOAST.get(fl, fl))

    # While auto-playing, manual controls are disabled (only the timer advances time).
    if playing:
        st.caption("⏸️ Pause **▶️ Auto-play** (above) to use the controls.")

    st.markdown("**⏱️ Time controls**")
    tc = st.columns(4)
    # st.rerun() after a manual move so the bank summary & leaderboard fragments refresh too.
    if tc[0].button("▶️ Advance 1 month", key=f"adv1_{wid}", disabled=s.done or playing,
                    use_container_width=True):
        s.advance(1); st.rerun()
    if tc[1].button("⏩ Advance 1 year", key=f"adv12_{wid}", disabled=s.done or playing,
                    use_container_width=True):
        s.advance(12); st.rerun()
    if tc[2].button("⏭️ To the end", key=f"advall_{wid}", disabled=s.done or playing,
                    use_container_width=True):
        s.advance(s.T); st.rerun()
    if tc[3].button("🔄 Restart", key=f"reset_{wid}", disabled=playing, use_container_width=True):
        bank[wid] = _build_wallet(st.session_state.bank_params[wid])
        st.rerun()
    if s.done:
        st.success("🏁 You reached the end of the available history. Press **Restart**, open another "
                   "wallet, or create one with **Simulated future** to keep playing forever.")

    st.markdown("**🏦 Bank controls** — add or take out capital whenever you like")
    bc = st.columns([1.4, 1, 1])
    amt = bc[0].number_input(f"Amount ({cur})", min_value=0, max_value=100_000_000,
                             value=5_000, step=1_000, key=f"amt_{wid}", disabled=playing)
    if bc[1].button("💰 Deposit", key=f"dep_{wid}", type="primary", disabled=playing,
                    use_container_width=True):
        s.deposit(float(amt)); st.rerun()
    if bc[2].button("🏧 Withdraw", key=f"wd_{wid}", disabled=playing or s.wealth <= 0,
                    use_container_width=True):
        s.withdraw(float(amt)); st.rerun()

    # figures (close them after rendering — this panel re-runs on every autoplay tick)
    g = st.columns([1.45, 1])
    fig_b = plot_balance(s); g[0].pyplot(fig_b); plots.plt.close(fig_b)
    fig_a = plot_allocation_bar(s); g[1].pyplot(fig_a); plots.plt.close(fig_a)
    st.caption("The dotted grey line is **buy & hold the market**. When the AI line stays above it "
               "*during crashes*, that's the RL brain protecting you.")

    # --- transparency: the decision & trade log (collapsed by default) ------- #
    _decision_log_expander(s)

    with st.expander("💼 Money in each asset, over time", expanded=False):
        fig_m = plot_asset_money(s)
        if fig_m is None:
            st.info("Advance a few months to see how your money gets split.")
        else:
            st.pyplot(fig_m); plots.plt.close(fig_m)
            st.caption("Each colored band is real money (NT$) parked in that asset. Watch it slosh from "
                       "stocks into bonds/gold (and cash) when the weather turns.")

    with st.expander("📊 Risk & return scorecard", expanded=False):
        mt = s.metrics()
        k = st.columns(4)
        k[0].metric("Yearly return", f"{mt['ann_return'] * 100:+.1f}%")
        k[1].metric("Volatility (bumpiness)", f"{mt['ann_vol'] * 100:.0f}%")
        k[2].metric("Sharpe (reward vs. stress)", f"{mt['sharpe']:.2f}")
        k[3].metric("Worst drop", f"{mt['max_drawdown'] * 100:.0f}%")
        st.caption("These ignore your deposits/withdrawals — they measure how the *investing* itself did. "
                   "A higher Sharpe and a smaller worst-drop mean a calmer ride to the goal.")

    with st.expander("🧾 Statement (every cash movement)", expanded=False):
        rows = [{"When": tx.label, "Type": tx.kind,
                 "Amount": f"{cur} {tx.amount:,.0f}",
                 "Balance after": f"{cur} {tx.balance_after:,.0f}"}
                for tx in reversed(s.transactions)]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.download_button("📥 Download statement (CSV)", data=_ledger_csv(s),
                           file_name=f"statement_{s.name}.csv", mime="text/csv",
                           key=f"dl_{wid}", use_container_width=True)


with t_bank:
    eyebrow("Live sandbox · simulated bank · public data · fake money")
    st.markdown('<h1>Your <span class="gbwm-serif">simulated bank</span>: '
                'play with the real market.</h1>', unsafe_allow_html=True)
    intro("🏦 What you'll see here",
          "A make-believe bank to experiment with. Open one or more **wallets** (you start with "
          "30,000 NT$), pick a real world market, and the **goal-based reinforcement-learning brain** "
          "<span class='gbwm-rl'>RL</span> decides on its own how much to invest and where, month by "
          "month. You can **deposit or withdraw** capital anytime — but here **nothing is real money**.",
          ["Open a wallet, press **▶️ Advance** (or **Auto-play**) and watch the balance grow (or fall).",
           "Open **🔍 How it's investing** to see *exactly* what it bought/sold each month and **why**.",
           "Public market history, or a **simulated future** generated by the model itself."])
    st.warning("🎮 **Educational game.** No real money is used or moved. It's public market data and "
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
        st.divider()

    # --- create a new wallet ------------------------------------------------ #
    with st.expander("➕ Open a new wallet", expanded=not order):
        with st.form("new_wallet", clear_on_submit=False):
            f1 = st.columns([1.3, 1])
            wname = f1[0].text_input("Wallet name", value=f"Wallet {st.session_state.bank_n + 1}")
            mode_label = f1[1].radio("Which data?", list(DATA_MODES), horizontal=False)
            mlabel = st.selectbox("Market", list(SANDBOX_MARKET_BY_LABEL),
                                  help="Real history of a public world market, or the multi-asset "
                                       "portfolio (S&P + international + bonds + gold).")
            f2 = st.columns(4)
            # NB: distinct names — `target`/`initial` are module globals used by the planner tabs.
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
            submitted = st.form_submit_button("🟢 Open wallet", type="primary")
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
                st.error(f"Could not open the wallet: {e}")
            else:
                wid = uuid.uuid4().hex[:8]
                bank[wid] = session
                st.session_state.bank_params[wid] = params
                order.append(wid)
                st.session_state.bank_n += 1
                st.success(f"✅ Opened **{session.name}**. Let's play!")
                st.rerun()

    # --- save / load the whole bank ----------------------------------------- #
    with st.expander("💾 Save / load your bank", expanded=False):
        sl = st.columns(2)
        if order:
            sl[0].download_button("💾 Save bank (JSON)", data=_bank_state_json(),
                                  file_name="bank_save.json", mime="application/json",
                                  use_container_width=True)
        else:
            sl[0].caption("Open a wallet first to save.")
        up = sl[1].file_uploader("📂 Load a saved bank (JSON)", type=["json"], key="bank_upload")
        if up is not None and sl[1].button("Load this save", use_container_width=True):
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
                st.error(f"Could not load that file: {e}")
            else:
                st.session_state.bank = new_bank
                st.session_state.bank_params = new_params
                st.session_state.bank_order = new_order
                st.session_state.bank_n = len(new_order)
                st.success(f"✅ Loaded {len(new_order)} wallet(s).")
                st.rerun()

    if not order:
        st.info("👆 Open your first wallet to start. Tip: try the **S&P 500** from **2008** to live "
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
            playing = pc[0].toggle("▶️ Auto-play", key=f"play_{active}", value=False,
                                   disabled=s.done,
                                   help="Advance months automatically. Pause to deposit/withdraw.")
            pc[1].selectbox("Speed", list(_SPEEDS), index=1, key=f"speedlbl_{active}")
            speed = _SPEEDS[st.session_state.get(f"speedlbl_{active}", "Normal")]
            if pc[2].button("🗑️ Close wallet", key=f"close_{active}"):
                bank.pop(active, None)
                st.session_state.bank_params.pop(active, None)
                order.remove(active)
                st.rerun()
            else:
                run_every = speed if (playing and not s.done) else None
                _render_fragment(_live_panel, active, run_every=run_every)
                # leaderboard + whole-bank export, also live during auto-play
                st.divider()
                _render_fragment(_ranking_export_panel, run_every=_bank_run_every())

    # --- the honest answer: does RL work in real markets? ------------------- #
    with st.expander("❓ Does this really work in real markets? (honest answer)", expanded=False):
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
            "**🕰️ Time machine** tab measures.")

# --------------------------------------------------------------------------- 2
with t_cmp:
    st.header("Compare the plans")
    intro("⚖️ What you'll see here",
          "A fair fight: every plan is tested on the **same** thousands of simulated futures. You'll "
          "see which gives the best chance of reaching the goal and how scary the ride would be "
          "(the worst dip). <span class='gbwm-rl'>RL</span> plans are the goal-based / regime-aware ones.",
          ["The longest green bar = the plan with the best chance of success.",
           "Check the **worst dip** column: a high ending with huge crashes is not a good plan."])
    st.caption("Every plan is tested on the *same* thousands of simulated futures — a fair fight.")
    res = run_comparison(key)
    st.pyplot(goal_chance_chart({n: res[n]["p_goal"] for n in res}))
    rows = [{"Plan": FRIENDLY_NAME[n], "Chance of reaching goal": pct(res[n]["p_goal"]),
             "Typical ending balance": money(res[n]["median"]),
             "If short, typical gap": money(res[n]["cvar"]),
             "Worst dip along the way": pct(res[n]["drawdown"])}
            for n in sorted(res, key=lambda x: res[x]["p_goal"], reverse=True)]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    with st.expander("📐 Full evaluation — the proposal's six metrics"):
        full = [{"Plan": FRIENDLY_NAME[n],
                 "Goal attainment": pct(res[n]["p_goal"]),
                 "Shortfall if missed": money(res[n]["cvar"]),
                 "Terminal wealth (10th–90th)": f"{money(res[n]['p10'])} – {money(res[n]['p90'])}",
                 "Max drawdown": pct(res[n]["drawdown"]),
                 "Turnover": f"{res[n]['turnover']:.3f}",
                 "Best in regime": REGIME_LABEL.get(res[n]["best_regime"], res[n]["best_regime"])}
                for n in sorted(res, key=lambda x: res[x]["p_goal"], reverse=True)]
        st.dataframe(pd.DataFrame(full), hide_index=True, use_container_width=True)
        st.caption("Matches proposal slide 11: goal attainment · shortfall · terminal wealth · drawdown · "
                   "turnover · regime behavior (visualized as the policy heatmap in *How the AI learns*).")
    with st.expander("What do these plans actually do?"):
        for n in ["Buy & Hold", "60/40", "Glide Path", "G-Learner", "Regime-Aware G-Learner"]:
            st.markdown(f"**{FRIENDLY_NAME[n]}** — {STRATEGY_BLURB[n]}")

# --------------------------------------------------------------------------- 3
with t_journey:
    st.header("Walk through one possible journey")
    intro("🔍 What you'll see here",
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
              delta="reached goal 🎉" if reached else "short of goal")
    m2.metric("Your goal", money(target))
    wfig = plots.plt.figure(figsize=(8, 3.3)); wax = wfig.add_subplot(111)
    wax.plot(hist["wealth"][0], color=INK, lw=1.9)
    wax.axhline(target, color=LOSS, lw=1.6, label="goal")
    wax.fill_between(range(hist["wealth"].shape[1]), hist["wealth"][0], target,
                     where=hist["wealth"][0] >= target, color=GOAL, alpha=0.15)
    wax.set(title="Your balance over time", xlabel="month", ylabel="balance ($)"); wax.legend()
    st.pyplot(wfig)
    st.pyplot(plots.plot_allocation_over_time(hist["weights"][0], ["stocks"]))
    st.caption("Risk level: under 33% stocks = **conservative**, 33–66% = **balanced**, over 66% = **aggressive**.")
    step = st.slider("Peek at a month", 1, cfg.total_steps - 1, cfg.total_steps // 2)
    sc = StepContext(weights=hist["weights"][0, step], prev_weights=hist["weights"][0, step - 1],
                     belief=hist["belief"][0, step], prev_belief=hist["belief"][0, step - 1],
                     wealth=float(hist["wealth"][0, step]), target=target, step=step,
                     n_steps=cfg.total_steps, steps_per_year=cfg.steps_per_year,
                     regime_names=cfg.market.regime_names, asset_names=hist["asset_names"])
    st.info("🗣️ " + ADVISOR.explain_step(sc))
    with st.expander("What did the model think the market was doing?"):
        st.pyplot(plots.plot_regime_beliefs(hist["belief"][0], [REGIME_LABEL[r] for r in hist["regime_names"]]))

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
    step = st.slider("Month (0 = start)", 0, T - 1, min(T - 1, T // 2), key="ma_step")
    mcol, icol = st.columns([1, 1])
    mcol.pyplot(plot_allocation_snapshot(dep, smart, step))
    reg = dep.regime_names[int(np.argmax(dep.belief(smart)[step]))]
    icol.metric("Month", f"{dep.dates[step]:%b %Y}")
    icol.metric("Detected weather", WEATHER_EN.get(reg, reg))
    icol.metric("In stocks (US + international)", pct(dep.stock_equity(smart)[step]),
                delta=f"{pct(dep.safe_haven(smart)[step])} in bonds+gold", delta_color="off")
    icol.metric("Bad-weather belief", pct(dep.risk_belief(smart)[step]))


with t_real:
    eyebrow("Out-of-sample · real market history · no look-ahead")
    st.markdown('<h1>If you had switched this on in <span class="gbwm-serif">1999</span>…</h1>',
                unsafe_allow_html=True)
    intro("🕰️ What you'll see here",
          "The honest test: we switch the plans on over the **real** history of a world market and make "
          "them live through dot-com, 2008 and COVID — with no cheating (the model only sees the past "
          "each month). This is where you judge the <span class='gbwm-rl'>RL</span> brain for real.",
          ["Pick a market and a year, then press **Run the time machine**.",
           "Compare the final balance **and** the worst crash: the smart plan wins on calm, not luck."])
    st.caption("Roll every plan over the **real** month-by-month history of a world market. The agent only "
               "ever sees the past — it discovers each crash *in real time* from returns, never with hindsight. "
               "This is the honest test the simulator was built for.")
    cc = st.columns([2, 1, 1, 1])
    label = cc[0].selectbox("Market", list(MARKET_BY_LABEL))
    mkey = MARKET_BY_LABEL[label]
    start_year = cc[1].slider("Start year", 1999, 2018, 1999)
    goal_h = cc[2].number_input("Goal for this run ($)", 50_000, 50_000_000, value=600_000, step=50_000)
    offline = cc[3].checkbox("Offline", value=False, help="No internet? Uses a synthetic stand-in.")
    st.caption(f"Using your sidebar plan: start **{money(st.session_state.initial)}**, "
               f"add **{money(st.session_state.contribution)}/mo**. {LONG_NOTE.get(mkey, '')}")
    if st.button("🕰️ Run the time machine", type="primary"):
        st.session_state["rd"] = True
    if st.session_state.get("rd"):
        dep = None
        try:
            dep = honest_deployment(mkey, f"{start_year}-01-01", float(st.session_state.initial),
                                    float(st.session_state.contribution), float(goal_h), offline)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not run {label} from {start_year}: {e}")
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
            card(cols[0], "📉", "All-in stocks — worst crash", pct(_val("Buy & Hold", "Max drawdown")), LOSS)
            card(cols[1], "🛡️", "Smart adaptive — worst crash",
                 pct(_val("Regime-Aware G-Learner", "Max drawdown")), GOAL)
            both = bool(_val("Buy & Hold", "Reached goal")) and bool(_val("Regime-Aware G-Learner", "Reached goal"))
            card(cols[2], "🎯", "Both reached the goal?",
                 "yes — at a fraction of the risk" if both else "depends on the goal you set", RL)

            st.pyplot(plot_journey(dep))

            st.subheader("The scorecard — the outcome **and** the ride")
            show = scorecard(dep)
            show.insert(0, "Plan", show["Strategy"].map(lambda n: FRIENDLY_NAME.get(n, n)))
            show = show.drop(columns="Strategy")
            show["Final balance"] = show["Final balance"].map(money)
            show["Reached goal"] = show["Reached goal"].map(lambda b: "✅ yes" if b else "—")
            for c in ["Max drawdown", "Worst 12-month", "Growth rate (CAGR)", "Avg equity"]:
                show[c] = show[c].map(pct)
            st.dataframe(show, hide_index=True, use_container_width=True)
            st.caption("Sorted safest-first. On one rising market the all-in plan wins on raw final balance — "
                       "but read the **Max drawdown** column: that's the 40–70% crash you'd have lived through. "
                       "The RL agents reach the goal with far less white-knuckle risk.")

            st.subheader("🗣️ What the agent did at the turning points")
            for line in diary_sentences(dep):
                st.markdown("- " + line)

            with st.expander("⏱️ Does *when* you start matter? — sequence-of-returns risk"):
                seq = seq_risk_cached(mkey, float(st.session_state.initial),
                                      float(st.session_state.contribution), 400_000.0, offline)
                if seq.empty:
                    st.info("Not enough history for this market to compare 1999 / 2000 / 2007 starts.")
                else:
                    st.pyplot(plot_sequence_risk(seq))
                    st.caption("Starting right before a crash (2000, 2007) is the real test. The regime-aware "
                               "agent's drawdown bar stays low even then — it saw the storm coming and de-risked.")
            st.caption("⚠️ Educational simulation — costs/taxes simplified, single risky asset, monthly "
                       "rebalancing. **Not financial advice.**")
    else:
        st.info("Pick a market and press **Run the time machine**.")

# --------------------------------------------------------------------------- 4b
with t_multi:
    eyebrow("Multi-asset live · S&P 500 + international + bonds + gold · no look-ahead")
    st.markdown('<h1>A portfolio that <span class="gbwm-serif">splits itself</span> '
                'across stocks, bonds and gold.</h1>', unsafe_allow_html=True)
    intro("🌍 What you'll see here",
          "Like the time machine, but with **four assets at once** (S&P, international stocks, bonds "
          "and gold). The smart <span class='gbwm-rl'>RL</span> plan loads up on stocks when it's "
          "sunny and rotates into bonds + gold when it's stormy — all over real history.",
          ["Adjust your numbers and press **Split my money live**.",
           "Watch the smart plan empty out of stocks during crises and fill up on safe havens."])
    st.caption("Here the plan splits your money across **four assets at once** and changes it month by "
               "month with the market weather — more stocks when sunny, more bonds and gold when stormy. "
               "All over **real** history, discovering each crisis in real time (never with the future).")

    uni = list_universe()
    ICONS = {"us_equity": "🇺🇸", "intl_equity": "🌍", "bonds": "🏦", "gold": "🪙"}
    ACCENT = {"us_equity": RL, "intl_equity": "#7048e8", "bonds": GOAL, "gold": RISK}
    ucols = st.columns(len(uni) + 1)
    for col, a in zip(ucols, uni):
        card(col, ICONS.get(a.key, "•"), a.label.split(" (")[0], a.note, ACCENT.get(a.key, INK))
    card(ucols[-1], "💵", "Cash", "The safe part, earning the risk-free rate.", INK)

    st.divider()
    cc = st.columns(4)
    ma_initial = cc[0].number_input("Starting money ($)", 0, 50_000_000, 100_000, step=5_000, key="ma_initial")
    ma_amount = cc[1].number_input("Top-up ($)", 0, 1_000_000, 800, step=100, key="ma_amount")
    ma_freq_label = cc[2].selectbox("How often do you add?", list(FREQ_EN), key="ma_freq")
    ma_start = cc[3].slider("Start year", 2001, 2018, 2005, key="ma_start")
    cc2 = st.columns([1, 2, 1])
    ma_target = cc2[0].number_input("Goal ($)", 50_000, 100_000_000, 600_000, step=50_000, key="ma_target")
    ma_mode_label = cc2[1].selectbox(
        "Regime honesty level", list(HONESTY_MODES), index=0, key="ma_mode",
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
        st.info("🧠 *Re-learn year by year* is the most honest and the slowest: it re-learns the regimes "
                "every year using only the past. The **first** run can take ~1–2 min; then it's cached.")

    if st.button("🚀 Split my money live", type="primary", key="ma_run"):
        st.session_state["ma_done"] = True
    if st.session_state.get("ma_done"):
        spinner_msg = ("Re-learning the regimes year by year over real history…"
                       if mode == "causal_walk_forward"
                       else "Splitting across S&P, international, bonds and gold…")
        ma_key = (ma_start, ma_initial, ma_amount, freq, ma_target, mode, ma_offline)
        dep = None
        try:
            with st.spinner(spinner_msg):
                dep = ma_deploy(*ma_key)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not run: {e}")
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
            card(kc[0], "🛡️", "Smart plan — worst crash",
                 pct(_vm("Regime-Aware Multi-Asset", "Max drawdown")), GOAL)
            card(kc[1], "📉", "All-in S&P — worst crash",
                 pct(_vm("All-in S&P 500", "Max drawdown")), LOSS)
            card(kc[2], "🎯", "Smart plan — final balance",
                 money(_vm("Regime-Aware Multi-Asset", "Final balance")), RL)

            st.subheader("How the 3 portfolios split over time")
            st.caption("Watch the **smart plan**: in the crisis bands it empties stocks (blue) and fills "
                       "up on bonds (green) and gold (amber). The other two never move.")
            st.pyplot(plot_allocation_grid(dep))

            st.subheader("Your money over real history")
            st.pyplot(plot_multi_balances(dep))

            st.subheader("The scorecard — the outcome **and** the scare")
            show = scorecard(dep)
            show.insert(0, "Portfolio", show["Strategy"].map(ma_friendly))
            show = show.drop(columns="Strategy")
            show["Final balance"] = show["Final balance"].map(money)
            show["Reached goal"] = show["Reached goal"].map(lambda b: "✅ yes" if b else "—")
            for c in ["Max drawdown", "Worst 12-month", "Growth rate (CAGR)", "Avg equity"]:
                show[c] = show[c].map(pct)
            show.columns = ["Portfolio", "Final balance", "Reached goal?", "Worst crash",
                            "Worst 12 months", "Growth (CAGR)", "Avg invested %"]
            st.dataframe(show, hide_index=True, use_container_width=True)
            st.caption("Sorted safest to riskiest. All-in S&P can end with more money in a bull market — "
                       "but read the **worst crash**: that's the scare you'd have lived. The smart plan "
                       "reaches the goal with a fraction of the risk.")

            st.subheader("🔴 Live: scrub through time and watch the portfolio change")
            ma_live_scrubber(ma_key)

            st.subheader("🗣️ What the plan did at the key moments")
            for line in multi_asset_diary(dep):
                st.markdown("- " + line)

            with st.expander("🔍 Why is it honest? (no look-ahead)"):
                st.markdown(
                    "- **Real aligned data**: S&P, international, bonds and gold, month by month, since all four exist.\n"
                    "- **The regime is read off the stock market** (the S&P): so a low-volatility asset (bonds) "
                    "doesn't mask the crashes.\n"
                    "- **The belief is causal**: it rises *after* the fall happens, never before.\n"
                    f"- **Regimes {mode_txt}**.")
            st.caption("⚠️ Educational simulation — costs/taxes simplified, monthly rebalancing. "
                       "**Not financial advice.**")
    else:
        st.info("Adjust your numbers and press **Split my money live**.")

# --------------------------------------------------------------------------- 5
with t_ai:
    st.header("How the AI learns — this is reinforcement learning")
    intro("🤖 What you'll see here",
          "A look *inside* the RL brain. There's no fixed formula: the computer treats your goal as a "
          "game and learns the strategy by playing it thousands of times. That's **reinforcement "
          "learning (RL)** <span class='gbwm-rl'>RL</span>.",
          ["The color map = how much stock to hold by balance and time (🔴 more risk, 🟢 safer).",
           "The curve shows the agent going from clueless to nearly the exact solution."])
    st.markdown("There's **no fixed formula** for the allocation. The computer treats your goal as a game "
                "and **learns a strategy by playing it thousands of times** against simulated markets. "
                "That's **reinforcement learning (RL)**.")
    st.markdown("- **Q-Learning** — learn by *trial and error*, no model of the market.\n"
                "- **G-Learning** *(our main agent)* — a smarter, entropy-regularized version of Q-learning "
                "we can solve exactly; greedy Q-learning is its zero-temperature limit.\n"
                "- **Deep RL (PPO / SAC)** — neural-network agents for many assets at once.")
    with st.expander("Show the math (optional)"):
        st.latex(r"Q(s,a)\leftarrow Q(s,a)+\alpha\big[r+\gamma\max_{a'}Q(s',a')-Q(s,a)\big]\quad\text{(Q-learning)}")
        st.latex(r"\pi(a\mid s)\propto\pi_0(a\mid s)\,e^{\beta G(s,a)},\quad F(s)=\tfrac1\beta\log\!\sum_a\pi_0(a\mid s)e^{\beta G(s,a)}\;\text{(G-learning)}")
        st.caption("As β→∞, G-learning becomes greedy value iteration (Q-learning). Dixon & Halperin, arXiv:2002.10990.")
    st.subheader("1) The strategy the AI learned")
    st.caption("Each square = how much to hold in stocks (🔴 more stocks/risk, 🟢 more cash/safe). It takes "
               "more risk when far **below** the goal and protects gains **above** it — nobody coded that.")
    hc = st.columns([1, 1])
    agent_label = hc[0].selectbox("Show the learned policy of",
                                  ["Smart adaptive plan", "Goal-based plan", "Self-taught (Q-learning)"])
    regime = None
    if agent_label == "Smart adaptive plan":
        regime = hc[1].selectbox("In which market mood?", cfg.market.regime_names,
                                 format_func=lambda n: REGIME_LABEL[n])
        pol = build_policies(key)["Regime-Aware G-Learner"]
    elif agent_label == "Goal-based plan":
        pol = build_policies(key)["G-Learner"]
    else:
        pol = trained_qlearner(key)
    st.pyplot(plots.plot_policy_heatmap(pol, target, cfg.steps_per_year, regime=regime))
    st.subheader("2) Watch Q-learning learn by trial and error")
    ql = trained_qlearner(key)
    exact = run_comparison(key)["G-Learner"]["p_goal"]
    curve = np.array(ql.learning_curve)
    cfig = plots.plt.figure(figsize=(8, 3.3)); cax = cfig.add_subplot(111)
    cax.plot(curve[:, 0], curve[:, 1], "-o", color=INK, lw=1.8, ms=3, label="Q-learner (learning)")
    cax.axhline(exact, color=GOAL, ls="--", lw=1.6, label="exact G-learning solution")
    cax.set(xlabel="training episodes (simulated lifetimes)", ylabel="chance of reaching goal",
            title="The agent starts clueless and learns", ylim=(0, 1)); cax.legend()
    st.pyplot(cfig)
    st.caption(f"From {pct(curve[0,1])} to {pct(curve[-1,1])} just by practising. Exact G-Learner reaches "
               f"{pct(exact)} — model-free RL approximates it from experience.")
    with st.expander("🧠 Advanced: train a deep-RL agent (PPO) live (needs the 'rl' extra; ~1–2 min)"):
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
                        title="Deep RL (PPO) learning live", ylim=(0, 1)); pax.legend()
                st.pyplot(pfig); st.success("A neural network learned the allocation from scratch.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Couldn't train PPO ({e}). Install:  pip install -e \".[rl]\"")

# --------------------------------------------------------------------------- 6
with t_simple:
    st.header("The simple version — climbing a mountain")
    intro("🧒 What you'll see here",
          "The simplest explanation, like for a kid: investing toward a goal is like climbing a "
          "mountain in good or bad weather. Here we tell it as that story and narrate one journey.")
    st.markdown(
        "- 🏔️ **The summit** is your money goal.\n- 🧗 **How high you are** is your current balance.\n"
        "- 🌦️ **The weather** is the market mood — calm, sunny, or stormy.\n"
        "- 🎒 **How risky a path you take** is how much you put in stocks vs. safe cash.\n\n"
        "If the weather is good and you're behind schedule, you climb faster (more stocks). If a storm "
        "rolls in and you're near the top, you slow down and protect what you have (more cash).\n\n"
        "The **smart adaptive plan** reacts to **time left**, **distance to your goal**, *and* the "
        "**market mood** — a regular target-date fund only reacts to time.")
    pick5 = st.selectbox("Narrate one journey", ["Regime-Aware G-Learner", "G-Learner"],
                         format_func=lambda n: FRIENDLY_NAME[n], key="eli5")
    hist5, _ = single_path(key, pick5, 7)
    ep = EpisodeContext.from_histories(hist5, target, cfg.steps_per_year)
    st.success("🗣️ " + ADVISOR.explain_episode(ep))

# --------------------------------------------------------------------------- 7
with t_cov:
    eyebrow("CME 241 · midterm proposal → working build")
    st.header("Everything in the proposal, covered")
    intro("📋 What you'll see here",
          "The technical part: how each slide of the course proposal became code that actually runs "
          "here (the MDP, the RL methods and the full coverage).")
    st.caption("This maps my pitch deck slide-by-slide to what actually runs here.")

    st.subheader("The MDP (proposal slide 6)")
    mdp = [("🎒", "STATE", "wealth · time left · gap to goal · market-regime belief", RL),
           ("🎛️", "ACTION", "how much in stocks — conservative / balanced / aggressive", RISK),
           ("🏁", "REWARD", "goal progress − shortfall − risk / turnover penalty", GOAL),
           ("🌐", "MARKET", "returns + your monthly contribution → next state", INK)]
    cols = st.columns(4)
    for col, (ico, lbl, val, acc) in zip(cols, mdp):
        card(col, ico, lbl, val, acc)
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
    st.dataframe(methods, hide_index=True, use_container_width=True)

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
