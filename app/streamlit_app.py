"""Regime-Aware GBWM — interactive goal planner + reinforcement-learning lab.

Styled to match the CME 241 proposal deck (docs/): Geist + Instrument Serif,
warm paper / ink palette, goal-green / risk-amber / RL-blue / loss-red accents.

Tabs: Your plan · Compare · Your journey · Real markets · How the AI learns ·
Simple analogy · Proposal coverage (slide-by-slide map of the deck → this build).

Educational simulation — NOT financial advice. Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
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

INK, PAPER, GOAL, RISK, RL, LOSS = plots.INK, plots.PAPER, plots.GOAL, plots.RISK, plots.RL, plots.LOSS

st.set_page_config(page_title="Goal Planner — Regime-Aware GBWM", page_icon="🎯", layout="wide")
ADVISOR = RuleBasedAdvisor()

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
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
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
</style>
""",
    unsafe_allow_html=True,
)


def eyebrow(text: str):
    st.markdown(f'<div class="gbwm-eyebrow">{text}</div>', unsafe_allow_html=True)


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

t_plan, t_cmp, t_journey, t_real, t_multi, t_ai, t_simple, t_cov = st.tabs(
    ["🎯 Your plan", "📊 Compare plans", "🔍 Your journey", "🕰️ Time machine (1999→today)",
     "🌍 Multi-activo en vivo", "🤖 How the AI learns", "🧒 Simple analogy", "📋 Proposal coverage"]
)

# --------------------------------------------------------------------------- 1
with t_plan:
    eyebrow("Goal-based wealth management · reinforcement learning")
    st.markdown('<h1>Reach your goal — <span class="gbwm-serif">safely</span>.</h1>', unsafe_allow_html=True)
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

# --------------------------------------------------------------------------- 2
with t_cmp:
    st.header("Compare the plans")
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
REGIME_ES = {"bull": "Bonanza ☀️", "stable": "Normal 🌤️",
             "high_vol": "Nervioso 🌧️", "bear": "Crisis ⛈️"}
HONESTY_MODES = {
    "Reaprender año a año (lo más honesto)": "causal_walk_forward",
    "Aprender una vez al inicio": "causal_calibrate",
    "Regímenes de sentido común (robusto)": "prior",
}
FREQ_ES = {"Mensual": "monthly", "Trimestral": "quarterly", "Anual": "annual"}


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
    step = st.slider("Mes (0 = inicio)", 0, T - 1, min(T - 1, T // 2), key="ma_step")
    mcol, icol = st.columns([1, 1])
    mcol.pyplot(plot_allocation_snapshot(dep, smart, step))
    reg = dep.regime_names[int(np.argmax(dep.belief(smart)[step]))]
    icol.metric("Mes", f"{dep.dates[step]:%b %Y}")
    icol.metric("Clima detectado", REGIME_ES.get(reg, reg))
    icol.metric("En bolsa (S&P + internacional)", pct(dep.stock_equity(smart)[step]),
                delta=f"{pct(dep.safe_haven(smart)[step])} en bonos+oro", delta_color="off")
    icol.metric("Creencia de mal tiempo", pct(dep.risk_belief(smart)[step]))


with t_real:
    eyebrow("Out-of-sample · real market history · no look-ahead")
    st.markdown('<h1>If you had switched this on in <span class="gbwm-serif">1999</span>…</h1>',
                unsafe_allow_html=True)
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
    eyebrow("Multi-activo en vivo · S&P 500 + internacional + bonos + oro · sin mirar el futuro")
    st.markdown('<h1>Una cartera que <span class="gbwm-serif">se reparte sola</span> '
                'entre acciones, bonos y oro.</h1>', unsafe_allow_html=True)
    st.caption("Aquí el plan reparte tu dinero entre **cuatro activos a la vez** y lo va cambiando mes a mes "
               "según el clima del mercado — más bolsa cuando hay sol, más bonos y oro cuando hay tormenta. "
               "Todo sobre la historia **real**, descubriendo cada crisis en tiempo real (jamás con el futuro).")

    uni = list_universe()
    ICONS = {"us_equity": "🇺🇸", "intl_equity": "🌍", "bonds": "🏦", "gold": "🪙"}
    ACCENT = {"us_equity": RL, "intl_equity": "#7048e8", "bonds": GOAL, "gold": RISK}
    ucols = st.columns(len(uni) + 1)
    for col, a in zip(ucols, uni):
        card(col, ICONS.get(a.key, "•"), a.label.split(" (")[0], a.note, ACCENT.get(a.key, INK))
    card(ucols[-1], "💵", "Efectivo", "La parte segura, que rinde el tipo libre de riesgo.", INK)

    st.divider()
    cc = st.columns(4)
    ma_initial = cc[0].number_input("Dinero inicial ($)", 0, 50_000_000, 100_000, step=5_000, key="ma_initial")
    ma_amount = cc[1].number_input("Aporte ($)", 0, 1_000_000, 800, step=100, key="ma_amount")
    ma_freq_label = cc[2].selectbox("¿Cada cuánto aportas?", list(FREQ_ES), key="ma_freq")
    ma_start = cc[3].slider("Año de inicio", 2001, 2018, 2005, key="ma_start")
    cc2 = st.columns([1, 2, 1])
    ma_target = cc2[0].number_input("Meta ($)", 50_000, 100_000_000, 600_000, step=50_000, key="ma_target")
    ma_mode_label = cc2[1].selectbox(
        "Nivel de honestidad de los regímenes", list(HONESTY_MODES), index=0, key="ma_mode",
        help="‘Reaprender año a año’ vuelve a aprender las estaciones del mercado cada año usando SOLO el "
             "pasado visto hasta ahí — lo más estricto.")
    ma_offline = cc2[2].checkbox("Offline", value=False, key="ma_offline",
                                 help="¿Sin internet? Usa datos sintéticos como sustituto.")
    freq = FREQ_ES[ma_freq_label]
    mode = HONESTY_MODES[ma_mode_label]
    per = {"monthly": 1, "quarterly": 3, "annual": 12}[freq]
    st.caption(f"Aportas **{money(ma_amount)}** ({ma_freq_label.lower()}) → equivale a "
               f"**{money(ma_amount / per)}/mes**.  Universo: "
               f"{' · '.join(a.label.split(' (')[0] for a in uni)} + efectivo.")
    if mode == "causal_walk_forward":
        st.info("🧠 *Reaprender año a año* es lo más honesto y lo más lento: re-aprende los regímenes cada año "
                "con solo el pasado. La **primera** vez puede tardar ~1–2 min; después queda en caché.")

    if st.button("🚀 Repartir mi dinero en vivo", type="primary", key="ma_run"):
        st.session_state["ma_done"] = True
    if st.session_state.get("ma_done"):
        spinner_msg = ("Reaprendiendo los regímenes año a año sobre la historia real…"
                       if mode == "causal_walk_forward"
                       else "Repartiendo entre S&P, internacional, bonos y oro…")
        ma_key = (ma_start, ma_initial, ma_amount, freq, ma_target, mode, ma_offline)
        dep = None
        try:
            with st.spinner(spinner_msg):
                dep = ma_deploy(*ma_key)
        except Exception as e:  # noqa: BLE001
            st.error(f"No se pudo ejecutar: {e}")
        if dep is not None:
            if dep.source == "synthetic":
                st.warning("Usando **datos sintéticos** (sin conexión). Solo ilustrativo.")
            else:
                tk = " · ".join(f"{k}={v}" for k, v in dep.tickers_used.items())
                st.success(f"Datos **reales** {dep.dates[0]:%b %Y} → {dep.dates[-1]:%b %Y} "
                           f"({dep.horizon_years} años). Proxies: {tk}. El agente solo vio el pasado en cada paso.")
            modo_txt = {"causal_walk_forward": "reaprendiendo los regímenes año a año (solo pasado)",
                        "causal_calibrate": "con regímenes aprendidos una vez (datos previos al inicio)",
                        "prior": "con regímenes de sentido común"}[dep.regime_mode]
            st.caption(f"Honestidad usada: **{modo_txt}**.")

            sc = scorecard(dep).set_index("Strategy")

            def _vm(name, col):
                return sc.loc[name, col] if name in sc.index else float("nan")

            kc = st.columns(3)
            card(kc[0], "🛡️", "Plan inteligente — peor caída",
                 pct(_vm("Regime-Aware Multi-Asset", "Max drawdown")), GOAL)
            card(kc[1], "📉", "Todo en S&P — peor caída",
                 pct(_vm("All-in S&P 500", "Max drawdown")), LOSS)
            card(kc[2], "🎯", "Plan inteligente — saldo final",
                 money(_vm("Regime-Aware Multi-Asset", "Final balance")), RL)

            st.subheader("Cómo se reparten las 3 carteras a lo largo del tiempo")
            st.caption("Mira el **plan inteligente**: en las bandas de crisis vacía la bolsa (azul) y se llena de "
                       "bonos (verde) y oro (ámbar). Las otras dos nunca se mueven.")
            st.pyplot(plot_allocation_grid(dep))

            st.subheader("Tu dinero sobre la historia real")
            st.pyplot(plot_multi_balances(dep))

            st.subheader("El marcador — el resultado **y** el sobresalto")
            show = scorecard(dep)
            show.insert(0, "Cartera", show["Strategy"].map(ma_friendly))
            show = show.drop(columns="Strategy")
            show["Final balance"] = show["Final balance"].map(money)
            show["Reached goal"] = show["Reached goal"].map(lambda b: "✅ sí" if b else "—")
            for c in ["Max drawdown", "Worst 12-month", "Growth rate (CAGR)", "Avg equity"]:
                show[c] = show[c].map(pct)
            show.columns = ["Cartera", "Saldo final", "¿Llegó a la meta?", "Peor caída",
                            "Peor 12 meses", "Crecimiento (CAGR)", "% invertido medio"]
            st.dataframe(show, hide_index=True, use_container_width=True)
            st.caption("Ordenado de más seguro a menos. El todo-en-S&P puede acabar con más dinero en un mercado "
                       "alcista — pero mira la **peor caída**: ese es el susto que vivirías. El plan inteligente "
                       "llega a la meta con una fracción del riesgo.")

            st.subheader("🔴 En vivo: muévete por el tiempo y mira cómo cambia la cartera")
            ma_live_scrubber(ma_key)

            st.subheader("🗣️ Qué hizo el plan en los momentos clave")
            for line in multi_asset_diary(dep):
                st.markdown("- " + line)

            with st.expander("🔍 ¿Por qué es honesto? (sin mirar el futuro)"):
                st.markdown(
                    "- **Datos reales alineados**: S&P, internacional, bonos y oro, mes a mes, desde que existen los cuatro.\n"
                    "- **El régimen se lee de la bolsa** (el S&P): así un activo de poca volatilidad (bonos) no tapa las caídas.\n"
                    "- **La creencia es causal**: sube *después* de que la caída ocurre, nunca antes.\n"
                    f"- **Regímenes {modo_txt}**.")
            st.caption("⚠️ Simulación educativa — costes/impuestos simplificados, rebalanceo mensual. "
                       "**No es asesoría financiera.**")
    else:
        st.info("Ajusta tus números y pulsa **Repartir mi dinero en vivo**.")

# --------------------------------------------------------------------------- 5
with t_ai:
    st.header("How the AI learns — this is reinforcement learning")
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
