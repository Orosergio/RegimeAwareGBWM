"""Regime-Aware GBWM — interactive goal planner + reinforcement-learning lab.

Plain-language for everyday users, but with a tab that makes the RL tangible:
the learned policy heatmap, a Q-learning convergence curve, and optional live
PPO training. Six screens:

  1. Your plan          — presets + a clear recommendation
  2. Compare plans      — which approach gives the best chance
  3. Your journey       — a sample path: balance vs goal, stocks vs cash, why
  4. Real markets       — pick S&P 500 / NASDAQ / Asia… , calibrate, backtest ALL
  5. How the AI learns  — RL explained + learned-policy heatmap + learning curves
  6. Simple analogy     — the mountain-climbing story

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

from gbwm import experiments as X  # noqa: E402
from gbwm.config import default_config  # noqa: E402
from gbwm.evaluation import plots  # noqa: E402
from gbwm.evaluation.harness import compare_policies, run_policy  # noqa: E402
from gbwm.explain import EpisodeContext, RuleBasedAdvisor, StepContext  # noqa: E402
from gbwm.policies import (  # noqa: E402
    BuyAndHold, GLearner, GlidePath, QLearner, RegimeAwareGLearner, SixtyForty,
)

st.set_page_config(page_title="Goal Planner — Regime-Aware GBWM", page_icon="🎯", layout="wide")
ADVISOR = RuleBasedAdvisor()


# --------------------------------------------------------------------------- #
def money(x: float) -> str:
    return f"${x:,.0f}"


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


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
    return {
        "Buy & Hold": BuyAndHold.from_config(cfg), "60/40": SixtyForty.from_config(cfg),
        "Glide Path": GlidePath.from_config(cfg), "G-Learner": GLearner.from_config(cfg),
        "Regime-Aware G-Learner": RegimeAwareGLearner.from_config(cfg),
    }


@st.cache_resource(show_spinner="Q-learning is learning by trial and error…")
def trained_qlearner(key):
    return QLearner.from_config(make_config(*key))


@st.cache_data(show_spinner="Simulating thousands of possible futures…")
def run_comparison(key):
    cfg = make_config(*key)
    results = compare_policies(build_policies(key), cfg)
    return {n: {"p_goal": r.p_goal, "median": r.median_terminal, "cvar": r.cvar_shortfall,
                "drawdown": r.avg_max_drawdown,
                "start_stock": float(r.histories["weights"][:, 0, :].sum(axis=1).mean())}
            for n, r in results.items()}


@st.cache_data(show_spinner="Rolling a sample journey…")
def single_path(key, agent_name, seed):
    cfg = make_config(*key)
    res = run_policy(build_policies(key)[agent_name], cfg, n_episodes=1, rng=np.random.default_rng(seed))
    return res.histories, float(res.terminal_wealth[0])


def goal_chance_chart(pgoals: dict):
    order = sorted(pgoals, key=pgoals.get)
    fig = plots.plt.figure(figsize=(7, 3.2)); ax = fig.add_subplot(111)
    vals = [pgoals[n] for n in order]
    colors = ["#2e9e5b" if n == order[-1] else "#9bb8d3" for n in order]
    ax.barh(range(len(order)), vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(min(v + 0.02, 0.98), i, pct(v), va="center", fontsize=9)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([FRIENDLY_NAME.get(n, n) for n in order], fontsize=9)
    ax.set_xlim(0, 1); ax.set_xlabel("chance of reaching your goal")
    ax.set_title("Which plan gives you the best chance?"); fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("🎯 Goal Planner")
st.sidebar.caption("See your chance of reaching a money goal — and which plan helps most.")
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
    rf = st.slider("Safe-cash yearly return", 0.0, 0.08, 0.03, step=0.005,
                   help="What your 'cash' bucket earns per year.")
    persistence = st.slider("How 'sticky' market moods are", 0.5, 0.99, 0.85, step=0.01,
                            help="Higher = good/bad spells last longer.")
    n_episodes = st.select_slider("How many futures to simulate", [500, 1000, 2000, 4000], value=1000)
st.sidebar.divider()
st.sidebar.caption("⚠️ Educational simulation — **not financial advice.** It explores assumptions; "
                   "it can't predict markets.")

cfg = make_config(st.session_state.initial, st.session_state.target, st.session_state.horizon,
                  st.session_state.contribution, rf, persistence, n_episodes)
key = cfg_key(cfg)
target = float(st.session_state.target)

t_plan, t_cmp, t_journey, t_real, t_ai, t_simple = st.tabs(
    ["🎯 Your plan", "📊 Compare plans", "🔍 Your journey", "📈 Real markets",
     "🤖 How the AI learns", "🧒 Simple analogy"]
)

# --------------------------------------------------------------------------- #
# 1 — plan + recommendation
# --------------------------------------------------------------------------- #
with t_plan:
    st.header("Your goal, your plan")
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
    st.info("👈 Try a **preset** or change your numbers, then open **Compare plans**. "
            "The *smart adaptive plan* helps most when your goal is ambitious for the time you have.")

# --------------------------------------------------------------------------- #
# 2 — compare
# --------------------------------------------------------------------------- #
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
    with st.expander("What do these plans actually do?"):
        for n in ["Buy & Hold", "60/40", "Glide Path", "G-Learner", "Regime-Aware G-Learner"]:
            st.markdown(f"**{FRIENDLY_NAME[n]}** — {STRATEGY_BLURB[n]}")

# --------------------------------------------------------------------------- #
# 3 — journey
# --------------------------------------------------------------------------- #
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
    wax.plot(hist["wealth"][0], color="#1f4e79", lw=1.9)
    wax.axhline(target, color="#e45756", lw=1.6, label="goal")
    wax.fill_between(range(hist["wealth"].shape[1]), hist["wealth"][0], target,
                     where=hist["wealth"][0] >= target, color="#2e9e5b", alpha=0.12)
    wax.set(title="Your balance over time", xlabel="month", ylabel="balance ($)"); wax.legend()
    st.pyplot(wfig)
    st.pyplot(plots.plot_allocation_over_time(hist["weights"][0], ["stocks"]))
    step = st.slider("Peek at a month", 1, cfg.total_steps - 1, cfg.total_steps // 2)
    sc = StepContext(weights=hist["weights"][0, step], prev_weights=hist["weights"][0, step - 1],
                     belief=hist["belief"][0, step], prev_belief=hist["belief"][0, step - 1],
                     wealth=float(hist["wealth"][0, step]), target=target, step=step,
                     n_steps=cfg.total_steps, steps_per_year=cfg.steps_per_year,
                     regime_names=cfg.market.regime_names, asset_names=hist["asset_names"])
    st.info("🗣️ " + ADVISOR.explain_step(sc))
    with st.expander("What did the model think the market was doing?"):
        st.pyplot(plots.plot_regime_beliefs(hist["belief"][0], [REGIME_LABEL[r] for r in hist["regime_names"]]))

# --------------------------------------------------------------------------- #
# 4 — real markets
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Fetching real market data…")
def fetch_monthly_logreturns(ticker, years, offline):
    from gbwm.data.providers import MarketDataProvider
    prov = MarketDataProvider(offline=offline)
    start = (pd.Timestamp.today() - pd.DateOffset(years=years + 1)).strftime("%Y-%m-%d")
    daily = prov.get_returns([ticker], start, kind="log")
    monthly = prov.resample_returns(daily, 12).iloc[:, 0].dropna()
    return monthly.to_numpy(), prov.last_source


@st.cache_data(show_spinner="Calibrating market moods + backtesting every plan on history…")
def real_data_analysis(ticker, years, offline, key):
    returns, source = fetch_monthly_logreturns(ticker, years, offline)
    results, used, calib = X.backtest_all_on_history(make_config(*key), returns, calibrate=True)
    used_returns = returns[-used.total_steps:]
    states = calib.detector.predict(used_returns)
    table = [{"Plan": FRIENDLY_NAME.get(n, n), "Ending balance": float(r.terminal_wealth[0]),
              "Reached goal?": bool(r.terminal_wealth[0] >= used.goal.target_wealth),
              "Worst dip": float(r.avg_max_drawdown)} for n, r in results.items()]
    ra = results["Regime-Aware G-Learner"]
    return {"source": source, "table": table, "wealth": ra.histories["wealth"][0],
            "weights": ra.histories["weights"][0], "belief": ra.histories["belief"][0],
            "regime_names": ra.histories["regime_names"], "mu": np.atleast_1d(calib.mu_annual).tolist(),
            "sigma": np.atleast_1d(calib.sigma_annual).tolist(), "states": states,
            "returns": used_returns, "years": used.goal.horizon_years,
            "target": used.goal.target_wealth, "initial": used.goal.initial_wealth}


with t_real:
    st.header("Try it on a real market")
    st.caption("Pick a real index/fund — US, NASDAQ, Asia, and more — learn its market 'moods' from "
               "history, then see how every plan would have navigated it.")
    cc = st.columns([2, 1, 1])
    market = cc[0].selectbox("Market", list(X.MARKETS))
    years = cc[1].slider("Years of history", 5, 20, 12)
    offline = cc[2].checkbox("Offline (demo data)", value=False, help="No internet? Uses synthetic data.")
    ticker = X.MARKETS[market]
    if st.button("📈 Analyze this market", type="primary"):
        st.session_state["rd"] = True
    if st.session_state.get("rd"):
        try:
            pack = real_data_analysis(ticker, years, offline, key)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not analyze data: {e}"); st.stop()
        if pack["source"] == "synthetic":
            st.warning("Using **synthetic stand-in data** (no live data available). Illustrative only.")
        else:
            st.success(f"Loaded **real {ticker}** data ({pack['source']}).")
        reg_df = pd.DataFrame({"Market mood": [REGIME_LABEL[n] for n in pack["regime_names"]],
                               "Avg yearly return": [pct(m) for m in pack["mu"]],
                               "Bumpiness (swings)": [pct(s) for s in pack["sigma"]]})
        left, right = st.columns([1, 2])
        left.markdown("**Market moods learned from history**"); left.dataframe(reg_df, hide_index=True)
        price = 100.0 * np.cumprod(1 + np.expm1(pack["returns"]))
        fig = plots.plt.figure(figsize=(8, 3.4)); ax = fig.add_subplot(111)
        colors = [plots.REGIME_COLORS.get(n, "#999") for n in pack["regime_names"]]
        for k in range(len(pack["regime_names"])):
            ax.fill_between(range(len(price)), 0, price.max() * 1.05, where=(pack["states"] == k),
                            color=colors[k], alpha=0.13, step="mid")
        ax.plot(price, color="#1f4e79", lw=1.6)
        ax.set(title=f"{ticker} history, shaded by detected market mood", xlabel="month",
               ylabel="growth of $100"); ax.set_ylim(0, price.max() * 1.05)
        right.pyplot(fig)

        st.subheader("How every plan would have done on this real history")
        tdf = pd.DataFrame(pack["table"]).sort_values("Ending balance", ascending=False)
        tdf["Ending balance"] = tdf["Ending balance"].map(money)
        tdf["Reached goal?"] = tdf["Reached goal?"].map(lambda b: "✅ yes" if b else "—")
        tdf["Worst dip"] = tdf["Worst dip"].map(pct)
        st.dataframe(tdf, hide_index=True, use_container_width=True)

        wfig = plots.plt.figure(figsize=(8, 3.1)); wax = wfig.add_subplot(111)
        wax.plot(pack["wealth"], color="#1f4e79", lw=1.9)
        wax.axhline(pack["target"], color="#e45756", lw=1.6, label="goal")
        wax.set(title="Smart adaptive plan — balance vs goal on real history",
                xlabel="month", ylabel="balance ($)"); wax.legend()
        st.pyplot(wfig)
        ep = EpisodeContext.from_histories(
            {"wealth": pack["wealth"][None, :], "weights": pack["weights"][None, :, :],
             "belief": pack["belief"][None, :, :], "regime": pack["states"][None, :],
             "regime_names": pack["regime_names"], "asset_names": ["stocks"]}, pack["target"], 12)
        st.info("🗣️ " + ADVISOR.explain_episode(ep))
        st.caption("⚠️ Illustration only — market moods are fit on this same window (in-sample); taxes, "
                   "fees and trading costs are simplified. **Not financial advice.**")
    else:
        st.info("Pick a market and press **Analyze this market**.")

# --------------------------------------------------------------------------- #
# 5 — how the AI learns (the RL tab)
# --------------------------------------------------------------------------- #
with t_ai:
    st.header("How the AI learns — this is reinforcement learning")
    st.markdown(
        "There's **no fixed formula** for the allocation. The computer treats your goal as a game and "
        "**learns a strategy by playing it millions of times** against simulated markets — taking an "
        "action (how much in stocks), seeing what happens to your balance, and improving. "
        "That's **reinforcement learning (RL)**."
    )
    st.markdown(
        "- **Q-Learning** — the classic RL method: learn by *trial and error*, no model of the market.\n"
        "- **G-Learning** *(our main agent)* — a smarter, *entropy-regularized* version of Q-learning that "
        "we can solve exactly; greedy Q-learning is its zero-temperature limit.\n"
        "- **Deep RL (PPO / SAC)** — neural-network agents for many assets at once."
    )
    with st.expander("Show the math (optional)"):
        st.markdown("**The decision** each month uses: time left, gap to your goal, and the market mood.")
        st.latex(r"Q(s,a)\;\leftarrow\;Q(s,a)+\alpha\big[\,r+\gamma\max_{a'}Q(s',a')-Q(s,a)\,\big]\quad\text{(Q-learning)}")
        st.latex(r"\pi(a\mid s)\;\propto\;\pi_0(a\mid s)\,e^{\beta\,G(s,a)},\qquad F(s)=\tfrac1\beta\log\!\sum_a \pi_0(a\mid s)\,e^{\beta G(s,a)}\quad\text{(G-learning)}")
        st.caption("As β→∞, G-learning becomes greedy value iteration (Q-learning). Reference: Dixon & "
                   "Halperin, arXiv:2002.10990.")

    st.subheader("1) The strategy the AI learned")
    st.caption("Each square = how much to hold in stocks (🔴 more stocks/risk, 🟢 more cash/safe) for a "
               "given time and balance. Notice it **takes more risk when far below the goal** and "
               "**protects gains above it** — nobody coded that; it was learned.")
    hc = st.columns([1, 1])
    agent_label = hc[0].selectbox("Show the learned policy of",
                                  ["Smart adaptive plan", "Goal-based plan", "Self-taught (Q-learning)"])
    regime = None
    if agent_label == "Smart adaptive plan":
        regname = hc[1].selectbox("In which market mood?", cfg.market.regime_names,
                                  format_func=lambda n: REGIME_LABEL[n])
        regime = regname
        pol = build_policies(key)["Regime-Aware G-Learner"]
    elif agent_label == "Goal-based plan":
        pol = build_policies(key)["G-Learner"]
    else:
        pol = trained_qlearner(key)
    st.pyplot(plots.plot_policy_heatmap(pol, target, cfg.steps_per_year, regime=regime))

    st.subheader("2) Watch Q-learning learn by trial and error")
    ql = trained_qlearner(key)
    res = run_comparison(key)
    exact = res["G-Learner"]["p_goal"]
    curve = np.array(ql.learning_curve)
    cfig = plots.plt.figure(figsize=(8, 3.3)); cax = cfig.add_subplot(111)
    cax.plot(curve[:, 0], curve[:, 1], "-o", color="#1f4e79", lw=1.8, ms=3, label="Q-learner (learning)")
    cax.axhline(exact, color="#2e9e5b", ls="--", lw=1.6, label="exact G-learning solution")
    cax.set(xlabel="training episodes (simulated lifetimes)", ylabel="chance of reaching goal",
            title="The agent starts clueless and learns", ylim=(0, 1)); cax.legend()
    st.pyplot(cfig)
    st.caption(f"From {pct(curve[0,1])} to {pct(curve[-1,1])} just by practising. The exact G-Learner "
               f"reaches {pct(exact)} by solving the equations — model-free RL approximates it from experience.")

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
                pax.axhline(exact, color="#2e9e5b", ls="--", lw=1.4, label="G-learning")
                pax.set(xlabel="training steps", ylabel="chance of reaching goal",
                        title="Deep RL (PPO) learning live", ylim=(0, 1)); pax.legend()
                st.pyplot(pfig)
                st.success("PPO trained — a neural network learned the allocation from scratch.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Couldn't train PPO ({e}). Install the RL extra:  pip install -e \".[rl]\"")

# --------------------------------------------------------------------------- #
# 6 — simple analogy
# --------------------------------------------------------------------------- #
with t_simple:
    st.header("The simple version — climbing a mountain")
    st.markdown(
        """
- 🏔️ **The summit** is your money goal.
- 🧗 **How high you are** is your current balance.
- 🌦️ **The weather** is the market mood — calm, sunny, or stormy.
- 🎒 **How risky a path you take** is how much you put in stocks vs. safe cash.

If the weather is good and you're behind schedule, you climb faster (more stocks).
If a storm rolls in and you're already near the top, you slow down and protect what you have (more cash).

The **smart adaptive plan** reacts to **time left**, **distance to your goal**, *and* the **market mood** —
a regular target-date fund only reacts to time.
        """
    )
    pick5 = st.selectbox("Narrate one journey", ["Regime-Aware G-Learner", "G-Learner"],
                         format_func=lambda n: FRIENDLY_NAME[n], key="eli5")
    hist5, _ = single_path(key, pick5, 7)
    ep = EpisodeContext.from_histories(hist5, target, cfg.steps_per_year)
    st.success("🗣️ " + ADVISOR.explain_episode(ep))
    st.caption("Built on a decoupled, tested core. ⚠️ Educational simulation — not financial advice.")
