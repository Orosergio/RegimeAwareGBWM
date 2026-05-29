# Regime-Aware GBWM Simulator

**A goal-based reinforcement-learning simulator for dynamic portfolio allocation
under changing market regimes.**

Most tools try to make you *as rich as possible*. This one tries to maximize the
**probability of reaching a specific financial goal** — e.g. "turn \$100k into
\$250k in 20 years, contributing \$500/month" — and to do it sensibly whether the
market is in a bull, stable, high-volatility, or bear regime.

> **Explain like I'm 10.** Imagine climbing a mountain. The summit is your goal,
> your height is your wealth, the weather is the market regime, and how risky a
> path you take is your portfolio allocation. If the weather is good and you're
> behind schedule, climb harder. If a storm rolls in and you're nearly there,
> slow down and protect what you've got. That adaptive decision-making is the
> whole project.

This extends the **G-Learner** of Dixon & Halperin (goal-based wealth management
via reinforcement learning) by adding **market regimes** and **HMM-based regime
detection**, then compares the regime-aware agent against classic baselines.

---

## Why it's different from a target-date fund

A glide path (target-date fund) reduces risk **as time passes** — one input.
Our agent is **state-dependent**: it reacts to *time*, *the gap between your
wealth and your goal*, and *the detected market regime* at once.

| Strategy | Adapts to time | Adapts to wealth gap | Adapts to regime |
|---|:---:|:---:|:---:|
| Buy & hold | ✗ | ✗ | ✗ |
| 60/40 | ✗ | ✗ | ✗ |
| Glide path | ✓ | ✗ | ✗ |
| G-Learner | ✓ | ✓ | ✗ |
| **Regime-aware G-Learner** | ✓ | ✓ | ✓ |

---

## Features

- **Custom Gymnasium environment** modeling the goal-based MDP (wealth, time,
  gap-to-goal, regime belief → allocation).
- **Market simulator**: multi-asset Geometric Brownian Motion with a
  **Markov-switching** hidden regime process (bull / stable / high-vol / bear).
- **Baselines**: buy & hold, 60/40, linear glide path.
- **G-Learner**: entropy-regularized G-learning, plus a **regime-aware** variant.
- **Deep RL**: PPO / SAC via Stable-Baselines3 as strong learned comparators.
- **Regime detection**: Gaussian **HMM** producing posterior regime beliefs, used
  online inside the environment.
- **Real data**: optional calibration from ETF prices (yfinance) and macro series
  (FRED), with on-disk caching and offline fallback.
- **Evaluation**: P(goal), expected shortfall, terminal-wealth distribution,
  regime-conditional P(goal), turnover, max drawdown.
- **Explainability**: plain-language explanations of agent decisions (rule-based
  by default; LLM provider pluggable).
- **Streamlit demo** deployable to Streamlit Cloud (loads pre-trained checkpoints).

---

## Install

```bash
# core only
pip install -e .

# everything (deep RL + real data + demo) — recommended
pip install -e ".[all,dev]"
```

## Quickstart (CLI)

```bash
# 1) (optional) calibrate regime parameters from real market data
gbwm calibrate --config configs/default.yaml

# 2) train the agents (writes versioned checkpoints to artifacts/)
gbwm train --config configs/default.yaml --agent g_learner
gbwm train --config configs/default.yaml --agent regime_aware_g_learner

# 3) compare every strategy with Monte-Carlo and print the results table
gbwm evaluate --config configs/default.yaml

# 4) launch the interactive demo
streamlit run app/streamlit_app.py
```

## Project structure

```
regime-aware-gbwm/
├── ARCHITECTURE.md            # system design + ADRs
├── configs/                   # default + multi-asset configs (YAML)
├── src/gbwm/                  # the decoupled core library
│   ├── simulation/            # GBM + Markov-switching regimes
│   ├── envs/                  # Gymnasium WealthEnv (the MDP)
│   ├── policies/              # baselines, G-Learner, RL agents
│   ├── detection/             # HMM regime inference
│   ├── data/                  # yfinance + FRED adapters, calibration
│   ├── evaluation/            # metrics, Monte-Carlo harness, plots
│   ├── explain/               # plain-language advisor (LLM pluggable)
│   ├── config.py · registry.py · utils/
│   └── cli.py
├── app/streamlit_app.py       # interactive demo
├── scripts/train_agents.py    # offline training entrypoint
├── tests/                     # pytest suite
└── notebooks/                 # exploratory notebooks
```

## The science, briefly

- **MDP** — state = (wealth, time fraction, gap-to-goal, regime belief); action =
  portfolio weights; reward = terminal goal utility − path penalties. The headline
  objective is `P(W_T ≥ G)`, not raw return.
- **Hidden regimes (POMDP)** — the true regime is hidden and switches via a Markov
  chain; the agent acts on an **HMM posterior belief**, not the true state.
- **G-Learning** — entropy-regularized RL with a reference (prior) policy and a
  free-energy Bellman backup; the regime-aware variant conditions on the belief.

See `ARCHITECTURE.md` for the full design and decision records.

## Interpreting results

A **good** result: the regime-aware G-Learner improves `P(goal)` and reduces
shortfall *without* extreme turnover or drawdowns. A **bad** result we explicitly
flag: an agent that hits the goal only by taking reckless path risk — "reaching
the goal by gambling" is not an acceptable policy.

## Reinforcement learning, made visible

Three RL methods are implemented and shown working, not just described:

- **Q-Learning** (`QLearner`) — learns the allocation purely by trial and error; the
  app shows its **success rate climbing** over training episodes toward the exact solution.
- **G-Learning** (`GLearner`, `RegimeAwareGLearner`) — the entropy-regularized
  generalization we use as the headline agent (greedy Q-learning is its β→∞ limit).
- **Deep RL** (PPO/SAC via Stable-Baselines3) — trainable live from the app with a
  learning curve.

The app's **How the AI learns** tab visualizes the *learned policy* as a heatmap
(how much to hold in stocks across wealth × time, per market regime) and lets you
watch Q-learning and PPO converge. The **Real markets** tab pulls actual history
for the S&P 500, NASDAQ, Asia-Pacific and more (via yfinance), learns the regimes
from it, and backtests every strategy on that real series.

## Using the demo (for non-experts)

The Streamlit app is written for everyday users — plain language, no jargon. It
has one-click **goal presets** (retirement, house, tuition, emergency fund), a
headline **recommendation** ("your best shot is X — about Y% chance"), a
walkthrough of a sample journey with plain explanations, and a **real market
data** tab that calibrates the regimes from a real ETF (e.g. SPY via yfinance)
and backtests on actual history.

> ⚠️ **Educational simulation — not financial advice.** It explores assumptions
> under a simplified market model (GBM + regimes); it does not predict markets,
> and taxes/fees/transaction costs are simplified.

## Status & roadmap

This is the foundation of the full **AI Wealth Decision Lab**: multi-asset
portfolios, real macro data, additional agents (model-based / robust RL), user
profiles & cashflows, taxes & fees, scripted stress tests, and an LLM advisor.
The architecture is built so each of these is additive — see ARCHITECTURE.md §8.

## License

MIT.
