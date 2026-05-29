# Architecture — Regime-Aware GBWM Simulator

> A goal-based reinforcement-learning lab for dynamic portfolio allocation under
> changing market regimes. This document is the system-design record: the
> problem framing, the layered architecture, the patterns, the technology
> choices, and the decisions (ADRs) behind them.

---

## 1. What we are building

Most portfolio tools optimize **return**. This project optimizes the **probability
of reaching a financial goal** — `P(W_T ≥ G)` — and minimizes expected shortfall,
while behaving sensibly as the market switches between regimes (bull, stable,
high-vol, bear).

The agent adapts to three things at once:

1. **Time** — how far from the deadline (like a target-date fund).
2. **Wealth gap** — how far current wealth is from the goal.
3. **Market regime** — the inferred "weather" of the market.

That third axis is the differentiator. A glide path changes risk only with time;
our regime-aware agent changes risk with time **and** wealth gap **and** detected
regime.

### The decision problem as an MDP

| Element | Definition |
|---|---|
| **State** `s_t` | normalized wealth, time fraction `t/T`, gap-to-goal `(G − W_t)/G`, and regime belief `b_t` (posterior probabilities over the 4 regimes) |
| **Action** `a_t` | portfolio weights over risky assets + cash (a point on the simplex; single-risky default ⇒ scalar equity weight ∈ [0,1]) |
| **Transition** | wealth evolves via multi-asset GBM whose drift/covariance depend on a hidden regime that follows a Markov chain |
| **Reward** | terminal goal utility (large credit for `W_T ≥ G`, shortfall penalty otherwise) minus path penalties (turnover/transaction cost, optional drawdown) |
| **Objective** | maximize `E[Σ γ^t r_t]`, which is dominated by terminal `P(goal)` and `−E[shortfall]` |

---

## 2. Architecture at a glance

The guiding principle is a **decoupled, UI-agnostic core**. The math knows nothing
about Streamlit, the CLI, or any future web frontend. Delivery layers are thin.

```
┌──────────────────────────────────────────────────────────────────┐
│  DELIVERY (thin, swappable)                                        │
│    app/streamlit_app.py   ·   gbwm CLI (argparse)   ·   [future API] │
└───────────────▲───────────────────────▲──────────────────────────┘
                │                        │
┌───────────────┴────────────────────────┴──────────────────────────┐
│  ORCHESTRATION                                                     │
│    evaluation.harness  ·  scripts/train_agents  ·  model registry  │
└───────────────▲────────────────────────────────────────────────── ┘
                │
┌───────────────┴────────────────────────────────────────────────── ┐
│  CORE LIBRARY  (src/gbwm, pure Python + NumPy; no UI imports)      │
│                                                                    │
│   simulation/    GBM paths + Markov-switching regimes              │
│   envs/          Gymnasium WealthEnv (the MDP)                     │
│   policies/      Policy protocol → baselines, G-Learner, RL agents │
│   detection/     HMM regime inference (offline + online filter)    │
│   data/          yfinance + FRED adapters, calibration, caching    │
│   evaluation/    metrics, Monte-Carlo harness, plots               │
│   explain/       advisor interface → rule-based (LLM pluggable)    │
│   config.py      dataclass config tree   ·   registry.py           │
└──────────────────────────────────────────────────────────────────┘
                │
┌───────────────┴────────────────────────────────────────────────── ┐
│  ARTIFACTS                                                         │
│    artifacts/checkpoints (versioned trained agents + metadata)     │
│    data/cache (parquet cache of real market/macro data)            │
└──────────────────────────────────────────────────────────────────┘
```

**Why this shape.** The expensive, non-deterministic work (training PPO/SAC,
fitting HMMs, downloading data) happens **offline** and is written to
`artifacts/` and `data/cache/`. The demo only does cheap work: load a checkpoint,
run vectorized Monte-Carlo, render. That makes a free Streamlit Cloud deployment
viable while still showcasing heavy RL.

---

## 3. Module map

| Module | Responsibility | Key types |
|---|---|---|
| `config.py` | Validated, mergeable configuration tree | `Config`, `MarketConfig`, `RewardConfig` |
| `registry.py` | Make policies/agents addressable by name | `Registry` |
| `simulation/gbm.py` | Vectorized multi-asset GBM returns | `GBMSimulator` |
| `simulation/regimes.py` | Markov-switching regime generator | `RegimeSimulator`, `MarketModel` |
| `envs/wealth_env.py` | The MDP as a Gymnasium env | `WealthEnv` |
| `policies/base.py` | Strategy-pattern policy contract | `Policy` (Protocol) |
| `policies/baselines.py` | Buy&hold, 60/40, glide path | — |
| `policies/g_learner.py` | Entropy-regularized G-learning | `GLearner` |
| `policies/regime_aware_g_learner.py` | Regime-conditioned G-learner | `RegimeAwareGLearner` |
| `policies/rl_agents.py` | SB3 PPO/SAC behind `Policy` | `PPOPolicy`, `SACPolicy` |
| `detection/hmm.py` | Gaussian HMM regime inference | `HMMRegimeDetector` |
| `data/providers.py` | Real data adapters + cache | `MarketDataProvider`, `FredProvider` |
| `data/calibration.py` | Estimate regime params from data | `calibrate_regimes` |
| `evaluation/metrics.py` | P(goal), shortfall, drawdown, turnover | `evaluate_paths` |
| `evaluation/harness.py` | Run any policy through Monte-Carlo | `run_policy`, `compare_policies` |
| `evaluation/plots.py` | Matplotlib/Plotly figures | — |
| `explain/` | Plain-language decision explanations | `Advisor`, `RuleBasedAdvisor` |
| `cli.py` | `gbwm train|evaluate|backtest|calibrate` | argparse app |

---

## 4. Design patterns

* **Strategy** — every allocator implements the same `Policy.act(obs) -> weights`
  contract, so baselines, the G-Learner and SB3 agents are interchangeable in the
  evaluation harness. New strategies require zero harness changes.
* **Factory + Registry** — policies/agents are registered by name and built from
  config, so experiments are declared in YAML rather than wired in code.
* **Adapter** — `yfinance` and `FRED` sit behind a `DataProvider` interface with
  on-disk caching and an offline fallback to bundled sample data.
* **Provider/Strategy (again)** — the explainability `Advisor` has a rule-based
  default and an LLM implementation behind the same interface; swapping is a
  config flag, never a code change in callers.
* **Dependency injection via config** — simulator, env, reward and agents all
  receive a validated `Config` subtree instead of reaching for globals.
* **Repository** — the model registry stores checkpoints with metadata
  (`config hash`, metrics, timestamp) so the app loads reproducible artifacts.

---

## 5. Technology choices

| Concern | Choice | Why |
|---|---|---|
| Language / packaging | Python 3.10+, `src/` layout, `pyproject.toml` | standard, importable, testable |
| Numerics | NumPy / SciPy / pandas | vectorized Monte-Carlo, stats |
| RL environment | **Gymnasium** | de-facto standard; SB3-compatible |
| Deep RL | **Stable-Baselines3** (PyTorch) PPO/SAC | reliable, well-tested implementations |
| Classical agent | **G-Learning** (Halperin) implemented in-repo | the paper the project extends |
| Regime detection | **NumPy** Gaussian HMM (hmmlearn optional) | transparent EM; no hard dep |
| Config | stdlib **dataclasses** + YAML | validated; **zero hard deps** + merge |
| CLI | **argparse** (+ rich if present) | stdlib, zero-dep, testable |
| Real data | **yfinance** + **pandas-datareader/FRED** | free, no API key required |
| Demo | **Streamlit** (+ Plotly) | fast to build, free public hosting |
| Tests / quality | **pytest**, **ruff**, **black** | correctness + consistent style |

---

## 6. Architecture Decision Records (ADRs)

**ADR-001 — Decouple the core library from any UI.**
*Decision:* all financial/ML logic lives in `src/gbwm` and imports no UI code.
*Consequence:* Streamlit ships now; a FastAPI+React frontend can be added later by
calling the same core — no rewrite. Tests run headless.

**ADR-002 — Offline training, online inference (the deployment model).**
*Context:* PPO/SAC and HMM fitting are too heavy for Streamlit Cloud at request
time. *Decision:* train/fit offline via the CLI, commit versioned checkpoints to
`artifacts/`, and have the app only load + simulate. *Consequence:* free public
demo stays responsive; reproducibility via stored metadata.

**ADR-003 — Hidden regimes via Markov-switching GBM; beliefs via HMM.**
*Decision:* the *true* data-generating process is a Markov chain over regimes,
each with its own GBM drift/covariance. The agent never sees the true regime — it
sees an **HMM posterior belief** estimated from returns. *Consequence:* this is a
POMDP approximated by belief-state augmentation, which is honest about real-world
partial observability and is exactly what makes "regime-aware" meaningful.

**ADR-004 — Single-risky-asset default, N-asset-ready core.**
*Decision:* the default world is equity + cash so baselines map cleanly to the
brief; the simulator, env and policies are written for `N` risky assets so the
multi-asset "big version" is a config change, not a refactor.

**ADR-005 — G-Learning as the headline classical agent.**
*Decision:* implement entropy-regularized G-learning (Gaussian reference policy,
free-energy Bellman backup) as the paper-faithful method, then add a
regime-aware variant that conditions on the belief vector. SB3 agents are
provided as strong learned-baseline comparators.

**ADR-006 — Explainability is a first-class, pluggable layer.**
*Decision:* a rule-based `Advisor` produces deterministic plain-language
explanations ("cut equity 70%→40% because bear belief rose 0.2→0.65"); an LLM
advisor implements the same interface and is **off by default** (no API key in a
public demo). *Consequence:* the demo is safe and free, the upgrade path is one flag.

**ADR-007 — Reproducibility by construction.**
*Decision:* explicit `numpy.random.Generator` threading, a single `seed`, antithetic
variates for Monte-Carlo variance reduction, and config hashing on checkpoints.

**ADR-008 — Dependency-light core; heavy capabilities are optional extras.**
*Context:* the simulator, environment, G-Learner, HMM, evaluation and CLI should
run anywhere (CI, minimal sandboxes) and stay easy to test. *Decision:* the core
depends only on NumPy/pandas/PyYAML/matplotlib; config uses stdlib dataclasses,
the HMM is implemented in NumPy, and the CLI uses argparse. Deep RL
(`gymnasium`+`stable-baselines3`+`torch`), real data (`yfinance`/FRED),
the demo (`streamlit`/`plotly`) and an accelerated HMM (`hmmlearn`/`scipy`) are
opt-in extras (`pip install -e ".[rl,data,app,accel]"`). *Consequence:* the whole
core is unit-testable without network or large wheels; missing optional deps fail
loudly only when that specific feature is invoked.

---

## 7. Data flow

**Training (offline, CLI):**
`config → calibrate regimes (optional, from real data) → build MarketModel →
WealthEnv → train agent (G-Learner / PPO / SAC) → save checkpoint + metadata`.

**Evaluation (offline or app):**
`load policy → harness runs N Monte-Carlo episodes through WealthEnv →
metrics (P(goal), shortfall, terminal-wealth dist, regime-conditional P(goal),
turnover, drawdown) → comparison table + plots`.

**Demo (online, cheap):**
`user sets goal/horizon → app loads checkpoints → vectorized Monte-Carlo →
charts (wealth paths, allocation over time, regime beliefs) → advisor explanation`.

---

## 8. Roadmap to the full "AI Wealth Decision Lab"

The scaffold already supports the big-version axes; remaining work is additive:

* **Multi-asset** — ship `configs/multi_asset.yaml` agents (done at config/sim level).
* **Real data & calibration** — `data/` adapters + `calibrate_regimes` (HMM on real returns).
* **More agents** — model-based & robust RL slot in behind `Policy`.
* **Profiles & cashflows** — retirement drawdown, tuition, house down-payment as goal presets.
* **Taxes & fees** — extend the reward/transaction-cost term in `WealthEnv`.
* **Stress testing** — scripted regime sequences (2008, COVID, inflation shock).
* **LLM advisor** — enable the pluggable provider.

---

## 9. Reproducibility & testing

* Deterministic seeding (`utils/seeding.py`); every stochastic component takes a
  `Generator`.
* `pytest` suite covers GBM moments, regime transition statistics, env API
  conformance (Gymnasium checker), reward monotonicity, and metric correctness.
* `ruff` + `black` enforce style; CI-ready via `pip install -e ".[all,dev]"`.
