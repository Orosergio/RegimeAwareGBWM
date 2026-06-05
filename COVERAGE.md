# Proposal → Implementation coverage

This maps every slide of the CME 241 midterm proposal deck
(`docs/RL_Goal_Based_Wealth_Management_Pitch.pptx`) to where it is implemented
and demonstrated in this repository, so the presentation and the working system
line up one-to-one. The app surfaces this same map in its **Proposal coverage** tab.

| # | Slide | Claim / topic | Where it lives | Status |
|---|-------|---------------|----------------|--------|
| 1 | Title | Regime-Aware Goal-Based Wealth Management with RL | whole repo · app header | ✅ |
| 2 | The problem | Goal-based, not return-maximizing (GPS analogy) | `objective.terminal_utility` (P(goal)); app **Your plan** + **Simple analogy** | ✅ |
| 3 | Why not a trading bot | Defensible objective + explainability | `explain/` advisor; goal reward; honest no-look-ahead real-history backtest (`backtesting/`, **Time machine** tab, `HISTORY.md`) | ✅ |
| 4 | Main paper | Dixon & Halperin G-Learner (replication) | `policies/g_learner.py`; cited in README / report / app math | ✅ |
| 5 | Course alignment | MDP · DP · Monte Carlo · TD · Policy gradients | env (MDP) · G-Learner (DP) · evaluation (MC) · Q-Learner (TD) · PPO/SAC (PG) — mapped in app **Proposal coverage** | ✅ |
| 6 | MDP formulation | State (wealth, time, gap, regime), action (conservative/balanced/aggressive), reward (goal − shortfall − risk) | `envs/wealth_env.py`, `config.RewardConfig`; diagram in app coverage tab | ✅ |
| 7 | Thesis | Maximize goal-attainment probability | `objective` + `evaluation/metrics.prob_goal` | ✅ |
| 8 | Technical extension | Regime-aware; 4 regimes; regime probs in state (Bauman et al. 2024) | `simulation/regimes.py`, `detection/`, `RegimeAwareGLearner`; Bauman cited in report | ✅ |
| 9 | Methods + stack | DP, Q/G-learning, PPO; Python, NumPy/pandas, PyTorch, Gymnasium, SB3, Matplotlib | full `src/gbwm`; `rl` extra; **`notebooks/01_pipeline.ipynb`** (Jupyter) | ✅ |
| 10 | Baselines | 60/40, buy-and-hold, glide path, G-Learner, regime-aware | `policies/baselines.py`, `g_learner.py` | ✅ |
| 11 | Evaluation | Goal attainment, shortfall, terminal wealth, drawdown, turnover, regime behavior | `evaluation/metrics.py`; app **Compare → Full evaluation** | ✅ |
| 12 | Real-world value | Robo-advisor, retirement, target-date, finance app, education | app goal **presets** + **🕰️ Time machine** tab (real history since 1999, no look-ahead) + `HISTORY.md` | ✅ |
| 13 | Execution plan | Paper → env → baselines → RL → regime → evaluation | delivered iteratively (see git history / ARCHITECTURE) | ✅ |
| 14 | Final ask | One-sentence pitch | README | ✅ |
| 15 | References | Dixon & Halperin; CME 241; Bauman et al.; Deep Hedging | report.md **References** + app coverage tab | ✅ |

### Honest notes for the defense
- **G-learning vs. the paper:** we solve the entropy-regularized G-learning problem by
  exact discretized backward induction (and deploy the greedy / β→∞ action by default);
  the paper's semi-analytic Gaussian-policy solution and **GIRL** (inverse RL) are not
  implemented. The stochastic Gibbs policy is available via `greedy: false`.
- **Q-learning** converges *toward* but not all the way to the exact G-learning optimum
  (sparse-reward sample inefficiency) — shown as a learning curve against the exact solution.
- **Regime estimation** from monthly returns is noisy; the in-env belief uses the
  true-parameter Bayes filter, so the agent is not bottlenecked by HMM estimation error.
