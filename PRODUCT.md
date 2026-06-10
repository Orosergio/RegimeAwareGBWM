# Product

## Register

product

## Users

- **Sergio (owner / presenter)** — builds and demos the project; judges everything by "does it prove the model works on real data?". Presents from a laptop, usually walking someone else (grader, recruiter, friend) through it live.
- **Evaluators / graders** — technical reviewers who know the CME-241-style proposal; they skim fast and look for honesty (no look-ahead), the MDP, the RL methods, and measured results.
- **Curious non-experts** — first-timers with zero finance/RL background; they need the ELI10 story and must never feel lost. UI is English; explanations kid-simple.

## Product Purpose

An interactive lab that demonstrates the Regime-Aware GBWM model (goal-based wealth management with reinforcement learning, extended with market regimes — pitch deck: https://orosergio.github.io/RegimeAwareGBWM/#1) working in a real environment: simulated futures, a live paper-trading bank (NT$), and honest backtests on real market history with no look-ahead.

Success = a first-time viewer can, within minutes, (1) understand how the model decides, (2) watch it decide live, and (3) believe the evidence — judged by drawdown/calm, not by terminal wealth.

## Brand Personality

Honest, kid-simple, quietly premium. Warm editorial paper-and-ink surface (matches the proposal deck), playful weather metaphor, zero hype. Transparency is the product: every RL decision is shown with its "why".

## Anti-references

- Generic SaaS dashboard slop (hero-metric templates, identical icon card grids, gradient text).
- Crypto / trading-bot UI (dark neon, candlesticks, urgency, profit bragging).
- Academic poster walls of text with unexplained charts.
- Emoji as the visual hierarchy.

## Design Principles

1. **Show, don't tell** — every claim gets a live, manipulable demonstration; text supports the demo, never replaces it.
2. **Judge by calm, not luck** — drawdown-first framing everywhere; never celebrate raw profit.
3. **No peeking is the product** — the no-look-ahead honesty must be visible at the moment of proof, not in a footnote.
4. **One thing at a time** — a guided Learn → Play → Prove journey; each screen has one primary focus and one clear next step.
5. **Plain words, expert numbers** — ELI10 language with real metrics (Sharpe, CVaR, drawdown) always glossed inline.

## Accessibility & Inclusion

WCAG AA contrast minimum (4.5:1 body text); never encode meaning by color alone (market weather = color + emoji + word); honor `prefers-reduced-motion`; visible keyboard focus; every chart paired with a plain-text takeaway.
