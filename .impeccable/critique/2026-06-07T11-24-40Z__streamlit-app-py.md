---
target: app/streamlit_app.py
total_score: 31
p0_count: 1
p1_count: 4
timestamp: 2026-06-07T11-24-40Z
slug: streamlit-app-py
---
# Critique — app/streamlit_app.py (Regime-Aware GBWM)

## Design Health Score — 31/40 (Good)

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 4 | Named spinners, progress bars, storm/goal toasts, live odometer, per-wallet provenance line. |
| 2 | Match System / Real World | 4 | Excellent ELI10 model (mountain/weather, "worst dip"); jargon (Sharpe/CVaR/turnover) always glossed. |
| 3 | User Control & Freedom | 3 | Restart/Close/Pause/save-load exist, but no Undo on Withdraw or Close wallet (both irreversible, no confirm). |
| 4 | Consistency & Standards | 2 | 4 charts skip `_finish_fig` → render on a different paper tone with default spines; 3 different greys; mixed header styles. |
| 5 | Error Prevention | 2 | Withdraw has no confirm and silently clamps to balance; Close wallet destroys state with no guard. |
| 6 | Recognition vs Recall | 3 | Presets/labels/glossaries are good, but 10 tabs force recall of where each feature lives. |
| 7 | Flexibility & Efficiency | 3 | Presets, Turbo, multi-wallet, CSV/JSON export; no deep-linking, no duplicate-wallet. |
| 8 | Aesthetic & Minimalist | 3 | Refined shell, but Live Bank stacks 6+ modules + 5 expanders on one tab. |
| 9 | Error Recovery | 3 | Friendly try/except messages, but raw `{e}` exception text shown to laypeople, no retry. |
| 10 | Help & Documentation | 4 | Best-in-class for the genre: dedicated "How it works" tab, per-tab intro callouts, inline glossaries, optional math. |
| **Total** | | **31/40** | **Good — solid foundation, address weak areas (Consistency, Error Prevention).** |

Cognitive load: fails 5 of 8 checks. Hierarchy and progressive disclosure are excellent; the failure mode is sheer breadth (10 top-level tabs; the new-wallet form exposes 11 inputs; dashboards fire many charts at once).

## Anti-Patterns Verdict

**Not slop — deliberately hand-tuned, and it shows.** The deterministic detector found ZERO gradient-text / glassmorphism / hero-metric / icon-grid hits. The only 9 findings are `overused-font` (Geist / Instrument Serif / Geist Mono sit on the AI-saturated font list) — a provenance signal, not a craft defect. The palette is warm and tokenized, shadows are subtle and token-driven, motion is a quiet 2px lift, charts are despined where `_finish_fig` runs. Residual tells: (1) **emoji as the primary visual hierarchy** on every tab/header/card/button — the single most templated-feeling reflex here, and an a11y liability; (2) the AI-saturated **font roster**; (3) a mild **side-stripe** (`border-left: 2px solid ink` on `.gbwm-intro`, line 118); (4) **chart-styling drift** — 4+ figures render on `plots.py`'s `#fafaf7` facecolor (vs page `#f7f4ee`) with default spines, reading like screenshots from a different app.

Browser overlay: not available (no browser automation / running server in this environment), so no live visual overlay was produced — findings are from source + deterministic scan + exact contrast math.

## Priority Issues

- **[P0] Color is the sole carrier of the most-repeated signal, and it fails contrast.** All four weather pills fail WCAG AA on their own tint (Sunny 3.59, Calm 4.23, Choppy/amber 2.87 — fails even the 3:1 large floor, Stormy 4.16), and muted `#7b746a` fails on paper (4.21) and sidebar (3.88) where eyebrows, captions and unselected tabs live. *Why:* the market "weather" drives every RL decision the app shows; colorblind/low-vision users can't read it. *Fix:* darken pill text to ≥4.5:1 on its tint (or invert to solid fills), add a darker `--muted-strong` token for on-paper/sidebar use. *Command:* `/impeccable colorize`.
- **[P1] The transparency centerpiece is collapsed by default.** The month-by-month decision & "why" log is in an `expander(expanded=False)` (line 727). *Why:* total transparency is the project's stated core principle and a grading criterion; the headline only shows the last month's one-liner, so most viewers never see the reasoning the whole app exists to show. *Fix:* default it open once decisions exist (or promote the last ~3 decisions inline). *Command:* `/impeccable layout`.
- **[P1] The proof-of-honesty charts look the least polished.** `goal_chance_chart` (L433), journey wealth (L1100), Q-learning (L1430) and PPO (L1448) skip `_finish_fig`, inherit a different facecolor and default spines, and `plots.py` uses a cooler palette (`#0a0b0d/#fafaf7/#4c6ef5`) than the CSS (`#15120d/#f7f4ee/#1c6b50`). *Why:* the visuals that prove "no peeking" are the ones that read as off-brand — weakest exactly where credibility matters. *Fix:* route all figures through `_finish_fig` + transparent canvas; unify one palette token source. *Command:* `/impeccable audit` → `/impeccable polish`.
- **[P1] Destructive bank actions have no confirm or feedback.** Withdraw silently clamps to the balance (asking 5,000 from 1,200 removes 1,200, no message); Close wallet wipes a built-up session with one click. *Why:* contradicts the reassuring "safe to experiment" contract. *Fix:* two-step confirm on Close; toast when a withdraw is capped. *Command:* `/impeccable harden`.
- **[P2] Ten sibling tabs overload the entry.** Fails ≤4-choices and single-focus; tabs wrap to 2-3 rows; several overlap (Time machine vs Multi-asset; AI-learns vs Simple analogy vs Coverage). *Fix:* group into ~3 phases (Learn / Play / Prove). *Command:* `/impeccable layout`.
- **[P2] Emoji carry hierarchy instead of type.** Slop tell + a11y issue (screen readers verbalize "🎯 target"); `_plain()` strips them for charts, so UI and chart labels disagree. *Fix:* demote emoji to optional accents, carry hierarchy with the existing type scale + accent dot. *Command:* `/impeccable typeset`.
- **[P2] The leaderboard defaults to sorting by "Market profit (%)".** *Why:* the app's whole thesis is "judge by drawdown / less fear, not by who made the most money" — yet the most prominent ranking defaults to the exact metric it argues against. *Fix:* default sort to "Worst drop" or Sharpe. *Command:* `/impeccable clarify`.

## Persona Red Flags

- **Jordan (first-timer):** lands on "Your plan" (tab order), not "How it works", so success %/CVaR appear before the mental model; 10 wrapping tabs read as an expert control panel; the new-wallet form asks 11 decisions before the first "aha"; Time machine silently depends on hidden sidebar values.
- **Sam (accessibility):** weather/state encoded by hue+emoji; emoji verbalized literally; custom CSS removes native focus rings without a guaranteed `:focus-visible` replacement; charts are images with no alt text (the decision-log table is a strong mitigation — if discovered).
- **Casey (mobile):** 4-up column rows become tall scroll walls; fixed-figsize charts overflow/shrink illegibly (most lack `use_container_width`); 7-10 column tables scroll horizontally; auto-play redraws raster charts on a timer with no reduced-motion respect.

## Minor Observations

Palette drift between CSS and `plots.py`; three distinct neutral greys for bars (`#b9c6dd`/`#b8b2a6`/`#cbc6bb`); `_emph()` re-implements a fragile mini markdown parser; markdown bold inside `st.subheader`; Turbo (0.3s) barely faster than Fast (0.35s) yet jumps 3 months/tick; toasts stack 3+ during Turbo; render-blocking Google Fonts `@import`.

## Questions to Consider

- If success is judged by drawdown and honesty, why does the most prominent ranking default to "Market profit"?
- Ten tabs each open with a "what you'll see here" callout — is that progressive disclosure, or a sign the IA is too broad to hold one narrative?
- The emotional spine is "no cheating" — but the charts that prove it are the ones rendered off-brand. Does the credibility argument land for a skeptical grader if the proof looks the least polished?
