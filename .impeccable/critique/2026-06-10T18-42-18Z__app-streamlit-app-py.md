---
target: app/streamlit_app.py
total_score: 28
p0_count: 2
p1_count: 2
timestamp: 2026-06-10T18-42-18Z
slug: app-streamlit-app-py
---
# Critique — app/streamlit_app.py (Regime-Aware GBWM)

Method: Assessment A (independent design review, source-only) + Assessment B (deterministic detector + exact WCAG math). No browser automation available in this environment; no live overlay produced. Synthesis weaves both.

## Design Health Score — 28/40 (Good, down from prior 31: deeper read of the same surface, not a regression)

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 3 | Narrative spinners + provenance banners excellent; but `rd`/`ma_done` latches never reset — after one press, every input change silently triggers full recomputes the button no longer represents. |
| 2 | Match System / Real World | 3 | Weather metaphor is 4-grade work; docked for Spanish strings inside the English proof charts and `$` sidebar next to an unrelated `NT$` bank. |
| 3 | User Control and Freedom | 2 | No undo on deposits/withdrawals; Restart wipes a wallet instantly (Close confirms, Restart doesn't); no URL state; refresh deletes the whole bank silently. |
| 4 | Consistency and Standards | 2 | Spanish vs English on evidence screens; same strategy named differently in chart legend vs table; half the pages get the eyebrow+h1 hero, half plain `st.header`; allocation drawn 3 ways (bar / stacked area / donut). |
| 5 | Error Prevention | 3 | Bounded inputs, offline fallbacks, clamped withdrawals; no guard for degenerate inputs (goal ≤ current wealth → meaningless 100% everywhere). |
| 6 | Recognition Rather Than Recall | 3 | Column tooltips gloss everything; but the policy heatmap shows one regime at a time behind a selectbox — the regime-dependence (the thesis!) can only be recalled, never seen. |
| 7 | Flexibility and Efficiency | 3 | Presets, Turbo, save/load, CSV; but no deep links, eager execution everywhere, any sidebar tweak invalidates every cache including Q-learner retraining. |
| 8 | Aesthetic and Minimalist Design | 2 | Visual layer tasteful; content layer not minimal: mountain analogy in full twice, intro bullets restated by captions, 4 disclaimers, emoji still anchoring buttons/headers. |
| 9 | Error Recovery | 4 | Best heuristic: human diagnosis + concrete fix + technical trace tucked away, consistently. |
| 10 | Help and Documentation | 3 | Abundant and honest, but front-loaded as a reading map; chart pedagogy lives in skippable small print. |
| **Total** | | **28/40** | **Good — solid foundation; the misses are editorial/structural, not visual.** |

Cognitive load: 5 fail / 2 partial / 1 pass of 8. Live Bank stacks ~15 blocks; wallet form asks 10 questions at once; everything renders eagerly so nothing is the focus.

## Anti-Patterns Verdict

**Above the slop line visually; below it editorially.** Deterministic scan: 9 findings, all `overused-font` (Geist / Geist Mono / Instrument Serif are on the AI-saturated roster) — provenance signal, accepted deliberately because the fonts match the proposal deck; zero gradient-text / glassmorphism / hero-metric / side-stripe / contrast hits. Exact WCAG math: every live token pair passes AA (body 9.9:1, captions 6.5:1, all four weather pills 5.3–6.4:1, primary button 17:1); the only fail (`--muted` 4.21:1) is a dead token never referenced by any rule. Where the surface loses trust is editorial, where the detector can't see: (1) Spanish remnants inside the English evidence charts ("Efectivo", "meta", "saldo", full Spanish ribbon title); (2) the Live Bank's two chart captions are swapped relative to their figures — text authored against code, not the rendered page; (3) the 10× repeated intro-card scaffold, applied inconsistently (5 hero pages, 5 plain headers); (4) LLM bold-sprinkle cadence with 84 em dashes in copy.

## Overall Impression

A genuinely hand-tuned, honest, warm surface whose two product promises — "guide me" and "demonstrate the model" — are the two things it doesn't yet do. Attention is structurally diffused (everything renders, everything weighs the same, every next step is prose instead of a control), and the model is described in text rather than demonstrated (the policy — the actual learned artifact — hides behind one selectbox-gated, red-green heatmap). The single biggest opportunity: make the learned policy itself the first interactive thing a visitor touches.

## What's Working

1. **The honesty thread is designed-in, not a footnote**: provenance stamped at the moment of proof, strictest honesty mode as default, a voluntarily skeptical "does this really work?" expander. Graders will notice.
2. **Plain-words/expert-numbers glossing at point of use**: every scorecard column carries an ELI10 tooltip; regimes are encoded color + emoji + word with per-tint AA-darkened text.
3. **Failure paths are first-class**: every network action has an offline fallback, a friendly error, a concrete recovery, and a collapsed trace.

## Priority Issues

- **[P0] The proof charts speak Spanish in an English app** (plotting.py 156/178–180/208–211/225, rendered on the Multi-asset evidence screens). Trust contamination exactly where belief is requested. Fix: translate; unify strategy names between chart legends and tables. *Command: /impeccable clarify.*
- **[P0] Charts narrate instead of teach** (owner complaint 2). Pedagogy exists only as post-hoc small-print captions (two of them swapped, 960–968); nothing annotated on-canvas; the thesis figure (regime-dependent policy) is one heatmap behind a selectbox, in colorblind-hostile `RdYlGn_r`. Fix: regime small-multiples in one row; annotate crisis reactions on the journey chart; per-chart reading keys adjacent to each figure; colorblind-safe ramp + contour labels. *Command: /impeccable polish.*
- **[P1] Attention is structurally diffused** (owner complaint 1): eager execution of all ten tab bodies (Q-learner trains on first load), equal-weight card chrome on everything, the verdict styled as a default alert, next steps as prose because tabs can't be linked. Fix: stateful page navigation with real "Next →" controls, one hero element per screen, reset latches when inputs change. *Command: /impeccable layout.*
- **[P1] The Live Bank's first decision is a 10-field form; the sidebar pretends to global scope.** Stage the form (open with defaults; customize optional); state the sidebar's actual scope. *Command: /impeccable onboard.*
- **[P2] Destructive actions inconsistent; the bank is volatile.** Restart unconfirmed while Close confirms; refresh erases the bank with save collapsed and unprompted. Fix: confirm Restart; quiet save nudge after months played. *Command: /impeccable harden.*

## Persona Red Flags

- **Jordan (first-timer):** first screen is an ~80-line text wall with zero interaction; the reading map lists 10 destinations with no clickable links; swapped captions sabotage the first attempt to learn chart-reading; `$` sidebar vs `NT$` bank unexplained; "Regime honesty level" and "Month (0 = start)" are engineer-speak islands.
- **Alex (power user):** no URL state; refresh destroys the bank; first load pays for Q-learner training never requested; one sidebar digit re-pays everything; latched runs make exploring five markets five forced waits.
- **Sam (a11y):** every chart is a silent PNG (captions are the only access path and one pair is mis-ordered); policy heatmap meaning carried entirely by a red-green ramp; "Reached goal" encoded ✅-vs-em-dash; medal-emoji ranks; auto-play DOM churn not covered by reduced-motion; custom cards are bare divs at .66rem labels. Credit: deliberate `:focus-visible` rings; AA-darkened pill text.

## Minor Observations

Figures passed to `st.pyplot` without `plt.close` (memory creep); two chart-styling families drift (app `_finish_fig` vs plotting.py rcParams); donut is the weakest allocation idiom and the only Spanish one; mountain analogy duplicated full-text in two tabs; disclaimers ×4; goal-chance chart hides the deck's named baseline (glide path) in neutral grey; naming drift (browser title vs sidebar vs none); `--muted` is a dead token; 84 em dashes.

## Questions to Consider

1. If the whole thesis is "the policy is regime-dependent," why can the user never see two regimes at once?
2. Where does a grader watch the deck's "+13pp vs a glide path" get reproduced — should there be one button, one number, one labeled bar?
3. Should the app open on a pre-opened S&P-2008 wallet already auto-playing — show first, explain on demand?
