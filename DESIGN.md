# Design

## Visual Theme

Warm editorial "paper & ink" — a quiet-luxury take on a research deck. Cream paper background, warm near-black ink, one deep-green accent, serif italics reserved for display moments. Charts blend into the page (transparent canvases, despined axes, hairline grids). The interface should read like a beautifully typeset working paper that happens to be alive.

## Color Palette

Tokens live in the CSS `:root` block of `app/streamlit_app.py` and are mirrored in `src/gbwm/evaluation/plots.py` (single source of truth for chart colors).

- `--paper` `#f7f4ee` — app background (also `.streamlit/config.toml` backgroundColor)
- `--card` `#fffdf8` — raised surfaces (cards, intro callouts, expanders)
- `--ink` `#15120d` — primary text and chart ink
- `--muted-strong` `#6a6157` — secondary text (passes AA on paper, card and sidebar; there is deliberately no lighter muted token)
- `--line` `rgba(21,18,13,.12)` / `--hair` `rgba(21,18,13,.07)` — borders
- `--accent` `#1c6b50` (deep green) + `--accent-soft` `rgba(28,107,80,.10)` — primary actions, selection, focus rings
- Chart semantics: `GOAL #2e9e5b` (green = safety/goal) · `RISK #cf8a2c` (amber = risk/regime) · `RL #4c6ef5` (blue = RL/data) · `LOSS #e0553a` (red = shortfall/crisis) · neutral `#bcb6a8` (cash / non-highlighted bars)
- Regime mapping: bull→GOAL, stable→RL, high_vol→RISK, bear→LOSS — always paired with emoji + word, never color alone
- Sidebar: `#efebe1` (second warm-neutral layer)

## Typography

- **Geist** 400/500/600/700 — UI, headings, body (one family carries the product)
- **Geist Mono** 400/500/600 — eyebrows, metric values, balances (`tabular-nums`), the RL badge
- **Instrument Serif** italic — display accents inside h1 only (`.gbwm-serif`)
- Scale: h1 2.55rem / −.042em · h2 1.5rem · h3 1.16rem · body 1rem / 1.6 · captions .82–.95rem
- Eyebrows / mono labels: .66–.72rem, uppercase, letter-spacing .15–.24em

## Components

- `.gbwm-card` — hairline border, 14px radius, soft layered shadow, hover lift (−2px); icon + mono label + value + colored status dot
- `.gbwm-intro` — per-page "what you'll see here" callout (card surface, full hairline box; no side-stripe)
- `.gbwm-rl` — small mono outline badge marking every point where the RL policy decides
- `.gbwm-pill` — weather pill (tinted background, AA-dark text of the same hue, emoji + word)
- `.gbwm-odo` — Geist Mono balance odometer (tabular numerals)
- `.gbwm-keys` — chart-reading key chips (tiny swatch + label) placed directly under each figure
- `.gbwm-verdict` — the one loud element per surface: accent-tinted block with a big mono number + deltas line
- `.gbwm-bar` — stocks/cash split bar for the live decision demo
- Nav pills: the two nav radios (`.st-key-nav_phase`, `.st-key-nav_page_*`) restyled as pills; phase = ink-filled, page = accent-soft; radio dot collapsed to zero size (keyboard focus preserved via `:focus-within`)
- Buttons: 10px radius, hairline border; primary = ink-filled; 2px accent `:focus-visible` ring everywhere
- Metrics, dataframes, expanders: card surface, hairline border, 12px radius

## Layout

- Max content width 1200px; top padding 2.6rem (1.4rem ≤640px)
- Guided-story navigation Learn → Play → Proof: stateful pages (one renders at a time), every page ends with a `next_step()` footer; "Open"/"Next" buttons navigate via `st.session_state["_goto"]`
- Charts: route every figure through `_finish_fig` (transparent canvas, despined, hairline y-grid, muted tick ink `#5f574c`); render via `show_fig` (closes figures). Policy heatmaps use the colorblind-safe `RdYlBu_r` ramp (red = risk-on, blue = safe) with labeled colorbar ticks — never red-green
- Mobile ≤640px: tightened padding, h1 2rem

## Motion

150–250ms, `cubic-bezier(.2,.7,.2,1)`, transform/opacity only (cards lift −2px, buttons −1px). Honors `prefers-reduced-motion` (everything collapses to ~0ms). No page-load choreography.
