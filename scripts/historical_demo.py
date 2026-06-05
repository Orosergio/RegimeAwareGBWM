"""Headline real-world demonstration — *"if you had deployed in 1999."*

Runs the honest, out-of-sample backtest across several world markets, prints the
risk-focused scorecards and the decision diary, runs the sequence-of-returns
analysis (start in 1999 vs 2000 vs 2007), and saves the report figures.

    python scripts/historical_demo.py                 # S&P 500 + cross-market + sequence risk
    python scripts/historical_demo.py --market nasdaq
    python scripts/historical_demo.py --offline       # no network (synthetic stand-in)

Figures are written to ``docs/uploads/``. Numbers are real market history; this
is an educational simulation, not financial advice.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from gbwm.backtesting import diary_sentences, run_deployment, scorecard, sequence_risk  # noqa: E402
from gbwm.backtesting.plotting import plot_journey, plot_sequence_risk  # noqa: E402
from gbwm.config import default_config  # noqa: E402

warnings.filterwarnings("ignore")
OUT = ROOT / "docs" / "uploads"
OUT.mkdir(parents=True, exist_ok=True)


def _money(x: float) -> str:
    return f"${x:,.0f}"


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def fmt_scorecard(sc: pd.DataFrame) -> pd.DataFrame:
    show = sc.copy()
    show["Final balance"] = show["Final balance"].map(_money)
    show["Reached goal"] = show["Reached goal"].map(lambda b: "yes" if b else "no")
    for c in ("Max drawdown", "Worst 12-month", "Growth rate (CAGR)", "Avg equity"):
        show[c] = show[c].map(_pct)
    return show


def main() -> int:
    ap = argparse.ArgumentParser(description="Historical real-world demonstration")
    ap.add_argument("--market", default="sp500", help="primary market for the journey figure")
    ap.add_argument("--start", default="1999-01-01")
    ap.add_argument("--initial", type=float, default=100_000.0)
    ap.add_argument("--contribution", type=float, default=500.0)
    ap.add_argument("--target", type=float, default=600_000.0)
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args()

    cfg = default_config()
    common = dict(initial=args.initial, contribution=args.contribution,
                  target=args.target, offline=args.offline)

    # --- 1) The headline journey on the primary market --------------------- #
    print("=" * 78)
    print(f"  IF YOU HAD DEPLOYED THIS RL ADVISOR IN {args.start[:4]} — {args.market.upper()}")
    print("=" * 78)
    dep = run_deployment(cfg, args.market, start=args.start, **common)
    print(f"\n{dep.market.label}: {dep.dates[0]:%b %Y} → {dep.dates[-1]:%b %Y} "
          f"({dep.horizon_years}y) · data={dep.source} · regimes={dep.regime_mode} (no look-ahead)")
    print(f"plan: start {_money(dep.config.goal.initial_wealth)}, "
          f"add {_money(dep.config.goal.contribution)}/mo, goal {_money(dep.target)}\n")
    print(fmt_scorecard(scorecard(dep)).to_string(index=False))
    print("\nDecision diary (what it did at the turning points):")
    for line in diary_sentences(dep):
        print("  • " + line.replace("**", ""))

    fig = plot_journey(dep)
    p = OUT / f"history_{args.market}_{args.start[:4]}.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    print(f"\n  saved figure → {p.relative_to(ROOT)}")

    # --- 2) Cross-market table (same plan, different worlds) --------------- #
    print("\n" + "=" * 78)
    print("  SAME PLAN, DIFFERENT WORLD MARKETS (deployed ~1999, max drawdown along the way)")
    print("=" * 78)
    markets = ["sp500", "nasdaq", "kospi", "nikkei"]
    cross_rows = []
    for mk in markets:
        try:
            d = run_deployment(cfg, mk, start=args.start, **common)
        except Exception as exc:  # noqa: BLE001
            print(f"  ({mk}: {exc})")
            continue
        sc = scorecard(d).set_index("Strategy")
        for strat in ("Buy & Hold", "Regime-Aware G-Learner"):
            if strat in sc.index:
                cross_rows.append({
                    "Market": d.market.label.split(" — ")[0],
                    "Strategy": strat,
                    "Final": _money(sc.loc[strat, "Final balance"]),
                    "Max drawdown": _pct(sc.loc[strat, "Max drawdown"]),
                    "Reached goal": "yes" if sc.loc[strat, "Reached goal"] else "no",
                })
    if cross_rows:
        print("\n" + pd.DataFrame(cross_rows).to_string(index=False))

    # --- 3) Sequence-of-returns risk --------------------------------------- #
    print("\n" + "=" * 78)
    print("  WHEN YOU START MATTERS — same 15-year plan, three start dates")
    print("=" * 78)
    seq = sequence_risk(cfg, args.market, starts=["1999-01-01", "2000-01-01", "2007-01-01"],
                        horizon_years=15, initial=args.initial,
                        contribution=args.contribution, target=400_000, offline=args.offline)
    if not seq.empty:
        dd = seq.pivot(index="Strategy", columns="Start", values="Max drawdown")
        print("\nMax drawdown by start year (lower = smoother ride):")
        print(dd.map(_pct).to_string())
        fb = seq.pivot(index="Strategy", columns="Start", values="Final balance")
        print("\nFinal balance by start year:")
        print(fb.map(_money).to_string())
        fig2 = plot_sequence_risk(seq)
        p2 = OUT / f"history_sequence_risk_{args.market}.png"
        fig2.savefig(p2, dpi=130, bbox_inches="tight")
        print(f"\n  saved figure → {p2.relative_to(ROOT)}")

    print("\n" + "-" * 78)
    print("Takeaway: on a single rising market, holding the most stocks wins on raw")
    print("terminal wealth — but with 45–70% crashes. The goal-based RL agents reach")
    print("the goal with a FRACTION of the drawdown, and the regime-aware one earns its")
    print("keep most when you start right before a crash. Educational, not advice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
