"""``gbwm`` command-line interface (argparse — stdlib, ADR-008).

Commands: train, evaluate, backtest, calibrate, ls. A thin shell over
:mod:`gbwm.experiments`.
"""

from __future__ import annotations

import argparse

import pandas as pd

from gbwm import experiments as X
from gbwm.checkpoints import ModelRegistry
from gbwm.config import default_config, load_config


def _load(args) -> "object":
    if args.config:
        return load_config(args.config, base=args.base)
    return default_config()


def _print_df(df: pd.DataFrame) -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(df.to_string(index=False))


def cmd_train(args):
    cfg = _load(args)
    reg = ModelRegistry(cfg.paths.checkpoints)
    agents = X.GLEARNER_KEYS + X.RL_KEYS if args.agent == "all" else [args.agent]
    for a in agents:
        X.train(cfg, a, reg, timesteps=args.timesteps, name=args.name)


def cmd_evaluate(args):
    cfg = _load(args)
    reg = ModelRegistry(cfg.paths.checkpoints)
    _, table = X.evaluate(cfg, n_episodes=args.episodes, which=args.agents, registry=reg)
    _print_df(table)


def cmd_backtest(args):
    cfg = _load(args)
    reg = ModelRegistry(cfg.paths.checkpoints)
    res, explanation = X.backtest(cfg, args.agent, seed=args.seed, registry=reg)
    print(f"terminal wealth: ${res.terminal_wealth[0]:,.0f}  (goal ${cfg.goal.target_wealth:,.0f})")
    print(explanation)


def cmd_history(args):
    from gbwm.backtesting.historical import (
        LONG_HISTORY_MARKETS, diary_sentences, list_markets, scorecard,
    )

    if args.market not in LONG_HISTORY_MARKETS:
        keys = ", ".join(m.key for m in list_markets())
        raise SystemExit(f"unknown market '{args.market}'. Choose one of: {keys}")
    cfg = _load(args)
    reg = ModelRegistry(cfg.paths.checkpoints)
    dep = X.history(
        cfg, args.market, start=args.start, end=args.end, target=args.target,
        initial=args.initial, contribution=args.contribution,
        regime_mode=args.regime_mode, offline=args.offline, registry=reg,
    )
    print(f"\n{dep.market.label}  —  deployed {dep.dates[0]:%b %Y} → {dep.dates[-1]:%b %Y} "
          f"({dep.horizon_years}y)")
    print(f"data: {dep.source} · regimes: {dep.regime_mode} (no look-ahead) · "
          f"goal: ${dep.target:,.0f}\n")

    sc = scorecard(dep)
    show = sc.copy()
    show["Final balance"] = show["Final balance"].map(lambda x: f"${x:,.0f}")
    show["Reached goal"] = show["Reached goal"].map(lambda b: "yes" if b else "no")
    for c in ("Max drawdown", "Worst 12-month", "Growth rate (CAGR)", "Avg equity"):
        show[c] = show[c].map(lambda x: f"{x * 100:.0f}%")
    _print_df(show)

    print("\nWhat the regime-aware agent did at the turning points:")
    for line in diary_sentences(dep):
        print("  • " + line.replace("**", ""))
    print("\nNote: terminal wealth rewards whoever held the most equity in a rising market; "
          "the goal-based agents optimize the *chance of reaching the goal with low drawdown*.")


def cmd_calibrate(args):
    cfg = _load(args)
    calib = X.calibrate(cfg, offline=args.offline, save_path=args.out)
    for nm, mu, sg in zip(calib.names, calib.mu_annual.ravel(), calib.sigma_annual.ravel()):
        print(f"  {nm:9s} mu={mu:+.3f}  sigma={sg:.3f}")


def cmd_ls(args):
    cfg = _load(args)
    for m in ModelRegistry(cfg.paths.checkpoints).list():
        print(f"  {m.name:28s} kind={m.kind:24s} P(goal)={m.metrics.get('p_goal', float('nan')):.3f}  {m.created}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gbwm", description="Regime-Aware GBWM simulator")
    p.add_argument("--config", help="path to a YAML config (default: bundled configs/default.yaml)")
    p.add_argument("--base", help="optional base config to merge under --config")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train/solve an agent and store a checkpoint")
    t.add_argument("--agent", default="all",
                   choices=["g_learner", "regime_aware_g_learner", "ppo", "sac", "all"])
    t.add_argument("--timesteps", type=int, default=None, help="override RL training timesteps")
    t.add_argument("--name", default=None, help="checkpoint name (default: agent key)")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", help="compare strategies via Monte-Carlo")
    e.add_argument("--agents", default="all", help="'all','baselines', or comma list of keys")
    e.add_argument("--episodes", type=int, default=None)
    e.set_defaults(func=cmd_evaluate)

    b = sub.add_parser("backtest", help="roll a single path and explain it")
    b.add_argument("--agent", default="regime_aware_g_learner")
    b.add_argument("--seed", type=int, default=None)
    b.set_defaults(func=cmd_backtest)

    h = sub.add_parser("history", help="honest out-of-sample backtest on real market history")
    h.add_argument("--market", default="sp500",
                   help="market key: sp500, nasdaq, kospi, nikkei, hangseng, dax, dowjones, mag7")
    h.add_argument("--start", default=None, help="ISO start date (default: market inception, ~1999)")
    h.add_argument("--end", default=None, help="ISO end date (default: latest available)")
    h.add_argument("--initial", type=float, default=100_000.0, help="starting wealth")
    h.add_argument("--contribution", type=float, default=500.0, help="monthly contribution")
    h.add_argument("--target", type=float, default=600_000.0, help="goal wealth")
    h.add_argument("--regime-mode", dest="regime_mode", default="prior",
                   choices=["prior", "causal_calibrate"],
                   help="prior = economic regimes (default, robust); causal_calibrate = HMM on pre-start data")
    h.add_argument("--offline", action="store_true", help="use synthetic data (no network)")
    h.set_defaults(func=cmd_history)

    c = sub.add_parser("calibrate", help="estimate regimes from market data")
    c.add_argument("--offline", action="store_true", help="use synthetic data (no network)")
    c.add_argument("--out", default=None, help="write a calibrated config to this path")
    c.set_defaults(func=cmd_calibrate)

    ls = sub.add_parser("ls", help="list stored checkpoints")
    ls.set_defaults(func=cmd_ls)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
