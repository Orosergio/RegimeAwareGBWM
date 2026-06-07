"""Live paper-trading sandbox: engine fidelity, bank cash-flow accounting, and
the no-look-ahead invariant.

All tests are network-free (``offline=True`` / synthetic data or hand-built
arrays), so they run in CI without yfinance.
"""

import numpy as np

from gbwm.config import default_config
from gbwm.evaluation.harness import run_on_returns
from gbwm.policies import RegimeAwareGLearner
from gbwm.sandbox import (
    SANDBOX_MARKETS,
    LiveSession,
    build_session,
    list_sandbox_markets,
    restore_session,
)
from gbwm.simulation.regimes import MarketModel


def _fast_cfg(years=6):
    cfg = default_config()
    cfg.goal.horizon_years = years
    cfg.goal.initial_wealth = 30_000.0
    cfg.goal.target_wealth = 100_000.0
    cfg.goal.contribution = 200.0
    cfg.agents.g_learner.update(dict(n_wealth_bins=121, n_actions=15))
    return cfg


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_sandbox_market_registry():
    keys = {m.key for m in list_sandbox_markets()}
    assert {"multi", "sp500", "nasdaq", "kospi"} <= keys
    assert [m for m in SANDBOX_MARKETS if m.key == "multi"][0].kind == "multi"
    assert [m for m in SANDBOX_MARKETS if m.key == "sp500"][0].kind == "single"


# --------------------------------------------------------------------------- #
# Engine FIDELITY: stepping the live session reproduces the validated backtest
# exactly over the same realized returns (same belief filter, same cashflow).
# --------------------------------------------------------------------------- #
def test_live_session_matches_run_on_returns():
    cfg = _fast_cfg()
    mm = MarketModel.from_config(cfg.market)
    policy = RegimeAwareGLearner.from_config(cfg, mm)
    T = cfg.total_steps
    rng = np.random.default_rng(3)
    simple = 0.005 + rng.normal(0, 0.04, size=T)
    simple[30] = -0.22  # a crash month

    res = run_on_returns(policy, cfg, simple, market_model=mm)
    ref_wealth = res.histories["wealth"][0]
    ref_weights = res.histories["weights"][0]
    ref_belief = res.histories["belief"][0]

    sess = LiveSession(
        name="x", mm=mm, policy=policy, returns=simple, target=cfg.goal.target_wealth,
        asset_labels=["x"], asset_keys=["x"], regime_names=mm.regime_names,
        initial=cfg.goal.initial_wealth, recurring=cfg.goal.contribution,
    )
    sess.advance(T)

    assert sess.done
    assert np.allclose(sess.wealth_curve(), ref_wealth, rtol=1e-9, atol=1e-6)
    assert np.allclose(np.array(sess.weights_hist), ref_weights, rtol=1e-9, atol=1e-9)
    assert np.allclose(sess.belief_curve(), ref_belief, rtol=1e-9, atol=1e-9)


# --------------------------------------------------------------------------- #
# The goal-based policy must be solved for the WALLET's goal, not config defaults
# (regression for: build_session ignoring target/initial/recurring).
# --------------------------------------------------------------------------- #
def test_policy_is_solved_for_the_wallets_goal():
    common = dict(market_key="sp500", data_mode="synthetic", horizon_years=10,
                  initial=30_000, recurring=0, seed=1)
    near = build_session(name="near", target=40_000, **common)    # 30k is 0.75*goal
    far = build_session(name="far", target=1_000_000, **common)   # 30k is 0.03*goal

    # the policy's baked-in goal + contribution match THIS wallet (not the 250k/500 default)
    assert near.policy.G == 40_000 and far.policy.G == 1_000_000
    assert near.policy.c == 0 and far.policy.c == 0
    # a wallet far below its goal takes strictly more risk than one near its goal
    assert far.equity_now() > near.equity_now()


def test_recurring_contribution_reaches_the_policy():
    s = build_session(name="r", market_key="sp500", data_mode="synthetic",
                      horizon_years=8, initial=30_000, target=200_000, recurring=750, seed=2)
    assert s.policy.c == 750


# --------------------------------------------------------------------------- #
# Bank cash-flow accounting
# --------------------------------------------------------------------------- #
def test_deposit_withdraw_accounting_invariant():
    sess = build_session(name="W", market_key="sp500", data_mode="real",
                         start="2008-01-01", horizon_years=6, initial=30_000,
                         target=100_000, offline=True)
    sess.advance(24)
    sess.deposit(5_000)
    sess.advance(12)
    taken = sess.withdraw(3_000)
    assert taken == 3_000
    # net market P&L excludes your own cash flows, by construction
    assert np.isclose(sess.net_pnl, sess.wealth + sess.total_withdrawn - sess.total_deposited)
    assert sess.total_deposited >= 35_000  # 30k open + 5k deposit
    assert sess.total_withdrawn == 3_000
    # ledger records every cash flow + the opening
    kinds = [t.kind for t in sess.transactions]
    assert kinds[0] == "Open"
    assert "Deposit" in kinds and "Withdraw" in kinds


def test_withdraw_capped_at_balance():
    sess = build_session(name="W", market_key="sp500", data_mode="real",
                         start="2010-01-01", horizon_years=5, initial=30_000,
                         target=100_000, offline=True)
    balance_before = sess.wealth
    taken = sess.withdraw(10_000_000)  # ask for far more than we have
    assert np.isclose(taken, balance_before)  # capped at the available balance
    assert sess.wealth == 0.0
    assert sess.withdraw(100) == 0.0   # nothing left to take


# --------------------------------------------------------------------------- #
# Multi-asset allocation is a valid simplex (risky + cash == 1)
# --------------------------------------------------------------------------- #
def test_multi_asset_allocation_is_valid():
    sess = build_session(name="M", market_key="multi", data_mode="real",
                         start="2008-01-01", horizon_years=8, initial=30_000,
                         target=120_000, offline=True)
    assert sess.A == 4
    sess.advance(40)
    alloc = sess.current_allocation()
    assert np.isclose(sum(alloc.values()), 1.0, atol=1e-9)
    assert all(v >= -1e-9 for v in alloc.values())
    # stock + safe-haven sleeves never exceed the invested fraction
    assert sess.stock_fraction() + sess.safe_haven_fraction() <= sess.equity_now() + 1e-9


# --------------------------------------------------------------------------- #
# Synthetic future: open-ended, no dates, runs to the configured horizon
# --------------------------------------------------------------------------- #
def test_synthetic_future_runs_to_horizon():
    sess = build_session(name="F", market_key="multi", data_mode="synthetic",
                         horizon_years=20, initial=30_000, target=200_000, seed=11)
    assert sess.dates is None
    assert sess.T == 20 * 12
    sess.advance(10_000)  # more than available -> caps at T
    assert sess.done
    assert len(sess.wealth_curve()) == sess.T + 1
    assert sess.belief_curve().shape == (sess.T, len(sess.regime_names))


# --------------------------------------------------------------------------- #
# Transparency: every step logs a decision with trades that reconcile
# --------------------------------------------------------------------------- #
def test_decision_log_records_reconciling_trades():
    sess = build_session(name="D", market_key="multi", data_mode="real",
                         start="2008-01-01", horizon_years=8, initial=30_000,
                         target=150_000, offline=True)
    sess.advance(36)
    assert len(sess.decisions) == 36
    d0 = sess.decisions[0]
    # the first decision invests out of cash: its trades equal its target allocation
    assert np.allclose(d0.trades_money, d0.alloc_money, atol=1e-6)
    # money allocated across assets + cash equals the invested amount, each month
    for d in sess.decisions:
        assert np.isclose(sum(d.alloc_money) + d.cash_money, d.invested, rtol=1e-9, atol=1e-6)
        assert isinstance(d.rationale, str) and len(d.rationale) > 10
    # asset-money matrix has the right shape (assets + cash)
    assert sess.asset_money_matrix().shape == (36, sess.A + 1)


# --------------------------------------------------------------------------- #
# Benchmark shadow + risk metrics
# --------------------------------------------------------------------------- #
def test_benchmark_and_metrics_tracked():
    sess = build_session(name="B", market_key="sp500", data_mode="real",
                         start="2008-01-01", horizon_years=8, initial=30_000,
                         target=120_000, offline=True)
    sess.advance(60)
    assert len(sess.bench_curve()) == len(sess.wealth_curve())
    m = sess.metrics()
    assert {"ann_return", "ann_vol", "sharpe", "max_drawdown"} <= set(m)
    assert 0.0 <= m["max_drawdown"] <= 1.0
    assert m["ann_vol"] >= 0.0


# --------------------------------------------------------------------------- #
# Trading costs (opt-in) reduce wealth and never break fidelity at zero cost
# --------------------------------------------------------------------------- #
def test_costs_reduce_wealth():
    base = dict(market_key="sp500", data_mode="synthetic", horizon_years=10,
                initial=30_000, target=100_000, seed=5)
    free = build_session(name="free", fee_bps=0, slippage_bps=0, **base)
    pricey = build_session(name="pricey", fee_bps=50, slippage_bps=50, **base)
    free.advance(120)
    pricey.advance(120)
    assert pricey.total_costs > 0
    assert pricey.wealth < free.wealth  # turnover costs drag the balance down


# --------------------------------------------------------------------------- #
# Save / load: replaying the action log reproduces the exact state
# --------------------------------------------------------------------------- #
def test_restore_reproduces_state():
    params = dict(name="R", market_key="multi", data_mode="real", start="2008-01-01",
                  horizon_years=8, initial=30_000, target=150_000, recurring=200,
                  offline=True)
    sess = build_session(**params)
    sess.advance(20)
    sess.deposit(7_000)
    sess.advance(10)
    sess.withdraw(3_000)
    sess.advance(5)

    clone = restore_session(params, sess.action_log)
    assert clone.t == sess.t
    assert np.isclose(clone.wealth, sess.wealth, rtol=1e-9, atol=1e-6)
    assert np.isclose(clone.total_deposited, sess.total_deposited)
    assert np.isclose(clone.total_withdrawn, sess.total_withdrawn)
    assert np.allclose(clone.wealth_curve(), sess.wealth_curve(), rtol=1e-9, atol=1e-6)
    assert len(clone.decisions) == len(sess.decisions)


# --------------------------------------------------------------------------- #
# Statement reconciles: every cash movement (incl. recurring) is recorded
# --------------------------------------------------------------------------- #
def test_recurring_contributions_appear_in_ledger():
    s = build_session(name="r", market_key="sp500", data_mode="synthetic",
                      horizon_years=5, initial=30_000, target=100_000, recurring=500, seed=3)
    s.advance(60)
    auto = [t for t in s.transactions if t.kind == "Auto-deposit"]
    assert len(auto) == 60
    # sum of all signed ledger amounts == net cash you put in
    total_in = sum(t.amount for t in s.transactions)
    assert np.isclose(total_in, s.total_deposited - s.total_withdrawn)


# --------------------------------------------------------------------------- #
# Trading costs now flow into the risk/return metrics (not just wealth)
# --------------------------------------------------------------------------- #
def test_costs_show_in_metrics():
    base = dict(market_key="sp500", data_mode="synthetic", horizon_years=10,
                initial=30_000, target=100_000, seed=5)
    free = build_session(name="free", fee_bps=0, slippage_bps=0, **base); free.advance(120)
    pricey = build_session(name="pricey", fee_bps=80, slippage_bps=20, **base); pricey.advance(120)
    assert pricey.metrics()["ann_return"] < free.metrics()["ann_return"]


# --------------------------------------------------------------------------- #
# Benchmark P&L stays consistent even when you withdraw more than it holds
# --------------------------------------------------------------------------- #
def test_withdraw_keeps_benchmark_pnl_consistent():
    s = build_session(name="b", market_key="sp500", data_mode="synthetic",
                      horizon_years=5, initial=30_000, target=80_000, seed=2)
    s.advance(30)
    s.withdraw(s.wealth)  # take everything out of the AI wallet
    assert np.isfinite(s.bench_pnl)
    assert np.isclose(s.bench_pnl, s.bench_wealth + s.bench_withdrawn - s.total_deposited)
