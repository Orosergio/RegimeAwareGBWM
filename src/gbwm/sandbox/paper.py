"""Step-by-step paper-trading engine for the Live Bank / sandbox.

A :class:`LiveSession` wraps one *wallet*: a starting balance, a goal, a chosen
market (real history or a simulated future) and a **goal-based reinforcement-
learning allocator**. You drive it one month at a time. At every step it asks the
*same* regime-aware RL policy the backtest uses for the allocation — given the
current balance, time left and the **online** regime belief (filtered from past
returns only, never the future) — applies that month's realized return, and
records a fully transparent decision: what it saw, what it now holds, **how much
money it moved into/out of each asset**, and *why* (a plain-English rationale).

Between steps you may :meth:`deposit` or :meth:`withdraw` capital, like a
(simulated) bank account. A buy-&-hold benchmark of the market index is tracked
alongside so you can see the AI vs. "just buying the market". Optional trading
costs (fee + slippage) can be switched on to show how turnover eats returns.

No real money is involved — this is an educational simulation. The class holds no
Streamlit dependency, so it is fully unit-testable and reusable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# The markets you can play on (single-index, basket, or the 4-asset panel)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SandboxMarket:
    """A market the sandbox can stream returns from."""

    key: str
    label: str
    note: str
    kind: str  # "single" (one risky index/basket) | "multi" (S&P+intl+bonds+gold)
    source_key: str = ""  # the backtesting market key for kind="single"


SANDBOX_MARKETS: list[SandboxMarket] = [
    SandboxMarket("multi", "Multi-asset portfolio 🌍 (S&P + international + bonds + gold)",
                  "The most complete plan: it splits money across 4 assets at once and rotates "
                  "into bonds/gold during storms.", kind="multi"),
    SandboxMarket("sp500", "S&P 500 🇺🇸 (500 big US companies)",
                  "The thermometer of the US stock market. Lived through dot-com, 2008 and COVID.",
                  kind="single", source_key="sp500"),
    SandboxMarket("nasdaq", "NASDAQ 💻 (US tech)",
                  "Very tech-heavy: the biggest dot-com bubble and the AI-era melt-up.",
                  kind="single", source_key="nasdaq"),
    SandboxMarket("kospi", "KOSPI 🇰🇷 (South Korea)",
                  "Korea's main index — hit hard by 2000 and 2008.",
                  kind="single", source_key="kospi"),
    SandboxMarket("nikkei", "Nikkei 225 🇯🇵 (Japan)",
                  "Japan — a very different regime history, flat for years.",
                  kind="single", source_key="nikkei"),
    SandboxMarket("hangseng", "Hang Seng 🇭🇰 (Hong Kong)",
                  "Hong Kong large caps — the China cycle, 2008 and the 2015 swings.",
                  kind="single", source_key="hangseng"),
    SandboxMarket("dax", "DAX 🇩🇪 (Germany)",
                  "Germany's benchmark index — a core European market.",
                  kind="single", source_key="dax"),
    SandboxMarket("mag7", "Magnificent 7 🚀 (mega-cap tech)",
                  "AAPL/MSFT/GOOGL/AMZN/NVDA/META/TSLA equal-weight — only complete from 2013.",
                  kind="single", source_key="mag7"),
]


def list_sandbox_markets() -> list[SandboxMarket]:
    return list(SANDBOX_MARKETS)


def _market_by_key(key: str) -> SandboxMarket:
    for m in SANDBOX_MARKETS:
        if m.key == key:
            return m
    raise KeyError(f"unknown sandbox market '{key}'")


# Friendly "weather" word per hidden regime (used in the rationale).
WEATHER_WORD = {"bull": "sunny", "stable": "calm", "high_vol": "choppy", "bear": "stormy"}


# --------------------------------------------------------------------------- #
# Records: transactions (cash flows) and decisions (what the model did)
# --------------------------------------------------------------------------- #
@dataclass
class Transaction:
    step: int
    label: str          # e.g. "Month 12 · Mar 2009" or "Start"
    kind: str           # "Open" | "Deposit" | "Withdraw" | "Auto-deposit"
    amount: float       # signed (+in / -out)
    balance_after: float


@dataclass
class Decision:
    """A fully transparent record of one month's allocation decision."""

    step: int
    label: str
    regime: str                 # most-likely current regime
    risk_belief: float          # P(bear)+P(high-vol) — the 'bad weather' gauge
    equity: float               # fraction in risky assets (1 - cash) this month
    prev_equity: float          # risky fraction last month
    invested: float             # money being allocated this month (NT$)
    alloc_money: list[float]    # NT$ held in each risky asset
    cash_money: float           # NT$ held in cash
    trades_money: list[float]   # NT$ bought (+) / sold (-) per risky asset vs last month
    cash_trade: float           # NT$ moved into (+) / out of (-) cash
    month_return: float         # realized portfolio return this month
    month_pnl: float            # NT$ gained/lost this month from the market
    rationale: str              # plain-English "why"
    flags: list[str] = field(default_factory=list)  # milestones: storm/recovery/goal/big_drop


# --------------------------------------------------------------------------- #
# The live session (one wallet)
# --------------------------------------------------------------------------- #
@dataclass
class LiveSession:
    """One wallet, advanced one month at a time by a goal-based RL allocator."""

    name: str
    mm: object                       # MarketModel
    policy: object                   # Policy (regime-aware, goal-based RL)
    returns: np.ndarray              # (T, A) realized SIMPLE returns to stream
    target: float
    asset_labels: list[str]
    asset_keys: list[str]
    regime_names: list[str]
    dates: pd.DatetimeIndex | None = None
    source: str = "synthetic"        # cache | yfinance | synthetic
    mode_label: str = ""             # human description of the data stream
    initial: float = 30_000.0
    recurring: float = 0.0           # per-step auto contribution
    currency: str = "NT$"
    fee_bps: float = 0.0             # trading fee, basis points of turnover
    slippage_bps: float = 0.0        # slippage, basis points of turnover

    # --- mutable state (set in __post_init__) ------------------------------ #
    t: int = field(default=0, init=False)
    wealth: float = field(default=0.0, init=False)
    bench_wealth: float = field(default=0.0, init=False)
    total_deposited: float = field(default=0.0, init=False)
    total_withdrawn: float = field(default=0.0, init=False)
    bench_withdrawn: float = field(default=0.0, init=False)
    total_costs: float = field(default=0.0, init=False)
    transactions: list[Transaction] = field(default_factory=list, init=False)
    decisions: list[Decision] = field(default_factory=list, init=False)
    wealth_hist: list[float] = field(default_factory=list, init=False)
    bench_hist: list[float] = field(default_factory=list, init=False)
    weights_hist: list[np.ndarray] = field(default_factory=list, init=False)
    belief_hist: list[np.ndarray] = field(default_factory=list, init=False)
    port_ret_hist: list[float] = field(default_factory=list, init=False)
    deposit_marks: list[tuple[int, float]] = field(default_factory=list, init=False)
    action_log: list[dict] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.returns = np.asarray(self.returns, dtype=float)
        if self.returns.ndim == 1:
            self.returns = self.returns[:, None]
        self.T, self.A = self.returns.shape
        self.filt = self.mm.make_filter()
        self.alpha = self.filt.reset()
        self.policy.reset(1)
        self.wealth = float(self.initial)
        self.bench_wealth = float(self.initial)
        self.total_deposited = float(self.initial)
        self._cost_rate = (float(self.fee_bps) + float(self.slippage_bps)) / 10_000.0
        self._prev_w = np.zeros(self.A)  # start fully in cash -> first month "invests"
        self.wealth_hist = [self.wealth]
        self.bench_hist = [self.bench_wealth]
        self.transactions = [Transaction(0, self._label(0), "Open",
                                         float(self.initial), self.wealth)]

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _label(self, step: int) -> str:
        if self.dates is not None and 0 <= step < len(self.dates):
            return f"Month {step} · {self.dates[step]:%b %Y}"
        if self.dates is not None and step >= len(self.dates):
            return f"Month {step} · {self.dates[-1]:%b %Y}"
        years = step // 12
        return f"Month {step}" + (f" (year {years})" if years else " (start)")

    @property
    def done(self) -> bool:
        return self.t >= self.T

    @property
    def progress(self) -> float:
        """Balance as a fraction of the goal (clipped at 1.0 for display)."""
        return float(min(self.wealth / self.target, 1.0)) if self.target > 0 else 0.0

    @property
    def reached(self) -> bool:
        return self.wealth >= self.target

    @property
    def net_pnl(self) -> float:
        """Profit/loss the *market* generated (excludes your own cash flows)."""
        return float(self.wealth + self.total_withdrawn - self.total_deposited)

    @property
    def net_pnl_pct(self) -> float:
        base = self.total_deposited
        return float(self.net_pnl / base) if base > 0 else 0.0

    @property
    def bench_pnl(self) -> float:
        """Profit/loss a plain buy-&-hold of the index would have made."""
        return float(self.bench_wealth + self.bench_withdrawn - self.total_deposited)

    @property
    def when(self) -> str:
        return self._label(self.t)

    def current_belief(self) -> np.ndarray:
        """Decision-time regime posterior (predict step), shape (K,)."""
        return self.filt.predict(self.alpha)

    def current_weights(self) -> np.ndarray:
        """Risky-asset weights the policy would pick *now*, shape (A,)."""
        from gbwm.policies.base import DecisionContext

        belief = self.current_belief()
        ctx = DecisionContext(
            step=self.t, n_steps=self.T, wealth=np.array([self.wealth]),
            target=self.target, belief=belief[None, :], n_assets=self.A,
            regime_names=self.regime_names,
        )
        return self.policy.weights(ctx)[0]

    def current_allocation(self) -> dict[str, float]:
        """Full allocation incl. cash: {asset_label: weight, 'Cash 💵': cash}."""
        w = self.current_weights()
        out = {lbl: float(wi) for lbl, wi in zip(self.asset_labels, w)}
        out["Cash 💵"] = float(max(0.0, 1.0 - w.sum()))
        return out

    def current_regime(self) -> str:
        return self.regime_names[int(np.argmax(self.current_belief()))]

    def current_risk_belief(self) -> float:
        """P(bear) + P(high-vol) right now — the 'bad weather' gauge."""
        bel = self.current_belief()
        idx = [i for i, n in enumerate(self.regime_names) if n in ("bear", "high_vol")]
        return float(bel[idx].sum())

    def _risk_belief_of(self, belief: np.ndarray) -> float:
        idx = [i for i, n in enumerate(self.regime_names) if n in ("bear", "high_vol")]
        return float(belief[idx].sum())

    def equity_now(self) -> float:
        """Fraction currently invested in risky assets (1 - cash)."""
        return float(self.current_weights().sum())

    def _stock_idx(self) -> list[int]:
        return [i for i, k in enumerate(self.asset_keys)
                if k in ("us_equity", "intl_equity")] or list(range(self.A))

    def stock_fraction(self) -> float:
        """Fraction in *stocks* (equity sleeves); on a single market = equity."""
        return float(self.current_weights()[self._stock_idx()].sum())

    def safe_haven_fraction(self) -> float:
        """Fraction in bonds + gold (0 on a single-equity market)."""
        idx = [i for i, k in enumerate(self.asset_keys) if k in ("bonds", "gold")]
        return float(self.current_weights()[idx].sum()) if idx else 0.0

    # ------------------------------------------------------------------ #
    # rationale — the plain-English "why" of each decision
    # ------------------------------------------------------------------ #
    def _make_rationale(self, equity: float, prev_equity: float, risk_belief: float,
                        regime: str, gap_frac: float) -> str:
        weather = WEATHER_WORD.get(regime, regime)
        d = equity - prev_equity
        if abs(d) < 0.03:
            act = f"kept risky assets steady (~{equity * 100:.0f}%)"
        elif d > 0:
            act = f"raised risky assets to {equity * 100:.0f}%"
        else:
            act = f"cut risky assets to {equity * 100:.0f}%"
        if risk_belief >= 0.5:
            why = "to shield your balance — it senses stormy weather"
        elif gap_frac > 0.2:
            why = "to push toward your goal while the weather allows"
        elif gap_frac < 0.0:
            why = "you're past the goal, so it locks in the gains"
        else:
            why = "to balance growth and safety"
        return f"Weather looks {weather}. It {act} {why}."

    # ------------------------------------------------------------------ #
    # the loop
    # ------------------------------------------------------------------ #
    def step(self) -> bool:
        """Advance one month. Returns False if the history is exhausted."""
        if self.done:
            return False
        t = self.t
        belief = self.current_belief()
        w = self.current_weights()
        cash_w = max(0.0, 1.0 - w.sum())
        r = self.returns[t]
        port_ret = float(w @ r + (1.0 - w.sum()) * self.mm.cash_return)

        # recurring contribution (added before growth, exactly like the backtest)
        if self.recurring:
            self.wealth += self.recurring
            self.total_deposited += self.recurring
            self.bench_wealth += self.recurring
        invested = self.wealth  # the money being allocated this month

        # trading cost on turnover (opt-in; 0 by default keeps backtest fidelity)
        prev_w = self._prev_w
        prev_cash = max(0.0, 1.0 - prev_w.sum())
        turnover = 0.5 * (np.abs(w - prev_w).sum() + abs(cash_w - prev_cash))  # one-way
        cost_rate = turnover * self._cost_rate
        cost_money = invested * cost_rate
        self.total_costs += cost_money

        self.wealth = invested * (1.0 + port_ret) * (1.0 - cost_rate)
        month_pnl = self.wealth - invested
        # the contribution IS a cash movement — record it so the statement reconciles
        if self.recurring:
            self.transactions.append(Transaction(
                self.t, self._label(self.t), "Auto-deposit", float(self.recurring), self.wealth))
        # benchmark: buy & hold the index (first risky asset), same cash flows, no costs
        self.bench_wealth = self.bench_wealth * (1.0 + float(r[0]))
        # the stats series uses the NET (after-cost) return so Sharpe/vol/drawdown
        # reflect the cost drag (cost_rate == 0 by default -> identical to pre-cost).
        net_ret = (1.0 + port_ret) * (1.0 - cost_rate) - 1.0

        # --- build the transparent decision record -------------------------- #
        regime = self.regime_names[int(np.argmax(belief))]
        rb = self._risk_belief_of(belief)
        gap_frac = (self.target - invested) / self.target if self.target > 0 else 0.0
        alloc_money = (w * invested)
        trades = (w - prev_w) * invested
        cash_trade = (cash_w - prev_cash) * invested
        flags: list[str] = []
        prev_rb = self.decisions[-1].risk_belief if self.decisions else 0.0
        if rb >= 0.6 > prev_rb:
            flags.append("storm")
        if rb < 0.4 <= prev_rb:
            flags.append("clearing")
        if port_ret <= -0.10:
            flags.append("big_drop")
        prev_wealth = self.wealth_hist[-1]
        if self.wealth >= self.target > prev_wealth:
            flags.append("goal")
        self.decisions.append(Decision(
            step=t, label=self._label(t), regime=regime, risk_belief=rb,
            equity=float(w.sum()), prev_equity=float(prev_w.sum()), invested=float(invested),
            alloc_money=[float(x) for x in alloc_money], cash_money=float(cash_w * invested),
            trades_money=[float(x) for x in trades], cash_trade=float(cash_trade),
            month_return=port_ret, month_pnl=float(month_pnl),
            rationale=self._make_rationale(float(w.sum()), float(prev_w.sum()), rb, regime, gap_frac),
            flags=flags,
        ))

        self.weights_hist.append(w)
        self.belief_hist.append(belief)
        self.port_ret_hist.append(net_ret)
        self._prev_w = w
        self.alpha = self.filt.update(self.alpha, self.mm.project_obs(np.log1p(r)))
        self.t += 1
        self.wealth_hist.append(self.wealth)
        self.bench_hist.append(self.bench_wealth)
        self.action_log.append({"type": "step"})
        return True

    def advance(self, n: int) -> int:
        """Advance up to ``n`` months; returns how many actually ran."""
        done = 0
        for _ in range(max(0, int(n))):
            if not self.step():
                break
            done += 1
        return done

    def deposit(self, amount: float, _log: bool = True) -> None:
        amount = float(max(0.0, amount))
        if amount <= 0:
            return
        self.wealth += amount
        self.bench_wealth += amount
        self.total_deposited += amount
        self.wealth_hist[-1] = self.wealth
        self.bench_hist[-1] = self.bench_wealth
        self.deposit_marks.append((self.t, amount))
        self.transactions.append(Transaction(
            self.t, self._label(self.t), "Deposit", amount, self.wealth))
        if _log:
            self.action_log.append({"type": "deposit", "amount": amount})

    def withdraw(self, amount: float, _log: bool = True) -> float:
        """Withdraw up to the available balance; returns the amount taken out."""
        amount = float(min(max(0.0, amount), self.wealth))
        if amount <= 0:
            return 0.0
        self.wealth -= amount
        bench_out = min(amount, self.bench_wealth)  # the benchmark may hold less
        self.bench_wealth -= bench_out
        self.bench_withdrawn += bench_out
        self.total_withdrawn += amount
        self.wealth_hist[-1] = self.wealth
        self.bench_hist[-1] = self.bench_wealth
        self.deposit_marks.append((self.t, -amount))
        self.transactions.append(Transaction(
            self.t, self._label(self.t), "Withdraw", -amount, self.wealth))
        if _log:
            self.action_log.append({"type": "withdraw", "amount": amount})
        return amount

    def replay(self, actions: list[dict]) -> None:
        """Re-apply a recorded action log to restore a saved session exactly."""
        for a in actions:
            kind = a.get("type")
            if kind == "step":
                self.step()
            elif kind == "deposit":
                self.deposit(float(a["amount"]))
            elif kind == "withdraw":
                self.withdraw(float(a["amount"]))

    # ------------------------------------------------------------------ #
    # views for plotting / metrics
    # ------------------------------------------------------------------ #
    def wealth_curve(self) -> np.ndarray:
        return np.asarray(self.wealth_hist, dtype=float)

    def bench_curve(self) -> np.ndarray:
        return np.asarray(self.bench_hist, dtype=float)

    def equity_curve(self) -> np.ndarray:
        """Total risky fraction at each *taken* step, shape (t,)."""
        if not self.weights_hist:
            return np.empty(0)
        return np.asarray(self.weights_hist, dtype=float).sum(axis=1)

    def belief_curve(self) -> np.ndarray:
        if not self.belief_hist:
            return np.empty((0, len(self.regime_names)))
        return np.asarray(self.belief_hist, dtype=float)

    def asset_money_matrix(self) -> np.ndarray:
        """NT$ held in each asset + cash at every taken step, shape (t, A+1)."""
        if not self.decisions:
            return np.empty((0, self.A + 1))
        return np.array([d.alloc_money + [d.cash_money] for d in self.decisions], dtype=float)

    def growth_index(self) -> np.ndarray:
        """Growth of 1 unit invested (cash flows removed) — for honest path risk."""
        pr = np.asarray(self.port_ret_hist, dtype=float)
        return np.concatenate([[1.0], np.cumprod(1.0 + pr)]) if pr.size else np.array([1.0])

    def metrics(self) -> dict:
        """Risk/return stats from the realized monthly returns (no cash-flow noise)."""
        pr = np.asarray(self.port_ret_hist, dtype=float)
        rf = self.mm.cash_return
        gi = self.growth_index()
        peaks = np.maximum.accumulate(gi)
        mdd = float(((peaks - gi) / peaks).max()) if gi.size > 1 else 0.0
        if pr.size < 2 or pr.std() == 0:
            return {"ann_return": float(pr.mean() * 12) if pr.size else 0.0,
                    "ann_vol": 0.0, "sharpe": 0.0, "max_drawdown": mdd}
        sd = pr.std(ddof=1)
        return {"ann_return": float(pr.mean() * 12),
                "ann_vol": float(sd * np.sqrt(12)),
                "sharpe": float((pr.mean() - rf) / sd * np.sqrt(12)),
                "max_drawdown": mdd}

    def date_axis(self) -> list:
        """X labels for the wealth curve (length t+1)."""
        if self.dates is not None:
            return [self.dates[min(i, len(self.dates) - 1)] for i in range(self.t + 1)]
        return list(range(self.t + 1))


# --------------------------------------------------------------------------- #
# Builders — produce (mm, policy, returns, dates, labels) for a wallet
# --------------------------------------------------------------------------- #
def _apply_goal(cfg, initial: float, target: float, recurring: float) -> None:
    """Write the wallet's goal economics into the config BEFORE the policy solves,
    so the goal-based RL allocator is optimized for *this* wallet's goal and
    contribution (not the config defaults). The DP bakes target + contribution in
    at solve time, so this must happen before ``from_config``."""
    cfg.goal.initial_wealth = float(initial)
    cfg.goal.target_wealth = float(target)
    cfg.goal.contribution = float(recurring)


def _build_single(market: SandboxMarket, start: str | None, horizon_years: int | None,
                  offline: bool, initial: float, target: float, recurring: float):
    from gbwm.backtesting.historical import fetch_market_returns
    from gbwm.config import default_config
    from gbwm.experiments import clone_config
    from gbwm.policies.g_learner import RegimeAwareGLearner
    from gbwm.simulation.regimes import MarketModel

    dates, r_log, source = fetch_market_returns(
        market.source_key, start=start, offline=offline)
    base = default_config()
    spy = base.steps_per_year
    n_years = horizon_years or max(1, len(r_log) // spy)
    keep = min(n_years, len(r_log) // spy) * spy  # whole years only -> T == policy.total_steps
    if keep < spy:
        raise ValueError(
            f"only {len(r_log)} months of {market.label} from {start or 'inception'}.")
    r_log = r_log[:keep]
    dates = dates[:keep]

    cfg = clone_config(base)
    cfg.goal.horizon_years = keep // spy
    _apply_goal(cfg, initial, target, recurring)
    mm = MarketModel.from_config(cfg.market)
    policy = RegimeAwareGLearner.from_config(cfg, mm)
    returns = np.expm1(r_log)[:, None]
    return mm, policy, returns, pd.DatetimeIndex(dates), [market.label.split(" (")[0]], \
        [market.source_key], mm.regime_names, source


def _build_multi(start: str | None, horizon_years: int | None, offline: bool,
                 initial: float, target: float, recurring: float):
    from gbwm.backtesting.multiasset import default_multi_asset_config, fetch_asset_panel
    from gbwm.experiments import clone_config
    from gbwm.policies.multi_asset import RegimeAwareMultiAsset
    from gbwm.simulation.regimes import MarketModel

    base = default_multi_asset_config()
    spy = base.steps_per_year
    panel = fetch_asset_panel(offline=offline, cache_dir=base.data.cache_dir)

    if start is not None:
        start_idx = int(np.searchsorted(panel.dates.values,
                                        np.datetime64(pd.Timestamp(start))))
    else:
        start_idx = 0
    realized = panel.r_log[start_idx:]
    realized_dates = panel.dates[start_idx:]
    if len(realized) < spy:
        raise ValueError("not enough common history for the 4 assets from that date.")
    n_years = horizon_years or max(1, len(realized) // spy)
    keep = min(n_years, len(realized) // spy) * spy  # whole years only
    realized = realized[:keep]
    realized_dates = realized_dates[:keep]

    cfg = clone_config(base)
    cfg.goal.horizon_years = keep // spy
    _apply_goal(cfg, initial, target, recurring)
    mm = MarketModel.from_config(cfg.market)
    policy = RegimeAwareMultiAsset.from_config(cfg, mm)
    returns = np.expm1(realized)
    short_labels = [lbl.split(" (")[0] for lbl in panel.labels]
    return mm, policy, returns, pd.DatetimeIndex(realized_dates), short_labels, \
        panel.keys, mm.regime_names, panel.source


def _build_synthetic(market: SandboxMarket, horizon_years: int, seed: int,
                     initial: float, target: float, recurring: float):
    """A simulated future: fresh regime-switching path from the market model."""
    from gbwm.backtesting.multiasset import default_multi_asset_config
    from gbwm.config import default_config
    from gbwm.experiments import clone_config
    from gbwm.policies.g_learner import RegimeAwareGLearner
    from gbwm.policies.multi_asset import RegimeAwareMultiAsset
    from gbwm.simulation.regimes import MarketModel

    base = default_multi_asset_config() if market.kind == "multi" else default_config()
    cfg = clone_config(base)
    cfg.goal.horizon_years = int(horizon_years)
    _apply_goal(cfg, initial, target, recurring)
    mm = MarketModel.from_config(cfg.market)
    rng = np.random.default_rng(seed)
    paths = mm.simulate(1, cfg.total_steps, rng, antithetic=False)
    returns = paths.risky_returns[0]  # (T, A)

    if market.kind == "multi":
        from gbwm.backtesting.multiasset import MULTI_ASSET_UNIVERSE
        policy = RegimeAwareMultiAsset.from_config(cfg, mm)
        labels = [a.label.split(" (")[0] for a in MULTI_ASSET_UNIVERSE]
        keys = [a.key for a in MULTI_ASSET_UNIVERSE]
    else:
        policy = RegimeAwareGLearner.from_config(cfg, mm)
        labels = [market.label.split(" (")[0]]
        keys = [market.source_key or market.key]
    return mm, policy, returns, None, labels, keys, mm.regime_names, "synthetic"


def build_session(
    *,
    name: str,
    market_key: str,
    data_mode: str = "real",          # "real" | "synthetic"
    start: str | None = None,
    horizon_years: int | None = None,
    initial: float = 30_000.0,
    target: float = 100_000.0,
    recurring: float = 0.0,
    currency: str = "NT$",
    fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    offline: bool = False,
    seed: int = 0,
) -> LiveSession:
    """Construct a ready-to-play :class:`LiveSession` for one wallet.

    ``data_mode="real"`` streams real public market history (no look-ahead);
    ``"synthetic"`` streams a fresh Monte-Carlo future from the market model.
    """
    market = _market_by_key(market_key)
    if data_mode == "synthetic":
        mm, policy, returns, dates, labels, keys, regimes, source = _build_synthetic(
            market, horizon_years or 25, seed, initial, target, recurring)
        mode_label = f"Simulated future · {market.label.split(' (')[0]} (model's Monte-Carlo)"
    else:
        if market.kind == "multi":
            mm, policy, returns, dates, labels, keys, regimes, source = _build_multi(
                start, horizon_years, offline, initial, target, recurring)
        else:
            mm, policy, returns, dates, labels, keys, regimes, source = _build_single(
                market, start, horizon_years, offline, initial, target, recurring)
        span = ""
        if dates is not None and len(dates):
            span = f" · {dates[0]:%b %Y} → {dates[-1]:%b %Y}"
        mode_label = f"Real history · {market.label.split(' (')[0]}{span}"

    return LiveSession(
        name=name, mm=mm, policy=policy, returns=returns, target=float(target),
        asset_labels=labels, asset_keys=keys, regime_names=regimes, dates=dates,
        source=source, mode_label=mode_label, initial=float(initial),
        recurring=float(recurring), currency=currency,
        fee_bps=float(fee_bps), slippage_bps=float(slippage_bps),
    )


def restore_session(params: dict, action_log: list[dict]) -> LiveSession:
    """Rebuild a wallet from its creation ``params`` and replay its ``action_log``.

    Because the return stream is deterministic (real history, or a seeded
    synthetic future), replaying the recorded actions reproduces the exact state —
    the basis for save/load."""
    sess = build_session(**params)
    sess.replay(action_log or [])
    return sess
