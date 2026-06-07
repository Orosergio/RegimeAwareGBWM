"""Interactive *paper-trading* sandbox — a simulated bank you can play with.

Where :mod:`gbwm.backtesting` rolls every strategy over a whole history in one
shot, this module exposes the **same validated core** (the regime-aware policy,
the online Bayes belief filter, the market model) as a **step-by-step session**
you can drive live: advance one month, deposit or withdraw capital into a wallet,
and watch the reinforcement-learning agent re-allocate in real time.

It is deliberately money-currency-agnostic (the numbers are just numbers — the UI
labels them, e.g. NT$). Two return streams are supported, both with **no
look-ahead**:

* **real public history** — month-by-month returns of a world market (S&P 500,
  NASDAQ, KOSPI, Nikkei, Hang Seng, DAX, Magnificent 7) or the 4-asset
  S&P+international+bonds+gold panel; the agent only ever sees the past, and
* **simulated future** — fresh regime-switching paths drawn from the market
  model (Monte-Carlo), for an open-ended game where the model itself dictates how
  the market evolves.

No real money is involved; this is an educational simulation.
"""

from __future__ import annotations

from gbwm.sandbox.paper import (
    LiveSession,
    SANDBOX_MARKETS,
    Decision,
    SandboxMarket,
    Transaction,
    build_session,
    list_sandbox_markets,
    restore_session,
)

__all__ = [
    "LiveSession",
    "SANDBOX_MARKETS",
    "Decision",
    "SandboxMarket",
    "Transaction",
    "build_session",
    "list_sandbox_markets",
    "restore_session",
]
