"""Honest, out-of-sample historical backtesting.

The :mod:`gbwm.backtesting.historical` module answers the question *"if this RL
advisor had been switched on in 1999, how would it have steered the portfolio
through the dot-com crash, the GFC and COVID?"* — using **real** market history
and **no look-ahead** (the agent only ever sees past realized returns).
"""

from __future__ import annotations

from gbwm.backtesting.historical import (
    CRISES,
    LONG_HISTORY_MARKETS,
    Crisis,
    Deployment,
    Market,
    decision_diary,
    diary_sentences,
    fetch_market_returns,
    list_markets,
    run_deployment,
    scorecard,
    sequence_risk,
)
from gbwm.backtesting.multiasset import (
    MULTI_ASSET_UNIVERSE,
    STRATEGY_LABELS,
    AssetPanel,
    AssetSpec,
    MultiAssetDeployment,
    build_multi_asset_policies,
    default_multi_asset_config,
    effective_step_contribution,
    fetch_asset_panel,
    list_universe,
    multi_asset_diary,
    run_multi_asset_deployment,
)

__all__ = [
    "CRISES",
    "LONG_HISTORY_MARKETS",
    "Crisis",
    "Deployment",
    "Market",
    "decision_diary",
    "diary_sentences",
    "fetch_market_returns",
    "list_markets",
    "run_deployment",
    "scorecard",
    "sequence_risk",
    # multi-asset
    "MULTI_ASSET_UNIVERSE",
    "STRATEGY_LABELS",
    "AssetPanel",
    "AssetSpec",
    "MultiAssetDeployment",
    "build_multi_asset_policies",
    "default_multi_asset_config",
    "effective_step_contribution",
    "fetch_asset_panel",
    "list_universe",
    "multi_asset_diary",
    "run_multi_asset_deployment",
]
