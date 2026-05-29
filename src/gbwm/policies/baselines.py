"""Classic, non-learning baselines: buy & hold, 60/40, linear glide path.

The glide path is the strongest baseline — it encodes the real-world intuition
that you should de-risk as the deadline approaches. Beating it is the bar.
"""

from __future__ import annotations

import numpy as np

from gbwm.config import Config
from gbwm.policies.base import DecisionContext, Policy, _equal_risky, policy_registry


@policy_registry.register("buy_and_hold")
class BuyAndHold(Policy):
    """Stay fully invested in risky assets (constant-mix at 100%)."""

    name = "Buy & Hold"

    def __init__(self, n_assets: int) -> None:
        self.n_assets = n_assets

    @classmethod
    def from_config(cls, config: Config) -> "BuyAndHold":
        return cls(config.market.n_risky)

    def weights(self, ctx: DecisionContext) -> np.ndarray:
        return _equal_risky(1.0, self.n_assets, len(ctx.wealth))


@policy_registry.register("sixty_forty")
class SixtyForty(Policy):
    """Constant 60% risky / 40% cash mix."""

    name = "60/40"

    def __init__(self, n_assets: int, risky_fraction: float = 0.6) -> None:
        self.n_assets = n_assets
        self.risky_fraction = risky_fraction

    @classmethod
    def from_config(cls, config: Config) -> "SixtyForty":
        frac = float(config.agents.extra.get("sixty_forty", {}).get("risky_fraction", 0.6))
        return cls(config.market.n_risky, frac)

    def weights(self, ctx: DecisionContext) -> np.ndarray:
        return _equal_risky(self.risky_fraction, self.n_assets, len(ctx.wealth))


@policy_registry.register("glide_path")
class GlidePath(Policy):
    """Linearly de-risk from ``start_equity`` to ``end_equity`` over the horizon."""

    name = "Glide Path"

    def __init__(self, n_assets: int, start_equity: float = 0.9, end_equity: float = 0.3) -> None:
        self.n_assets = n_assets
        self.start_equity = start_equity
        self.end_equity = end_equity

    @classmethod
    def from_config(cls, config: Config) -> "GlidePath":
        gp = config.agents.glide_path
        return cls(
            config.market.n_risky,
            float(gp.get("start_equity", 0.9)),
            float(gp.get("end_equity", 0.3)),
        )

    def weights(self, ctx: DecisionContext) -> np.ndarray:
        frac = self.start_equity + (self.end_equity - self.start_equity) * ctx.time_frac
        return _equal_risky(frac, self.n_assets, len(ctx.wealth))


BASELINE_NAMES = ["buy_and_hold", "sixty_forty", "glide_path"]
