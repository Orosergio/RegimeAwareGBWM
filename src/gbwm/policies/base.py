"""Policy contract (Strategy pattern, ADR / §4).

Every allocator — baselines, the G-Learner, and SB3 agents — implements the same
small interface, so the evaluation harness treats them interchangeably. The
contract is vectorized over Monte-Carlo paths for speed: a policy is asked for
the risky-asset weights of *all* paths at a given step, given each path's wealth
and regime belief.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from gbwm.registry import Registry


@dataclass
class DecisionContext:
    """Everything a policy may condition on at one decision step."""

    step: int  # current step index t in [0, n_steps)
    n_steps: int  # horizon T
    wealth: np.ndarray  # (N,) current wealth per path
    target: float  # goal G
    belief: np.ndarray  # (N, K) regime posterior per path
    n_assets: int  # number of risky assets A
    regime_names: list[str]

    @property
    def time_frac(self) -> float:
        return self.step / self.n_steps

    @property
    def gap(self) -> np.ndarray:
        """(N,) relative gap to goal, (G - W)/G."""
        return (self.target - self.wealth) / self.target


class Policy(ABC):
    """Base class for all allocation policies."""

    name: str = "policy"
    requires_belief: bool = False

    def reset(self, n_paths: int) -> None:  # noqa: D401 - optional hook
        """Reset any per-episode internal state (default: nothing)."""

    @abstractmethod
    def weights(self, ctx: DecisionContext) -> np.ndarray:
        """Return risky-asset weights, shape ``(N, n_assets)``; cash = 1 - sum."""

    # convenience for single-state callers (e.g. plotting one path)
    def weight_single(self, ctx: DecisionContext) -> np.ndarray:
        return self.weights(ctx)[0]


# Global registry so policies are addressable by name from config/CLI.
policy_registry: Registry[Policy] = Registry("policy")


def _equal_risky(fraction: float, n_assets: int, n_paths: int) -> np.ndarray:
    """Spread ``fraction`` of wealth equally across the risky assets."""
    per = fraction / n_assets
    return np.full((n_paths, n_assets), per, dtype=float)
