"""Explainability layer (Provider/Strategy pattern, ADR-006).

A plain-language advisor turns the agent's state and decisions into sentences a
non-expert understands ("cut equity 70%→40% because bear probability rose
0.20→0.65"). The rule-based advisor is deterministic and the default; an LLM
advisor implements the same interface and is off by default (no API key in a
public demo).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

RISKY_REGIMES = {"bear", "high_vol"}


@dataclass
class StepContext:
    weights: np.ndarray
    prev_weights: np.ndarray
    belief: np.ndarray
    prev_belief: np.ndarray
    wealth: float
    target: float
    step: int
    n_steps: int
    steps_per_year: int
    regime_names: list[str]
    asset_names: list[str]

    @property
    def equity(self) -> float:
        return float(np.sum(self.weights))

    @property
    def prev_equity(self) -> float:
        return float(np.sum(self.prev_weights))

    @property
    def gap(self) -> float:
        return (self.target - self.wealth) / self.target

    @property
    def years_left(self) -> float:
        return (self.n_steps - self.step) / self.steps_per_year


@dataclass
class EpisodeContext:
    wealth: np.ndarray  # (T+1,)
    weights: np.ndarray  # (T, A)
    belief: np.ndarray  # (T, K)
    regime: np.ndarray  # (T,)
    target: float
    regime_names: list[str]
    asset_names: list[str]
    steps_per_year: int

    @classmethod
    def from_histories(cls, histories: dict, target: float, steps_per_year: int, path: int = 0):
        return cls(
            wealth=histories["wealth"][path],
            weights=histories["weights"][path],
            belief=histories["belief"][path],
            regime=histories["regime"][path],
            target=target,
            regime_names=histories["regime_names"],
            asset_names=histories["asset_names"],
            steps_per_year=steps_per_year,
        )


class Advisor(ABC):
    name: str = "advisor"

    @abstractmethod
    def explain_step(self, ctx: StepContext) -> str: ...

    @abstractmethod
    def explain_episode(self, ctx: EpisodeContext) -> str: ...
