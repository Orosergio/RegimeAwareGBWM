"""Pluggable LLM advisor — same interface as the rule-based one, off by default.

Provide a ``complete_fn(prompt: str) -> str`` (wrapping any LLM/API client) to
enable it. With no function it transparently falls back to the rule-based advisor
so the demo is safe and free (ADR-006).
"""

from __future__ import annotations

from collections.abc import Callable

from gbwm.explain.base import Advisor, EpisodeContext, StepContext
from gbwm.explain.rule_based import RuleBasedAdvisor


class LLMAdvisor(Advisor):
    name = "llm"

    def __init__(self, complete_fn: Callable[[str], str] | None = None, model: str | None = None):
        self.complete_fn = complete_fn
        self.model = model
        self._fallback = RuleBasedAdvisor()

    def _prompt(self, facts: str) -> str:
        return (
            "You are a financial-planning assistant. In 2-3 plain sentences, explain "
            "the portfolio decision to a non-expert. Be concrete and avoid jargon.\n\n"
            f"Facts:\n{facts}"
        )

    def explain_step(self, ctx: StepContext) -> str:
        if self.complete_fn is None:
            return self._fallback.explain_step(ctx)
        facts = self._fallback.explain_step(ctx)  # grounded facts to rephrase
        return self.complete_fn(self._prompt(facts))

    def explain_episode(self, ctx: EpisodeContext) -> str:
        if self.complete_fn is None:
            return self._fallback.explain_episode(ctx)
        facts = self._fallback.explain_episode(ctx)
        return self.complete_fn(self._prompt(facts))


def make_advisor(kind: str = "rule_based", complete_fn=None, model=None) -> Advisor:
    """Factory: 'rule_based' (default) or 'llm' (needs complete_fn to be active)."""
    if kind == "llm":
        return LLMAdvisor(complete_fn=complete_fn, model=model)
    return RuleBasedAdvisor()
