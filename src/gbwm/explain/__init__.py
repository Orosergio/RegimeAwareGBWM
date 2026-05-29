"""Plain-language explainability advisor (rule-based default; LLM pluggable)."""
from gbwm.explain.base import Advisor, StepContext, EpisodeContext
from gbwm.explain.rule_based import RuleBasedAdvisor
from gbwm.explain.llm import LLMAdvisor, make_advisor

__all__ = ["Advisor","StepContext","EpisodeContext","RuleBasedAdvisor","LLMAdvisor","make_advisor"]
