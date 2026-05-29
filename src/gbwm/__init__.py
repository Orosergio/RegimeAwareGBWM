"""Regime-Aware Goal-Based Wealth Management (GBWM) simulator.

A goal-based reinforcement-learning lab for dynamic portfolio allocation under
changing market regimes.  The public surface is intentionally small and
UI-agnostic so the same core powers the CLI, the test suite and the Streamlit
demo (and, later, any web API).
"""
from __future__ import annotations

__version__ = "0.1.0"

from gbwm.config import Config, default_config, load_config

__all__ = ["Config", "load_config", "default_config", "__version__"]
