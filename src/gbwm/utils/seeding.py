"""Deterministic seeding helpers.

We prefer explicit ``numpy.random.Generator`` objects threaded through the code
over global state, but :func:`set_global_seed` is provided for libraries (torch,
SB3) that read global seeds.
"""
from __future__ import annotations

import os
import random

import numpy as np


def make_rng(seed: int | None) -> np.random.Generator:
    """Return a fresh NumPy Generator (PCG64) for reproducible sampling."""
    return np.random.default_rng(seed)


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch global RNGs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # torch is an optional dependency (only needed for SB3 agents)
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # pragma: no cover - torch optional
        pass
