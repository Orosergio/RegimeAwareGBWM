"""The goal-based terminal utility — the single source of truth for the objective.

Used by both the Gymnasium environment (terminal reward) and the G-Learner
(terminal value of the backward induction), so the learned policy and the
evaluated reward agree by construction.
"""

from __future__ import annotations

import numpy as np

from gbwm.config import RewardConfig


def terminal_utility(wealth, target: float, reward: RewardConfig):
    """Terminal utility of ending wealth relative to the goal ``target``.

    Works on scalars or NumPy arrays. The default ``'threshold'`` form directly
    rewards reaching the goal and penalizes relative shortfall (the headline
    P(goal) objective); ``'power'`` is CRRA utility; ``'linear'`` is relative
    terminal wealth.
    """
    rel = np.asarray(wealth, dtype=float) / target
    kind = reward.utility
    if kind == "linear":
        out = rel - 1.0
    elif kind == "power":
        ra = reward.risk_aversion
        rel = np.clip(rel, 1e-8, None)
        if abs(ra - 1.0) < 1e-8:
            out = np.log(rel)
        else:
            out = (rel ** (1.0 - ra) - 1.0) / (1.0 - ra)
    else:  # threshold
        success = (rel >= 1.0).astype(float)
        shortfall = np.clip(1.0 - rel, 0.0, None)
        overshoot = np.log(np.clip(rel, 1.0, None))
        out = (
            reward.goal_bonus * success
            - reward.shortfall_penalty * shortfall
            + reward.overshoot_credit * overshoot
        )
    return float(out) if np.isscalar(wealth) or np.ndim(wealth) == 0 else out
