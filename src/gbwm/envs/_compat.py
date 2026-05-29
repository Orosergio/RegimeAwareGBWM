"""Gymnasium compatibility shim.

If ``gymnasium`` is installed (the ``rl`` extra), we use it directly so the env
is fully compatible with Stable-Baselines3. In minimal environments (e.g. CI
without the RL extra) we fall back to a tiny ``Env``/``Box`` so the environment
*logic* remains importable and unit-testable headless.
"""

from __future__ import annotations

import numpy as np

try:  # real gymnasium when available
    import gymnasium as gym
    from gymnasium import spaces

    Env = gym.Env
    Box = spaces.Box
    HAS_GYMNASIUM = True
except Exception:  # pragma: no cover - exercised only without the rl extra
    HAS_GYMNASIUM = False

    class Box:  # minimal Box supporting what WealthEnv + tests need
        def __init__(self, low, high, shape=None, dtype=np.float32):
            low = np.asarray(low, dtype=dtype)
            high = np.asarray(high, dtype=dtype)
            if shape is None:
                shape = low.shape if low.ndim else high.shape
            self.low = np.broadcast_to(low, shape).astype(dtype)
            self.high = np.broadcast_to(high, shape).astype(dtype)
            self.shape = tuple(shape)
            self.dtype = dtype

        def sample(self, rng=None):
            r = rng or np.random.default_rng()
            return r.uniform(self.low, self.high).astype(self.dtype)

        def contains(self, x) -> bool:
            x = np.asarray(x)
            return bool(
                x.shape == self.shape
                and np.all(x >= self.low - 1e-6)
                and np.all(x <= self.high + 1e-6)
            )

    class _Spaces:
        Box = Box

    spaces = _Spaces()  # type: ignore

    class Env:  # minimal Gymnasium-like base
        metadata: dict = {"render_modes": []}
        observation_space = None
        action_space = None

        def reset(self, *, seed=None, options=None):  # noqa: D401
            raise NotImplementedError

        def step(self, action):
            raise NotImplementedError

        def render(self):
            return None

        def close(self):
            return None
