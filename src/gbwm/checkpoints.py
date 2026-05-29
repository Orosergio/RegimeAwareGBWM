"""Versioned model registry (Repository pattern, ADR-002 / ADR-007).

Persists trained policies with metadata (kind, config hash, metrics, timestamp)
so the offline-trained agents can be loaded reproducibly by the demo. G-Learners
are stored as ``.npz``; SB3 agents delegate to their own ``save``/``load``.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from gbwm.config import Config

_GLEARNER_KINDS = {"GLearner", "RegimeAwareGLearner"}
_KIND_TO_REGISTRY = {"GLearner": "g_learner", "RegimeAwareGLearner": "regime_aware_g_learner"}


def config_hash(config: Config) -> str:
    blob = json.dumps(config.to_dict(), sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


@dataclass
class CheckpointMeta:
    name: str
    kind: str  # policy-registry key: g_learner | regime_aware_g_learner | ppo | sac
    created: str
    config_hash: str
    metrics: dict = field(default_factory=dict)


class ModelRegistry:
    def __init__(self, root: str = "artifacts/checkpoints") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def dir(self, name: str) -> Path:
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save(self, name: str, policy, config: Config, metrics: dict | None = None) -> CheckpointMeta:
        d = self.dir(name)
        cls = type(policy).__name__
        if cls in _GLEARNER_KINDS:
            policy.save(d / "model.npz")
            kind = _KIND_TO_REGISTRY[cls]
        elif hasattr(policy, "algo"):  # SB3Policy
            policy.save(str(d / "model"))
            kind = policy.algo
        else:
            raise TypeError(f"don't know how to persist policy of type {cls}")
        meta = CheckpointMeta(
            name=name,
            kind=kind,
            created=time.strftime("%Y-%m-%dT%H:%M:%S"),
            config_hash=config_hash(config),
            metrics=metrics or {},
        )
        (d / "meta.json").write_text(json.dumps(asdict(meta), indent=2))
        return meta

    def load(self, name: str, config: Config):
        d = self.root / name
        meta = json.loads((d / "meta.json").read_text())
        kind = meta["kind"]
        from gbwm.policies.g_learner import GLearner, RegimeAwareGLearner

        if kind == "g_learner":
            return GLearner.load(d / "model.npz", config)
        if kind == "regime_aware_g_learner":
            return RegimeAwareGLearner.load(d / "model.npz", config)
        if kind in ("ppo", "sac"):
            from gbwm.policies.rl_agents import PPOPolicy, SACPolicy

            cls = PPOPolicy if kind == "ppo" else SACPolicy
            return cls.from_checkpoint(str(d / "model"), config)
        raise ValueError(f"unknown checkpoint kind '{kind}'")

    def exists(self, name: str) -> bool:
        return (self.root / name / "meta.json").exists()

    def list(self) -> list[CheckpointMeta]:
        out = []
        for d in sorted(self.root.glob("*")):
            mp = d / "meta.json"
            if mp.exists():
                out.append(CheckpointMeta(**json.loads(mp.read_text())))
        return out
