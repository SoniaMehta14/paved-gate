from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from paved_gate.policy.models import Policy


@dataclass(frozen=True)
class LoadedPolicy:
    policy: Policy
    policy_hash: str
    source: str


def policy_hash(policy: Policy) -> str:
    """sha256 of the validated, canonicalised policy, so formatting changes don't alter it."""
    canonical = json.dumps(policy.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def from_policy(policy: Policy, source: str = "<in-memory>") -> LoadedPolicy:
    return LoadedPolicy(policy=policy, policy_hash=policy_hash(policy), source=source)


def load_policy(path: str | Path) -> LoadedPolicy:
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    return from_policy(Policy.model_validate(data), source=str(p))
