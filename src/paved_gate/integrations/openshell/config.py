"""Egress policy for the OpenShell supervisor middleware (policy as code)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from paved_gate.policy.models import ALL_DETECTORS, AuditConfig, DetectorName, EvaluatorConfig


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ScoredQuestion(_Strict):
    instructions: str
    criteria: list[str] = Field(min_length=2)  # ordered low -> high; reported on a 1-based scale
    threshold: float  # strictly above this (1-based) triggers the action

    @model_validator(mode="after")
    def _threshold_on_scale(self) -> Self:
        if not 1 <= self.threshold <= len(self.criteria):
            raise ValueError(f"threshold must be within [1, {len(self.criteria)}]")
        return self


class SensitiveQuestion(_Strict):
    instructions: str
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class EgressQuestions(_Strict):
    sensitive: SensitiveQuestion
    exfiltration: ScoredQuestion
    injection: ScoredQuestion  # asked of responses returned to the sandbox (tool results)
    tainted_exfiltration_threshold: float = Field(
        default=2.5, description="stricter exfiltration threshold after an injection was seen in this sandbox"
    )


class EgressPolicy(_Strict):
    version: Literal[1]
    name: str
    evaluator: EvaluatorConfig = EvaluatorConfig()
    local_detectors: list[DetectorName] = Field(default_factory=lambda: list(ALL_DETECTORS))
    deny_on_local_detection: bool = True
    mask_before_evaluate: bool = True
    response_mode: Literal["flag", "block"] = "flag"
    taint_ttl_seconds: int = Field(default=600, gt=0)
    max_inspect_chars: int = Field(default=16_000, gt=0)
    questions: EgressQuestions
    audit: AuditConfig = AuditConfig(path="./logs/openshell_egress.audit.jsonl")


# Per-sandbox overrides accepted in the OpenShell policy's `network_middlewares.<name>.config`.
SandboxMode = Literal["enforce", "audit"]
CONFIG_KEYS: dict[str, tuple[str, ...]] = {"mode": ("enforce", "audit"), "response_mode": ("flag", "block")}


def load_egress_policy(path: str | Path) -> EgressPolicy:
    return EgressPolicy.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def egress_policy_hash(policy: EgressPolicy) -> str:
    """sha256 of the validated, canonicalised policy (same scheme as the main gate's policy_hash)."""
    canonical = json.dumps(policy.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
