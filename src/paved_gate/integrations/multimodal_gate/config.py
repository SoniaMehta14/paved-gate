"""Policy for the multimodal gate: typed checks over an image or PDF plus text (policy as code)."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from paved_gate.policy.models import EvaluatorConfig

Action = Literal["review", "block"]
_KEY = re.compile(r"^[A-Za-z0-9_-]{1,64}$")  # question keys the Decisions API accepts


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TruthCheck(_Strict):
    instructions: str
    expect: Literal["yes", "no"] = "yes"
    min_probability: float = Field(default=0.7, ge=0.0, le=1.0)  # probability of the expected answer
    on_fail: Action = "review"


class ChoiceCheck(_Strict):
    instructions: str
    criteria: dict[str, str] = Field(min_length=2)
    expect: str | None = None  # the option that must be chosen; the command line can set it with --expect
    min_confidence: float = Field(default=0.6, ge=0.0, le=1.0)  # probability of the chosen option
    on_fail: Action = "block"

    @model_validator(mode="after")
    def _expect_offered(self) -> Self:
        if self.expect is not None and self.expect not in self.criteria:
            raise ValueError(f"expect {self.expect!r} is not one of the options {sorted(self.criteria)}")
        return self


class ScoreCheck(_Strict):
    instructions: str
    criteria: list[str] = Field(min_length=2)  # ordered low to high
    min_level: float  # 1 based; the check fails below it
    on_fail: Action = "review"

    @model_validator(mode="after")
    def _level_on_scale(self) -> Self:
        if not 1 <= self.min_level <= len(self.criteria):
            raise ValueError(f"min_level must be within [1, {len(self.criteria)}]")
        return self


class MultimodalPolicy(_Strict):
    version: Literal[1]
    name: str
    evaluator: EvaluatorConfig = EvaluatorConfig(provider="gemini_decision")
    max_media_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    truth: dict[str, TruthCheck] = Field(default_factory=dict)
    choice: dict[str, ChoiceCheck] = Field(default_factory=dict)
    score: dict[str, ScoreCheck] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _keys(self) -> Self:
        keys = [*self.truth, *self.choice, *self.score]
        if not keys:
            raise ValueError("the policy needs at least one check")
        if len(keys) != len(set(keys)):
            raise ValueError("check names must be unique across truth, choice and score")
        bad = [k for k in keys if not _KEY.match(k)]
        if bad:
            raise ValueError(f"check names must match [A-Za-z0-9_-] and be 1 to 64 characters: {bad}")
        return self


def load_multimodal_policy(path: str | Path) -> MultimodalPolicy:
    return MultimodalPolicy.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def multimodal_policy_hash(policy: MultimodalPolicy) -> str:
    """sha256 of the validated, canonicalised policy (same scheme as the main gate's policy_hash)."""
    canonical = json.dumps(policy.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
