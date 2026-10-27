"""Adoption policy for the Harness² edit adoption gate (policy as code)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from paved_gate.policy.models import EvaluatorConfig

Access = Literal["editable", "review", "protected"]
TriggerMode = Literal["allow", "review", "deny"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ComponentPolicy(_Strict):
    access: Access
    # fnmatch patterns relative to the harness root; `*` also matches `/`.
    paths: list[str] = Field(min_length=1)


class PathRule(_Strict):
    pattern: str
    access: Access
    reason: str | None = None


class SkillTriggerFormat(_Strict):
    """Where Harness² appends a trigger line when a skill is added, and the line's exact shape.

    Defaults are the OpenCode surface in harness2/substrate/opencode.py at commit 1cfe254.
    """

    file: str = "guardrails.md"
    template: str = "- When {desc}, load the `{name}` skill."
    skill_file: str = "skills/{name}/SKILL.md"

    @model_validator(mode="after")
    def _placeholders(self) -> Self:
        for field, text in (("template", self.template), ("skill_file", self.skill_file)):
            if "{name}" not in text:
                raise ValueError(f"skill_trigger_format.{field} must contain {{name}}")
        if "{desc}" not in self.template:
            raise ValueError("skill_trigger_format.template must contain {desc}")
        return self


class RegressionQuestion(_Strict):
    instructions: str
    criteria: list[str] = Field(min_length=2)  # ordered low to high; reported on a 1 based scale
    threshold: float  # strictly above this (1 based) rejects

    @model_validator(mode="after")
    def _threshold_on_scale(self) -> Self:
        if not 1 <= self.threshold <= len(self.criteria):
            raise ValueError(f"threshold must be within [1, {len(self.criteria)}]")
        return self


class ConstraintQuestion(_Strict):
    instructions: str
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)  # strictly above this rejects


class RegressionConfig(_Strict):
    regression: RegressionQuestion
    constraint_removal: ConstraintQuestion
    max_diff_chars: int = Field(default=16_000, gt=0)
    max_output_chars: int = Field(default=8_000, gt=0)


class AdoptionPolicy(_Strict):
    version: Literal[1]
    name: str
    components: dict[str, ComponentPolicy] = Field(min_length=1)  # first matching component wins
    rules: list[PathRule] = Field(default_factory=list)  # first match wins; overrides the component access
    unmapped_files: Access = "review"
    skill_trigger_lines: TriggerMode = "review"
    skill_trigger_format: SkillTriggerFormat = SkillTriggerFormat()
    # provider heuristic selects the local deterministic regression evaluator; jev selects the Jev adapter.
    # fail_mode closed rejects when the evaluator fails; open downgrades that to needs_review.
    evaluator: EvaluatorConfig = EvaluatorConfig(provider="heuristic")
    regression: RegressionConfig


def load_adoption_policy(path: str | Path) -> AdoptionPolicy:
    return AdoptionPolicy.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def adoption_policy_hash(policy: AdoptionPolicy) -> str:
    """sha256 of the validated, canonicalised policy (same scheme as the main gate's policy_hash)."""
    canonical = json.dumps(policy.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
