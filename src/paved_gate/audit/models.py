from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, JsonValue

from paved_gate.types import Outcome, Route


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class EvaluatorInfo(BaseModel):
    name: str
    model: str | None
    mode: str


class SensitiveInfo(BaseModel):
    flag: bool
    noul_probability: float | None
    local_detectors: list[str]
    masked_before_evaluate: bool


class LatencyInfo(BaseModel):
    evaluator_ms: float | None
    gate_ms: float


class UsageInfo(BaseModel):
    input_tokens: int | None
    output_tokens: int | None


class GateDecisionEntry(BaseModel):
    """One per request, allowed or blocked."""

    event: Literal["gate_decision"] = "gate_decision"
    ts: str = Field(default_factory=_now)
    request_id: str
    policy_name: str
    policy_hash: str
    evaluator: EvaluatorInfo
    outcome: Outcome
    route: Route | None
    target: str | None
    reasons: list[str]
    scores: dict[str, JsonValue]
    raw_evaluator_response: dict[str, JsonValue] | None
    sensitive: SensitiveInfo
    latency: LatencyInfo
    usage: UsageInfo
    payload_sha256: str
    payload_masked: str | None = None
    error: str | None = None


class SensitiveDataEvent(BaseModel):
    """Written in addition to the decision entry whenever PII/PHI is flagged."""

    event: Literal["sensitive_data"] = "sensitive_data"
    ts: str = Field(default_factory=_now)
    request_id: str
    policy_hash: str
    outcome: Outcome
    route: Route | None
    sources: list[str]
    noul_probability: float | None
    local_detectors: list[str]
    masked_before_evaluate: bool
    payload_sha256: str


class EgressDecisionEntry(BaseModel):
    """One OpenShell supervisor-middleware evaluation (an outbound request or a returned response).

    Never stores payload content: only its sha256, size, audit-safe finding types and scores.
    """

    event: Literal["openshell_egress"] = "openshell_egress"
    ts: str = Field(default_factory=_now)
    direction: Literal["request", "response"]
    request_id: str
    sandbox_id: str
    policy_name: str
    policy_hash: str
    host: str
    port: int
    method: str
    path: str
    decision: Literal["allow", "deny", "flag"]
    enforced: bool
    reason_code: str | None
    findings: list[str]
    scores: dict[str, float]
    evaluator: EvaluatorInfo | None
    tainted: bool
    latency: LatencyInfo
    body_sha256: str
    body_bytes: int
    error: str | None = None


class AdoptionComponentChange(BaseModel):
    component: str
    added: list[str]
    removed: list[str]
    modified: list[str]
    lines_added: int
    lines_removed: int


class AdoptionFileResult(BaseModel):
    path: str
    component: str | None
    change: Literal["added", "removed", "modified"]
    access: Literal["editable", "review", "protected"]
    source: str
    result: Literal["ok", "review", "reject"]
    reason: str


class AdoptionTriggerLink(BaseModel):
    """A line added to the trigger file and the skill it names."""

    line: str
    skill: str | None
    skill_file: str | None
    skill_added: bool
    skill_policy_ok: bool
    regression_passed: bool | None
    accepted: bool
    reason: str


class AdoptionRegression(BaseModel):
    ran: bool
    passed: bool | None
    regression: float | None
    regression_threshold: float
    constraint_removal: float | None
    constraint_removal_threshold: float
    used_outputs: bool
    raw_evaluator_response: dict[str, JsonValue] | None
    error: str | None


class HarnessAdoptionEntry(BaseModel):
    """One Harness² adoption decision: may this composed harness replace its baseline?

    Stores no file or task output bodies: paths, line counts, policy results, scores and hashes, plus
    the added skill trigger lines and the lesson fields Harness² recorded for the adoption.
    Paths are relative to the Harness² output directory the gate was run over.
    """

    event: Literal["harness2_adoption"] = "harness2_adoption"
    ts: str = Field(default_factory=_now)
    task_id: str
    mode: str
    k: int
    namespace: str
    adopted_label: str
    adopted_is_base: bool
    policy_name: str
    policy_hash: str
    decision: Literal["accept", "reject", "needs_review"]
    reasons: list[str]
    components: list[AdoptionComponentChange]
    files: list[AdoptionFileResult]
    skill_trigger_lines: Literal["allow", "review", "deny"]
    skill_triggers: list[AdoptionTriggerLink]
    regression: AdoptionRegression
    evaluator: EvaluatorInfo | None
    latency: LatencyInfo
    baseline_sha256: str
    adopted_sha256: str
    selection: str
    reflection_note: str | None
    lesson: str | None
    experience: dict[str, str] | None


AuditRecord = GateDecisionEntry | SensitiveDataEvent | EgressDecisionEntry | HarnessAdoptionEntry
