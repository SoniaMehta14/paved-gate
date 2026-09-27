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


AuditRecord = GateDecisionEntry | SensitiveDataEvent
