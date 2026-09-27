from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class Route(StrEnum):
    DIRECT_CODE_EXEC = "DIRECT_CODE_EXEC"
    FRONTIER_AGENT_REQUIRED = "FRONTIER_AGENT_REQUIRED"
    POLICY_VIOLATION = "POLICY_VIOLATION"


Outcome = Literal["blocked", "deterministic", "frontier"]


class GateState(BaseModel):
    """The inbound request as the gate sees it. `input` is never masked here."""

    model_config = ConfigDict(frozen=True)

    request_id: str
    input: str
    metadata: dict[str, str] = Field(default_factory=dict)


class GateDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    request_id: str
    outcome: Outcome
    route: Route | None
    reasons: list[str]
    intent_confidence: float | None
    risk_score: float | None
    sensitive: bool
    evaluator_ms: float | None
    gate_ms: float
    policy_hash: str


class FrontierResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    model: str
    text: str
    stop_reason: str | None
    input_tokens: int | None = None
    output_tokens: int | None = None


class Blocked(BaseModel):
    kind: Literal["blocked"] = "blocked"
    decision: GateDecision


class Deterministic(BaseModel):
    kind: Literal["deterministic"] = "deterministic"
    decision: GateDecision
    handler: str
    output: dict[str, JsonValue]


class Frontier(BaseModel):
    """`response` is None when no frontier handler is configured; the caller owns the call."""

    kind: Literal["frontier"] = "frontier"
    decision: GateDecision
    state: GateState
    response: FrontierResponse | None


GateResult = Blocked | Deterministic | Frontier
