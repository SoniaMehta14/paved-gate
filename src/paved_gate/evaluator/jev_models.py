"""Pydantic models for Jev's System One REST contract.

Source: public docs/guides as of 2026-09 (see README "Jev API contract"). Request
models are strict; response models allow unknown fields so additive API changes
don't break the gate. Verify against TypeSafe's official reference before production.

    POST {base_url}/systemone
    Authorization: Bearer <TYPESAFE_API_KEY>
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class _Req(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Resp(BaseModel):
    model_config = ConfigDict(extra="allow")


class JevNoulQuestion(_Req):
    type: Literal["noul"] = "noul"
    instructions: str


class JevChoiceQuestion(_Req):
    type: Literal["choice"] = "choice"
    instructions: str
    criteria: dict[str, str]


class JevScoreQuestion(_Req):
    type: Literal["score"] = "score"
    instructions: str
    criteria: list[str]


JevQuestion = Annotated[JevNoulQuestion | JevChoiceQuestion | JevScoreQuestion, Field(discriminator="type")]


class JevRequest(_Req):
    model: str
    state: str | dict[str, JsonValue]
    questions: dict[str, JevQuestion]


class JevNoulAnswer(_Resp):
    type: Literal["noul"] | None = None
    noul: float = Field(ge=0.0, le=1.0)


class JevChoiceAnswer(_Resp):
    type: Literal["choice"] | None = None
    choice: str
    probabilities: dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)


class JevScoreAnswer(_Resp):
    type: Literal["score"] | None = None
    score: float
    probabilities: dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)
    legend: dict[str, str] = Field(default_factory=dict)


# Matched on required fields rather than `type`, so answers parse even if `type` is omitted.
JevAnswer = Annotated[JevNoulAnswer | JevChoiceAnswer | JevScoreAnswer, Field(union_mode="left_to_right")]


class JevUsage(_Resp):
    input_tokens: int
    output_tokens: int = 0


class JevResponse(_Resp):
    model: str
    answers: dict[str, JevAnswer]
    usage: JevUsage | None = None
