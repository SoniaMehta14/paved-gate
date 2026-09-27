"""Vendor-neutral fast evaluator contract.

A FastEvaluator takes one piece of state plus a set of named, typed questions and
returns typed answers. Adapters decide how to fan out (one batched request, or
concurrent per-question calls) but must answer every question.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, JsonValue

EvaluatorState = str | dict[str, JsonValue]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ChoiceQuestion(_Frozen):
    """Pick exactly one label. `criteria` maps label -> description."""

    kind: Literal["choice"] = "choice"
    instructions: str
    criteria: dict[str, str]


class ScoreQuestion(_Frozen):
    """Place the state on an ordered rubric. `criteria` is ordered low -> high."""

    kind: Literal["score"] = "score"
    instructions: str
    criteria: list[str] = Field(min_length=2)


class TruthQuestion(_Frozen):
    """Estimate the probability that a yes/no statement is true."""

    kind: Literal["truth"] = "truth"
    instructions: str


Question = Annotated[ChoiceQuestion | ScoreQuestion | TruthQuestion, Field(discriminator="kind")]


class ChoiceAnswer(_Frozen):
    kind: Literal["choice"] = "choice"
    choice: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]


class ScoreAnswer(_Frozen):
    """`score` is a continuous level index in [0, len(criteria) - 1]."""

    kind: Literal["score"] = "score"
    score: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: dict[str, float]


class TruthAnswer(_Frozen):
    kind: Literal["truth"] = "truth"
    probability: float = Field(ge=0.0, le=1.0)


Answer = Annotated[ChoiceAnswer | ScoreAnswer | TruthAnswer, Field(discriminator="kind")]


class EvaluationResult(_Frozen):
    evaluator: str
    model: str
    mode: str
    answers: dict[str, Answer]
    raw: dict[str, JsonValue]
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None


class EvaluatorError(RuntimeError):
    """Raised when an evaluator cannot produce a complete, valid set of answers."""


@runtime_checkable
class FastEvaluator(Protocol):
    name: str

    async def evaluate(self, state: EvaluatorState, questions: Mapping[str, Question]) -> EvaluationResult: ...

    async def aclose(self) -> None: ...
