"""Gemini Decisions API response shapes (pydantic), as `decisions.create` returns them.

Written from the google-genai 3.2.0 types under google/genai/_gaos/types/decisions and the response
reference in Google's Decisions API guide, and checked against a live response recorded on 2026-11-11
(tests/fixtures/gemini_decision/recorded_response.json).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class _Resp(BaseModel):
    model_config = ConfigDict(extra="allow")


class GeminiNoulAnswer(_Resp):
    type: Literal["noul"]
    noul: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)


class GeminiChoiceAnswer(_Resp):
    type: Literal["choice"]
    choice: str
    probabilities: dict[str, float]
    confidence: float = Field(ge=0.0, le=1.0)


class GeminiScoreAnswer(_Resp):
    """`score` is a continuous, probability weighted level index in `[0, len(criteria) - 1]`."""

    type: Literal["score"]
    score: float = Field(ge=0.0)
    probabilities: dict[str, float]
    legend: dict[str, str] = Field(default_factory=dict)
    confidence: float = Field(ge=0.0, le=1.0)


GeminiAnswer = Annotated[GeminiNoulAnswer | GeminiChoiceAnswer | GeminiScoreAnswer, Field(discriminator="type")]
KNOWN_ANSWER_TYPES = frozenset({"noul", "choice", "score"})


class GeminiPromptFeedback(_Resp):
    block_reason: str | None = None
    block_reason_message: str | None = None


class GeminiUsage(_Resp):
    total_input_tokens: int | None = None
    total_output_tokens: int | None = None


class GeminiDecisionResponse(_Resp):
    answers: dict[str, GeminiAnswer] = Field(default_factory=dict)
    prompt_feedback: GeminiPromptFeedback | None = None
    usage: GeminiUsage | None = None
