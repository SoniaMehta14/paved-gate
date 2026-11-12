"""Gemini Decision 1 adapter for the FastEvaluator protocol (Google's Decisions API, `gemini-decision-1`).

All questions go out in ONE `decisions.create` call: the input is read once in a single prefill pass and
every question is answered from it, with no output tokens billed. Paved Gate's question kinds map one to
one: TruthQuestion to `noul`, ChoiceQuestion to `choice`, ScoreQuestion to `score` (0 based levels, the
same index the gates already add 1 to).

`mode="mock"` uses `gemini_decision_mock` and needs neither the SDK nor a key. `mode="live"` needs the
`gemini` extra (google-genai 3.2 or later) and GEMINI_API_KEY.

Retries: the SDK's own retries are turned off and this adapter retries 429 and 5xx responses itself with
exponential backoff and jitter, so each result can report how many retries it took and how long it
waited. A 429 that outlasts the retries raises RateLimitedError.

Request and response shapes follow the google-genai 3.2.0 types and Google's Decisions API guide
(ai.google.dev/gemini-api/docs/decisions-api, read 2026-11-11): model ID `gemini-decision-1` on the v1beta
Developer API, `confidence` equal to the top outcome's probability (`max(noul, 1 - noul)` for yes or no),
safety blocks returned as HTTP 400. They have not yet been checked against a live response.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from collections.abc import Mapping
from typing import Literal, Protocol

from pydantic import JsonValue, ValidationError

from paved_gate.evaluator.base import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    EvaluationResult,
    EvaluatorError,
    EvaluatorState,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    TruthAnswer,
)
from paved_gate.evaluator.gemini_decision_mock import MOCK_MODEL, mock_decisions
from paved_gate.evaluator.gemini_decision_models import (
    KNOWN_ANSWER_TYPES,
    GeminiChoiceAnswer,
    GeminiDecisionResponse,
    GeminiNoulAnswer,
    GeminiScoreAnswer,
)
from paved_gate.policy.models import EvaluatorConfig

DEFAULT_MODEL = "gemini-decision-1"
_RETRYABLE_MIN_SERVER = 500
_RATE_LIMITED = 429


class RateLimitedError(EvaluatorError):
    """The API kept answering 429 after every retry: a quota or rate limit was hit."""


class DecisionsAPI(Protocol):
    """The slice of `client.aio.decisions` this adapter uses; tests inject a fake."""

    async def create(
        self, *, input: str, model: str, questions: dict[str, dict[str, JsonValue]], timeout: float
    ) -> object: ...


def to_gemini_question(q: Question) -> dict[str, JsonValue]:
    if isinstance(q, ChoiceQuestion):
        return {"type": "choice", "instructions": q.instructions, "criteria": dict(q.criteria)}
    if isinstance(q, ScoreQuestion):
        return {"type": "score", "instructions": q.instructions, "criteria": list(q.criteria)}
    return {"type": "noul", "instructions": q.instructions}


def state_text(state: EvaluatorState) -> str:
    """The Decisions API takes text (or content blocks); a dict state is sent as canonical JSON."""
    return state if isinstance(state, str) else json.dumps(state, sort_keys=True, ensure_ascii=False)


def _to_answer(qid: str, question: Question, raw: object) -> Answer:
    if isinstance(question, ChoiceQuestion) and isinstance(raw, GeminiChoiceAnswer):
        if raw.choice not in question.criteria:
            raise EvaluatorError(f"{qid}: choice {raw.choice!r} is not one of the offered options")
        return ChoiceAnswer(choice=raw.choice, confidence=raw.confidence, probabilities=raw.probabilities)
    if isinstance(question, ScoreQuestion) and isinstance(raw, GeminiScoreAnswer):
        top = len(question.criteria) - 1
        if not 0.0 <= raw.score <= top:
            raise EvaluatorError(f"{qid}: score {raw.score} outside [0, {top}]")
        return ScoreAnswer(score=raw.score, confidence=raw.confidence, probabilities=raw.probabilities)
    if question.kind == "truth" and isinstance(raw, GeminiNoulAnswer):
        return TruthAnswer(probability=raw.noul)
    raise EvaluatorError(f"{qid}: answer type does not match question kind {question.kind!r}")


def _status(exc: BaseException) -> int | None:
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def _as_dict(response: object) -> dict[str, JsonValue]:
    if isinstance(response, dict):
        return response
    dump = getattr(response, "model_dump", None)
    if callable(dump):
        data = dump(mode="json", exclude_none=True)
        if isinstance(data, dict):
            return data
    raise EvaluatorError(f"Gemini response is not a JSON object: {type(response).__name__}")


class GeminiDecisionEvaluator:
    name = "gemini_decision"

    def __init__(
        self,
        *,
        mode: Literal["mock", "live"] = "mock",
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        timeout_s: float = 3.0,
        max_retries: int = 2,
        backoff_base_s: float = 0.5,
        backoff_cap_s: float = 8.0,
        decisions_api: DecisionsAPI | None = None,
        mock_latency_ms: tuple[float, float] = (150.0, 300.0),
        rng: random.Random | None = None,
    ) -> None:
        if mode == "live" and decisions_api is None and not api_key:
            raise EvaluatorError("Gemini Decision live mode requires an API key (set GEMINI_API_KEY)")
        self.mode = mode
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_cap_s = backoff_cap_s
        self.mock_latency_ms = mock_latency_ms
        self._rng = rng or random.Random()
        self._client: object | None = None
        self._api = decisions_api
        if mode == "live" and decisions_api is None:
            self._api, self._client = _live_api(api_key or "")

    @classmethod
    def from_config(cls, cfg: EvaluatorConfig, **overrides: object) -> GeminiDecisionEvaluator:
        return cls(
            mode=cfg.mode,
            model=cfg.model,
            api_key=os.environ.get(cfg.api_key_env),
            timeout_s=cfg.timeout_ms / 1000,
            max_retries=cfg.max_retries,
            **overrides,  # type: ignore[arg-type]
        )

    def retry_policy(self) -> str:
        return (
            f"up to {self.max_retries} retries on 429 and 5xx, exponential backoff "
            f"{self.backoff_base_s:g} s doubling to a {self.backoff_cap_s:g} s cap, plus up to 10% jitter; "
            "SDK retries off"
        )

    async def evaluate(self, state: EvaluatorState, questions: Mapping[str, Question]) -> EvaluationResult:
        text = state_text(state)
        wire = {qid: to_gemini_question(q) for qid, q in questions.items()}
        retries, wait_ms = 0, 0.0
        started = time.perf_counter()
        if self.mode == "mock":
            payload = await mock_decisions(text, wire, latency_ms=self.mock_latency_ms, rng=self._rng)
        else:
            payload, retries, wait_ms = await self._create(text, wire)
        latency_ms = (time.perf_counter() - started) * 1000
        return self._result(questions, payload, latency_ms, retries, wait_ms)

    def _result(
        self,
        questions: Mapping[str, Question],
        payload: dict[str, JsonValue],
        latency_ms: float,
        retries: int,
        wait_ms: float,
    ) -> EvaluationResult:
        answers_raw = payload.get("answers")
        if isinstance(answers_raw, dict):
            for qid, a in answers_raw.items():
                kind = a.get("type") if isinstance(a, dict) else None
                if kind not in KNOWN_ANSWER_TYPES:
                    raise EvaluatorError(f"{qid}: Gemini returned an unknown answer type {kind!r}")
        try:
            response = GeminiDecisionResponse.model_validate(payload)
        except ValidationError as exc:
            raise EvaluatorError(f"Gemini response did not match the expected schema: {exc}") from exc

        feedback = response.prompt_feedback
        if feedback is not None and feedback.block_reason:
            why = f": {feedback.block_reason_message}" if feedback.block_reason_message else ""
            raise EvaluatorError(f"Gemini blocked the input ({feedback.block_reason}){why}")
        missing = set(questions) - set(response.answers)
        if missing:
            raise EvaluatorError(f"Gemini response is missing answers for {sorted(missing)}")

        answers = {qid: _to_answer(qid, q, response.answers[qid]) for qid, q in questions.items()}
        usage = response.usage
        return EvaluationResult(
            evaluator=self.name,
            model=MOCK_MODEL if self.mode == "mock" else self.model,
            mode=self.mode,
            answers=answers,
            raw=payload,
            latency_ms=round(latency_ms, 2),
            input_tokens=usage.total_input_tokens if usage else None,
            output_tokens=usage.total_output_tokens if usage else None,
            retries=retries,
            retry_wait_ms=round(wait_ms, 2),
        )

    async def _create(
        self, text: str, wire: dict[str, dict[str, JsonValue]]
    ) -> tuple[dict[str, JsonValue], int, float]:
        assert self._api is not None
        attempt, waited = 0, 0.0
        while True:
            try:
                response = await self._api.create(input=text, model=self.model, questions=wire, timeout=self.timeout_s)
                return _as_dict(response), attempt, waited * 1000
            except EvaluatorError:
                raise
            except Exception as exc:  # SDK and transport errors; retried or mapped to EvaluatorError
                status = _status(exc)
                retryable = status == _RATE_LIMITED or (status is not None and status >= _RETRYABLE_MIN_SERVER)
                if retryable and attempt < self.max_retries:
                    attempt += 1
                    delay = self._backoff_s(attempt)
                    waited += delay
                    await asyncio.sleep(delay)
                    continue
                if status == _RATE_LIMITED:
                    raise RateLimitedError(
                        f"Gemini Decisions API kept returning 429 after {attempt} retries: quota or rate limit hit"
                    ) from exc
                detail = f"HTTP {status}" if status is not None else type(exc).__name__
                raise EvaluatorError(f"Gemini Decisions request failed ({detail}): {str(exc)[:300]}") from exc

    def _backoff_s(self, attempt: int) -> float:
        base = min(self.backoff_cap_s, self.backoff_base_s * 2.0 ** (attempt - 1))
        return base + self._rng.uniform(0, base * 0.1)

    async def aclose(self) -> None:
        close = getattr(self._client, "aclose", None)
        if callable(close):
            await close()


def _live_api(api_key: str) -> tuple[DecisionsAPI, object]:
    """`client.aio.decisions` on v1beta (where the guide says the API lives), SDK retries off; lazy import."""
    try:
        import google.genai as genai
    except ImportError as exc:  # pragma: no cover (depends on the optional extra)
        raise EvaluatorError("Gemini Decision live mode needs the gemini extra: uv sync --extra gemini") from exc
    options = genai.types.HttpOptions(api_version="v1beta", retry_options=genai.types.HttpRetryOptions(attempts=1))
    client = genai.Client(api_key=api_key, http_options=options)
    return client.aio.decisions, client.aio
