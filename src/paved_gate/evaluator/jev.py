"""Jev (TypeSafe AI) adapter for the FastEvaluator protocol.

All questions go out in ONE System One request. Jev evaluates the questions of a
request in parallel on the server, so batching beats issuing three concurrent
HTTP calls: one TLS round trip, one queue slot, one state tokenisation.

`mode="mock"` uses `jev_mock` and needs no key; `mode="live"` calls the API.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from collections.abc import Mapping
from typing import Literal

import httpx
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
    TruthQuestion,
)
from paved_gate.evaluator.jev_mock import mock_system_one
from paved_gate.evaluator.jev_models import (
    JevChoiceAnswer,
    JevChoiceQuestion,
    JevNoulAnswer,
    JevNoulQuestion,
    JevQuestion,
    JevRequest,
    JevResponse,
    JevScoreAnswer,
    JevScoreQuestion,
)
from paved_gate.policy.models import EvaluatorConfig

_RETRYABLE = {429, 529}


class JevAPIError(EvaluatorError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"Jev API returned HTTP {status}: {body[:300]}")
        self.status = status


def to_jev_question(q: Question) -> JevQuestion:
    if isinstance(q, ChoiceQuestion):
        return JevChoiceQuestion(instructions=q.instructions, criteria=dict(q.criteria))
    if isinstance(q, ScoreQuestion):
        return JevScoreQuestion(instructions=q.instructions, criteria=list(q.criteria))
    return JevNoulQuestion(instructions=q.instructions)


def _to_answer(qid: str, question: Question, raw: object) -> Answer:
    if isinstance(question, ChoiceQuestion) and isinstance(raw, JevChoiceAnswer):
        if raw.choice not in question.criteria:
            raise EvaluatorError(f"{qid}: choice {raw.choice!r} is not one of the offered options")
        return ChoiceAnswer(choice=raw.choice, confidence=raw.confidence, probabilities=raw.probabilities)
    if isinstance(question, ScoreQuestion) and isinstance(raw, JevScoreAnswer):
        top = len(question.criteria) - 1
        if not 0.0 <= raw.score <= top:
            raise EvaluatorError(f"{qid}: score {raw.score} outside [0, {top}]")
        return ScoreAnswer(score=raw.score, confidence=raw.confidence, probabilities=raw.probabilities)
    if isinstance(question, TruthQuestion) and isinstance(raw, JevNoulAnswer):
        return TruthAnswer(probability=raw.noul)
    raise EvaluatorError(f"{qid}: answer type does not match question kind {question.kind!r}")


class JevEvaluator:
    name = "jev"

    def __init__(
        self,
        *,
        mode: Literal["mock", "live"] = "mock",
        model: str = "jev-latest",
        base_url: str = "https://api.typesafe.ai/v1",
        api_key: str | None = None,
        timeout_s: float = 0.4,
        max_retries: int = 1,
        http_client: httpx.AsyncClient | None = None,
        mock_latency_ms: tuple[float, float] = (60.0, 120.0),
        rng: random.Random | None = None,
    ) -> None:
        if mode == "live" and not api_key:
            raise EvaluatorError("Jev live mode requires an API key (set TYPESAFE_API_KEY)")
        self.mode = mode
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.mock_latency_ms = mock_latency_ms
        self._rng = rng or random.Random()
        self._api_key = api_key
        self._owns_client = http_client is None and mode == "live"
        self._client = http_client
        if self._owns_client:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(timeout_s),
                limits=httpx.Limits(max_keepalive_connections=20, keepalive_expiry=30),
            )

    @classmethod
    def from_config(cls, cfg: EvaluatorConfig, **overrides: object) -> JevEvaluator:
        return cls(
            mode=cfg.mode,
            model=cfg.model,
            base_url=cfg.base_url,
            api_key=os.environ.get(cfg.api_key_env),
            timeout_s=cfg.timeout_ms / 1000,
            max_retries=cfg.max_retries,
            **overrides,  # type: ignore[arg-type]
        )

    def build_request(self, state: EvaluatorState, questions: Mapping[str, Question]) -> JevRequest:
        return JevRequest(
            model=self.model,
            state=state,
            questions={qid: to_jev_question(q) for qid, q in questions.items()},
        )

    async def evaluate(self, state: EvaluatorState, questions: Mapping[str, Question]) -> EvaluationResult:
        request = self.build_request(state, questions)
        started = time.perf_counter()
        if self.mode == "mock":
            payload = await mock_system_one(request, latency_ms=self.mock_latency_ms, rng=self._rng)
        else:
            payload = await self._post(request)
        latency_ms = (time.perf_counter() - started) * 1000

        try:
            response = JevResponse.model_validate(payload)
        except ValidationError as exc:
            raise EvaluatorError(f"Jev response did not match the expected schema: {exc}") from exc

        missing = set(questions) - set(response.answers)
        if missing:
            raise EvaluatorError(f"Jev response is missing answers for {sorted(missing)}")

        answers = {qid: _to_answer(qid, q, response.answers[qid]) for qid, q in questions.items()}
        return EvaluationResult(
            evaluator=self.name,
            model=response.model,
            mode=self.mode,
            answers=answers,
            raw=payload,
            latency_ms=round(latency_ms, 2),
            input_tokens=response.usage.input_tokens if response.usage else None,
            output_tokens=response.usage.output_tokens if response.usage else None,
        )

    async def _post(self, request: JevRequest) -> dict[str, JsonValue]:
        assert self._client is not None
        body = request.model_dump(mode="json")
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        attempt = 0
        while True:
            try:
                resp = await self._client.post(f"{self.base_url}/systemone", json=body, headers=headers)
            except httpx.HTTPError as exc:
                raise EvaluatorError(f"Jev request failed: {exc!r}") from exc
            if resp.status_code in _RETRYABLE and attempt < self.max_retries:
                attempt += 1
                await asyncio.sleep(self._backoff_s(attempt, resp.headers.get("retry-after")))
                continue
            if resp.status_code != 200:
                raise JevAPIError(resp.status_code, resp.text)
            data = resp.json()
            if not isinstance(data, dict):
                raise EvaluatorError("Jev response body is not a JSON object")
            return data

    def _backoff_s(self, attempt: int, retry_after: str | None) -> float:
        # The gate enforces the overall deadline; keep backoff a small fraction of it.
        cap = self.timeout_s / 4
        if retry_after is not None:
            try:
                return min(float(retry_after), cap)
            except ValueError:
                pass
        return min(cap, 0.025 * 2.0**attempt + self._rng.uniform(0, 0.01))

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
