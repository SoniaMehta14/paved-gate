"""OpenAI chat-completions adapter for the FastEvaluator protocol (Choice questions).

Used as a baseline next to Jev: the same ChoiceQuestion (instructions plus labelled criteria) becomes a system
prompt listing the labels, and a strict JSON schema restricts the reply to one of them. Optional few-shot examples
are sent as user/assistant message pairs before the state.

Retries are done here rather than inside the SDK (`max_retries=0` on the client) so every result records how many
attempts it took and the latency of the attempt that succeeded. `insufficient_quota` is never retried.

The API key is read by the SDK from `OPENAI_API_KEY` unless passed explicitly; it is never logged or returned.
The adapter does not estimate confidence: answers carry confidence 1.0 for the chosen label.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import JsonValue

from paved_gate.evaluator.base import (
    ChoiceAnswer,
    ChoiceQuestion,
    EvaluationResult,
    EvaluatorError,
    EvaluatorState,
    Question,
)


class OpenAIAPIError(EvaluatorError):
    """A non-retryable API error, or retries exhausted."""

    def __init__(self, message: str, *, code: str | None = None, fatal: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.fatal = fatal  # e.g. insufficient_quota or an invalid key: stop the whole run


class InvalidChoiceError(EvaluatorError):
    """The model answered, but not with one of the offered labels (refusal, bad JSON, unknown label).

    The call was billed, so the raw reply and token usage are kept for the caller to record.
    """

    def __init__(self, message: str, *, raw: dict[str, JsonValue], input_tokens: int, output_tokens: int) -> None:
        super().__init__(message)
        self.raw = raw
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


def system_prompt(question: ChoiceQuestion) -> str:
    listing = "\n".join(f"- {label}: {desc}" for label, desc in question.criteria.items())
    return f'{question.instructions} Reply with JSON: {{"choice": "<label>"}}.\n\n{listing}'


def response_schema(question: ChoiceQuestion) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "choice",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"choice": {"type": "string", "enum": sorted(question.criteria)}},
                "required": ["choice"],
                "additionalProperties": False,
            },
        },
    }


def _state_text(state: EvaluatorState) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, sort_keys=True)


class OpenAIEvaluator:
    name = "openai"

    def __init__(
        self,
        *,
        model: str,
        temperature: float | None = 0.0,
        examples: Sequence[tuple[str, str]] = (),
        reasoning_effort: str | None = None,
        prompt_cache_key: str | None = None,
        max_output_tokens: int = 50,
        timeout_s: float = 60.0,
        max_retries: int = 6,
        api_key: str | None = None,
        client: Any | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.examples = list(examples)
        self.reasoning_effort = reasoning_effort
        self.prompt_cache_key = prompt_cache_key  # routes identical prompt prefixes to the same cache
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self._rng = rng or random.Random()
        self._owns_client = client is None
        if client is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=api_key, timeout=timeout_s, max_retries=0)
        self._client = client

    def build_messages(self, state: EvaluatorState, question: ChoiceQuestion) -> list[dict[str, str]]:
        messages = [{"role": "system", "content": system_prompt(question)}]
        for text, label in self.examples:
            if label not in question.criteria:
                raise EvaluatorError(f"few-shot example label {label!r} is not one of the offered options")
            messages.append({"role": "user", "content": text})
            messages.append({"role": "assistant", "content": json.dumps({"choice": label})})
        messages.append({"role": "user", "content": _state_text(state)})
        return messages

    def build_request(self, state: EvaluatorState, question: ChoiceQuestion) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": self.build_messages(state, question),
            "response_format": response_schema(question),
            "max_completion_tokens": self.max_output_tokens,
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            request["reasoning_effort"] = self.reasoning_effort
        if self.prompt_cache_key is not None:
            request["prompt_cache_key"] = self.prompt_cache_key
        return request

    async def evaluate(self, state: EvaluatorState, questions: Mapping[str, Question]) -> EvaluationResult:
        if len(questions) != 1:
            raise EvaluatorError("OpenAIEvaluator answers exactly one question per call")
        ((qid, question),) = questions.items()
        if not isinstance(question, ChoiceQuestion):
            raise EvaluatorError(f"OpenAIEvaluator supports choice questions only, not {question.kind!r}")

        request = self.build_request(state, question)
        started = time.perf_counter()
        response, attempts, attempt_ms = await self._create(request)
        total_ms = (time.perf_counter() - started) * 1000

        choice0 = response.choices[0]
        message = choice0.message
        usage = response.usage
        cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", None) if usage else None
        reasoning = (
            getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None) if usage else None
        )
        input_tokens = int(usage.prompt_tokens) if usage else 0
        output_tokens = int(usage.completion_tokens) if usage else 0
        raw: dict[str, JsonValue] = {
            "model": response.model,
            "content": message.content,
            "refusal": message.refusal,
            "finish_reason": choice0.finish_reason,
            "system_fingerprint": getattr(response, "system_fingerprint", None),
            "usage": {
                "input_tokens": input_tokens,
                "cached_tokens": int(cached or 0),
                "output_tokens": output_tokens,
                "reasoning_tokens": int(reasoning or 0),
            },
            "attempts": attempts,
            "total_ms": round(total_ms, 2),
        }

        label = self._parse(message.content, message.refusal, question)
        if label is None:
            raise InvalidChoiceError(
                f"{qid}: model did not return one of the offered options",
                raw=raw,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        return EvaluationResult(
            evaluator=self.name,
            model=str(response.model),
            mode="live",
            answers={qid: ChoiceAnswer(choice=label, confidence=1.0, probabilities={label: 1.0})},
            raw=raw,
            latency_ms=round(attempt_ms, 2),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    @staticmethod
    def _parse(content: str | None, refusal: str | None, question: ChoiceQuestion) -> str | None:
        if refusal or not content:
            return None
        try:
            value = json.loads(content).get("choice")
        except (json.JSONDecodeError, AttributeError):
            return None
        return value if isinstance(value, str) and value in question.criteria else None

    async def _create(self, request: dict[str, Any]) -> tuple[Any, int, float]:
        """Returns (response, attempts, latency of the successful attempt in ms)."""
        import openai

        attempt = 0
        while True:
            attempt += 1
            started = time.perf_counter()
            try:
                response = await self._client.chat.completions.create(**request)
                return response, attempt, (time.perf_counter() - started) * 1000
            except openai.RateLimitError as exc:
                code = _error_code(exc)
                if code == "insufficient_quota":
                    raise OpenAIAPIError("OpenAI quota exhausted", code=code, fatal=True) from exc
                retry_after = exc.response.headers.get("retry-after") if exc.response is not None else None
                await self._retry_or_raise(attempt, exc, code, retry_after)
            except (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError) as exc:
                await self._retry_or_raise(attempt, exc, _error_code(exc), None)
            except openai.AuthenticationError as exc:
                raise OpenAIAPIError("OpenAI rejected the API key", code=_error_code(exc), fatal=True) from exc
            except openai.APIStatusError as exc:
                raise OpenAIAPIError(
                    f"OpenAI returned HTTP {exc.status_code}: {str(exc)[:200]}", code=_error_code(exc)
                ) from exc

    async def _retry_or_raise(self, attempt: int, exc: Exception, code: str | None, retry_after: str | None) -> None:
        if attempt > self.max_retries:
            raise OpenAIAPIError(f"OpenAI request failed after {attempt} attempts: {type(exc).__name__}", code=code)
        await asyncio.sleep(self._backoff_s(attempt, retry_after))

    def _backoff_s(self, attempt: int, retry_after: str | None) -> float:
        if retry_after is not None:
            try:
                return min(float(retry_after), 60.0)
            except ValueError:
                pass
        return min(60.0, 0.5 * 2.0 ** (attempt - 1)) + self._rng.uniform(0, 0.25)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.close()


def _error_code(exc: Exception) -> str | None:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        code = body.get("code")
        if code is None and isinstance(body.get("error"), dict):
            code = body["error"].get("code")
        return str(code) if code is not None else None
    return None
