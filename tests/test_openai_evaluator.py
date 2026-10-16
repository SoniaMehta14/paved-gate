"""OpenAIEvaluator: request construction, parsing, usage accounting and retries, with a fake client."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

from paved_gate import ChoiceQuestion, OpenAIEvaluator, ScoreQuestion
from paved_gate.evaluator.base import EvaluatorError
from paved_gate.evaluator.openai_eval import InvalidChoiceError, OpenAIAPIError

Q = ChoiceQuestion(instructions="Which intent?", criteria={"get_refund": "wants money back", "track_order": "where"})


def _response(content: str | None, *, refusal: str | None = None, cached: int = 0, reasoning: int = 0) -> Any:
    usage = SimpleNamespace(
        prompt_tokens=600,
        completion_tokens=7 + reasoning,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
    )
    message = SimpleNamespace(content=content, refusal=refusal)
    return SimpleNamespace(
        model="gpt-test",
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=usage,
        system_fingerprint="fp_1",
    )


class FakeClient:
    def __init__(self, *outcomes: object) -> None:
        self.outcomes = list(outcomes)
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **request: Any) -> Any:
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    async def close(self) -> None:
        return None


def _status_error(
    cls: type[openai.APIStatusError], status: int, code: str | None, retry_after: str | None = None
) -> Any:
    headers = {"retry-after": retry_after} if retry_after else {}
    resp: Any = httpx.Response(status, headers=headers, request=httpx.Request("POST", "https://api.openai.com/v1/x"))
    return cls("error", response=resp, body={"code": code} if code else None)


async def test_zero_shot_request_and_answer() -> None:
    client = FakeClient(_response('{"choice": "get_refund"}'))
    ev = OpenAIEvaluator(model="gpt-test", client=client)
    result = await ev.evaluate("my blender broke, money back please", {"intent": Q})
    req = client.requests[0]
    assert req["temperature"] == 0.0 and "reasoning_effort" not in req and "prompt_cache_key" not in req
    assert [m["role"] for m in req["messages"]] == ["system", "user"]
    assert "- get_refund: wants money back" in req["messages"][0]["content"]
    schema = req["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["choice"]["enum"] == ["get_refund", "track_order"]
    assert result.answers["intent"].choice == "get_refund"  # type: ignore[union-attr]
    assert result.input_tokens == 600 and result.output_tokens == 7
    assert result.raw["attempts"] == 1


async def test_few_shot_examples_become_message_pairs() -> None:
    client = FakeClient(_response('{"choice": "track_order"}'))
    ev = OpenAIEvaluator(
        model="gpt-test", client=client, examples=[("where is it", "track_order")], prompt_cache_key="k"
    )
    await ev.evaluate("status of my order?", {"intent": Q})
    msgs = client.requests[0]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert json.loads(msgs[2]["content"]) == {"choice": "track_order"}
    assert client.requests[0]["prompt_cache_key"] == "k"


async def test_reasoning_model_omits_temperature() -> None:
    client = FakeClient(_response('{"choice": "get_refund"}', cached=512, reasoning=40))
    ev = OpenAIEvaluator(model="gpt-r", client=client, temperature=None, reasoning_effort="low")
    result = await ev.evaluate("refund", {"intent": Q})
    assert "temperature" not in client.requests[0] and client.requests[0]["reasoning_effort"] == "low"
    usage = result.raw["usage"]
    assert isinstance(usage, dict) and usage["cached_tokens"] == 512 and usage["reasoning_tokens"] == 40


@pytest.mark.parametrize(
    "content,refusal", [('{"choice": "cancel_everything"}', None), ("not json", None), (None, "I can't help")]
)
async def test_invalid_or_refused_answers_keep_raw_and_usage(content: str | None, refusal: str | None) -> None:
    ev = OpenAIEvaluator(model="gpt-test", client=FakeClient(_response(content, refusal=refusal)))
    with pytest.raises(InvalidChoiceError) as exc:
        await ev.evaluate("x", {"intent": Q})
    assert exc.value.input_tokens == 600 and exc.value.raw["content"] == content


async def test_rate_limits_are_retried_and_counted() -> None:
    client = FakeClient(
        _status_error(openai.RateLimitError, 429, "rate_limit_exceeded", "0"), _response('{"choice": "get_refund"}')
    )
    ev = OpenAIEvaluator(model="gpt-test", client=client)
    result = await ev.evaluate("x", {"intent": Q})
    assert result.raw["attempts"] == 2 and len(client.requests) == 2


async def test_quota_exhaustion_and_bad_key_are_fatal_and_not_retried() -> None:
    for err in (
        _status_error(openai.RateLimitError, 429, "insufficient_quota"),
        _status_error(openai.AuthenticationError, 401, "invalid_api_key"),
    ):
        client = FakeClient(err, _response('{"choice": "get_refund"}'))
        with pytest.raises(OpenAIAPIError) as exc:
            await OpenAIEvaluator(model="gpt-test", client=client).evaluate("x", {"intent": Q})
        assert exc.value.fatal and len(client.requests) == 1


async def test_retries_are_bounded() -> None:
    errors = [_status_error(openai.InternalServerError, 500, None, "0") for _ in range(3)]
    ev = OpenAIEvaluator(model="gpt-test", client=FakeClient(*errors), max_retries=2)
    with pytest.raises(OpenAIAPIError) as exc:
        await ev.evaluate("x", {"intent": Q})
    assert not exc.value.fatal


async def test_only_single_choice_questions() -> None:
    ev = OpenAIEvaluator(model="gpt-test", client=FakeClient())
    with pytest.raises(EvaluatorError):
        await ev.evaluate("x", {"s": ScoreQuestion(instructions="rate", criteria=["low", "high"])})
