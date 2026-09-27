from __future__ import annotations

import json

import httpx
import pytest
import respx

from paved_gate import ChoiceAnswer, EvaluatorError, JevEvaluator, Question, ScoreAnswer, TruthAnswer
from paved_gate.evaluator.base import ChoiceQuestion, ScoreQuestion, TruthQuestion
from paved_gate.evaluator.jev import JevAPIError
from paved_gate.evaluator.jev_mock import mock_system_one
from paved_gate.evaluator.jev_models import JevResponse

URL = "https://api.typesafe.ai/v1/systemone"

QUESTIONS: dict[str, Question] = {
    "is_sponsor_inquiry": TruthQuestion(instructions="Does `description` ask to sponsor the site or newsletter?"),
    "product_category": ChoiceQuestion(
        instructions="What kind of product is described by `name` and `description`?",
        criteria={
            "dev_tool": "Developer tools, hosting, APIs, SaaS for developers",
            "course": "Courses, books, or training",
            "unrelated": "Anything not aimed at developers",
        },
    ),
    "message_quality": ScoreQuestion(
        instructions="How specific is the request?",
        criteria=[
            "Generic template, no reference to this site",
            "Mentions the site but no concrete ask",
            "Concrete ask with a timeframe or product named",
        ],
    ),
}

# The documented example response, verbatim.
DOC_RESPONSE = {
    "model": "jev-1.13.0",
    "answers": {
        "is_sponsor_inquiry": {"type": "noul", "noul": 0.99},
        "product_category": {
            "type": "choice",
            "choice": "dev_tool",
            "probabilities": {"dev_tool": 0.97, "course": 0.01, "unrelated": 0.02},
            "confidence": 0.95,
        },
        "message_quality": {
            "type": "score",
            "score": 1.9,
            "legend": {
                "0": "Generic template, no reference to this site",
                "1": "Mentions the site but no concrete ask",
                "2": "Concrete ask with a timeframe or product named",
            },
            "probabilities": {"0": 0.0, "1": 0.1, "2": 0.9},
            "confidence": 0.86,
        },
    },
    "usage": {"input_tokens": 210, "output_tokens": 31},
}


def live(**kw: object) -> JevEvaluator:
    return JevEvaluator(mode="live", api_key="test-key", timeout_s=2.0, **kw)  # type: ignore[arg-type]


@respx.mock
async def test_request_matches_documented_contract() -> None:
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=DOC_RESPONSE))
    ev = live()
    await ev.evaluate({"name": "Managed Postgres"}, QUESTIONS)
    await ev.aclose()

    sent = route.calls.last.request
    assert sent.headers["authorization"] == "Bearer test-key"
    body = json.loads(sent.content)
    assert body["model"] == "jev-latest"
    assert body["state"] == {"name": "Managed Postgres"}
    assert body["questions"]["is_sponsor_inquiry"] == {
        "type": "noul",
        "instructions": "Does `description` ask to sponsor the site or newsletter?",
    }
    assert body["questions"]["product_category"]["type"] == "choice"
    assert body["questions"]["message_quality"]["criteria"][0].startswith("Generic")
    assert route.call_count == 1  # all questions in one request


@respx.mock
async def test_parses_documented_response() -> None:
    respx.post(URL).mock(return_value=httpx.Response(200, json=DOC_RESPONSE))
    result = await live().evaluate("state", QUESTIONS)
    assert result.answers["is_sponsor_inquiry"] == TruthAnswer(probability=0.99)
    choice = result.answers["product_category"]
    assert isinstance(choice, ChoiceAnswer) and choice.choice == "dev_tool"
    score = result.answers["message_quality"]
    assert isinstance(score, ScoreAnswer) and score.score == 1.9
    assert result.model == "jev-1.13.0"
    assert result.input_tokens == 210
    assert result.raw == DOC_RESPONSE


@respx.mock
async def test_answers_without_type_field_still_parse() -> None:
    stripped = json.loads(json.dumps(DOC_RESPONSE))
    for a in stripped["answers"].values():
        a.pop("type")
    respx.post(URL).mock(return_value=httpx.Response(200, json=stripped))
    result = await live().evaluate("state", QUESTIONS)
    assert isinstance(result.answers["message_quality"], ScoreAnswer)


@respx.mock
async def test_retries_on_429_then_succeeds() -> None:
    route = respx.post(URL).mock(
        side_effect=[httpx.Response(429, headers={"retry-after": "0"}), httpx.Response(200, json=DOC_RESPONSE)]
    )
    await live(max_retries=1).evaluate("state", QUESTIONS)
    assert route.call_count == 2


@respx.mock
async def test_auth_error_is_raised() -> None:
    respx.post(URL).mock(return_value=httpx.Response(401, json={"error": "invalid key"}))
    with pytest.raises(JevAPIError) as exc:
        await live().evaluate("state", QUESTIONS)
    assert exc.value.status == 401


@respx.mock
async def test_missing_answer_is_an_error() -> None:
    partial = json.loads(json.dumps(DOC_RESPONSE))
    del partial["answers"]["message_quality"]
    respx.post(URL).mock(return_value=httpx.Response(200, json=partial))
    with pytest.raises(EvaluatorError, match="missing answers"):
        await live().evaluate("state", QUESTIONS)


@respx.mock
async def test_choice_outside_options_is_an_error() -> None:
    bad = json.loads(json.dumps(DOC_RESPONSE))
    bad["answers"]["product_category"]["choice"] = "something_else"
    respx.post(URL).mock(return_value=httpx.Response(200, json=bad))
    with pytest.raises(EvaluatorError, match="not one of the offered options"):
        await live().evaluate("state", QUESTIONS)


def test_live_mode_requires_key() -> None:
    with pytest.raises(EvaluatorError, match="API key"):
        JevEvaluator(mode="live", api_key=None)


async def test_mock_output_satisfies_contract() -> None:
    ev = JevEvaluator(mode="mock", mock_latency_ms=(0, 0))
    payload = await mock_system_one(ev.build_request("What is 2+2?", QUESTIONS), latency_ms=(0, 0))
    parsed = JevResponse.model_validate(payload)
    assert set(parsed.answers) == set(QUESTIONS)
    result = await ev.evaluate("What is 2+2?", QUESTIONS)
    assert result.mode == "mock"
