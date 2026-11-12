"""Mock Gemini Decisions backend: returns responses in the `decisions.create` shape. No SDK, no network.

Answers come from the same keyword heuristics as the Jev mock, so the gate behaves plausibly end to end
without a key; they are not Gemini judgments. `confidence` follows the SDK's documented definition (the
probability of the winning outcome). Latency is simulated as one round trip; usage reports input tokens
only, since output is not billed.

The response shape matches a live response recorded on 2026-11-11; the answers are not Gemini's.
"""

from __future__ import annotations

import asyncio
import random
import re
from collections.abc import Mapping

from pydantic import JsonValue

from paved_gate.evaluator import signals
from paved_gate.evaluator.jev_mock import route_probabilities, score_distribution

MOCK_MODEL = "gemini-decision-1-mock"
_PII_QUESTION = re.compile(r"\b(PII|PHI|personal|sensitive|health information)\b", re.I)


def _noul(text: str, instructions: str) -> dict[str, JsonValue]:
    p = round(signals.sensitive(text) if _PII_QUESTION.search(instructions) else 0.5, 3)
    return {"type": "noul", "noul": p, "confidence": round(max(p, 1 - p), 3)}


def _choice(text: str, criteria: Mapping[str, str]) -> dict[str, JsonValue]:
    probs = route_probabilities(text, list(criteria))
    choice = max(probs, key=lambda k: probs[k])
    return {"type": "choice", "choice": choice, "probabilities": dict(probs), "confidence": probs[choice]}


def _score(text: str, criteria: list[str]) -> dict[str, JsonValue]:
    score, confidence, probs = score_distribution(text, len(criteria))
    return {
        "type": "score",
        "score": score,
        "probabilities": dict(probs),
        "legend": {str(i): c for i, c in enumerate(criteria)},
        "confidence": confidence,
    }


async def mock_decisions(
    text: str,
    questions: Mapping[str, Mapping[str, JsonValue]],
    *,
    latency_ms: tuple[float, float] = (150.0, 300.0),
    rng: random.Random | None = None,
) -> dict[str, JsonValue]:
    rng = rng or random.Random()
    answers: dict[str, JsonValue] = {}
    for qid, q in questions.items():
        instructions = str(q.get("instructions", ""))
        criteria = q.get("criteria")
        if q["type"] == "choice" and isinstance(criteria, dict):
            answers[qid] = _choice(text, {str(k): str(v) for k, v in criteria.items()})
        elif q["type"] == "score" and isinstance(criteria, list):
            answers[qid] = _score(text, [str(c) for c in criteria])
        else:
            answers[qid] = _noul(text, instructions)

    lo, hi = latency_ms
    if hi > 0:
        await asyncio.sleep(rng.uniform(lo, hi) / 1000)

    input_tokens = max(1, len(text) // 4) + sum(len(str(q)) // 4 for q in questions.values())
    return {
        "answers": answers,
        "usage": {"total_input_tokens": input_tokens, "total_output_tokens": 0},
    }
