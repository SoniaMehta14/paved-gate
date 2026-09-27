"""Mock Jev backend: returns responses in the exact System One JSON shape.

Answers are derived from keyword heuristics so the gate behaves plausibly
end-to-end without an API key. Latency is simulated as a single server-side
round trip, because Jev evaluates all questions in one request in parallel.
"""

from __future__ import annotations

import asyncio
import math
import random
import re

from pydantic import JsonValue

from paved_gate.evaluator import signals
from paved_gate.evaluator.jev_models import (
    JevChoiceQuestion,
    JevNoulQuestion,
    JevRequest,
    JevScoreQuestion,
)
from paved_gate.types import Route

MOCK_MODEL = "jev-mock-1.13.0"
_PII_QUESTION = re.compile(r"\b(PII|PHI|personal|sensitive|health information)\b", re.I)


def _state_text(state: str | dict[str, JsonValue]) -> str:
    return state if isinstance(state, str) else str(state)


def _round_probs(probs: dict[str, float]) -> dict[str, float]:
    total = sum(probs.values())
    return {k: round(v / total, 3) for k, v in probs.items()}


def route_probabilities(text: str, labels: list[str]) -> dict[str, float]:
    if set(labels) != {r.value for r in Route}:
        # Unknown choice question: mild preference for the first option.
        return _round_probs({label: (2.0 if i == 0 else 1.0) for i, label in enumerate(labels)})
    if signals.violation(text):
        weights = {
            Route.POLICY_VIOLATION: 0.92,
            Route.FRONTIER_AGENT_REQUIRED: 0.06,
            Route.DIRECT_CODE_EXEC: 0.02,
        }
    elif signals.deterministic(text) and signals.injection_risk(text) < 0.5:
        weights = {
            Route.DIRECT_CODE_EXEC: 0.88,
            Route.FRONTIER_AGENT_REQUIRED: 0.11,
            Route.POLICY_VIOLATION: 0.01,
        }
    else:
        risk = signals.injection_risk(text)
        weights = {
            Route.FRONTIER_AGENT_REQUIRED: 0.9 - 0.4 * risk,
            Route.POLICY_VIOLATION: 0.02 + 0.4 * risk,
            Route.DIRECT_CODE_EXEC: 0.03,
        }
    return _round_probs({k.value: v for k, v in weights.items()})


def score_distribution(text: str, levels: int) -> tuple[float, float, dict[str, float]]:
    """(score, confidence, probabilities) over `levels` ordered levels, centred on the risk signal."""
    center = signals.injection_risk(text) * (levels - 1)
    raw = {str(i): math.exp(-((i - center) ** 2) / 0.6) for i in range(levels)}
    probs = _round_probs(raw)
    score = round(sum(i * probs[str(i)] for i in range(levels)), 2)
    return score, round(max(probs.values()), 2), probs


async def mock_system_one(
    request: JevRequest,
    *,
    latency_ms: tuple[float, float] = (60.0, 120.0),
    rng: random.Random | None = None,
) -> dict[str, JsonValue]:
    rng = rng or random.Random()
    text = _state_text(request.state)
    answers: dict[str, JsonValue] = {}
    for qid, q in request.questions.items():
        if isinstance(q, JevNoulQuestion):
            p = signals.sensitive(text) if _PII_QUESTION.search(q.instructions) else 0.5
            answers[qid] = {"type": "noul", "noul": round(p, 3)}
        elif isinstance(q, JevChoiceQuestion):
            probs = route_probabilities(text, list(q.criteria))
            choice = max(probs, key=lambda k: probs[k])
            answers[qid] = {
                "type": "choice",
                "choice": choice,
                "probabilities": dict(probs),
                "confidence": round(probs[choice] * 0.97, 2),
            }
        elif isinstance(q, JevScoreQuestion):
            score, confidence, probs = score_distribution(text, len(q.criteria))
            answers[qid] = {
                "type": "score",
                "score": score,
                "legend": {str(i): c for i, c in enumerate(q.criteria)},
                "probabilities": dict(probs),
                "confidence": confidence,
            }

    lo, hi = latency_ms
    if hi > 0:
        await asyncio.sleep(rng.uniform(lo, hi) / 1000)

    input_tokens = max(1, len(text) // 4) + sum(len(q.model_dump_json()) // 4 for q in request.questions.values())
    return {
        "model": MOCK_MODEL,
        "answers": answers,
        "usage": {"input_tokens": input_tokens, "output_tokens": 12 * len(answers)},
    }
