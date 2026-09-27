"""Vendor-free evaluator for tests and offline demos.

Answers each question in its own coroutine and gathers them concurrently, which
is the fan-out pattern for any backend that exposes one endpoint per question
type rather than a batched call.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping

from paved_gate.evaluator import signals
from paved_gate.evaluator.base import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    EvaluationResult,
    EvaluatorState,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    TruthAnswer,
)
from paved_gate.evaluator.jev_mock import route_probabilities, score_distribution


class HeuristicEvaluator:
    name = "heuristic"

    def __init__(self, *, per_question_delay_ms: float = 0.0) -> None:
        self.per_question_delay_ms = per_question_delay_ms

    async def _answer(self, text: str, q: Question) -> Answer:
        if self.per_question_delay_ms:
            await asyncio.sleep(self.per_question_delay_ms / 1000)
        if isinstance(q, ChoiceQuestion):
            probs = route_probabilities(text, list(q.criteria))
            choice = max(probs, key=lambda k: probs[k])
            return ChoiceAnswer(choice=choice, confidence=probs[choice], probabilities=probs)
        if isinstance(q, ScoreQuestion):
            score, confidence, probs = score_distribution(text, len(q.criteria))
            return ScoreAnswer(score=score, confidence=confidence, probabilities=probs)
        return TruthAnswer(probability=signals.sensitive(text))

    async def evaluate(self, state: EvaluatorState, questions: Mapping[str, Question]) -> EvaluationResult:
        text = state if isinstance(state, str) else str(state)
        started = time.perf_counter()
        ids = list(questions)
        results = await asyncio.gather(*(self._answer(text, questions[qid]) for qid in ids))
        answers = dict(zip(ids, results, strict=True))
        return EvaluationResult(
            evaluator=self.name,
            model="keyword-heuristics",
            mode="local",
            answers=answers,
            raw={qid: a.model_dump(mode="json") for qid, a in answers.items()},
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    async def aclose(self) -> None:
        return None
