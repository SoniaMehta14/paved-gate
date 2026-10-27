"""A local, deterministic FastEvaluator for the harness regression check. No network.

It reads the state the regression check builds (`diff`, `baseline_outputs`, `adopted_outputs`) and
answers by rule, so tests and the demo give the same answer every time:

- Score questions: regression level from the outputs, compared per run. No adopted output where the
  baseline had one is the top level; error markers absent from every baseline run add two levels, and
  adopted output shorter than half the mean baseline run adds one.
- Truth questions: probability that a constraint was removed, from removed diff lines that state a rule
  (must, never, always, only, do not, ...) and do not reappear among the added lines.
- Choice questions: the first label.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping

from pydantic import JsonValue

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

_CONSTRAINT = re.compile(
    r"\b(must|never|always|only|do not|don't|required|forbid(?:den)?|prohibit(?:ed)?|not allowed)\b", re.I
)
_ERROR = re.compile(r"\b(Traceback|Exception|Error:|FAILED)\b")


def _text(state: Mapping[str, JsonValue], key: str) -> str:
    value = state.get(key)
    return value if isinstance(value, str) else ""


def _runs(state: Mapping[str, JsonValue], key: str) -> list[str]:
    value = state.get(key)
    if not isinstance(value, dict):
        return []
    return [v.strip() for v in value.values() if isinstance(v, str) and v.strip()]


def regression_level(baseline: list[str], adopted: list[str], levels: int) -> int:
    top = levels - 1
    if not baseline:
        return 0
    if not adopted:
        return top
    level = 0
    if any(_ERROR.search(a) for a in adopted) and not any(_ERROR.search(b) for b in baseline):
        level += 2
    mean_baseline = sum(len(b) for b in baseline) / len(baseline)
    if min(len(a) for a in adopted) < mean_baseline / 2:
        level += 1
    return min(level, top)


def removed_constraints(diff: str) -> list[str]:
    added = {line[1:].strip() for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")}
    return [
        line[1:].strip()
        for line in diff.splitlines()
        if line.startswith("-")
        and not line.startswith("---")
        and _CONSTRAINT.search(line)
        and line[1:].strip() not in added
    ]


class HarnessRegressionHeuristic:
    name = "harness-regression-heuristic"

    def _answer(self, state: Mapping[str, JsonValue], q: Question) -> Answer:
        if isinstance(q, ScoreQuestion):
            levels = len(q.criteria)
            level = regression_level(_runs(state, "baseline_outputs"), _runs(state, "adopted_outputs"), levels)
            probs = {str(i): (1.0 if i == level else 0.0) for i in range(levels)}
            return ScoreAnswer(score=float(level), confidence=1.0, probabilities=probs)
        if isinstance(q, ChoiceQuestion):
            labels = list(q.criteria)
            return ChoiceAnswer(
                choice=labels[0], confidence=1.0, probabilities={k: float(i == 0) for i, k in enumerate(labels)}
            )
        n = len(removed_constraints(_text(state, "diff")))
        return TruthAnswer(probability=0.0 if n == 0 else min(0.95, 0.45 + 0.15 * n))

    async def evaluate(self, state: EvaluatorState, questions: Mapping[str, Question]) -> EvaluationResult:
        started = time.perf_counter()
        st: Mapping[str, JsonValue] = state if isinstance(state, dict) else {"diff": state}
        answers = {qid: self._answer(st, q) for qid, q in questions.items()}
        return EvaluationResult(
            evaluator=self.name,
            model="rules-v1",
            mode="local",
            answers=answers,
            raw={qid: a.model_dump(mode="json") for qid, a in answers.items()},
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    async def aclose(self) -> None:
        return None
