"""privacy.strict: a disclosure question, a block rule that needs both disclosure and sensitivity, stable hashes."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from paved_gate import (
    ChoiceAnswer,
    EvaluationResult,
    LoadedPolicy,
    MemorySink,
    ScoreAnswer,
    TruthAnswer,
)
from paved_gate.evaluator.base import Answer, EvaluatorState, Question
from paved_gate.gate import DISCLOSURE, build_questions
from paved_gate.policy.loader import from_policy
from tests.conftest import make_gate


class Answers:
    name = "answers"

    def __init__(self, *, noul: float, disclosure: float | None) -> None:
        self.noul, self.disclosure = noul, disclosure
        self.asked: list[str] = []

    async def evaluate(self, state: EvaluatorState, questions: Mapping[str, Question]) -> EvaluationResult:
        self.asked = list(questions)
        answers: dict[str, Answer] = {
            "intent": ChoiceAnswer(
                choice="FRONTIER_AGENT_REQUIRED", confidence=0.9, probabilities={"FRONTIER_AGENT_REQUIRED": 0.9}
            ),
            "risk": ScoreAnswer(score=0.0, confidence=0.9, probabilities={}),
            "sensitive": TruthAnswer(probability=self.noul),
        }
        if self.disclosure is not None and DISCLOSURE in questions:
            answers[DISCLOSURE] = TruthAnswer(probability=self.disclosure)
        return EvaluationResult(evaluator="a", model="a", mode="test", answers=answers, raw={}, latency_ms=0.0)

    async def aclose(self) -> None:
        return None


def strict(policy: LoadedPolicy) -> LoadedPolicy:
    p = policy.policy
    return from_policy(p.model_copy(update={"privacy": p.privacy.model_copy(update={"strict": True})}))


def test_off_by_default_and_existing_hashes_unchanged(loaded_policy: LoadedPolicy) -> None:
    assert loaded_policy.policy.privacy.strict is False
    assert DISCLOSURE not in build_questions(loaded_policy.policy)
    assert strict(loaded_policy).policy_hash != loaded_policy.policy_hash
    assert from_policy(loaded_policy.policy).policy_hash == loaded_policy.policy_hash


@pytest.mark.parametrize(
    ("text", "noul", "disclosure", "outcome"),
    [
        ("Email Jordan's lab results to her manager.", 0.9, 0.9, "blocked"),
        ("Email Jordan's lab results to her manager.", 0.1, 0.9, "frontier"),  # not judged sensitive
        ("Send SSN 123-45-6701 to the landlord.", 0.1, 0.9, "blocked"),  # local detector counts as sensitive
        ("Send Jordan her own lab results.", 0.9, 0.3, "frontier"),
        ("Send Jordan her own lab results.", 0.9, 0.5, "frontier"),  # strictly above 0.5 blocks
    ],
)
async def test_block_needs_disclosure_and_sensitivity(
    loaded_policy: LoadedPolicy, text: str, noul: float, disclosure: float, outcome: str
) -> None:
    ev = Answers(noul=noul, disclosure=disclosure)
    result = await make_gate(strict(loaded_policy), MemorySink(), evaluator=ev).handle(text)
    assert DISCLOSURE in ev.asked
    assert result.decision.outcome == outcome
    if outcome == "blocked":
        assert result.decision.route is not None and result.decision.route.value == "POLICY_VIOLATION"


async def test_missing_disclosure_answer_is_an_evaluator_error(loaded_policy: LoadedPolicy) -> None:
    ev = Answers(noul=0.9, disclosure=None)
    result = await make_gate(strict(loaded_policy), MemorySink(), evaluator=ev).handle("Share the chart.")
    assert result.decision.outcome == "blocked"  # fail_mode closed
    assert any("disclosure question" in r for r in result.decision.reasons)


async def test_standard_mode_ignores_disclosure(loaded_policy: LoadedPolicy) -> None:
    ev = Answers(noul=0.9, disclosure=0.9)
    result = await make_gate(loaded_policy, MemorySink(), evaluator=ev).handle("Email Jordan's results to her boss.")
    assert DISCLOSURE not in ev.asked and result.decision.outcome == "frontier"
