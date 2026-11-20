"""profile: strict runs a secondary evaluator on every request; either can block, both must agree to auto execute."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
from pydantic import ValidationError

from paved_gate import EvaluationResult, LoadedPolicy, MemorySink, PavedGate, default_registry, paved_gate
from paved_gate.evaluator.base import EvaluatorError, EvaluatorState, Question
from paved_gate.evaluator.heuristic import HeuristicEvaluator
from paved_gate.policy.loader import from_policy
from paved_gate.policy.models import EvaluatorConfig, Policy
from tests.conftest import ScriptedEvaluator


class Failing:
    name = "failing"

    async def evaluate(self, state: EvaluatorState, questions: Mapping[str, Question]) -> EvaluationResult:
        raise EvaluatorError("down")

    async def aclose(self) -> None:
        return None


def gate(policy: LoadedPolicy, primary: object, secondary: object) -> PavedGate:
    return PavedGate(
        policy,
        evaluator=primary,  # type: ignore[arg-type]
        audit=MemorySink(),
        deterministic=default_registry(),
        frontier=None,
        secondary=secondary,  # type: ignore[arg-type]
    )


def test_strict_needs_a_secondary_and_default_hash_is_unchanged(loaded_policy: LoadedPolicy) -> None:
    data = loaded_policy.policy.model_dump()
    with pytest.raises(ValidationError):
        Policy.model_validate({**data, "profile": "strict"})
    strict = Policy.model_validate({**data, "profile": "strict", "secondary_evaluator": {"provider": "heuristic"}})
    assert from_policy(strict).policy_hash != loaded_policy.policy_hash
    assert from_policy(Policy.model_validate(data)).policy_hash == loaded_policy.policy_hash


@pytest.mark.parametrize(
    ("primary_risk", "secondary_risk", "outcome"),
    [(0.0, 0.0, "frontier"), (4.0, 0.0, "blocked"), (0.0, 4.0, "blocked"), (4.0, 4.0, "blocked")],
)
async def test_either_evaluator_can_block(
    loaded_policy: LoadedPolicy, primary_risk: float, secondary_risk: float, outcome: str
) -> None:
    primary = ScriptedEvaluator(risk_level=primary_risk)
    secondary = ScriptedEvaluator(risk_level=secondary_risk)
    result = await gate(loaded_policy, primary, secondary).handle("anything")
    assert result.decision.outcome == outcome
    assert primary.seen_states and secondary.seen_states
    assert any(r.startswith("secondary (scripted-1): ") for r in result.decision.reasons)


async def test_auto_execute_needs_both(loaded_policy: LoadedPolicy) -> None:
    both = gate(loaded_policy, ScriptedEvaluator(route="DIRECT_CODE_EXEC"), ScriptedEvaluator(route="DIRECT_CODE_EXEC"))
    assert (await both.handle("What is 2 + 2?")).decision.outcome == "deterministic"
    one = gate(loaded_policy, ScriptedEvaluator(route="DIRECT_CODE_EXEC"), ScriptedEvaluator())
    result = await one.handle("What is 2 + 2?")
    assert result.decision.outcome == "frontier"
    assert any("did not agree on DIRECT_CODE_EXEC" in r for r in result.decision.reasons)


async def test_a_failing_secondary_falls_to_fail_mode(loaded_policy: LoadedPolicy) -> None:
    result = await gate(loaded_policy, ScriptedEvaluator(), Failing()).handle("hello")
    assert result.decision.outcome == "blocked"  # fail_mode closed
    assert any("evaluator error" in r for r in result.decision.reasons)


def test_factory_builds_the_secondary_from_the_policy(loaded_policy: LoadedPolicy) -> None:
    p = loaded_policy.policy.model_copy(
        update={"profile": "strict", "secondary_evaluator": EvaluatorConfig(provider="heuristic")}
    )
    g = paved_gate(from_policy(p), audit=MemorySink(), frontier=None)
    assert isinstance(g.secondary, HeuristicEvaluator)
    assert paved_gate(loaded_policy, audit=MemorySink(), frontier=None).secondary is None
