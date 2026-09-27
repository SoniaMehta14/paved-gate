from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path

import pytest

from paved_gate import (
    ChoiceAnswer,
    EvaluationResult,
    FrontierResponse,
    GateState,
    JevEvaluator,
    LoadedPolicy,
    MemorySink,
    PavedGate,
    Question,
    ScoreAnswer,
    TruthAnswer,
    default_registry,
    load_policy,
)
from paved_gate.evaluator.base import Answer, EvaluatorState
from paved_gate.handlers.base import FrontierHandler

POLICY_PATH = Path(__file__).resolve().parent.parent / "policy" / "paved_gate.policy.yaml"


@pytest.fixture
def loaded_policy() -> LoadedPolicy:
    return load_policy(POLICY_PATH)


@pytest.fixture
def sink() -> MemorySink:
    return MemorySink()


def make_gate(
    policy: LoadedPolicy,
    sink: MemorySink,
    *,
    evaluator: object | None = None,
    frontier: FrontierHandler | None = None,
) -> PavedGate:
    return PavedGate(
        policy,
        evaluator=evaluator or JevEvaluator(mode="mock", mock_latency_ms=(0, 0)),  # type: ignore[arg-type]
        audit=sink,
        deterministic=default_registry(),
        frontier=frontier,
    )


class ScriptedEvaluator:
    """Returns fixed answers and records what it was sent."""

    name = "scripted"

    def __init__(
        self,
        *,
        route: str = "FRONTIER_AGENT_REQUIRED",
        confidence: float = 0.9,
        risk_level: float = 0.0,
        noul: float = 0.05,
        delay_s: float = 0.0,
    ) -> None:
        self.route, self.confidence, self.risk_level, self.noul = route, confidence, risk_level, noul
        self.delay_s = delay_s
        self.seen_states: list[EvaluatorState] = []

    async def evaluate(self, state: EvaluatorState, questions: Mapping[str, Question]) -> EvaluationResult:
        self.seen_states.append(state)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        answers: dict[str, Answer] = {
            "intent": ChoiceAnswer(choice=self.route, confidence=self.confidence, probabilities={self.route: 1.0}),
            "risk": ScoreAnswer(score=self.risk_level, confidence=0.9, probabilities={}),
            "sensitive": TruthAnswer(probability=self.noul),
        }
        return EvaluationResult(
            evaluator=self.name,
            model="scripted-1",
            mode="test",
            answers=answers,
            raw={"scripted": True},
            latency_ms=0.0,
        )

    async def aclose(self) -> None:
        return None


class RecordingFrontier:
    provider = "recording"

    def __init__(self) -> None:
        self.states: list[GateState] = []

    async def respond(self, state: GateState) -> FrontierResponse:
        self.states.append(state)
        return FrontierResponse(provider=self.provider, model="fake", text="ok", stop_reason="end_turn")
