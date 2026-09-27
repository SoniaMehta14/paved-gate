"""Paved Gate: a fast System 1 ingestion gate for AI agent architectures."""

from paved_gate.audit.models import AuditRecord, GateDecisionEntry, SensitiveDataEvent
from paved_gate.audit.sink import AuditSink, JsonlFileSink, MemorySink, StdoutSink
from paved_gate.evaluator.base import (
    ChoiceAnswer,
    ChoiceQuestion,
    EvaluationResult,
    EvaluatorError,
    FastEvaluator,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    TruthAnswer,
    TruthQuestion,
)
from paved_gate.evaluator.heuristic import HeuristicEvaluator
from paved_gate.evaluator.jev import JevEvaluator
from paved_gate.gate import PavedGate, paved_gate
from paved_gate.handlers.base import DeterministicHandler, FrontierHandler
from paved_gate.handlers.deterministic import DeterministicRegistry, default_registry
from paved_gate.policy.loader import LoadedPolicy, load_policy
from paved_gate.policy.models import Policy
from paved_gate.types import (
    Blocked,
    Deterministic,
    Frontier,
    FrontierResponse,
    GateDecision,
    GateResult,
    GateState,
    Route,
)

__all__ = [
    "AuditRecord",
    "AuditSink",
    "Blocked",
    "ChoiceAnswer",
    "ChoiceQuestion",
    "Deterministic",
    "DeterministicHandler",
    "DeterministicRegistry",
    "EvaluationResult",
    "EvaluatorError",
    "FastEvaluator",
    "Frontier",
    "FrontierHandler",
    "FrontierResponse",
    "GateDecision",
    "GateDecisionEntry",
    "GateResult",
    "GateState",
    "HeuristicEvaluator",
    "JevEvaluator",
    "JsonlFileSink",
    "LoadedPolicy",
    "MemorySink",
    "PavedGate",
    "Policy",
    "Question",
    "Route",
    "ScoreAnswer",
    "ScoreQuestion",
    "SensitiveDataEvent",
    "StdoutSink",
    "TruthAnswer",
    "TruthQuestion",
    "default_registry",
    "load_policy",
    "paved_gate",
]
