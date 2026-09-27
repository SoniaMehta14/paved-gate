"""Paved Gate orchestration: mask -> evaluate (one parallel round) -> decide -> audit -> route."""

from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import JsonValue

from paved_gate.audit.models import (
    EvaluatorInfo,
    GateDecisionEntry,
    LatencyInfo,
    SensitiveDataEvent,
    SensitiveInfo,
    UsageInfo,
)
from paved_gate.audit.sink import AuditSink, sink_from_config
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
from paved_gate.handlers.base import FrontierHandler
from paved_gate.handlers.deterministic import DeterministicRegistry, default_registry
from paved_gate.policy.loader import LoadedPolicy, load_policy
from paved_gate.policy.models import DownstreamConfig, EvaluatorConfig, Policy
from paved_gate.privacy.detect import detect
from paved_gate.privacy.mask import mask
from paved_gate.types import (
    Blocked,
    Deterministic,
    Frontier,
    GateDecision,
    GateResult,
    GateState,
    Outcome,
    Route,
)

INTENT, RISK, SENSITIVE = "intent", "risk", "sensitive"
DEFAULT_POLICY_PATH = Path("policy/paved_gate.policy.yaml")


def build_questions(policy: Policy) -> dict[str, Question]:
    q = policy.questions
    return {
        INTENT: ChoiceQuestion(instructions=q.intent.instructions, criteria=dict(q.intent.criteria)),
        RISK: ScoreQuestion(instructions=q.risk.instructions, criteria=list(q.risk.criteria)),
        SENSITIVE: TruthQuestion(instructions=q.sensitive.instructions),
    }


@dataclass(frozen=True)
class _Answers:
    intent: ChoiceAnswer
    risk: ScoreAnswer
    sensitive: TruthAnswer


def _extract(evaluation: EvaluationResult) -> _Answers:
    """Validate that the evaluator answered all three questions with the right types."""
    a = evaluation.answers
    intent, risk, sensitive = a.get(INTENT), a.get(RISK), a.get(SENSITIVE)
    if not (isinstance(intent, ChoiceAnswer) and isinstance(risk, ScoreAnswer) and isinstance(sensitive, TruthAnswer)):
        raise EvaluatorError("evaluator did not return intent/risk/sensitive answers of the expected types")
    if intent.choice not in {r.value for r in Route}:
        raise EvaluatorError(f"unknown intent route {intent.choice!r}")
    return _Answers(intent, risk, sensitive)


@dataclass(frozen=True)
class _Verdict:
    outcome: Outcome
    route: Route | None
    reasons: list[str]
    intent_confidence: float | None
    risk_score: float | None
    noul: float | None


class PavedGate:
    def __init__(
        self,
        policy: LoadedPolicy,
        *,
        evaluator: FastEvaluator,
        audit: AuditSink,
        deterministic: DeterministicRegistry,
        frontier: FrontierHandler | None,
    ) -> None:
        self.loaded = policy
        self.policy = policy.policy
        self.evaluator = evaluator
        self.audit = audit
        self.deterministic = deterministic
        self.frontier = frontier
        self.questions: Mapping[str, Question] = build_questions(self.policy)

    async def handle(
        self,
        text: str,
        *,
        request_id: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> GateResult:
        started = time.perf_counter()
        state = GateState(request_id=request_id or str(uuid.uuid4()), input=text, metadata=metadata or {})
        privacy = self.policy.privacy
        findings = detect(text, privacy.detectors)
        detectors: list[str] = sorted({f.detector for f in findings})
        outbound = mask(text, findings) if privacy.mask_before_evaluate else text

        evaluation: EvaluationResult | None = None
        answers: _Answers | None = None
        error: str | None = None
        try:
            async with asyncio.timeout(self.policy.evaluator.timeout_ms / 1000):
                evaluation = await self.evaluator.evaluate(outbound, self.questions)
            answers = _extract(evaluation)
        except TimeoutError:
            error = f"evaluator timed out after {self.policy.evaluator.timeout_ms}ms"
        except Exception as exc:  # any evaluator failure is handled by fail_mode
            error = f"evaluator error: {type(exc).__name__}: {exc}"

        verdict = self._decide(answers, error)
        # Pre-routing target: the deterministic registry may still fall through to frontier.
        target: str | None = None
        det: tuple[str, dict[str, JsonValue]] | None = None
        if verdict.outcome == "deterministic":
            det = self.deterministic.run(text)
            if det is None:
                verdict = _Verdict(
                    "frontier",
                    Route.FRONTIER_AGENT_REQUIRED,
                    [*verdict.reasons, "no deterministic handler matched; routed to frontier"],
                    verdict.intent_confidence,
                    verdict.risk_score,
                    verdict.noul,
                )
            else:
                target = f"deterministic:{det[0]}"
        if verdict.outcome == "frontier":
            target = f"frontier:{self.frontier.provider}" if self.frontier else "frontier:passthrough"

        noul_flag = verdict.noul is not None and verdict.noul >= self.policy.questions.sensitive.threshold
        local_flag = bool(detectors) and privacy.local_detection_counts_as_sensitive
        sensitive = noul_flag or local_flag
        gate_ms = round((time.perf_counter() - started) * 1000, 2)

        decision = GateDecision(
            request_id=state.request_id,
            outcome=verdict.outcome,
            route=verdict.route,
            reasons=verdict.reasons,
            intent_confidence=verdict.intent_confidence,
            risk_score=verdict.risk_score,
            sensitive=sensitive,
            evaluator_ms=evaluation.latency_ms if evaluation else None,
            gate_ms=gate_ms,
            policy_hash=self.loaded.policy_hash,
        )
        payload_sha = "sha256:" + hashlib.sha256(text.encode()).hexdigest()
        await self._write_audit(
            decision,
            evaluation,
            target,
            detectors,
            verdict.noul,
            noul_flag,
            outbound,
            payload_sha,
            error,
        )

        if verdict.outcome == "blocked":
            return Blocked(decision=decision)
        if det is not None:
            return Deterministic(decision=decision, handler=det[0], output=det[1])
        response = await self.frontier.respond(state) if self.frontier else None
        return Frontier(decision=decision, state=state, response=response)

    def _decide(self, answers: _Answers | None, error: str | None) -> _Verdict:
        q = self.policy.questions
        if answers is None:
            reason = f"{error}; fail_mode={self.policy.evaluator.fail_mode}"
            if self.policy.evaluator.fail_mode == "closed":
                return _Verdict("blocked", None, [reason], None, None, None)
            return _Verdict("frontier", Route.FRONTIER_AGENT_REQUIRED, [reason], None, None, None)

        intent, risk, noul = answers.intent, answers.risk, answers.sensitive
        route = Route(intent.choice)
        risk_score = round(q.risk.scale_min + risk.score, 2)
        reasons = [
            f"intent={route.value} (confidence={intent.confidence:.2f})",
            f"risk={risk_score} (block_above={q.risk.block_above})",
        ]
        v = (intent.confidence, risk_score, noul.probability)

        if route is Route.POLICY_VIOLATION:
            return _Verdict("blocked", route, [*reasons, "blocked: intent is POLICY_VIOLATION"], *v)
        if risk_score > q.risk.block_above:
            return _Verdict("blocked", route, [*reasons, "blocked: risk above threshold"], *v)
        if route is Route.DIRECT_CODE_EXEC:
            if intent.confidence >= q.intent.min_confidence:
                return _Verdict("deterministic", route, reasons, *v)
            reasons.append(f"intent confidence below min_confidence={q.intent.min_confidence}; routed to frontier")
            return _Verdict("frontier", Route.FRONTIER_AGENT_REQUIRED, reasons, *v)
        return _Verdict("frontier", route, reasons, *v)

    async def _write_audit(
        self,
        decision: GateDecision,
        evaluation: EvaluationResult | None,
        target: str | None,
        detectors: list[str],
        noul: float | None,
        noul_flag: bool,
        outbound: str,
        payload_sha: str,
        error: str | None,
    ) -> None:
        cfg = self.policy.evaluator
        masked = self.policy.privacy.mask_before_evaluate
        scores: dict[str, JsonValue] = (
            {qid: a.model_dump(mode="json") for qid, a in evaluation.answers.items()} if evaluation else {}
        )
        if decision.risk_score is not None:
            scores["risk_on_policy_scale"] = decision.risk_score
        await self.audit.write(
            GateDecisionEntry(
                request_id=decision.request_id,
                policy_name=self.policy.name,
                policy_hash=decision.policy_hash,
                evaluator=EvaluatorInfo(
                    name=getattr(self.evaluator, "name", type(self.evaluator).__name__),
                    model=evaluation.model if evaluation else None,
                    mode=evaluation.mode if evaluation else cfg.mode,
                ),
                outcome=decision.outcome,
                route=decision.route,
                target=target,
                reasons=decision.reasons,
                scores=scores,
                raw_evaluator_response=evaluation.raw if evaluation else None,
                sensitive=SensitiveInfo(
                    flag=decision.sensitive,
                    noul_probability=noul,
                    local_detectors=detectors,
                    masked_before_evaluate=masked,
                ),
                latency=LatencyInfo(evaluator_ms=decision.evaluator_ms, gate_ms=decision.gate_ms),
                usage=UsageInfo(
                    input_tokens=evaluation.input_tokens if evaluation else None,
                    output_tokens=evaluation.output_tokens if evaluation else None,
                ),
                payload_sha256=payload_sha,
                payload_masked=outbound if self.policy.audit.log_masked_payload and masked else None,
                error=error,
            )
        )
        if decision.sensitive:
            sources = [f"local:{d}" for d in detectors]
            if noul_flag:
                sources.insert(0, "evaluator:noul")
            await self.audit.write(
                SensitiveDataEvent(
                    request_id=decision.request_id,
                    policy_hash=decision.policy_hash,
                    outcome=decision.outcome,
                    route=decision.route,
                    sources=sources,
                    noul_probability=noul,
                    local_detectors=detectors,
                    masked_before_evaluate=masked,
                    payload_sha256=payload_sha,
                )
            )

    async def aclose(self) -> None:
        await self.evaluator.aclose()


def evaluator_from_config(cfg: EvaluatorConfig) -> FastEvaluator:
    if cfg.provider == "heuristic":
        return HeuristicEvaluator()
    return JevEvaluator.from_config(cfg)


def frontier_from_config(cfg: DownstreamConfig) -> FrontierHandler | None:
    if cfg.provider == "anthropic":
        from paved_gate.handlers.anthropic_handler import AnthropicFrontierHandler

        return AnthropicFrontierHandler(
            model=cfg.model or "claude-opus-5", max_tokens=cfg.max_tokens, system=cfg.system_prompt
        )
    if cfg.provider == "openai":
        from paved_gate.handlers.openai_handler import OpenAIFrontierHandler

        return OpenAIFrontierHandler(model=cfg.model or "gpt-4o", max_tokens=cfg.max_tokens, system=cfg.system_prompt)
    return None


def paved_gate(
    policy: str | Path | LoadedPolicy = DEFAULT_POLICY_PATH,
    *,
    evaluator: FastEvaluator | None = None,
    audit: AuditSink | None = None,
    deterministic: DeterministicRegistry | None = None,
    frontier: FrontierHandler | Literal["from_policy"] | None = "from_policy",
) -> PavedGate:
    """Build a gate from a policy file (or a pre-loaded policy).

    Every dependency defaults to what the policy describes and can be overridden,
    e.g. `frontier=None` to handle FRONTIER_AGENT_REQUIRED requests yourself.
    """
    loaded = policy if isinstance(policy, LoadedPolicy) else load_policy(policy)
    p = loaded.policy
    return PavedGate(
        loaded,
        evaluator=evaluator or evaluator_from_config(p.evaluator),
        audit=audit or sink_from_config(p.audit),
        deterministic=deterministic or default_registry(),
        frontier=frontier_from_config(p.downstream) if frontier == "from_policy" else frontier,
    )
