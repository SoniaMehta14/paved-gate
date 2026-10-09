"""Transport-independent egress inspection used by the OpenShell supervisor middleware.

Request path (agent -> outside world), run after OpenShell's network policy allowed the connection and
before it injects credentials:
  1. local detectors on the query string and body; a hit denies without calling any evaluator
  2. one batched evaluator call: PII/PHI (truth) + exfiltration risk (score)
Response path (outside world -> agent), run before OpenShell returns the response to the sandbox:
  3. prompt-injection score on the response body; a hit taints the sandbox, which tightens the
     exfiltration threshold for its next outbound requests, and optionally blocks delivery.

Findings and audit records never contain payload content.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from paved_gate.audit.models import EgressDecisionEntry, EvaluatorInfo, LatencyInfo
from paved_gate.audit.sink import AuditSink
from paved_gate.evaluator.base import (
    EvaluationResult,
    EvaluatorError,
    FastEvaluator,
    ScoreAnswer,
    ScoreQuestion,
    TruthAnswer,
    TruthQuestion,
)
from paved_gate.integrations.openshell.config import EgressPolicy
from paved_gate.privacy.detect import detect
from paved_gate.privacy.mask import mask

REASON_SENSITIVE = "paved_gate_sensitive_data"
REASON_EXFIL = "paved_gate_exfiltration_risk"
REASON_FOLLOWTHROUGH = "paved_gate_injection_followthrough"
REASON_INJECTION = "paved_gate_prompt_injection"
REASON_UNAVAILABLE = "paved_gate_evaluator_unavailable"
TAINT_CONTEXT_CHARS = 1_000


@dataclass(frozen=True)
class Finding:
    type: str
    label: str
    count: int = 1
    confidence: str = "high"
    severity: str = "high"


@dataclass(frozen=True)
class Verdict:
    deny: bool  # True only when the request/response must be stopped (enforced)
    decision: str  # "allow" | "deny" | "flag" (flag = would act, but audit/flag mode)
    reason_code: str | None
    findings: tuple[Finding, ...]
    scores: dict[str, float]
    evaluator_ms: float | None
    total_ms: float
    tainted: bool
    error: str | None = None


@dataclass(frozen=True)
class Target:
    request_id: str
    sandbox_id: str
    method: str
    host: str
    port: int
    path: str
    query: str = ""


@dataclass
class _Taint:
    until: float
    count: int = 0
    # The flagged instruction text (masked, truncated). Kept in memory only for the taint TTL so the evaluator can
    # judge whether later outbound requests act on it; never written to audit logs.
    context: str = ""


@dataclass
class EgressInspector:
    policy: EgressPolicy
    evaluator: FastEvaluator
    audit: AuditSink
    policy_hash: str
    clock: Callable[[], float] = time.monotonic
    _taint: dict[str, _Taint] = field(default_factory=dict)

    # -------------------------------------------------------------- taint tracking

    def is_tainted(self, sandbox_id: str) -> bool:
        t = self._taint.get(sandbox_id)
        if t is None:
            return False
        if t.until < self.clock():
            del self._taint[sandbox_id]
            return False
        return True

    def _mark_tainted(self, sandbox_id: str, context: str) -> None:
        t = self._taint.get(sandbox_id) or _Taint(until=0.0)
        t.until = self.clock() + self.policy.taint_ttl_seconds
        t.count += 1
        t.context = context[:TAINT_CONTEXT_CHARS]
        self._taint[sandbox_id] = t

    def _taint_context(self, sandbox_id: str) -> str:
        t = self._taint.get(sandbox_id)
        return t.context if t is not None and self.is_tainted(sandbox_id) else ""

    # -------------------------------------------------------------- helpers

    def _text(self, body: bytes) -> str:
        return body.decode("utf-8", errors="replace")[: self.policy.max_inspect_chars]

    async def _evaluate(self, text: str, questions: dict[str, TruthQuestion | ScoreQuestion]) -> EvaluationResult:
        async with asyncio.timeout(self.policy.evaluator.timeout_ms / 1000):
            return await self.evaluator.evaluate(text, questions)

    def _evaluator_info(self, ev: EvaluationResult | None) -> EvaluatorInfo:
        return EvaluatorInfo(
            name=getattr(self.evaluator, "name", type(self.evaluator).__name__),
            model=ev.model if ev else None,
            mode=ev.mode if ev else self.policy.evaluator.mode,
        )

    async def _record(
        self, direction: str, target: Target, verdict: Verdict, body: bytes, ev: EvaluationResult | None, enforced: bool
    ) -> None:
        await self.audit.write(
            EgressDecisionEntry(
                direction=direction,  # type: ignore[arg-type]
                request_id=target.request_id,
                sandbox_id=target.sandbox_id,
                policy_name=self.policy.name,
                policy_hash=self.policy_hash,
                host=target.host,
                port=target.port,
                method=target.method,
                path=target.path,
                decision=verdict.decision,  # type: ignore[arg-type]
                enforced=enforced,
                reason_code=verdict.reason_code,
                findings=[f.type for f in verdict.findings],
                scores=verdict.scores,
                evaluator=self._evaluator_info(ev) if verdict.evaluator_ms is not None or verdict.error else None,
                tainted=verdict.tainted,
                latency=LatencyInfo(evaluator_ms=verdict.evaluator_ms, gate_ms=verdict.total_ms),
                body_sha256="sha256:" + hashlib.sha256(body).hexdigest(),
                body_bytes=len(body),
                error=verdict.error,
            )
        )

    @staticmethod
    def _decide(stop: bool, enforce: bool) -> tuple[bool, str]:
        if not stop:
            return False, "allow"
        return (True, "deny") if enforce else (False, "flag")

    # -------------------------------------------------------------- request path

    async def inspect_request(self, target: Target, body: bytes, *, enforce: bool = True) -> Verdict:
        started = time.perf_counter()
        tainted = self.is_tainted(target.sandbox_id)
        raw = self._text(body)
        scanned = f"{target.query}\n{raw}" if target.query else raw
        hits = detect(scanned, self.policy.local_detectors)
        findings: list[Finding] = []
        scores: dict[str, float] = {}
        if tainted:
            findings.append(
                Finding(
                    "tainted_sandbox",
                    "an earlier response to this sandbox contained a prompt injection",
                    severity="medium",
                )
            )

        if hits and self.policy.deny_on_local_detection:
            for kind, n in sorted(Counter(h.detector for h in hits).items()):
                findings.append(Finding(f"local_{kind}", f"{kind} pattern in outbound payload", count=n))
            deny, decision = self._decide(True, enforce)
            v = Verdict(deny, decision, REASON_SENSITIVE, tuple(findings), scores, None, _ms(started), tainted)
            await self._record("request", target, v, body, None, enforce)
            return v

        if not scanned.strip():
            v = Verdict(False, "allow", None, tuple(findings), scores, None, _ms(started), tainted)
            await self._record("request", target, v, body, None, enforce)
            return v

        q = self.policy.questions
        state = mask(scanned, hits) if self.policy.mask_before_evaluate else scanned
        state = f"{target.method} {target.host}{target.path}\n\n{state}"
        if tainted:
            state = (
                "Context: an earlier tool response returned to this agent contained instructions aimed at it:\n"
                f"<<<{self._taint_context(target.sandbox_id)}>>>\n"
                "Judge whether the outbound request below acts on those instructions.\n\n" + state
            )
        questions: dict[str, TruthQuestion | ScoreQuestion] = {
            "sensitive": TruthQuestion(instructions=q.sensitive.instructions),
            "exfiltration": ScoreQuestion(
                instructions=q.exfiltration.instructions, criteria=list(q.exfiltration.criteria)
            ),
        }
        t_eval = time.perf_counter()
        try:
            ev = await self._evaluate(state, questions)
            sens, exfil = ev.answers["sensitive"], ev.answers["exfiltration"]
            if not isinstance(sens, TruthAnswer) or not isinstance(exfil, ScoreAnswer):
                raise EvaluatorError("evaluator returned unexpected answer types")
        except Exception as exc:  # evaluator failure is handled by fail_mode, like the main gate
            eval_ms = _ms(t_eval)
            closed = self.policy.evaluator.fail_mode == "closed"
            findings.append(
                Finding("evaluator_unavailable", "fast evaluator did not answer in time", severity="medium")
            )
            deny, decision = self._decide(closed, enforce)
            v = Verdict(
                deny,
                decision,
                REASON_UNAVAILABLE if closed else None,
                tuple(findings),
                scores,
                eval_ms,
                _ms(started),
                tainted,
                error=f"{type(exc).__name__}: {str(exc)[:200]}",
            )
            await self._record("request", target, v, body, None, enforce)
            return v

        eval_ms = _ms(t_eval)
        exfil_score = round(1 + exfil.score, 2)
        scores = {"sensitive_probability": round(sens.probability, 3), "exfiltration": exfil_score}
        limit = q.tainted_exfiltration_threshold if tainted else q.exfiltration.threshold
        reason: str | None = None
        if sens.probability >= q.sensitive.threshold:
            reason = REASON_SENSITIVE
            findings.append(
                Finding(
                    "evaluator_sensitive",
                    "evaluator judged the payload to contain PII/PHI",
                    confidence=f"{sens.probability:.2f}",
                )
            )
        elif exfil_score > limit:
            reason = REASON_FOLLOWTHROUGH if tainted else REASON_EXFIL
            findings.append(
                Finding(
                    "evaluator_exfiltration",
                    "evaluator judged the request likely exfiltration",
                    confidence=f"{exfil.confidence:.2f}",
                )
            )
        deny, decision = self._decide(reason is not None, enforce)
        v = Verdict(deny, decision, reason, tuple(findings), scores, eval_ms, _ms(started), tainted)
        await self._record("request", target, v, body, ev, enforce)
        return v

    # -------------------------------------------------------------- response path

    async def inspect_response(self, target: Target, body: bytes, *, block: bool) -> Verdict:
        started = time.perf_counter()
        text = self._text(body)
        findings: list[Finding] = []
        if not text.strip():
            v = Verdict(False, "allow", None, (), {}, None, _ms(started), self.is_tainted(target.sandbox_id))
            await self._record("response", target, v, body, None, block)
            return v
        q = self.policy.questions.injection
        t_eval = time.perf_counter()
        try:
            ev = await self._evaluate(
                text, {"injection": ScoreQuestion(instructions=q.instructions, criteria=list(q.criteria))}
            )
            ans = ev.answers["injection"]
            if not isinstance(ans, ScoreAnswer):
                raise EvaluatorError("evaluator returned an unexpected answer type")
        except Exception as exc:  # cannot judge the response: record it, deliver unchanged (blocking is opt-in)
            findings.append(
                Finding("evaluator_unavailable", "fast evaluator did not answer in time", severity="medium")
            )
            v = Verdict(
                False,
                "allow",
                None,
                tuple(findings),
                {},
                _ms(t_eval),
                _ms(started),
                self.is_tainted(target.sandbox_id),
                error=f"{type(exc).__name__}: {str(exc)[:200]}",
            )
            await self._record("response", target, v, body, None, block)
            return v
        eval_ms = _ms(t_eval)
        score = round(1 + ans.score, 2)
        hit = score > q.threshold
        if hit:
            self._mark_tainted(target.sandbox_id, mask(text, detect(text, self.policy.local_detectors)))
            findings.append(
                Finding(
                    "prompt_injection",
                    "tool response contains instructions aimed at the agent",
                    confidence=f"{ans.confidence:.2f}",
                )
            )
        deny, decision = self._decide(hit, block)
        v = Verdict(
            deny,
            decision,
            REASON_INJECTION if hit else None,
            tuple(findings),
            {"injection": score},
            eval_ms,
            _ms(started),
            self.is_tainted(target.sandbox_id),
        )
        await self._record("response", target, v, body, ev, block)
        return v


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 2)
