"""The multimodal gate: typed checks over an image or PDF plus text, answered in one evaluator call."""

from __future__ import annotations

import hashlib
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from paved_gate.audit.models import (
    EvaluatorInfo,
    LatencyInfo,
    MultimodalArtifact,
    MultimodalCheckResult,
    MultimodalDecisionEntry,
    UsageInfo,
)
from paved_gate.evaluator.base import (
    ChoiceAnswer,
    ChoiceQuestion,
    EvaluationResult,
    FastEvaluator,
    MediaPart,
    MultimodalEvaluator,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    TruthAnswer,
    TruthQuestion,
)
from paved_gate.gate import evaluator_from_config
from paved_gate.integrations.multimodal_gate.config import (
    MultimodalPolicy,
    load_multimodal_policy,
    multimodal_policy_hash,
)

# File types the gate accepts, by suffix, with the leading bytes each must start with.
FILE_TYPES: dict[str, tuple[str, tuple[bytes, ...]]] = {
    ".png": ("image/png", (b"\x89PNG\r\n\x1a\n",)),
    ".jpg": ("image/jpeg", (b"\xff\xd8\xff",)),
    ".jpeg": ("image/jpeg", (b"\xff\xd8\xff",)),
    ".webp": ("image/webp", (b"RIFF",)),
    ".pdf": ("application/pdf", (b"%PDF-",)),
}


Decision = Literal["allow", "review", "block"]


class MultimodalGateError(ValueError):
    """The artifact or the evaluator cannot be used; raised before anything is sent."""


def read_artifact(path: Path, max_bytes: int) -> MediaPart:
    """Check type and size from the file system first, then read and check the leading bytes."""
    kind = FILE_TYPES.get(path.suffix.lower())
    if kind is None:
        raise MultimodalGateError(f"{path.name}: unsupported file type; use one of {sorted(FILE_TYPES)}")
    size = path.stat().st_size
    if size > max_bytes:
        raise MultimodalGateError(f"{path.name} is {size:,} bytes, over the policy limit of {max_bytes:,} bytes")
    data = path.read_bytes()
    mime, magic = kind
    if not data.startswith(magic):
        raise MultimodalGateError(f"{path.name}: content is not a {mime} file")
    return MediaPart(mime_type=mime, data=data, name=path.name)


def build_questions(policy: MultimodalPolicy) -> dict[str, Question]:
    qs: dict[str, Question] = {k: TruthQuestion(instructions=c.instructions) for k, c in policy.truth.items()}
    qs |= {k: ChoiceQuestion(instructions=c.instructions, criteria=c.criteria) for k, c in policy.choice.items()}
    qs |= {k: ScoreQuestion(instructions=c.instructions, criteria=c.criteria) for k, c in policy.score.items()}
    return qs


def check_answers(
    policy: MultimodalPolicy, result: EvaluationResult, expect: Mapping[str, str]
) -> list[MultimodalCheckResult]:
    out: list[MultimodalCheckResult] = []
    for name, t in policy.truth.items():
        ans = result.answers[name]
        assert isinstance(ans, TruthAnswer)
        p = ans.probability if t.expect == "yes" else 1 - ans.probability
        ok = p >= t.min_probability
        reason = (
            f"{name}: probability of {t.expect!r} is {p:.2f}, {'at or above' if ok else 'below'} {t.min_probability}"
        )
        out.append(
            MultimodalCheckResult(
                name=name,
                kind="truth",
                passed=ok,
                on_fail=t.on_fail,
                value=round(p, 4),
                threshold=t.min_probability,
                expected=t.expect,
                reason=reason,
            )
        )
    for name, c in policy.choice.items():
        ans = result.answers[name]
        assert isinstance(ans, ChoiceAnswer)
        expected = expect.get(name, c.expect)
        p = ans.probabilities.get(ans.choice, ans.confidence)
        if expected is not None and ans.choice != expected:
            ok, reason = False, f"{name}: the artifact shows {ans.choice!r}, expected {expected!r}"
        elif p < c.min_confidence:
            ok, reason = False, f"{name}: {ans.choice!r} at probability {p:.2f}, below {c.min_confidence}"
        else:
            ok, reason = True, f"{name}: {ans.choice!r} at probability {p:.2f}"
        out.append(
            MultimodalCheckResult(
                name=name,
                kind="choice",
                passed=ok,
                on_fail=c.on_fail,
                value=ans.choice,
                threshold=c.min_confidence,
                expected=expected,
                reason=reason,
            )
        )
    for name, s in policy.score.items():
        ans = result.answers[name]
        assert isinstance(ans, ScoreAnswer)
        level = round(ans.score + 1, 2)
        ok = level >= s.min_level
        reason = f"{name}: level {level:g} of {len(s.criteria)}, {'at or above' if ok else 'below'} {s.min_level:g}"
        out.append(
            MultimodalCheckResult(
                name=name,
                kind="score",
                passed=ok,
                on_fail=s.on_fail,
                value=level,
                threshold=s.min_level,
                expected=None,
                reason=reason,
            )
        )
    return out


@dataclass
class MultimodalGate:
    policy: MultimodalPolicy
    evaluator: MultimodalEvaluator
    policy_hash: str
    questions: dict[str, Question] = field(init=False)

    def __post_init__(self) -> None:
        self.questions = build_questions(self.policy)

    async def evaluate(
        self, artifact: Path, text: str = "", expect: Mapping[str, str] | None = None
    ) -> MultimodalDecisionEntry:
        started = time.perf_counter()
        expect = dict(expect or {})
        unknown = sorted(set(expect) - set(self.policy.choice))
        if unknown:
            raise MultimodalGateError(f"--expect names no choice check: {unknown}")
        for name, value in expect.items():
            if value not in self.policy.choice[name].criteria:
                raise MultimodalGateError(
                    f"--expect {name}={value}: not one of {sorted(self.policy.choice[name].criteria)}"
                )
        part = read_artifact(artifact, self.policy.max_media_bytes)
        info = MultimodalArtifact(
            name=part.name, mime_type=part.mime_type, bytes=len(part.data), sha256=hashlib.sha256(part.data).hexdigest()
        )
        text_sha = hashlib.sha256(text.encode()).hexdigest() if text else None

        def entry(decision: Decision, reasons: list[str], **fields: object) -> MultimodalDecisionEntry:
            return MultimodalDecisionEntry.model_validate(
                {
                    "request_id": uuid.uuid4().hex,
                    "policy_name": self.policy.name,
                    "policy_hash": self.policy_hash,
                    "artifact": info,
                    "text_sha256": text_sha,
                    "decision": decision,
                    "reasons": reasons,
                    **fields,
                }
            )

        try:
            result = await self.evaluator.evaluate_media(text, [part], self.questions)
            checks = check_answers(self.policy, result, expect)
        except Exception as exc:  # any evaluator failure is handled by fail_mode, like the main gate
            fail_mode = self.policy.evaluator.fail_mode
            error = f"{type(exc).__name__}: {exc}"
            return entry(
                "block" if fail_mode == "closed" else "review",
                [f"evaluator failed ({error}); fail_mode is {fail_mode}"],
                checks=[],
                evaluator=None,
                latency=LatencyInfo(evaluator_ms=None, gate_ms=round((time.perf_counter() - started) * 1000, 2)),
                usage=UsageInfo(input_tokens=None, output_tokens=None),
                error=error,
            )
        failed = [c for c in checks if not c.passed]
        decision: Decision = "block" if any(c.on_fail == "block" for c in failed) else "review" if failed else "allow"
        return entry(
            decision,
            [c.reason for c in failed] or ["every check passed"],
            checks=checks,
            evaluator=EvaluatorInfo(name=result.evaluator, model=result.model, mode=result.mode),
            latency=LatencyInfo(
                evaluator_ms=result.latency_ms, gate_ms=round((time.perf_counter() - started) * 1000, 2)
            ),
            usage=UsageInfo(input_tokens=result.input_tokens, output_tokens=result.output_tokens),
        )

    async def aclose(self) -> None:
        await self.evaluator.aclose()


def build_gate(policy: MultimodalPolicy | str | Path, *, evaluator: FastEvaluator | None = None) -> MultimodalGate:
    p = policy if isinstance(policy, MultimodalPolicy) else load_multimodal_policy(policy)
    ev = evaluator or evaluator_from_config(p.evaluator)
    if not isinstance(ev, MultimodalEvaluator):
        raise MultimodalGateError(
            f"evaluator {ev.name!r} reads text only in this adapter; the multimodal gate needs gemini_decision"
        )
    return MultimodalGate(policy=p, evaluator=ev, policy_hash=multimodal_policy_hash(p))
