"""Multimodal gate: decisions over the synthetic PNG and PDF, media limits, and what the audit record keeps.

No network and no credentials. The PNG is the published synthetic report screenshot; the PDF is generated
by examples/multimodal_gate/make_pdf.py and must stay free of metadata.
"""

from __future__ import annotations

import base64
import hashlib
import random
import struct
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
from pydantic import JsonValue

from paved_gate import ChoiceAnswer, EvaluationResult, EvaluatorError, JevEvaluator, ScoreAnswer, TruthAnswer
from paved_gate.evaluator.base import Answer, EvaluatorState, MediaPart, Question
from paved_gate.evaluator.gemini_decision import GeminiDecisionEvaluator, Input, MediaError, media_input
from paved_gate.integrations.multimodal_gate import (
    MultimodalGateError,
    MultimodalPolicy,
    build_gate,
    load_multimodal_policy,
)

ROOT = Path(__file__).resolve().parent.parent
POLICY = ROOT / "policy" / "multimodal_gate.policy.yaml"
PNG = ROOT / "docs" / "harness2" / "1_synthetic_fixture.png"
PDF = ROOT / "tests" / "fixtures" / "multimodal" / "synthetic_policy.pdf"
sys.path.insert(0, str(ROOT / "examples" / "multimodal_gate"))

from make_pdf import LINES, build_pdf  # noqa: E402


class ScriptedMedia:
    """A multimodal evaluator with fixed answers; records the media it was given."""

    name = "scripted"

    def __init__(
        self, *, synthetic: float = 0.95, decision: str = "reject", legibility: float = 3.0, fail: bool = False
    ):
        self.synthetic, self.decision, self.legibility, self.fail = synthetic, decision, legibility, fail
        self.seen: list[tuple[str, list[MediaPart]]] = []

    async def evaluate(self, state: EvaluatorState, questions: Mapping[str, Question]) -> EvaluationResult:
        raise AssertionError("the multimodal gate must call evaluate_media")

    async def evaluate_media(
        self, state: str, media: Sequence[MediaPart], questions: Mapping[str, Question]
    ) -> EvaluationResult:
        self.seen.append((state, list(media)))
        if self.fail:
            raise EvaluatorError("scripted outage")
        probs = {k: (0.9 if k == self.decision else 0.05) for k in ("accept", "needs_review", "reject")}
        answers: dict[str, Answer] = {
            "synthetic_label": TruthAnswer(probability=self.synthetic),
            "overall_decision": ChoiceAnswer(choice=self.decision, confidence=0.9, probabilities=probs),
            "legibility": ScoreAnswer(score=self.legibility, confidence=0.8, probabilities={}),
        }
        return EvaluationResult(
            evaluator=self.name, model="scripted", mode="test", answers=answers, raw={}, latency_ms=1.0
        )

    async def aclose(self) -> None:
        return None


@pytest.fixture
def policy() -> MultimodalPolicy:
    return load_multimodal_policy(POLICY)


# The committed artifacts


def test_pdf_is_reproducible_and_has_no_metadata() -> None:
    data = PDF.read_bytes()
    assert data == build_pdf(LINES)
    assert data.startswith(b"%PDF-1.4") and b"SYNTHETIC TEST DOCUMENT" in data
    for marker in (b"/Info", b"/Author", b"/Producer", b"/Creator", b"/CreationDate", b"/Metadata", b"/Users"):
        assert marker not in data


def test_png_has_no_metadata_chunks() -> None:
    data, i, chunks = PNG.read_bytes(), 8, set()
    while i < len(data):
        (n,) = struct.unpack(">I", data[i : i + 4])
        chunks.add(data[i + 4 : i + 8].decode("latin1"))
        i += 12 + n
    assert chunks == {"IHDR", "IDAT", "IEND"}


# Decisions


async def test_all_checks_pass_allows(policy: MultimodalPolicy) -> None:
    ev = ScriptedMedia()
    entry = await build_gate(policy, evaluator=ev).evaluate(PNG, "audit says reject", {"overall_decision": "reject"})
    assert entry.decision == "allow" and entry.reasons == ["every check passed"]
    [(state, [part])] = ev.seen
    assert state == "audit says reject" and part.mime_type == "image/png" and part.name == PNG.name


async def test_choice_mismatch_blocks_and_low_truth_reviews(policy: MultimodalPolicy) -> None:
    blocked = await build_gate(policy, evaluator=ScriptedMedia(decision="accept")).evaluate(
        PNG, expect={"overall_decision": "reject"}
    )
    assert blocked.decision == "block"
    assert blocked.reasons == ["overall_decision: the artifact shows 'accept', expected 'reject'"]
    review = await build_gate(policy, evaluator=ScriptedMedia(synthetic=0.2)).evaluate(PDF)
    assert review.decision == "review" and review.reasons[0].startswith("synthetic_label: probability of 'yes' is 0.20")


@pytest.mark.parametrize(("fail_mode", "decision"), [("closed", "block"), ("open", "review")])
async def test_evaluator_failure_follows_fail_mode(policy: MultimodalPolicy, fail_mode: str, decision: str) -> None:
    p = policy.model_copy(update={"evaluator": policy.evaluator.model_copy(update={"fail_mode": fail_mode})})
    entry = await build_gate(p, evaluator=ScriptedMedia(fail=True)).evaluate(PDF)
    assert entry.decision == decision and entry.error == "EvaluatorError: scripted outage" and entry.checks == []


async def test_gemini_mock_end_to_end_and_audit_keeps_no_bytes(policy: MultimodalPolicy) -> None:
    ev = GeminiDecisionEvaluator(mode="mock", mock_latency_ms=(0, 0), rng=random.Random(0))
    for artifact, mime in ((PNG, "image/png"), (PDF, "application/pdf")):
        data = artifact.read_bytes()
        entry = await build_gate(policy, evaluator=ev).evaluate(artifact, "synthetic text state")
        assert entry.artifact.mime_type == mime and entry.artifact.bytes == len(data)
        assert entry.artifact.sha256 == hashlib.sha256(data).hexdigest()
        assert entry.evaluator is not None and (entry.evaluator.name, entry.evaluator.mode) == (
            "gemini_decision",
            "mock",
        )
        assert {c.name for c in entry.checks} == {"synthetic_label", "overall_decision", "legibility"}
        record = entry.model_dump_json()
        assert len(record) < 4000 and base64.b64encode(data[:48]).decode()[:40] not in record
        assert "synthetic text state" not in record and entry.text_sha256 is not None


async def test_gemini_live_sends_one_call_with_text_and_media_blocks(policy: MultimodalPolicy) -> None:
    class Fake:
        def __init__(self) -> None:
            self.inputs: list[Input] = []

        async def create(
            self, *, input: Input, model: str, questions: dict[str, dict[str, JsonValue]], timeout: float
        ) -> object:
            self.inputs.append(input)
            raise EvaluatorError("stop after recording the request")

    api = Fake()
    ev = GeminiDecisionEvaluator(mode="live", decisions_api=api, max_retries=0)
    await build_gate(policy, evaluator=ev).evaluate(PDF, "state text")
    [blocks] = api.inputs
    assert blocks == [
        {"type": "text", "text": "state text\n"},
        {"type": "document", "mime_type": "application/pdf", "data": base64.b64encode(PDF.read_bytes()).decode()},
    ]


# Limits and refusals


def test_media_size_and_type_are_checked_before_encoding() -> None:
    big = MediaPart(mime_type="image/png", data=b"\x89PNG\r\n\x1a\n" + b"0" * 100, name="big.png")
    with pytest.raises(MediaError, match="over the 50 byte limit"):
        media_input("", [big], max_bytes=50)
    with pytest.raises(MediaError, match="type 'image/gif' is not one of"):
        media_input("", [MediaPart(mime_type="image/gif", data=b"GIF89a", name="a.gif")], max_bytes=1000)


async def test_gate_rejects_large_unsupported_or_mislabelled_files(policy: MultimodalPolicy, tmp_path: Path) -> None:
    gate = build_gate(policy.model_copy(update={"max_media_bytes": 500}), evaluator=ScriptedMedia())
    with pytest.raises(MultimodalGateError, match="over the policy limit of 500 bytes"):
        await gate.evaluate(PNG)
    fake_png = tmp_path / "fake.png"
    fake_png.write_bytes(PDF.read_bytes()[:200])
    with pytest.raises(MultimodalGateError, match="content is not a image/png file"):
        await gate.evaluate(fake_png)
    gif = tmp_path / "a.gif"
    gif.write_bytes(b"GIF89a")
    with pytest.raises(MultimodalGateError, match="unsupported file type"):
        await gate.evaluate(gif)
    with pytest.raises(MultimodalGateError, match="names no choice check"):
        await gate.evaluate(PDF, expect={"nope": "x"})


def test_text_only_evaluator_is_refused(policy: MultimodalPolicy) -> None:
    with pytest.raises(MultimodalGateError, match="reads text only"):
        build_gate(policy, evaluator=JevEvaluator(mode="mock"))


def test_policy_validation() -> None:
    data = load_multimodal_policy(POLICY).model_dump(mode="json")
    with pytest.raises(ValueError, match="not one of the options"):
        MultimodalPolicy.model_validate(
            {**data, "choice": {"d": {**data["choice"]["overall_decision"], "expect": "x"}}}
        )
    with pytest.raises(ValueError, match="must match"):
        MultimodalPolicy.model_validate({**data, "truth": {"bad key!": data["truth"]["synthetic_label"]}})
