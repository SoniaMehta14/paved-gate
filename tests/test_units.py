from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from paved_gate import HeuristicEvaluator, JevEvaluator, LoadedPolicy, MemorySink, Policy, load_policy
from paved_gate.gate import build_questions
from paved_gate.handlers.deterministic import (
    Arithmetic,
    DateDiff,
    ExpressionError,
    JsonValidate,
    UnitConvert,
    evaluate_arithmetic,
)
from paved_gate.policy.loader import policy_hash
from paved_gate.privacy.detect import detect
from paved_gate.privacy.mask import mask
from tests.conftest import POLICY_PATH, make_gate

# ---------------------------------------------------------------- policy


def test_sample_policy_loads_and_hash_is_stable(loaded_policy: LoadedPolicy) -> None:
    assert loaded_policy.policy.questions.risk.scale_max == 5
    assert loaded_policy.policy_hash == load_policy(POLICY_PATH).policy_hash
    assert loaded_policy.policy_hash.startswith("sha256:")


def test_hash_changes_when_threshold_changes(loaded_policy: LoadedPolicy) -> None:
    p = loaded_policy.policy
    risk = p.questions.risk.model_copy(update={"block_above": 2.5})
    changed = p.model_copy(update={"questions": p.questions.model_copy(update={"risk": risk})})
    assert policy_hash(changed) != loaded_policy.policy_hash


def _raw_policy(**questions_override: object) -> dict[str, object]:
    base: dict[str, object] = {
        "intent": {
            "instructions": "x",
            "criteria": {"DIRECT_CODE_EXEC": "a", "FRONTIER_AGENT_REQUIRED": "b", "POLICY_VIOLATION": "c"},
        },
        "risk": {"instructions": "x", "criteria": ["1", "2", "3"], "block_above": 2},
        "sensitive": {"instructions": "x"},
    }
    base.update(questions_override)
    return {"version": 1, "name": "t", "questions": base}


def test_intent_criteria_must_be_the_three_routes() -> None:
    bad = _raw_policy(intent={"instructions": "x", "criteria": {"DIRECT_CODE_EXEC": "a"}})
    with pytest.raises(ValidationError, match=r"intent\.criteria keys"):
        Policy.model_validate(bad)


def test_block_above_must_be_on_scale() -> None:
    bad = _raw_policy(risk={"instructions": "x", "criteria": ["1", "2", "3"], "block_above": 9})
    with pytest.raises(ValidationError, match="block_above"):
        Policy.model_validate(bad)


def test_unknown_policy_keys_are_rejected() -> None:
    raw = _raw_policy()
    raw["surprise"] = True
    with pytest.raises(ValidationError):
        Policy.model_validate(raw)


# ---------------------------------------------------------------- privacy

ALL = ["email", "phone", "ssn", "credit_card", "mrn", "dob"]


@pytest.mark.parametrize(
    ("text", "detector"),
    [
        ("reach me at jane.doe@example.org", "email"),
        ("call (415) 555-0134 today", "phone"),
        ("ssn 123-45-6789", "ssn"),
        ("card 4111 1111 1111 1111 exp 09/29", "credit_card"),
        ("MRN: 00482913", "mrn"),
        ("DOB: 1984-03-12", "dob"),
    ],
)
def test_detectors(text: str, detector: str) -> None:
    assert [f.detector for f in detect(text, ALL)] == [detector]  # type: ignore[arg-type]


def test_card_without_valid_luhn_is_ignored() -> None:
    assert detect("order 4111 1111 1111 1112", ["credit_card"]) == []


def test_mask_replaces_spans() -> None:
    text = "SSN 123-45-6789, email a@b.io"
    assert mask(text, detect(text, ALL)) == "SSN [REDACTED:SSN], email [REDACTED:EMAIL]"  # type: ignore[arg-type]


# ---------------------------------------------------------------- deterministic handlers


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("2 + 3 * 4", "14"),
        ("(2 + 3) * 4", "20"),
        ("2 ^ 3 ^ 2", "512"),
        ("-2 ** 2", "-4"),
        ("10 / 4", "2.5"),
        ("7 % 3", "1"),
        ("0.1 + 0.2", "0.3"),
    ],
)
def test_arithmetic(expr: str, expected: str) -> None:
    assert Arithmetic().try_run(f"calculate {expr}") == {"expression": expr, "ok": True, "result": expected}


@pytest.mark.parametrize("expr", ["__import__('os').system('ls')", "2 +", "(1 + 2", "1 / 0", "9 ^ 9 ^ 9", "()"])
def test_arithmetic_rejects_bad_input(expr: str) -> None:
    with pytest.raises(ExpressionError):
        evaluate_arithmetic(expr)


def test_arithmetic_reports_division_by_zero() -> None:
    out = Arithmetic().try_run("what is 1 / 0")
    assert out is not None and out["ok"] is False


def test_unit_convert() -> None:
    assert UnitConvert().try_run("convert 5 km to miles") == {
        "ok": True,
        "from": {"value": "5", "unit": "km"},
        "to": {"value": "3.106856", "unit": "mi"},
    }
    out = UnitConvert().try_run("100 F to C")
    assert out is not None and out["to"] == {"value": "37.777778", "unit": "c"}
    out = UnitConvert().try_run("convert 1000 kg in feet")
    assert out is not None and out["ok"] is False


def test_json_validate() -> None:
    assert JsonValidate().try_run('is this valid json? {"a": [1, 2]}') == {
        "ok": True,
        "valid": True,
        "top_level_type": "dict",
        "size": 1,
    }
    bad = JsonValidate().try_run('validate this json: {"a": 1,\n "b": }')
    assert bad is not None and bad["valid"] is False and bad["line"] == 2


def test_date_diff() -> None:
    assert DateDiff().try_run("how many days between 2026-01-01 and 2026-09-26?") == {
        "ok": True,
        "from": "2026-01-01",
        "to": "2026-09-26",
        "days": 268,
    }


# ---------------------------------------------------------------- parallelism / latency


async def test_heuristic_evaluator_answers_questions_concurrently(loaded_policy: LoadedPolicy) -> None:
    ev = HeuristicEvaluator(per_question_delay_ms=80)
    started = time.perf_counter()
    await ev.evaluate("hello", build_questions(loaded_policy.policy))
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert elapsed_ms < 160, f"3 x 80ms questions took {elapsed_ms:.0f}ms; expected ~80ms if parallel"


async def test_gate_with_jev_mock_stays_near_100ms(loaded_policy: LoadedPolicy) -> None:
    gate = make_gate(loaded_policy, MemorySink(), evaluator=JevEvaluator(mode="mock", mock_latency_ms=(60, 120)))
    timings = []
    for _ in range(5):
        result = await gate.handle("Draft a customer email about the outage")
        timings.append(result.decision.gate_ms)
    assert max(timings) < 150


async def test_jsonl_sink_writes_one_line_per_record(tmp_path: Path, loaded_policy: LoadedPolicy) -> None:
    from paved_gate import JsonlFileSink

    path = tmp_path / "audit.jsonl"
    gate = make_gate(loaded_policy, MemorySink())
    gate.audit = JsonlFileSink(path)
    await asyncio.gather(*(gate.handle(f"What is {i} + 1?") for i in range(10)))
    await gate.handle("my ssn is 123-45-6789")
    lines = path.read_text().splitlines()
    assert len(lines) == 12  # 11 decisions + 1 sensitive_data event
