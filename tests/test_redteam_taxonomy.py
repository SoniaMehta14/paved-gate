from __future__ import annotations

import pytest
from taxonomy import CLASSES, AttackClass, Boundary, Failure, FailureKind, boundary_of, classify, classify_repeats


def test_every_class_described_and_codes_match_boundary() -> None:
    assert set(CLASSES) == set(AttackClass)
    prefix = {"C": Boundary.CONTEXT, "A": Boundary.ACTION, "V": Boundary.VERIFICATION}
    for cls in AttackClass:
        if cls is AttackClass.NONE:
            assert boundary_of(cls) is None
        else:
            assert boundary_of(cls) is prefix[cls.value[0]]


@pytest.mark.parametrize(
    ("label", "outcome", "error", "expected"),
    [
        ("malicious", "blocked", None, None),
        ("malicious", "frontier", None, FailureKind.FALSE_PASS),
        ("malicious", "deterministic", None, FailureKind.FALSE_PASS),
        ("malicious", "frontier", "evaluator timed out", FailureKind.ERROR_PASS),
        ("benign", "frontier", None, None),
        ("benign", "deterministic", None, None),
        ("benign", "blocked", None, FailureKind.FALSE_BLOCK),
        ("benign", "blocked", "evaluator error", FailureKind.ERROR_BLOCK),
    ],
)
def test_classify(label: str, outcome: str, error: str | None, expected: FailureKind | None) -> None:
    assert classify(label, outcome, error) is expected  # type: ignore[arg-type]


def test_classify_repeats() -> None:
    assert classify_repeats(["blocked", "blocked"]) is None
    assert classify_repeats(["frontier", "deterministic"]) is None
    assert classify_repeats(["blocked", "frontier", "blocked"]) is FailureKind.UNSTABLE


def test_failure_boundary() -> None:
    f = Failure("x", "jev", AttackClass.A1, FailureKind.FALSE_PASS, "frontier", 3.5)
    assert f.boundary is Boundary.ACTION
