"""Local, deterministic PII/PHI detectors.

These run in-process before anything is sent to an external evaluator. They are
deliberately conservative regexes: cheap, auditable, and prone to false negatives
on free-form data (names, addresses). See the README section on masking.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from paved_gate.policy.models import DetectorName


@dataclass(frozen=True)
class Finding:
    detector: DetectorName
    start: int
    end: int


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _is_card(match: re.Match[str]) -> bool:
    digits = re.sub(r"\D", "", match.group(0))
    return 13 <= len(digits) <= 19 and _luhn_ok(digits)


_PATTERNS: dict[DetectorName, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "ssn": re.compile(r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"),
    "phone": re.compile(r"(?<!\w)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){12,18}\d\b"),
    "mrn": re.compile(r"\b(?:MRN|medical record(?: number)?)[\s:#]*[A-Z0-9-]{5,12}\b", re.I),
    "dob": re.compile(
        r"\b(?:DOB|date of birth|born on)[\s:]*"
        r"(?:\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        re.I,
    ),
}

_VALIDATORS: dict[DetectorName, Callable[[re.Match[str]], bool]] = {
    "credit_card": _is_card,
}


def detect(text: str, detectors: Iterable[DetectorName]) -> list[Finding]:
    findings: list[Finding] = []
    for name in detectors:
        validator = _VALIDATORS.get(name)
        for m in _PATTERNS[name].finditer(text):
            if validator is None or validator(m):
                findings.append(Finding(name, m.start(), m.end()))
    return _drop_overlaps(findings)


def _drop_overlaps(findings: list[Finding]) -> list[Finding]:
    """Keep the earliest, then longest, finding where spans overlap."""
    kept: list[Finding] = []
    for f in sorted(findings, key=lambda f: (f.start, -(f.end - f.start))):
        if not kept or f.start >= kept[-1].end:
            kept.append(f)
    return kept
