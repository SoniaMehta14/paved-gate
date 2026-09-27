"""Keyword heuristics shared by the Jev mock and the HeuristicEvaluator.

These exist so offline demos and tests behave plausibly. They are not a safety
control; the real judgement comes from the evaluator model.
"""

from __future__ import annotations

import re

from paved_gate.handlers.deterministic import DeterministicRegistry, default_registry

_INJECTION = [
    (
        re.compile(
            r"\b(ignore|disregard|forget)\b.{0,30}\b(previous|prior|above|all|earlier)\b.{0,20}\b(instructions?|rules|prompts?)\b",
            re.I | re.S,
        ),
        0.95,
    ),
    (
        re.compile(
            r"\b(reveal|print|show|leak|dump)\b.{0,30}"
            r"\b(system prompt|hidden instructions|api key|secrets?|credentials)\b",
            re.I | re.S,
        ),
        0.9,
    ),
    (re.compile(r"\b(jailbreak|DAN mode|developer mode|do anything now)\b", re.I), 0.9),
    (
        re.compile(
            r"\byou are now\b|\bpretend (to be|you are)\b|\bact as\b.{0,40}\bno (rules|restrictions)\b",
            re.I,
        ),
        0.7,
    ),
    (re.compile(r"rm\s+-rf\s+/|;\s*drop\s+table|<script\b|curl\s+[^|]+\|\s*(ba)?sh", re.I), 0.95),
    (re.compile(r"\b(base64|rot13)\b.{0,40}\b(decode|execute|follow)\b", re.I | re.S), 0.75),
    (re.compile(r"\bexfiltrat\w*|\bsend (it|them|this|the data) to\b", re.I), 0.8),
]

_VIOLATION = re.compile(
    r"\b(build|make|synthesi[sz]e|create|write)\b.{0,40}\b(bomb|explosive|nerve agent|meth(amphetamine)?|"
    r"ransomware|keylogger|malware|phishing kit|credential stealer)\b"
    r"|\bsteal\b.{0,30}\b(passwords?|credentials|credit cards?)\b",
    re.I | re.S,
)

_SENSITIVE = re.compile(
    r"\[REDACTED:[A-Z_]+\]|\b(patient|diagnos\w+|medical record|prescription|lab results?|"
    r"social security|date of birth|passport number|driver'?s licen[cs]e|home address|salary|"
    r"bank account|HIV|mental health|insurance id)\b",
    re.I,
)

_COMPUTE_CUE = re.compile(
    r"\b(calculate|compute|evaluate|what is|what's|how much is|convert|how many days|"
    r"days between|validate|is this valid|solve)\b|=\s*\?",
    re.I,
)

_registry: DeterministicRegistry = default_registry()


def injection_risk(text: str) -> float:
    """0..1 likelihood of prompt injection / policy risk."""
    score = max((w for pat, w in _INJECTION if pat.search(text)), default=0.05)
    if violation(text):
        score = max(score, 0.85)
    return score


def violation(text: str) -> bool:
    return bool(_VIOLATION.search(text))


def deterministic(text: str) -> bool:
    return bool(_COMPUTE_CUE.search(text)) and _registry.can_handle(text)


def sensitive(text: str) -> float:
    hits = len(_SENSITIVE.findall(text))
    return min(0.97, 0.08 + 0.45 * hits) if hits else 0.04
