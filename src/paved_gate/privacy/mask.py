from __future__ import annotations

from paved_gate.privacy.detect import Finding


def mask(text: str, findings: list[Finding]) -> str:
    """Replace each finding with a typed placeholder, e.g. `[REDACTED:SSN]`.

    `findings` must be non-overlapping, as returned by `detect()`.
    """
    out: list[str] = []
    cursor = 0
    for f in sorted(findings, key=lambda f: f.start):
        out.append(text[cursor : f.start])
        out.append(f"[REDACTED:{f.detector.upper()}]")
        cursor = f.end
    out.append(text[cursor:])
    return "".join(out)
