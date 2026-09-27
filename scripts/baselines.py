"""SIMULATED BASELINES for frontier arms that could not run (no API key).

Used only with `benchmark.py --simulate-missing`, for layout demos. These are
ASSUMPTIONS, not measurements: every arm built from them is labelled
"simulated" in the JSON, the terminal table, and the dashboard. Do not publish
numbers derived from them as a comparison; run the arm live instead.

Latency samples are drawn log-normally around the assumed p50. Cost uses the
real list price from pricing.py times an estimated token count (prompt chars / 4
for input, the assumed output size below).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Baseline:
    p50_ms: float
    sigma: float  # log-normal spread; 0.35 puts p95 at roughly 1.8x p50
    output_tokens: int
    source: str


SIMULATED_BASELINES: dict[str, Baseline] = {
    "gpt-4o": Baseline(600.0, 0.35, 30, "assumed ~600 ms p50 for a short JSON classification"),
    "gpt-4o-mini": Baseline(450.0, 0.35, 30, "assumed ~450 ms p50 for a short JSON classification"),
    "claude-sonnet-5": Baseline(800.0, 0.35, 60, "assumed ~800 ms p50 at effort low"),
}
