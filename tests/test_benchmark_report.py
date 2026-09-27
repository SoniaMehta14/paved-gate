from __future__ import annotations

import json
import random
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
from benchmark import SAMPLE_TIMINGS, ArmResult, Mode, Report, build_report, simulate_frontier_arm, write_outputs
from dashboard import PLACEHOLDER, render_dashboard


def _embedded(html: str) -> object:
    m = re.search(r'<script id="benchmark-data" type="application/json">(.*?)</script>', html, re.S)
    assert m is not None
    return json.loads(m.group(1))


def test_arm_metrics() -> None:
    arm = ArmResult("jev", "Paved Gate (Jev)", "jev-latest", "mock", "simulated", "estimated")
    arm.latencies_ms = [float(v) for v in range(1, 41)]  # 1..40
    arm.costs_usd = [0.00001, 0.00003]
    out = arm.to_json()
    assert out["latency_ms"] == {"p50": 20.5, "p95": 38.0, "mean": 20.5}  # nearest-rank p95
    assert out["cost_usd"]["per_million_requests"] == 20.0
    assert out["n"] == 40
    assert len(out["sample_timings_ms"]) == SAMPLE_TIMINGS


def test_not_run_arm_has_null_metrics() -> None:
    out = ArmResult("claude", "Claude zero-shot", "claude-opus-5", "not_run", "ANTHROPIC_API_KEY not set").to_json()
    assert out["mode"] == "not_run"
    assert out["latency_ms"] == {"p50": None, "p95": None, "mean": None}
    assert out["cost_usd"]["per_million_requests"] is None
    assert out["sample_timings_ms"] == []


def test_report_and_dashboard_round_trip(tmp_path: Path) -> None:
    started = datetime(2026, 9, 27, 18, 0, tzinfo=UTC)
    arm = ArmResult("jev", "Paved Gate (Jev)", "jev-latest", "mock", "simulated", "estimated")
    arm.latencies_ms = [90.0, 110.0]
    report = build_report([arm], started_at=started, iterations=1, policy_hash="sha256:abc")
    json_path, html_path = write_outputs(report, tmp_path, started)

    assert json_path.name == "benchmark-20260927-180000.json"
    assert json.loads(json_path.read_text()) == report
    html = html_path.read_text()
    assert PLACEHOLDER not in html
    assert _embedded(html) == report


def test_data_cannot_break_out_of_script_tag() -> None:
    html = render_dashboard({"arms": [{"label": "</script><script>alert(1)</script>"}]})
    assert "</script><script>alert(1)" not in html
    assert _embedded(html) == {"arms": [{"label": "</script><script>alert(1)</script>"}]}


def test_missing_placeholder_is_an_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import dashboard

    broken = tmp_path / "template.html"
    broken.write_text("<html></html>")
    monkeypatch.setattr(dashboard, "TEMPLATE", broken)
    with pytest.raises(RuntimeError, match="placeholder"):
        render_dashboard({})


def _arm(arm_id: str, mode: Mode, p50: float, cost: float) -> ArmResult:
    arm = ArmResult(arm_id, arm_id, f"{arm_id}-model", mode, "detail", "basis")
    arm.latencies_ms = [p50]
    arm.costs_usd = [cost / 1_000_000]
    return arm


def _report(*arms: ArmResult) -> Report:
    return build_report(list(arms), started_at=datetime(2026, 9, 27, tzinfo=UTC), iterations=1, policy_hash="h")


def test_comparison_uses_best_frontier_arm_per_metric() -> None:
    comp = _report(
        _arm("jev", "live", 100.0, 20.0),
        _arm("openai", "live", 600.0, 3000.0),  # fastest frontier arm
        _arm("claude", "live", 800.0, 1500.0),  # cheapest frontier arm
    )["comparison"]
    assert comp["measured"] is True
    lat, cost = comp["latency_p50"], comp["cost_per_million"]
    assert lat is not None and lat["baseline_arm"] == "openai" and lat["reduction_pct"] == 83.3
    assert cost is not None and cost["baseline_arm"] == "claude" and cost["reduction_pct"] == 98.7


def test_comparison_is_flagged_when_any_side_is_not_live() -> None:
    report = _report(_arm("jev", "mock", 100.0, 20.0), _arm("openai", "simulated", 600.0, 3000.0))
    assert report["comparison"]["measured"] is False
    assert report["caveats"][0].startswith("Arms marked 'simulated'")


def test_comparison_is_empty_without_frontier_data() -> None:
    comp = _report(_arm("jev", "live", 100.0, 20.0), ArmResult("openai", "GPT-4o", "gpt-4o", "not_run", "no key"))[
        "comparison"
    ]
    assert comp["latency_p50"] is None and comp["cost_per_million"] is None and comp["measured"] is False


def test_simulated_arm_is_labelled_and_reproducible() -> None:
    a = simulate_frontier_arm("claude", "Claude", "claude-sonnet-5", "system", 1, random.Random(0))
    b = simulate_frontier_arm("claude", "Claude", "claude-sonnet-5", "system", 1, random.Random(0))
    assert a.mode == "simulated" and "SIMULATED BASELINE" in a.mode_detail
    assert a.latencies_ms == b.latencies_ms and len(a.latencies_ms) == 12
    assert all(c > 0 for c in a.costs_usd)


def test_no_simulated_baseline_for_unknown_model() -> None:
    arm = simulate_frontier_arm("claude", "Claude", "claude-unknown", "system", 1, random.Random(0))
    assert arm.mode == "not_run" and arm.latencies_ms == []
