from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest
from benchmark import SAMPLE_TIMINGS, ArmResult, build_report, write_outputs
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
