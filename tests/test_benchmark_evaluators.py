"""Jev vs Gemini Decision benchmark runner, offline: mock runs, live mode without keys, checkpoint, quota stop.

No test reaches an API: `.env` loading is stubbed out and both keys are removed wherever live mode is used.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import benchmark_evaluators as be
import pytest

from paved_gate.evaluator.gemini_decision import GeminiDecisionEvaluator


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(be, "load_dotenv", lambda *a, **k: None)
    for env in be.KEY_ENV.values():
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr(be, "RESULTS", tmp_path)
    return tmp_path


def _outputs(tmp_path: Path) -> tuple[dict[str, object], str]:
    [js] = sorted(tmp_path.glob("evaluators-*.json"))
    return json.loads(js.read_text()), js.with_suffix(".md").read_text()


def test_mock_run_is_a_labelled_pipeline_check(offline: Path) -> None:
    assert be.main(["--mode", "mock", "--limit", "6", "--fanout-runs", "1"]) == 0
    report, md = _outputs(offline)
    assert report["kind"] == "pipeline check (mock), not results" and report["mode"] == "mock"
    assert report["models"] == {"jev": "jev-mock", "gemini_decision": "gemini-decision-1-mock"}
    assert report["settings"] == {
        "mode": "mock",
        "workloads": list(be.WORKLOADS),
        "limit": 6,
        "fanout_runs": 1,
        "fanout_sizes": list(be.FANOUT_SIZES),
        "harness2_cases": None,
    }
    assert report["timing_method"] == be.TIMING_METHOD and report["calibration_definition"] == be.CALIBRATION
    retry = report["retry_policies"]
    assert "date" in report and "sdk_versions" in report and isinstance(retry, dict) and set(retry) == set(be.ARMS)
    assert "**pipeline check (mock), not results**" in md and "results (live)" not in md
    assert md.count(be.MOCK_CELL) > 20 and "| Jev | 6 | 0 |" in md
    assert "Decision agreement between arms: not measured (mock)." in md


def test_rehearsal_flags_limit_fanout_sizes_and_harness2_cases(offline: Path) -> None:
    args = ["--mode", "mock", "--limit", "2", "--fanout-runs", "2", "--fanout-sizes", "1", "10"]
    assert be.main([*args, "--harness2-cases", "6"]) == 0
    report, _ = _outputs(offline)
    workloads = report["workloads"]
    assert isinstance(workloads, dict)
    assert workloads["fanout"]["sizes"] == [1, 10] and set(workloads["fanout"]["cells"]) == {
        "jev:1",
        "jev:10",
        "gemini_decision:1",
        "gemini_decision:10",
    }
    cats = workloads["harness2"]["arms"]["jev"]["categories"]
    assert {c: x["n"] for c, x in cats.items()} == {
        "guardrail_weakening": 2,
        "benign_prompt_edit": 2,
        "empty_output": 2,
    }
    tickets, _ = be.load_sample(be.DEFAULT_SAMPLE)
    rows = be.estimate(be.Settings("live", list(be.WORKLOADS), 2, 2, [1, 10], 6), tickets[:2])
    assert rows[0]["calls"] == be.WARMUP_CALLS + 2 + 6 + 2 * 2


def test_live_without_keys_marks_arms_not_run_and_calls_nothing(offline: Path) -> None:
    assert be.main(["--mode", "live", "--yes", "--workloads", "tickets", "--limit", "3"]) == 0
    report, md = _outputs(offline)
    assert report["kind"] == "results (live)"
    assert report["arms"] == {
        "jev": {"status": "not_run", "reason": "TYPESAFE_API_KEY not set"},
        "gemini_decision": {"status": "not_run", "reason": "GEMINI_API_KEY not set"},
    }
    assert "| Jev | not_run | TYPESAFE_API_KEY not set |" in md


def test_live_requires_yes(offline: Path) -> None:
    assert be.main(["--mode", "live"]) == 2
    assert not list(offline.glob("evaluators-*"))


def test_estimate_prints_calls_and_cost(capsys: pytest.CaptureFixture[str], offline: Path) -> None:
    tickets, _ = be.load_sample(be.DEFAULT_SAMPLE)
    rows = be.estimate(be.Settings("live", list(be.WORKLOADS), 10, 2), tickets[:10])
    calls = be.WARMUP_CALLS + 10 + len(be.harness2_cases()) + 2 * len(be.FANOUT_SIZES)
    assert [r["calls"] for r in rows] == [calls, calls]
    assert all(isinstance(r["est_cost_usd"], float) and r["est_cost_usd"] > 0 for r in rows)
    assert be.main(["--estimate", "--limit", "10", "--fanout-runs", "2"]) == 0
    out = capsys.readouterr().out
    assert "Gemini Decision 1" in out and "no API calls were made" in out
    assert not list(offline.glob("evaluators-*"))


def test_ece_and_percentile_by_hand() -> None:
    # bin 0.9: accuracy 0.5 vs 0.9 -> 0.4 x 0.5; bin 0.6: accuracy 1.0 vs 0.6 -> 0.4 x 0.5
    assert be.ece([0.9, 0.9, 0.6, 0.6], [True, False, True, True]) == 0.4
    assert be.ece([1.0], [True]) == 0.0
    assert be.percentile([5.0, 1.0, 3.0, 2.0, 4.0], 0.5) == 3.0
    assert be.percentile([float(i) for i in range(1, 21)], 0.95) == 19.0
    assert be.percentile([], 0.5) is None


def test_retried_calls_are_excluded_from_latency_only() -> None:
    recs = [
        be.CallRecord("fanout", "jev", "1:0", True, latency_ms=10.0, retries=0),
        be.CallRecord("fanout", "jev", "1:1", True, latency_ms=900.0, retries=1),
        be.CallRecord("fanout", "jev", "1:2", True, latency_ms=12.0, retries=0),
    ]
    assert be.latency(recs) == {"p50_ms": 10.0, "p95_ms": 12.0, "runs": 2, "excluded_retried": 1}


def test_unsupported_size_is_shown_not_chunked(monkeypatch: pytest.MonkeyPatch, offline: Path) -> None:
    caps = {**be.CAPS, "gemini_decision": {**be.CAPS["gemini_decision"], "questions_per_call": 10}}
    monkeypatch.setattr(be, "CAPS", caps)
    assert be.main(["--mode", "mock", "--workloads", "fanout", "--fanout-runs", "1"]) == 0
    report, md = _outputs(offline)
    cells = report["workloads"]["fanout"]["cells"]  # type: ignore[index]
    assert cells["gemini_decision:25"] == {"supported": False} and cells["gemini_decision:50"] == {"supported": False}
    assert "| Gemini Decision 1 | 25 | not supported at this size | n/a |" in md


def test_checkpoint_resumes_only_with_the_same_settings(tmp_path: Path) -> None:
    path = tmp_path / "ckpt.jsonl"
    first = be.Checkpoint(path, "abc", fresh=False)
    first.add(be.CallRecord("tickets", "jev", "t1", True, latency_ms=5.0, retries=0, data={"pred": "x"}))
    resumed = be.Checkpoint(path, "abc", fresh=False)
    assert ("tickets", "jev", "t1") in resumed.done and resumed.done[("tickets", "jev", "t1")].data == {"pred": "x"}
    with pytest.raises(SystemExit, match="different settings"):
        be.Checkpoint(path, "other", fresh=False)
    assert be.Checkpoint(path, "other", fresh=True).done == {}


class _Always429:
    async def create(self, **kwargs: object) -> object:
        err = Exception("rate limited")
        err.status_code = 429  # type: ignore[attr-defined]
        raise err


async def test_quota_hit_stops_cleanly_and_keeps_progress(tmp_path: Path) -> None:
    gem = GeminiDecisionEvaluator(
        mode="live", decisions_api=_Always429(), max_retries=1, backoff_base_s=0.001, rng=random.Random(0)
    )
    arms = {"gemini_decision": be.Arm("gemini_decision", gem, "ok", "fake")}
    ckpt = be.Checkpoint(tmp_path / "ckpt.jsonl", "abc", fresh=False)
    settings = be.Settings("live", ["tickets"], 2, 1)
    tickets, _ = be.load_sample(be.DEFAULT_SAMPLE)
    with pytest.raises(be.QuotaHit, match="Gemini Decision 1 hit a quota or rate limit"):
        await be.run_workloads(settings, arms, tickets[:2], ckpt, print)
