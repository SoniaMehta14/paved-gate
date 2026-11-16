"""Jev vs Gemini Decision benchmark runner, offline: mock runs, live mode without keys, checkpoint, quota stop.

No test reaches an API: `.env` loading is stubbed out and both keys are removed wherever live mode is used.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from pathlib import Path

import benchmark_evaluators as be
import pytest

from paved_gate import EvaluationResult
from paved_gate.evaluator.base import EvaluatorState, Question
from paved_gate.evaluator.gemini_decision import GeminiDecisionEvaluator


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setattr(be, "load_dotenv", lambda *a, **k: None)
    for env in be.KEY_ENV.values():
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr(be, "RESULTS", tmp_path)
    monkeypatch.setattr(be, "CHECKPOINT", tmp_path / "evaluators-checkpoint.jsonl")  # never the real one
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
        "gemini_tpm": 45000,
        "gemini_tier": None,
        "resumed_from_checkpoint": False,
        "calls_restored": 0,
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


def test_limit_spreads_tickets_across_intents() -> None:
    tickets, _ = be.load_sample(be.DEFAULT_SAMPLE)
    picked = be.spread(tickets, 27)
    assert len(picked) == 27 and len({t.intent for t in picked}) == 27
    assert be.spread(tickets, None) == tickets and be.spread(tickets, 5000) == tickets


class FakeClock:
    """A clock that only moves when the pacer sleeps, so waits are exact and nothing really sleeps."""

    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.t

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.t += seconds


async def test_pacer_waits_for_the_sliding_window() -> None:
    clock = FakeClock()
    pacer = be.TokenPacer(1000, clock=clock, sleep=clock.sleep)
    assert await pacer.wait(400) == 0.0
    pacer.record("tickets", 400)  # t=0
    clock.t = 10.0
    assert await pacer.wait(pacer.expected("tickets", 50)) == 0.0  # 800 of 1000 after this one
    pacer.record("tickets", 400)  # t=10
    clock.t = 20.0
    # 800 used; another 400 would cross 1000, so wait until the t=0 entry leaves the window at t=60
    assert await pacer.wait(400) == 40.0
    assert clock.t == 60.0 and pacer.used() == 400
    assert pacer.waited_s == 40.0 and clock.sleeps == [40.0]


async def test_pacer_refuses_a_call_larger_than_the_limit() -> None:
    clock = FakeClock()
    pacer = be.TokenPacer(1000, clock=clock, sleep=clock.sleep)
    assert pacer.expected("fanout:50", 1200) == 1200
    with pytest.raises(be.CallTooLarge, match="over the 1,000 per minute limit"):
        await pacer.wait(1200)
    assert clock.sleeps == []


async def test_oversized_fanout_cell_is_not_supported_and_not_sent() -> None:
    sent: list[int] = []

    class Counting(GeminiDecisionEvaluator):
        async def evaluate(self, state: EvaluatorState, questions: Mapping[str, Question]) -> EvaluationResult:
            sent.append(len(questions))
            return await super().evaluate(state, questions)

    clock = FakeClock()
    pacer = be.TokenPacer(1500, clock=clock, sleep=clock.sleep)
    gem = Counting(mode="mock", mock_latency_ms=(0, 0), rng=random.Random(0))
    arms = {"gemini_decision": be.Arm("gemini_decision", gem, "ok", "fake", pacer=pacer)}
    settings = be.Settings("mock", ["fanout"], None, 2, [1, 50])
    out = await be.run_workloads(settings, arms, [], be.Checkpoint(None, "x", fresh=False), lambda _: None)
    cells = out["fanout"]["cells"]
    assert cells["gemini_decision:1"]["supported"] is True
    assert cells["gemini_decision:50"]["supported"] is False
    assert cells["gemini_decision:50"]["reason"].startswith("pacing: one call needs about")
    assert 50 not in sent


def test_pacing_and_tier_do_not_change_the_resume_hash() -> None:
    base = be.Settings("live", list(be.WORKLOADS), None, 20, list(be.FANOUT_SIZES), None)
    # the hash in the header of the checkpoint the interrupted full run wrote on 2026-11-16
    assert base.digest() == "57b626b84b604451"
    paced = be.Settings("live", list(be.WORKLOADS), None, 20, list(be.FANOUT_SIZES), None, 30_000, "free")
    assert paced.digest() == base.digest()


def test_bootstrap_interval_is_reproducible_and_brackets_the_estimate() -> None:
    correct = [True] * 90 + [False] * 10
    first = be.bootstrap_ci(len(correct), be.accuracy_stat(correct))
    assert first == be.bootstrap_ci(len(correct), be.accuracy_stat(correct))  # fixed seed
    assert first is not None and first[0] < 0.9 < first[1] and first[0] > 0.8 and first[1] <= 1.0
    assert be.bootstrap_ci(0, be.accuracy_stat([])) is None


def _record(workload: str, arm: str, item: str, **data: object) -> dict[str, object]:
    rec = be.CallRecord(workload, arm, item, True, latency_ms=10.0, retries=0, input_tokens=100, output_tokens=0)
    rec.data = dict(data)
    out: dict[str, object] = be.asdict(rec)
    return out


def test_rebuild_recomputes_from_the_calls_file_without_api_calls(tmp_path: Path) -> None:
    # Jev right on t0 to t4, Gemini right on t2 to t5: both right 3, only Jev 2, only Gemini 1, both wrong 0.
    jev_right, gem_right = {0, 1, 2, 3, 4}, {2, 3, 4, 5}
    calls: list[dict[str, object]] = [{"settings_hash": "x"}]
    for arm, right in (("jev", jev_right), ("gemini_decision", gem_right)):
        for i in range(6):
            pred = "cancel_order" if i in right else "track_order"
            calls.append(
                _record("tickets", arm, f"t{i}", true="cancel_order", pred=pred, top_p=0.9, confidence_field=0.9)
            )
        for i, cat in enumerate(("guardrail_weakening", "benign_prompt_edit", "empty_output")):
            calls.append(_record("harness2", arm, f"h{i}", category=cat, flagged=cat != "benign_prompt_edit"))
        calls += [_record("fanout", arm, f"1:{r}") for r in range(2)]
    stem = tmp_path / "evaluators-live-x"
    Path(f"{stem}.calls.jsonl").write_text("\n".join(json.dumps(c) for c in calls) + "\n")
    original = {
        "title": "T",
        "kind": "results (live)",
        "mode": "live",
        "date": "d",
        "models": {"jev": "jev-1.14.0", "gemini_decision": "gemini-decision-1"},
        "sdk_versions": {},
        "settings": {},
        "retry_policies": {},
        "caps": {},
        "arms": {a: {"status": "ok", "reason": "real"} for a in be.ARMS},
        "workloads": {
            "tickets": {},
            "harness2": {},
            "fanout": {"sizes": [1], "cells": {"jev:1": {"supported": True}, "gemini_decision:1": {"supported": True}}},
        },
        "notes": ["Timing method: kept", "Costs use an old wording"],
    }
    Path(f"{stem}.json").write_text(json.dumps(original))

    assert be.main(["--rebuild", f"{stem}.json"]) == 0
    report = json.loads(Path(f"{stem}.json").read_text())
    md = Path(f"{stem}.md").read_text()
    paired = report["workloads"]["tickets"]["paired_accuracy"]
    assert (paired["both_correct"], paired["only_jev_correct"], paired["only_gemini_decision_correct"]) == (3, 2, 1)
    assert paired["both_wrong"] == 0 and paired["p_value"] == 1.0
    jev = report["workloads"]["tickets"]["arms"]["jev"]
    assert (
        jev["accuracy"] == round(5 / 6, 4)
        and len(jev["accuracy_ci95"]) == 2
        and len(jev["ece_top_probability_ci95"]) == 2
    )
    assert report["workloads"]["harness2"]["agreement"] == {"agree": 3, "n": 3, "rate": 1.0}
    assert report["rebuilt"]["api_calls"] == 0 and report["rebuilt"]["calls_read"] == 22
    assert "Agreement on flag or no flag: 3 of 3 cases." in md
    assert "| Only Jev correct | 2 |" in md and "| Only Gemini Decision 1 correct | 1 |" in md
    assert "Timing method: kept" in report["notes"] and "Costs use an old wording" not in report["notes"]
    assert be.gemini_cost_note(None) in report["notes"] and be.STATS_NOTE in report["notes"]
    assert be.gemini_cost_note("free").startswith(
        "This run used the Gemini free tier, so its Gemini calls cost nothing."
    )
