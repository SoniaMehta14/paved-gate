from __future__ import annotations

import csv
import json
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pytest
from build_sample import (
    DEFAULT_SAMPLE,
    Ticket,
    load_sample,
    read_rows,
    stratified_sample,
    write_sample,
)
from dashboard import render_dashboard
from intents import INTENTS, INVALID, labels, require_descriptions
from metrics import compute_metrics


def _csv(tmp_path: Path, counts: dict[str, int]) -> Path:
    path = tmp_path / "data.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["flags", "instruction", "category", "intent", "response"])
        w.writeheader()
        for intent, n in counts.items():
            for i in range(n):
                w.writerow(
                    {
                        "flags": "B",
                        "instruction": f"{intent} message {i}",
                        "category": "X",
                        "intent": intent,
                        "response": "r",
                    }
                )
    return path


def test_stratified_sample_caps_each_intent_and_is_deterministic(tmp_path: Path) -> None:
    rows = read_rows(_csv(tmp_path, {"x": 10, "y": 3}))
    a = stratified_sample(rows, per_intent=5, seed=1)
    assert Counter(t.intent for t in a) == {"x": 5, "y": 3}  # y has fewer than the cap
    assert a == stratified_sample(rows, per_intent=5, seed=1)
    assert {t.row_index for t in a} != {t.row_index for t in stratified_sample(rows, per_intent=5, seed=2)}


def test_sample_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    tickets = [Ticket("bitext-1", 1, "hello", "x"), Ticket("bitext-2", 2, "world", "y")]
    sample = tmp_path / "s.jsonl"
    manifest = write_sample(tickets, sample, per_intent=1, seed=0)
    loaded, m = load_sample(sample)
    assert loaded == tickets and m["sample_sha256"] == manifest["sample_sha256"] and m["n"] == 2
    sample.write_text(sample.read_text().replace("hello", "HELLO"))
    with pytest.raises(ValueError, match="does not match its manifest"):
        load_sample(sample)


def test_missing_intent_description_is_an_error() -> None:
    with pytest.raises(ValueError, match="missing descriptions"):
        require_descriptions({*INTENTS, "brand_new_intent"})


def test_committed_sample_is_intact_and_stratified() -> None:
    tickets, manifest = load_sample(DEFAULT_SAMPLE)  # raises if the file no longer matches its hash
    counts = Counter(t.intent for t in tickets)
    assert set(counts) == set(INTENTS) and len(counts) == 27
    assert set(counts.values()) == {manifest["per_intent"]}
    assert manifest["license"] == "CDLA-Sharing-1.0" and manifest["revision"]


# ---------------------------------------------------------------- run_eval (no network)


def test_estimate_makes_no_api_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    import openai
    import run_eval

    def boom(*_: object, **__: object) -> None:
        raise AssertionError("estimate must not construct an API client")

    monkeypatch.setattr(openai, "AsyncOpenAI", boom)
    tickets, _ = load_sample(DEFAULT_SAMPLE)
    rows = run_eval.estimate(tickets, ["gpt-4o", "gpt-4o-mini"])
    assert [r[1] for r in rows] == [len(tickets)] * 3
    assert all(r[3] > 0 for r in rows)


def test_parse_openai_intent() -> None:
    from run_eval import parse_openai_intent

    assert parse_openai_intent('{"intent": "track_order"}') == "track_order"
    assert parse_openai_intent('{"intent": "not_a_label"}') == INVALID
    assert parse_openai_intent("not json") == INVALID
    assert parse_openai_intent(None) == INVALID


def test_report_with_classification_renders(tmp_path: Path) -> None:
    from benchmark import ArmResult
    from run_eval import build_report

    arm = ArmResult("jev", "Jev (typed Choice)", "jev-latest", "live", "real", "reported tokens x list price")
    arm.latencies_ms = [100.0, 120.0]
    arm.costs_usd = [0.00002, 0.00002]
    lab = labels()
    metrics = compute_metrics([lab[0], lab[1]], [lab[0], INVALID], lab, INVALID)
    cls = {
        "id": "jev",
        "label": arm.label,
        "model": arm.model,
        "mode": "live",
        "measured": True,
        "attempted": 2,
        "api_errors": 0,
        "metrics": metrics,
        "note": "",
    }
    manifest = {
        "dataset": "d",
        "source_url": "https://example.org",
        "revision": "r",
        "license": "CDLA-Sharing-1.0",
        "attribution": "a",
        "per_intent": 1,
        "seed": 0,
        "n": 2,
        "sample_sha256": "0" * 64,
    }
    report = build_report([arm], [cls], started_at=datetime(2026, 9, 30, tzinfo=UTC), manifest=manifest, concurrency=4)  # type: ignore[list-item]
    html = render_dashboard(report)
    m = re.search(r'<script id="benchmark-data" type="application/json">(.*?)</script>', html, re.S)
    assert m is not None
    embedded = json.loads(m.group(1))
    assert embedded["kind"] == "classification"
    assert embedded["classification"]["arms"][0]["metrics"]["accuracy"] == 0.5
