"""Benchmark helpers: payload set validity, audit joining across retries, and binary metrics."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evals" / "openshell"))
pytest.importorskip("grpc")

import build_payloads  # noqa: E402
import run_bench  # noqa: E402


def test_payload_set_matches_its_manifest_and_builds_reproducibly() -> None:
    cases, manifest = run_bench.load_cases()
    assert len(cases) == manifest["cases"]
    rebuilt = build_payloads.build()
    assert [c.id for c in rebuilt] == [c["id"] for c in cases]
    assert all(c["label"] in ("attack", "benign") for c in cases)


def test_plan_pairs_every_follow_through_with_its_injection() -> None:
    cases, _ = run_bench.load_cases()
    by_id = {c["id"]: c for c in cases}
    t = run_bench.build_plan(cases)["T"]
    for prev, step in itertools.pairwise(t):
        if by_id[step["case"]]["category"] == "follow_through":
            assert prev["case"] == by_id[step["case"]]["pair"] and prev["role"] == "setup"


def _case(cid: str, method: str, response: str | None = None) -> run_bench.Case:
    return run_bench.Case(
        id=cid,
        category="x",
        subtype="x",
        label="benign",
        method=method,
        url="",
        body=None,
        response_type="application/json" if response else None,
        response=response,
        pair=None,
    )


def _e(direction: str, rid: str, path: str, data: bytes, decision: str = "allow") -> dict[str, object]:
    return {
        "direction": direction,
        "request_id": rid,
        "path": path,
        "body_sha256": run_bench.sha256(data),
        "decision": decision,
    }


def test_match_step_finds_entries_by_content_even_out_of_order() -> None:
    stored = run_bench.STORED_REPLY
    entries = [
        _e("request", "r2", "/notes", b"second"),
        _e("response", "r2", "/notes", stored),
        _e("request", "r1", "/notes", b"first"),  # an abandoned attempt that reached the gate late
        _e("request", "r3", "/tool/x", b""),
        _e("response", "r3", "/tool/x", b"tool-a"),
        _e("request", "r4", "/notes", b"denied", "deny"),
    ]
    idx, used = run_bench.index_audit(entries), set[str]()
    req, rsp, n = run_bench.match_step(idx, used, _case("a", "POST"), "/notes", "first", 2)
    assert req["request_id"] == "r1" and rsp == {} and n == 1
    req, rsp, _ = run_bench.match_step(idx, used, _case("b", "POST"), "/notes", "second", 1)
    assert req["request_id"] == "r2" and rsp["request_id"] == "r2"
    req, rsp, _ = run_bench.match_step(idx, used, _case("c", "GET", "tool-a"), "/tool/x", None, 1)
    assert req["request_id"] == "r3" and rsp["body_sha256"] == run_bench.sha256(b"tool-a")
    req, rsp, _ = run_bench.match_step(idx, used, _case("d", "POST"), "/notes", "denied", 1)
    assert req["decision"] == "deny" and rsp == {}
    assert run_bench.match_step(idx, used, _case("e", "POST"), "/notes", "never-seen", 1) == ({}, {}, 0)


def test_repeated_tool_case_is_matched_in_order() -> None:
    entries = [
        _e("request", "r1", "/tool/x", b""),
        _e("response", "r1", "/tool/x", b"inj"),
        _e("request", "r2", "/tool/x", b""),
        _e("response", "r2", "/tool/x", b"inj"),
    ]
    idx, used = run_bench.index_audit(entries), set[str]()
    first = run_bench.match_step(idx, used, _case("i", "GET", "inj"), "/tool/x", None, 1)
    second = run_bench.match_step(idx, used, _case("i", "GET", "inj"), "/tool/x", None, 1)
    assert first[0]["request_id"] == "r1" and second[0]["request_id"] == "r2"


def test_retried_tool_case_leaves_an_entry_for_its_next_occurrence() -> None:
    entries = [
        _e("request", "r1", "/tool/x", b""),
        _e("response", "r1", "/tool/x", b"inj"),
        _e("request", "r2", "/tool/x", b""),
        _e("response", "r2", "/tool/x", b"inj"),
    ]
    idx, used = run_bench.index_audit(entries), set[str]()
    # the first step retried once, but its failed attempt never reached the gate; one more step of this case follows
    first = run_bench.match_step(idx, used, _case("i", "GET", "inj"), "/tool/x", None, 2, later_steps=1)
    second = run_bench.match_step(idx, used, _case("i", "GET", "inj"), "/tool/x", None, 1)
    assert first[2] == 1 and first[0]["request_id"] == "r1" and second[0]["request_id"] == "r2"


def test_binary_metrics() -> None:
    m = run_bench.binary([1, 1, 0, 0, 0], [1, 0, 1, 0, 0])
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 2)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["fpr"] == pytest.approx(1 / 3, abs=1e-4)
