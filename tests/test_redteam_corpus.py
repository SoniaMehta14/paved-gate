from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import build_corpus
import pytest
from taxonomy import AttackClass

from paved_gate.integrations.harness2.config import load_adoption_policy
from paved_gate.integrations.harness2.diff import diff_trees
from paved_gate.integrations.openshell.config import load_egress_policy
from paved_gate.integrations.openshell.inspector import TAINT_CONTEXT_CHARS
from paved_gate.privacy.detect import detect

ROOT = build_corpus.ROOT


@pytest.fixture(scope="module")
def items() -> list[dict[str, Any]]:
    return build_corpus.build()


def test_committed_corpus_matches_generator_and_manifest(items: list[dict[str, Any]]) -> None:
    data = "".join(json.dumps(i, ensure_ascii=False, sort_keys=True) + "\n" for i in items)
    manifest = json.loads(build_corpus.MANIFEST.read_text())
    assert build_corpus.OUT.read_text(encoding="utf-8") == data
    assert manifest["sha256"] == hashlib.sha256(data.encode()).hexdigest()
    assert manifest["items"] == len(items)


def test_schema(items: list[dict[str, Any]]) -> None:
    codes = {c.value for c in AttackClass}
    for i in items:
        assert i["set"] in {"gate", "detector", "harness2", "openshell"}
        assert i["label"] in {"malicious", "benign"}
        assert i["attack_class"] in codes
        assert i["id"].startswith(i["set"] + ".")


def test_gate_design_is_balanced(items: list[dict[str, Any]]) -> None:
    gate = [i for i in items if i["set"] == "gate"]
    assert sum(i["label"] == "malicious" for i in gate) == 45 * 11
    assert sum(i["label"] == "benign" for i in gate) == 30 * 5
    benign_transforms = {i["transform"] for i in gate if i["label"] == "benign"}
    assert benign_transforms <= {"plain", "padding", *build_corpus.C4_TRANSFORMS}


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def test_synthetic_identifiers_only(items: list[dict[str, Any]]) -> None:
    # Gate variants are encodings of the seeds, so check the seeds themselves plus every non gate payload.
    seeds = [s for _, _, plain, para in build_corpus.MALICIOUS for s in (plain, para)]
    seeds += [text for _, _, text in build_corpus.BENIGN]
    text = "\n".join([*seeds, *(s for i in items if i["set"] != "gate" for s in _strings(i["payload"]))])
    for ssn in re.findall(r"\b\d{3}-\d{2}-\d{4}\b", text):
        assert ssn.startswith("123-45-"), ssn
    for domain in re.findall(r"@([A-Za-z0-9.-]+\.[A-Za-z]{2,})", text):
        assert domain.endswith(("example.com", "example.net", "example.org", "example")), domain
    for ip in re.findall(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", text):
        assert ip.startswith(("203.0.113.", "198.51.100.", "192.0.2.")), ip


def test_detector_canonical_cases_are_detected(items: list[dict[str, Any]]) -> None:
    for i in items:
        if i["set"] == "detector" and i["transform"] == "canonical":
            found = {f.detector for f in detect(i["payload"]["text"], [i["payload"]["detector"]])}
            assert found == {i["payload"]["detector"]}, i["id"]


def test_harness2_attacks_leave_protected_guardrails_untouched(items: list[dict[str, Any]]) -> None:
    for i in items:
        if i["set"] == "harness2":
            p = i["payload"]
            assert p["adopted"]["guardrails.md"] == p["baseline"]["guardrails.md"], i["id"]


def test_harness2_truncated_change_falls_past_diff_limit(items: list[dict[str, Any]]) -> None:
    """The weakened file comes after more than max_diff_chars of other diff, which hid it from a joined clip."""
    cfg = load_adoption_policy(ROOT / "policy" / "harness2_adoption.policy.yaml").regression
    for i in items:
        if i["set"] == "harness2" and i["transform"] == "truncated":
            p = i["payload"]
            joined = "".join(c.unified for c in diff_trees(p["baseline"], p["adopted"]))
            assert joined.index("systemprompt.md") > cfg.max_diff_chars, i["id"]


def test_openshell_padding_places_injection_past_limits(items: list[dict[str, Any]]) -> None:
    # Every injected instruction names a path such as /notes/...; the filler contains no slash.
    limit = load_egress_policy(ROOT / "policy" / "openshell_egress.policy.yaml").max_inspect_chars
    cut = {"pad_taint_context": TAINT_CONTEXT_CHARS, "pad_past_inspect_limit": limit}
    seen = set()
    for i in items:
        if i["set"] == "openshell" and i["transform"] in cut:
            body = i["payload"]["response"]["body"]
            assert "/" not in body[: cut[i["transform"]]] and "/" in body, i["id"]
            assert len(body) < 2 * cut[i["transform"]] + 2_000, i["id"]
            seen.add(i["transform"])
    assert seen == set(cut)


def test_heldout_privacy_set_is_synthetic_and_matches_manifest() -> None:
    import build_heldout_privacy as hp

    items = hp.build()
    data = "".join(json.dumps(i, ensure_ascii=False, sort_keys=True) + "\n" for i in items)
    assert hp.OUT.read_text(encoding="utf-8") == data
    assert json.loads(hp.MANIFEST.read_text())["sha256"] == hashlib.sha256(data.encode()).hexdigest()
    text = "\n".join(i["payload"]["text"] for i in items)
    assert all(s.startswith("123-45-69") for s in re.findall(r"\b\d{3}-\d{2}-\d{4}\b", text))
    assert all(d.startswith("example.") for d in re.findall(r"@([\w.-]+)", text))
