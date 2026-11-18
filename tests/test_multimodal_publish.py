"""The sanitized multimodal export and its section on the Pages dashboard, offline."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals" / "multimodal"))
sys.path.insert(0, str(ROOT / "examples" / "multimodal_gate"))

import publish  # noqa: E402
from evaluators_dashboard import page_data, render_html  # noqa: E402

PUBLISHED = ROOT / "docs" / "benchmarks" / "multimodal-live-20261118.json"
REPORT = ROOT / "docs" / "benchmarks" / "evaluators-live-20261116T192356Z.json"


def load() -> dict[str, object]:
    data = json.loads(PUBLISHED.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


@pytest.mark.parametrize(
    "leak",
    [
        "/Users/someone/project",
        "/private/tmp/x",
        "results/multimodal/run/calls.jsonl",
        '{"namespace": "synthetic_oc_agg1"}',
        '{"policy_hash": "sha256:00"}',
        "data_base64",
        "Bearer abc",
        "A" * 240,
    ],
)
def test_check_clean_refuses_leaks(leak: str) -> None:
    with pytest.raises(SystemExit, match="refusing to publish"):
        publish.check_clean(json.dumps({"x": leak}))


def test_published_export_is_clean_and_complete() -> None:
    text = PUBLISHED.read_text(encoding="utf-8")
    publish.check_clean(text)
    publish.check_clean(PUBLISHED.with_suffix(".md").read_text(encoding="utf-8"))
    data = load()
    cases = data["cases"]
    assert isinstance(cases, list) and len(cases) == 24
    assert {k for c in cases for k in c} == {
        "id",
        "source",
        "edits",
        "mime_type",
        "sha256",
        "group",
        "counted",
        "labels",
        "record_says",
        "arms",
    }
    for c in cases:
        for arm, cells in c["arms"].items():
            assert arm in ("gemini_decision", "jev_image", "jev_text")
            for q in publish.QKEYS:
                cell = cells[q]
                assert 0.0 <= cell["p"] <= 1.0
                assert cell["close"] == (abs(cell["p"] - 0.5) <= 0.1)
                assert cell["correct"] is None or cell["correct"] == (cell["answer"] == c["labels"][q])
    uncounted = [c["id"] for c in cases if not c["counted"]]
    assert uncounted == ["pdf_4pt"]
    pdf_rows = data["pdf_as_png"]["rows"]  # type: ignore[index]
    assert len(pdf_rows) == 7
    assert all(set(r["columns"]) == {"gemini_pdf", "gemini_png", "jev_png", "jev_text"} for r in pdf_rows)


def test_caveats_are_derived_from_the_data() -> None:
    data = load()
    notes = publish.caveats(data)
    assert notes == data["caveats"]
    text = " ".join(notes)
    for needed in ("cannot see the image", "preview", "24 cases", "page image plus the text", "4 pt"):
        assert needed in text
    assert not re.search("[\\u2010-\\u2015\\u2212-]", text)  # no dash characters in the published prose


def test_dashboard_embeds_the_multimodal_section() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    mm = load()
    html = render_html(page_data(report, REPORT.name, mm, PUBLISHED.name))
    m = re.search(r'<script id="results-data" type="application/json">(.*?)</script>', html, re.S)
    assert m is not None
    embedded = json.loads(m.group(1))
    assert embedded["multimodal"]["cases"] == mm["cases"]
    assert embedded["multimodal_files"] == {"json": PUBLISHED.name, "md": PUBLISHED.with_suffix(".md").name}
    assert 'id="mm"' in html and "renderMultimodal" in html


def test_dashboard_without_multimodal_data() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    data = page_data(report, REPORT.name)
    assert data["multimodal"] is None and data["multimodal_files"] is None


def test_published_page_matches_the_generator() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    expected = render_html(page_data(report, REPORT.name, load(), PUBLISHED.name))
    assert (ROOT / "docs" / "benchmarks" / "index.html").read_text(encoding="utf-8") == expected


def test_edited_screenshots_group_counts_only_edited_png_cases() -> None:
    data = load()
    cases = data["cases"]
    assert isinstance(cases, list)
    picked = [c for c in cases if c["group"] == "edited" and c["mime_type"] == "image/png"]
    assert len(picked) == 11
    counts = data["counts"]
    assert isinstance(counts, dict)
    assert list(counts) == ["unedited", "edited", "edited_conflict", "edited_png", "controls"]
    assert counts["edited_png"] == publish.edited_png_counts(cases)
    for arm in ("gemini_decision", "jev_image", "jev_text"):
        for q in publish.QKEYS:
            right = sum(c["arms"][arm][q]["correct"] is True for c in picked)
            assert counts["edited_png"][arm][q] == [right, 11]
    # an edited PDF case or an uncounted case must not change the group
    pdf = next(c for c in cases if c["group"] == "edited" and c["mime_type"] == "application/pdf")
    extra = {**picked[0], "counted": False}
    assert publish.edited_png_counts([*picked, pdf, extra]) == counts["edited_png"]
