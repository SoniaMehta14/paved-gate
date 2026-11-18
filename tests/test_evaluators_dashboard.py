from __future__ import annotations

import json
import re
from pathlib import Path

import evaluators_dashboard
import pytest
from evaluators_dashboard import PLACEHOLDER, page_data, render_html

REPORT = Path(__file__).resolve().parents[1] / "docs" / "benchmarks" / "evaluators-live-20261116T192356Z.json"


def embedded(html: str) -> dict[str, object]:
    m = re.search(r'<script id="results-data" type="application/json">(.*?)</script>', html, re.S)
    assert m is not None
    data = json.loads(m.group(1))
    assert isinstance(data, dict)
    return data


def test_renders_published_report() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    html = render_html(page_data(report, REPORT.name))
    assert PLACEHOLDER not in html
    data = embedded(html)
    assert data["date"] == report["date"]
    assert data["notes"] == report["notes"]
    assert data["tickets"] == report["workloads"]["tickets"]
    assert data["files"] == {
        "json": REPORT.name,
        "md": REPORT.with_suffix(".md").name,
        "calls": "evaluators-live-20261116T192356Z.calls.jsonl",
    }
    assert data["calls_total"] == 2380


def test_page_makes_no_network_requests() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    html = render_html(page_data(report, REPORT.name))
    assert re.search(r"<(script|link|img)[^>]+(src|href)=", html) is None
    assert "fetch(" not in html


def test_data_cannot_close_the_script_element() -> None:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    report["title"] = "</script><script>alert(1)</script>"
    html = render_html(page_data(report, REPORT.name))
    assert "</script><script>alert(1)" not in html
    assert embedded(html)["title"] == report["title"]


def test_missing_placeholder_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    broken = tmp_path / "template.html"
    broken.write_text("<html></html>", encoding="utf-8")
    monkeypatch.setattr(evaluators_dashboard, "TEMPLATE", broken)
    with pytest.raises(RuntimeError, match="placeholder"):
        render_html({})
