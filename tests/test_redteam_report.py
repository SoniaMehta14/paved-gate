"""The red team report rebuilds from the committed run data alone, and says what it should about omitted arms."""

from __future__ import annotations

import json

import redteam_report as report


def test_report_rebuilds_byte_for_byte_from_committed_data() -> None:
    runs = json.loads(report.RUNS.read_text())
    built = report.build(runs)
    summary_md = (report.OUT / "SUMMARY.md").read_text(encoding="utf-8")
    assert json.loads((report.OUT / "results.json").read_text()) == json.loads(json.dumps(built, default=str))
    assert (report.OUT / "RESULTS.md").read_text() == report.render_markdown(built) + "\n"
    assert (report.OUT / "index.html").read_text() == report.render_html(built, summary_md)


def test_omitted_gpt_arms_appear_in_no_table() -> None:
    built = report.build(json.loads(report.RUNS.read_text()))
    assert set(built["omitted_arms"]) == {"gpt-4o", "gpt-4o-mini"}
    tables = json.dumps({k: v for k, v in built.items() if k not in ("omitted_arms", "spend_usd_by_evaluator")})
    assert "gpt" not in tables.lower()
    md = (report.OUT / "RESULTS.md").read_text()
    assert all("GPT" not in line for line in md.splitlines() if line.startswith("|"))


def test_headline_numbers_match_the_runs() -> None:
    built = report.build(json.loads(report.RUNS.read_text()))
    gate = built["baseline"]["gate"]
    assert (gate["jev"]["attack_success"]["k"], gate["jev"]["attack_success"]["n"]) == (48, 495)
    assert (gate["gemini_decision"]["attack_success"]["k"], gate["gemini_decision"]["attack_success"]["n"]) == (52, 495)
    assert len(built["passed_both_live"]) == 15  # from the two single evaluator runs
    assert built["strict_profile"]["gate"]["dual"]["attack_success"]["k"] == 15  # a separate live run
