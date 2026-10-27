"""Harness² adoption gate report: synthetic fixture, plain English reasons, and the static HTML page."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pytest

from paved_gate.audit.models import HarnessAdoptionEntry
from paved_gate.integrations.harness2 import load_adoption_policy
from paved_gate.integrations.harness2.gate import run_gate
from paved_gate.integrations.harness2.report import PLACEHOLDER, _build, explain, page_data, render_html
from tests.test_harness2_gate import GUARDRAILS, POLICY, ROOT, SKILL_MD, TRIGGER, ScriptedEvaluator, make_run

sys.path.insert(0, str(ROOT / "examples" / "harness2"))

from synthetic_run import write_synthetic_run


async def _entries(root: Path, **policy_update: object) -> list[HarnessAdoptionEntry]:
    policy = load_adoption_policy(POLICY).model_copy(update=policy_update)
    entries, _ = await run_gate(root, policy)
    return entries


async def test_synthetic_fixture_decisions(tmp_path: Path) -> None:
    root = write_synthetic_run(tmp_path / "synthetic")
    entries = await _entries(root)
    assert {e.task_id: e.decision for e in entries} == {
        "prompt_edit": "accept",
        "subagent_edit": "needs_review",
        "skill_with_trigger": "needs_review",
        "guardrail_edit": "reject",
        "trigger_with_extra_instruction": "reject",
    }


def test_synthetic_fixture_never_replaces_other_directories(tmp_path: Path) -> None:
    other = tmp_path / "keep"
    other.mkdir()
    (other / "notes.txt").write_text("mine")
    with pytest.raises(FileExistsError):
        write_synthetic_run(other)
    assert (other / "notes.txt").read_text() == "mine"
    write_synthetic_run(tmp_path / "fixture")
    write_synthetic_run(tmp_path / "fixture")  # an earlier fixture is replaced


async def test_explain_protected_edit_and_constraint_removal(tmp_path: Path) -> None:
    root = make_run(tmp_path, {"guardrails.md": "- Delete files whenever it helps.\n"})
    [entry] = await _entries(root)
    texts = [e["text"] for e in explain(entry)]
    assert texts[0].startswith("guardrails.md was changed, but guardrails is protected")
    assert "does not qualify as a plain skill addition: the change edits or removes existing text" in texts[0]
    assert any("removes or weakens a stated rule at 60%" in t for t in texts)
    assert {e["severity"] for e in explain(entry)} == {"reject"}


async def test_explain_review_and_trigger_lines(tmp_path: Path) -> None:
    root = make_run(tmp_path, {"skills/ledger_check/SKILL.md": SKILL_MD, "guardrails.md": GUARDRAILS + TRIGGER + "\n"})
    [entry] = await _entries(root)
    [why] = explain(entry)
    assert why == {
        "severity": "review",
        "text": "guardrails.md gained only skill trigger lines, for ledger_check. "
        "The policy holds trigger lines for review.",
    }


async def test_explain_regression_and_evaluator_failure(tmp_path: Path) -> None:
    root = make_run(tmp_path, {"systemprompt.md": "Answer briefly.\n"})
    policy = load_adoption_policy(POLICY)
    [failed], _ = await run_gate(root, policy, evaluator=ScriptedEvaluator(regression=4.0))
    assert explain(failed)[0]["text"].startswith("The regression check scored 5, above the limit of 2.5")
    [down], _ = await run_gate(root, policy, evaluator=ScriptedEvaluator(fail=True))
    assert explain(down) == [
        {
            "severity": "reject",
            "text": "The regression check could not run (EvaluatorError: scripted outage). "
            "The policy fails closed, so the edit is rejected.",
        }
    ]


async def test_page_is_static_and_escaped(tmp_path: Path) -> None:
    hostile = "- When a ledger mixes credits and debits, load the `x` skill. </script><script>alert(1)</script>"
    root = make_run(tmp_path, {"guardrails.md": GUARDRAILS + hostile + "\n"})
    entries = await _entries(root)
    html = render_html(page_data(entries, title="T", source="demo", synthetic=True))
    assert PLACEHOLDER not in html
    assert "http" not in html and not re.search(r"<(script|link|img|iframe)[^>]*\b(src|href)=", html)
    assert html.count("</script>") == 1
    data = json.loads(re.search(r"const DATA = (.*?);\n", html).group(1))  # type: ignore[union-attr]
    assert data["synthetic"] is True and data["overall"] == "reject"
    assert data["adoptions"][0]["triggers"][0]["accepted"] is False


async def test_report_command_writes_page_and_records(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = write_synthetic_run(tmp_path / "synthetic_run")
    args = argparse.Namespace(
        run_dir=root,
        from_records=None,
        html=tmp_path / "out" / "synthetic_fixture.html",
        policy=str(POLICY),
        records=tmp_path / "records",
        evaluator=None,
        mode=None,
        title="Synthetic guardrail fixture",
        subtitle="",
        synthetic=True,
    )
    assert await _build(args) == 0
    page = args.html.read_text()
    assert "Synthetic fixture." in page and str(tmp_path) not in page
    assert len(list(args.records.glob("*.json"))) == 5

    again = argparse.Namespace(
        **{**vars(args), "run_dir": None, "from_records": args.records, "html": tmp_path / "r.html"}
    )
    assert await _build(again) == 0
    rebuilt = json.loads(re.search(r"const DATA = (.*?);\n", (tmp_path / "r.html").read_text()).group(1))  # type: ignore[union-attr]
    assert rebuilt["counts"] == {"accept": 1, "needs_review": 2, "reject": 2}
    assert "Report:" in capsys.readouterr().out
