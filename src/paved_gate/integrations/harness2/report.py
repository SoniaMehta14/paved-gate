"""Build a single file HTML report of Harness² adoption gate decisions from the JSON audit records.

    python -m paved_gate.integrations.harness2.report ../harness2/runs/demo --html results/harness2/default.html
    python -m paved_gate.integrations.harness2.report --from-records logs/harness2 --html results/harness2/r.html

With a run directory, the gate runs first and its audit records are written to --records. With
--from-records, existing audit records are read instead. The page has inline CSS, JS and data and
makes no network requests. Paths in it are relative to the run directory; the run directory itself
is shown by its last path component only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from paved_gate.audit.models import AdoptionFileResult, HarnessAdoptionEntry
from paved_gate.integrations.harness2.cli import print_summary, with_evaluator, write_records
from paved_gate.integrations.harness2.config import load_adoption_policy
from paved_gate.integrations.harness2.gate import run_gate
from paved_gate.integrations.harness2.run_dir import RunDirError, Skipped

TEMPLATE = Path(__file__).resolve().parent / "report_template.html"
PLACEHOLDER = "/*__REPORT__*/null"
UNMAPPED = "(unmapped)"
_ORDER = {"ok": 0, "accept": 0, "review": 1, "needs_review": 1, "reject": 2}
_VERB = {"added": "was added", "removed": "was removed", "modified": "was changed"}
_ACCESS = {"editable": "editable", "review": "held for review", "protected": "protected"}


def _file_reason(entry: HarnessAdoptionEntry, f: AdoptionFileResult) -> str:
    verb = _VERB[f.change]
    if f.source == "skill_trigger":
        names = ", ".join(t.skill for t in entry.skill_triggers if t.skill)
        if f.result == "review":
            return f"{f.path} gained only skill trigger lines, for {names}. The policy holds trigger lines for review."
        return f"{f.path} gained only skill trigger lines, for {names}, and the policy does not allow them."
    if f.source.startswith("rule:"):
        text = f"{f.path} {verb}. The path rule {f.source[5:]} marks it as {_ACCESS[f.access]}."
    elif f.source == "unmapped":
        text = f"{f.path} {verb}. It belongs to no component in the policy, so it is {_ACCESS[f.access]}."
    elif f.access == "protected":
        text = f"{f.path} {verb}, but {f.component} is protected: the editor may never change it."
    else:
        text = f"{f.path} {verb}. Changes to {f.component} need human review before they are adopted."
    if entry.skill_trigger_note and f.path == entry.skill_trigger_file:
        text += " It does not qualify as a plain skill addition: " + entry.skill_trigger_note.split(": ", 1)[-1] + "."
        text += "".join(f' Added line "{t.line}": {t.reason}.' for t in entry.skill_triggers if not t.accepted)
    return text


def explain(entry: HarnessAdoptionEntry) -> list[dict[str, str]]:
    """Plain English for every reason the edit was rejected or held for review, most severe first."""
    out = [{"severity": f.result, "text": _file_reason(entry, f)} for f in entry.files if f.result != "ok"]
    reg = entry.regression
    if reg.error:
        closed = reg.fail_mode == "closed"
        out.append(
            {
                "severity": "reject" if closed else "review",
                "text": f"The regression check could not run ({reg.error}). The policy fails "
                + ("closed, so the edit is rejected." if closed else "open, so the edit is held for review."),
            }
        )
    elif reg.passed is False:
        if reg.regression is not None and reg.regression > reg.regression_threshold:
            out.append(
                {
                    "severity": "reject",
                    "text": f"The regression check scored {reg.regression:g}, above the limit of "
                    f"{reg.regression_threshold:g} (1 means no sign of regression): the adopted harness likely "
                    "makes results worse than the baseline.",
                }
            )
        if reg.constraint_removal is not None and reg.constraint_removal > reg.constraint_removal_threshold:
            out.append(
                {
                    "severity": "reject",
                    "text": f"The regression check puts the chance that this edit removes or weakens a stated rule "
                    f"at {reg.constraint_removal:.0%}, above the limit of {reg.constraint_removal_threshold:.0%}.",
                }
            )
    return sorted(out, key=lambda e: -_ORDER[e["severity"]])


def _components(entry: HarnessAdoptionEntry) -> list[dict[str, object]]:
    lines = {c.component: (c.lines_added, c.lines_removed) for c in entry.components}
    grouped: dict[str, list[AdoptionFileResult]] = {}
    for f in entry.files:
        grouped.setdefault(f.component or UNMAPPED, []).append(f)
    return [
        {
            "component": name,
            "files": [
                {"path": f.path, "change": f.change, "access": f.access, "source": f.source, "result": f.result}
                for f in files
            ],
            "lines_added": lines.get(name, (0, 0))[0],
            "lines_removed": lines.get(name, (0, 0))[1],
            "result": max((f.result for f in files), key=lambda r: _ORDER[r]),
        }
        for name, files in grouped.items()
    ]


def _adoption(entry: HarnessAdoptionEntry) -> dict[str, object]:
    reg = entry.regression
    return {
        "task_id": entry.task_id,
        "mode": entry.mode,
        "label": entry.adopted_label,
        "namespace": entry.namespace,
        "decision": entry.decision,
        "explanations": explain(entry),
        "summary": entry.reasons[0] if entry.decision == "accept" and entry.reasons else "",
        "components": _components(entry),
        "regression": reg.model_dump(mode="json", exclude={"raw_evaluator_response"}),
        "evaluator": entry.evaluator.model_dump(mode="json") if entry.evaluator else None,
        "trigger_mode": entry.skill_trigger_lines,
        "trigger_note": entry.skill_trigger_note,
        "triggers": [t.model_dump(mode="json") for t in entry.skill_triggers],
        "links": {"selection": entry.selection, "reflection_note": entry.reflection_note, "lesson": entry.lesson},
        "lesson": {k: v for k, v in (entry.experience or {}).items() if k in ("cue", "lesson")},
        "baseline_sha256": entry.baseline_sha256,
        "adopted_sha256": entry.adopted_sha256,
    }


def page_data(
    entries: Sequence[HarnessAdoptionEntry],
    skipped: Sequence[Skipped] = (),
    *,
    title: str,
    subtitle: str = "",
    source: str = "",
    synthetic: bool = False,
    root: Path | None = None,
) -> dict[str, object]:
    ordered = sorted(entries, key=lambda e: (-_ORDER[e.decision], e.task_id, e.mode))
    decisions = [e.decision for e in entries]
    overall: str | None = None
    if decisions:
        overall = max(decisions, key=_ORDER.__getitem__)
    evaluator = next((e.evaluator for e in entries if e.evaluator), None)
    now = datetime.now(UTC)
    return {
        "title": title,
        "subtitle": subtitle,
        "source": source,
        "synthetic": synthetic,
        "generated": f"{now:%B} {now.day}, {now.year}",
        "policy": sorted({f"{e.policy_name} ({e.policy_hash[:15]})" for e in entries}),
        "evaluator": evaluator.model_dump(mode="json") if evaluator else None,
        "overall": overall,
        "counts": {d: decisions.count(d) for d in ("accept", "needs_review", "reject")},
        "skipped": [
            {
                "selection": (item.selection_path.relative_to(root) if root else item.selection_path).as_posix(),
                "reason": item.reason,
            }
            for item in skipped
        ],
        "adoptions": [_adoption(e) for e in ordered],
    }


def render_html(data: dict[str, object]) -> str:
    payload = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    template = TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise RuntimeError(f"{TEMPLATE.name} is missing the data placeholder")
    return template.replace(PLACEHOLDER, payload)


def load_records(directory: Path) -> list[HarnessAdoptionEntry]:
    return [
        HarnessAdoptionEntry.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted(directory.glob("*.json"))
    ]


async def _build(args: argparse.Namespace) -> int:
    skipped: list[Skipped] = []
    root: Path | None = None
    if args.from_records:
        entries = load_records(args.from_records)
        source = args.from_records.name
    else:
        root = args.run_dir
        policy = with_evaluator(load_adoption_policy(args.policy), args.evaluator, args.mode)
        try:
            entries, skipped = await run_gate(root, policy)
        except RunDirError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        write_records(entries, args.records)
        print_summary(entries, skipped, root, args.records)
        source = root.resolve().name
    if not entries:
        print("error: no adoption records to report", file=sys.stderr)
        return 2
    data = page_data(
        entries, skipped, title=args.title, subtitle=args.subtitle, source=source, synthetic=args.synthetic, root=root
    )
    args.html.parent.mkdir(parents=True, exist_ok=True)
    args.html.write_text(render_html(data), encoding="utf-8")
    print(f"Report: {args.html}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, nargs="?", help="Harness² output directory, e.g. ../harness2/runs/demo")
    ap.add_argument("--from-records", type=Path, help="build from existing JSON audit records instead of a run")
    ap.add_argument("--html", type=Path, required=True, help="where to write the report")
    ap.add_argument("--policy", default="policy/harness2_adoption.policy.yaml")
    ap.add_argument("--records", type=Path, default=Path("logs/harness2"), help="where to write the audit records")
    ap.add_argument("--evaluator", choices=["heuristic", "jev"], help="override the policy's evaluator provider")
    ap.add_argument("--mode", choices=["mock", "live"], help="override the policy's evaluator mode (jev only)")
    ap.add_argument("--title", default="Harness² adoption gate report")
    ap.add_argument("--subtitle", default="")
    ap.add_argument("--synthetic", action="store_true", help="label the report as built from a synthetic fixture")
    args = ap.parse_args()
    if (args.run_dir is None) == (args.from_records is None):
        ap.error("give either a run directory or --from-records")
    return asyncio.run(_build(args))


if __name__ == "__main__":
    sys.exit(main())
