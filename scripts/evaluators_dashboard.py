"""Render a Jev vs Gemini Decision 1 report into a single self-contained HTML dashboard.

    uv run python scripts/evaluators_dashboard.py docs/benchmarks/<report>.json            # writes <report>.html
    uv run python scripts/evaluators_dashboard.py docs/benchmarks/<report>.json --publish  # writes docs/benchmarks/
    uv run python scripts/evaluators_dashboard.py docs/benchmarks/<report>.json --publish \
        --multimodal docs/benchmarks/multimodal-live-<date>.json                           # adds the multimodal test

Reads the JSON written by scripts/benchmark_evaluators.py and, optionally, the sanitized multimodal results
written by evals/multimodal/publish.py. Every number on the page comes from those files; the page makes no
network requests.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TEMPLATE = HERE / "evaluators_dashboard_template.html"
PLACEHOLDER = "/*__RESULTS__*/null"
PUBLISH_DIR = ROOT / "docs" / "benchmarks"
REPO_URL = "https://github.com/SoniaMehta14/paved-gate"
ARM_LABELS = {"jev": "Jev", "gemini_decision": "Gemini Decision 1"}

J = Mapping[str, object]


def obj(v: object) -> J:
    assert isinstance(v, Mapping), v
    return v


def page_data(
    report: J, json_name: str, multimodal: J | None = None, multimodal_name: str | None = None
) -> dict[str, object]:
    workloads = obj(report["workloads"])
    rebuilt = obj(report["rebuilt"]) if "rebuilt" in report else None
    files = {"json": json_name, "md": str(Path(json_name).with_suffix(".md"))}
    if rebuilt is not None:
        files["calls"] = str(rebuilt["from"])
    return {
        "title": report["title"],
        "mode": report["mode"],
        "date": report["date"],
        "models": report["models"],
        "settings": report["settings"],
        "arms": report["arms"],
        "arm_labels": ARM_LABELS,
        "tickets": workloads.get("tickets"),
        "harness2": workloads.get("harness2"),
        "fanout": workloads.get("fanout"),
        "notes": report["notes"],
        "calls_total": rebuilt["calls_read"] if rebuilt is not None else None,
        "files": files,
        "repo": REPO_URL,
        "multimodal": multimodal,
        "multimodal_files": (
            {"json": multimodal_name, "md": str(Path(multimodal_name).with_suffix(".md"))} if multimodal_name else None
        ),
    }


def render_html(data: Mapping[str, object]) -> str:
    # "<" escaped so no string in the data can close the <script> element early.
    payload = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    template = TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise RuntimeError(f"{TEMPLATE.name} is missing the data placeholder")
    return template.replace(PLACEHOLDER, payload)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", type=Path, help="JSON written by scripts/benchmark_evaluators.py")
    ap.add_argument("--publish", action="store_true", help="write docs/benchmarks/index.html (GitHub Pages)")
    ap.add_argument("--multimodal", type=Path, help="sanitized multimodal results from evals/multimodal/publish.py")
    args = ap.parse_args()
    src: Path = args.report
    report = json.loads(src.read_text(encoding="utf-8"))
    mm: Path | None = args.multimodal
    multimodal = json.loads(mm.read_text(encoding="utf-8")) if mm else None
    html = render_html(page_data(report, src.name, multimodal, mm.name if mm else None))
    out = PUBLISH_DIR / "index.html" if args.publish else src.with_suffix(".html")
    out.write_text(html, encoding="utf-8")
    print(f"Dashboard: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
