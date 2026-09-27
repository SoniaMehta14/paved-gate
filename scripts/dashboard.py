"""Render a benchmark JSON report into a single self-contained HTML dashboard.

    uv run python scripts/dashboard.py results/benchmark-20260927-140312.json [--open]

The JSON is embedded in the page, so the HTML opens straight from disk (file://)
with no server, no build step, and no network access: charts are inline SVG.
"""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from collections.abc import Mapping
from pathlib import Path

TEMPLATE = Path(__file__).with_name("dashboard_template.html")
PLACEHOLDER = "/*__BENCHMARK_DATA__*/null"


def render_dashboard(report: Mapping[str, object]) -> str:
    # "<" escaped so no string in the data can close the <script> element early.
    data = json.dumps(report, separators=(",", ":")).replace("<", "\\u003c")
    template = TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise RuntimeError(f"{TEMPLATE.name} is missing the data placeholder")
    return template.replace(PLACEHOLDER, data)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", type=Path, help="benchmark JSON written by scripts/benchmark.py")
    ap.add_argument("-o", "--output", type=Path, help="HTML path (default: next to the JSON)")
    ap.add_argument("--open", action="store_true", help="open the dashboard in a browser")
    args = ap.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    out = args.output or args.report.with_suffix(".html")
    out.write_text(render_dashboard(report), encoding="utf-8")
    print(f"Dashboard: {out}")
    if args.open:
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
