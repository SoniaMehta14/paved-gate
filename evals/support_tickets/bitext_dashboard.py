"""Render the full Bitext benchmark results into a single self-contained HTML dashboard.

    uv run python evals/support_tickets/bitext_dashboard.py <dir>            # writes <dir>/dashboard.html
    uv run python evals/support_tickets/bitext_dashboard.py <dir> --publish  # writes docs/bitext/

Reads results.json, per_class.csv and manifest.json from the results directory written by full_bench.py.
Every number on the page comes from those files; the page makes no network requests.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TEMPLATE = HERE / "bitext_dashboard_template.html"
PLACEHOLDER = "/*__RESULTS__*/null"
PUBLISH_DIR = ROOT / "docs" / "bitext"
REPO_URL = "https://github.com/SoniaMehta14/paved-gate"
# Paired tests with p in [low, high) are called borderline significant on the page.
BORDERLINE_P = (0.01, 0.10)

J = Mapping[str, object]


def obj(v: object) -> J:
    assert isinstance(v, Mapping), v
    return v


def lst(v: object) -> list[object]:
    assert isinstance(v, list), v
    return v


def num(v: object) -> float:
    assert isinstance(v, int | float), v
    return float(v)


def per_class(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Mean and sample SD over runs of per-intent recall (per-class accuracy) and F1, per config and scope."""
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        groups[(r["config"], r["scope"], r["label"])].append(r)
    out: list[dict[str, object]] = []
    for (config, scope, label), runs in sorted(groups.items()):
        entry: dict[str, object] = {"config": config, "scope": scope, "label": label, "runs": len(runs)}
        for k in ("recall", "f1"):
            vals = [float(r[k]) for r in runs]
            entry[k] = round(statistics.fmean(vals), 4)
            entry[k + "_sd"] = round(statistics.stdev(vals), 4) if len(vals) > 1 else None
        entry["support"] = int(runs[0]["support"])
        out.append(entry)
    return out


def page_data(results: J, manifest: J, class_rows: list[dict[str, str]]) -> dict[str, object]:
    low, high = BORDERLINE_P
    config_keys = ("id", "label", "model", "prompt", "scope", "temperature")
    configs = [{k: obj(c)[k] for k in config_keys} for c in lst(manifest["configs"])]
    run_keys = ("config", "scope", "run", "accuracy", "api_errors")
    runs = [{k: obj(s)[k] for k in run_keys} for s in lst(results["summary"])]
    return {
        "configs": configs,
        "aggregate": results["aggregate"],
        "latency": results["latency"],
        "run_agreement": results["run_agreement"],
        "paired_tests": results["paired_tests"],
        "runs": runs,
        "per_class": per_class(class_rows),
        "borderline": {"low": low, "high": high},
        "scopes": {obj(a)["scope"]: obj(a)["n"] for a in lst(results["aggregate"])},
        "meta": {
            "dataset": manifest["dataset"],
            "source_url": manifest["source_url"],
            "revision": str(manifest["revision"])[:12],
            "license": manifest["license"],
            "intents": len({r["label"] for r in class_rows}),
            "total_spend_usd": num(manifest["total_spend_usd"]),
            "price_sources": {m: obj(p)["source"] for m, p in obj(manifest["prices_per_mtok"]).items()},
            "finalized_at": manifest["finalized_at"],
            "repo": REPO_URL,
        },
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
    ap.add_argument("results_dir", type=Path, help="directory written by full_bench.py --finalize")
    ap.add_argument("--publish", action="store_true", help="write to docs/bitext/ (GitHub Pages)")
    args = ap.parse_args()
    src: Path = args.results_dir
    results = json.loads((src / "results.json").read_text(encoding="utf-8"))
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    with (src / "per_class.csv").open(encoding="utf-8", newline="") as f:
        class_rows = list(csv.DictReader(f))
    html = render_html(page_data(results, manifest, class_rows))
    if args.publish:
        PUBLISH_DIR.mkdir(parents=True, exist_ok=True)
        out = PUBLISH_DIR / "index.html"
        for name in ("results.json", "per_class.csv"):
            shutil.copy(src / name, PUBLISH_DIR / name)
    else:
        out = src / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"Dashboard: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
