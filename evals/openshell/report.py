"""Turn a benchmark results JSON into the published artifacts. Every number comes from the JSON.

    uv run python evals/openshell/report.py results/openshell-bench-<ts>.json            # writes next to the JSON
    uv run python evals/openshell/report.py results/openshell-bench-<ts>.json --publish  # writes docs/openshell/

Outputs:
  results.json               the full results (published copy)
  RESULTS.md                 markdown tables plus a Mermaid sequence diagram
  index.html                 single file dashboard: inline CSS, JS and SVG, no external requests
  img/linkedin-1200x1200.png headline tiles and the "who caught what" chart (headless Chrome)
  img/linkedin-1200x627.png
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TEMPLATE = HERE / "dashboard_template.html"
PLACEHOLDER = "/*__RESULTS__*/null"
PUBLISH_DIR = ROOT / "docs" / "openshell"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
REPO_URL = "https://github.com/SoniaMehta14/paved-gate"
FRICTION_LOG_URL = f"{REPO_URL}/blob/main/docs/openshell/FRICTION_LOG.md"
CARDS = {"square": (1200, 1200), "wide": (1200, 627)}

J = Mapping[str, object]


def obj(v: object) -> J:
    assert isinstance(v, Mapping), v
    return v


def lst(v: object) -> list[object]:
    assert isinstance(v, list), v
    return v


def page_data(report: J) -> dict[str, object]:
    """The dashboard's data: the report without per-step outcomes and per-case taint rows (kept in results.json)."""
    data = {k: v for k, v in report.items() if k != "outcomes"}
    taint = {
        arm: {
            **{k: v for k, v in obj(t).items() if k not in ("follow_through", "tainted_legit")},
            "follow_through": {k: v for k, v in obj(obj(t)["follow_through"]).items() if k != "cases"},
            "tainted_legit": {k: v for k, v in obj(obj(t)["tainted_legit"]).items() if k != "cases"},
        }
        for arm, t in obj(report["taint_effect"]).items()
    }
    data["taint_effect"] = taint
    data["mermaid"] = mermaid(report)
    data["links"] = {
        "repo": REPO_URL,
        "friction_log": FRICTION_LOG_URL,
        "results_json": "results.json",
        "results_md": f"{REPO_URL}/blob/main/docs/openshell/RESULTS.md",
    }
    return data


def render_html(report: J) -> str:
    data = json.dumps(page_data(report), separators=(",", ":")).replace("<", "\\u003c")
    template = TEMPLATE.read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise RuntimeError(f"{TEMPLATE.name} is missing the data placeholder")
    return template.replace(PLACEHOLDER, data)


# ------------------------------------------------------------------ markdown


def f_pct(v: object) -> str:
    return f"{float(str(v)) * 100:.1f}%" if isinstance(v, int | float) else "n/a"


def f_ms(v: object) -> str:
    return f"{float(str(v)):.1f} ms" if isinstance(v, int | float) else "n/a"


def f_num(v: object) -> str:
    return str(v) if v is not None else "n/a"


def mermaid(report: J) -> str:
    seq = report.get("sequence_example")
    th = obj(obj(report["policy"])["thresholds"])
    if not isinstance(seq, Mapping):
        return ""
    fts = obj(seq["follow_through_scores"])
    lines = [
        "sequenceDiagram",
        "    autonumber",
        "    participant A as Agent (sandbox)",
        "    participant S as OpenShell supervisor",
        "    participant G as Paved Gate middleware",
        "    participant J as Jev",
        "    participant T as Tool API (allowed host)",
        "    A->>S: GET /tool/... (host and port in policy)",
        "    S->>G: EvaluateHttpRequest (PRE_CREDENTIALS)",
        "    G-->>S: ALLOW",
        "    S->>T: forward request",
        "    T-->>S: 200, body carries instructions for the agent",
        "    S->>G: HttpResponsePreReturn.Evaluate (PRE_RETURN, WHOLE_BODY_BYTES)",
        "    G->>J: injection score",
        f"    J-->>G: {seq['injection_score']} (threshold {th['injection']})",
        f"    G-->>S: pass_through, reason_code {seq['injection_reason']}",
        "    Note over G: sandbox tainted: stricter exfiltration threshold, flagged text kept as context",
        "    S-->>A: tool result (flag mode delivers it)",
        f"    A->>S: POST {seq['follow_through_path']} (acting on the injected instruction)",
        "    S->>G: EvaluateHttpRequest (PRE_CREDENTIALS)",
        "    G->>J: sensitive + exfiltration, with the flagged instruction as context",
        f"    J-->>G: exfiltration {fts.get('exfiltration')} (tainted threshold {th['tainted_exfiltration']})",
        f"    G-->>S: DENY, reason_code {seq['follow_through_reason']}",
        "    S-->>A: 403 middleware_denied (never reaches the API)",
    ]
    return "\n".join(lines)


def render_markdown(report: J) -> str:
    run = obj(report["run"])
    arms = obj(report["arms"])
    full = obj(arms["full"])
    head = obj(full["headline"])
    lat = obj(report["latency"])
    mode = str(run["evaluator_mode"])
    tag = "LIVE" if mode == "live" else "MOCK (keyword heuristics, simulated latency: not real model judgments)"
    out: list[str] = []
    w = out.append
    w("# OpenShell + Paved Gate benchmark results")
    w("")
    w(
        f"Run {run['started_at']} to {run['finished_at']} UTC on {run['runtime']} "
        f"(OpenShell {run['openshell_version']}, Docker {run['docker_version']}). "
        f"Full arm evaluator: **Jev {tag}**. Synthetic data only."
    )
    if run.get("meta_reconstructed"):
        w("")
        w(f"_Run times reconstructed: {run['meta_reconstructed']}._")
    w("")
    w(
        "Generated by `evals/openshell/report.py` from [results.json](results.json). "
        "Dashboard: [index.html](index.html). Friction log: [FRICTION_LOG.md](FRICTION_LOG.md)."
    )
    w("")
    w("## Headline (full arm)")
    w("")
    w(
        "| Requests tested | Blocked by OpenShell | Denied by Paved Gate | Injected tool responses flagged "
        "| Attack requests leaked to upstream | Median added latency |"
    )
    w("|---|---|---|---|---|---|")
    added = obj(obj(lat["per_arm"])["full"])["added_ms"]
    w(
        f"| {head['requests_tested']} | {head['blocked_by_openshell']} | {head['denied_by_paved_gate']} | "
        f"{head['injected_responses_flagged']} of {head['injected_responses']} | "
        f"{head['leaked_to_upstream']} of {head['attack_requests']} | "
        f"{f_ms(obj(added)['p50'])} |"
    )
    w("")
    w(f"_{head['definition']}._")
    w("")
    w("## Who caught what (full arm)")
    w("")
    w("| Request type | n | OpenShell | Paved Gate | Passed | Error |")
    w("|---|---|---|---|---|---|")
    for g in lst(full["caught_by"]):
        g = obj(g)
        w(f"| {g['label']} | {g['n']} | {g['openshell']} | {g['paved_gate']} | {g['passed']} | {g['error']} |")
    w("")
    w("## Three arms side by side")
    w("")
    w(
        "| Arm | Evaluator | Added latency p50 | Added latency p95 | Recall: synthetic PHI | Recall: injected tool "
        "responses | Recall: follow-through | FPR (untainted) | FPR (tainted) | Steps |"
    )
    w("|---|---|---|---|---|---|---|---|---|---|")
    for arm_id, a in arms.items():
        a = obj(a)
        acc = obj(a["accuracy"])
        rbt = obj(acc["recall_by_type"])
        added_a = obj(obj(lat["per_arm"])[arm_id])["added_ms"]
        p50 = f_ms(obj(added_a)["p50"]) if added_a else "baseline"
        p95 = f_ms(obj(added_a)["p95"]) if added_a else "baseline"
        w(
            f"| {a['label']} | {a['evaluator']} | {p50} | {p95} | {f_pct(obj(rbt['phi'])['recall'])} "
            f"(n={obj(rbt['phi'])['n']}) | {f_pct(obj(rbt['injection'])['recall'])} (n={obj(rbt['injection'])['n']}) | "
            f"{f_pct(obj(rbt['follow_through'])['recall'])} (n={obj(rbt['follow_through'])['n']}) | "
            f"{f_pct(obj(acc['untainted'])['fpr'])} (n={obj(acc['untainted'])['negatives']}) | "
            f"{f_pct(obj(acc['tainted'])['fpr'])} (n={obj(acc['tainted'])['negatives']}) | {a['steps']} |"
        )
    w("")
    w("Added latency: " + str(obj(lat["definitions"])["added_ms"]) + ".")
    w("")
    w("## Paved Gate accuracy (full arm)")
    w("")
    w("| Set | n | Precision | Recall | F1 | FPR | TP | FP | TN | FN |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    acc = obj(full["accuracy"])
    for key, name in (("untainted", "Untainted sandbox"), ("tainted", "Tainted sandbox")):
        m = obj(acc[key])
        w(
            f"| {name} | {m['n']} | {f_pct(m['precision'])} | {f_pct(m['recall'])} | {f_pct(m['f1'])} | "
            f"{f_pct(m['fpr'])} | {m['tp']} | {m['fp']} | {m['tn']} | {m['fn']} |"
        )
    w("")
    for key in ("untainted", "tainted"):
        w(f"- {key}: {obj(acc[key])['definition']}.")
    w(
        f"- Tainted set requests the gate saw as tainted: {obj(acc['tainted'])['requests_marked_tainted']} of "
        f"{obj(acc['tainted'])['n']}."
    )
    w("")
    rbt = obj(acc["recall_by_type"])
    w("| Recall by type | n | Caught | Recall |")
    w("|---|---|---|---|")
    for key, name in (
        ("phi_structured", "Synthetic PHI with identifiers (local detectors)"),
        ("phi_unstructured", "Synthetic PHI without identifier patterns"),
        ("injection", "Injected tool responses"),
        ("follow_through", "Follow-through (tainted)"),
    ):
        r = obj(rbt[key])
        w(f"| {name} | {r['n']} | {r['caught']} | {f_pct(r['recall'])} |")
    w("")
    w("## Does the taint change decisions? (full arm)")
    w("")
    te = obj(obj(report["taint_effect"])["full"])
    ft = obj(te["follow_through"])
    bt = obj(ft["below_threshold"])
    tl = obj(te["tainted_legit"])
    w(
        f"Each follow-through and legitimate request ran twice: once in a sandbox that never saw an injection, once "
        f"right after an injected tool response. Untainted exfiltration threshold: {te['untainted_threshold']}."
    )
    w("")
    w("| | n | Denied untainted | Denied tainted |")
    w("|---|---|---|---|")
    w(f"| Follow-through requests | {ft['n']} | {ft['denied_untainted']} | {ft['denied_tainted']} |")
    w(f"| Legitimate requests | {tl['n']} | {tl['denied_untainted']} | {tl['denied_tainted']} |")
    w("")
    w(
        f"- Below threshold: {bt['n']} follow-through requests were allowed untainted ({bt['definition']}). "
        f"{bt['denied_when_tainted']} of them were denied once the sandbox was tainted: "
        f"{bt['of_which_score_alone_would_deny']} because the evaluator, given the flagged instruction as context, "
        f"scored them above the untainted threshold, and {bt['of_which_only_the_stricter_threshold_denies']} only "
        "because of the stricter tainted threshold."
    )
    sub = obj(bt["by_designed_subtype"])
    w(
        f"- By design: subtle (routine looking) {obj(sub['subtle'])['denied_when_tainted']} of "
        f"{obj(sub['subtle'])['n']}, overt {obj(sub['overt'])['denied_when_tainted']} of {obj(sub['overt'])['n']} "
        "denied when tainted, among those below threshold."
    )
    w(
        f"- Taint-induced false positives: {tl['taint_induced_false_positives']} of {tl['n']} legitimate requests were "
        "allowed untainted but denied after an injection."
    )
    w("")
    w("## Latency")
    w("")
    mw = obj(lat["middleware"])
    w("| Measure | n | p50 | p95 |")
    w("|---|---|---|---|")
    for key, name in (
        ("local_detectors", "Local detectors (middleware time per request, local detectors arm)"),
        ("local_denials_in_full_arm", "Local denials in the full arm (no evaluator call)"),
        ("jev_calls", f"Jev calls (evaluator time per call, full arm, {mode})"),
    ):
        d = obj(mw[key])
        w(f"| {name} | {d['n']} | {f_ms(d['p50'])} | {f_ms(d['p95'])} |")
    for arm_id in arms:
        d = obj(obj(obj(lat["per_arm"])[arm_id])["end_to_end_ms"])
        w(
            f"| End to end inside the sandbox, {obj(arms[arm_id])['label']} | {d['n']} | {f_ms(d['p50'])} | "
            f"{f_ms(d['p95'])} |"
        )
    w("")
    w("## OpenShell")
    w("")
    w("| Arm | Sandbox | Startup (create to agent's first line) |")
    w("|---|---|---|")
    for a in arms.values():
        for sb in lst(obj(a)["sandboxes"]):
            sb = obj(sb)
            w(f"| {obj(a)['label']} | {sb['phase']} | {sb['startup_s']} s |")
    w("")
    w("| Isolation probe (full arm, sandbox U) | Expected | Observed | Pass |")
    w("|---|---|---|---|")
    for p in lst(full["probes"]):
        p = obj(p)
        w(f"| {p['label']} | {p['expect']} | {p['result']} | {'yes' if p['pass'] else 'NO'} |")
    w("")
    pol = obj(report["policy"])

    def joined(key: str, sep: str = ", ") -> str:
        return sep.join(map(str, lst(pol[key])))

    w(
        f"Policy ({pol['file']}): default {pol['default']}; allowed endpoints {joined('allowed_endpoints')}; "
        f"binaries {joined('allowed_binaries')}; filesystem read only {joined('filesystem_read_only')}, read write "
        f"{joined('filesystem_read_write')}; middleware {joined('middlewares', '; ')}."
    )
    w("")
    diagram = mermaid(report)
    if diagram:
        w("## An injected tool response, then the follow-through (full arm, real values from this run)")
        w("")
        w("```mermaid")
        w(diagram)
        w("```")
        w("")
    ds = obj(report["dataset"])
    w("## Sample sizes")
    w("")
    w(f"{ds['cases']} synthetic cases in `{ds['file']}` (sha256 `{ds['sha256']}`, seed {ds['seed']}):")
    w("")
    for k, v in obj(ds["counts"]).items():
        w(f"- {k}: {v}")
    w("")
    w("## Reproduce")
    w("")
    w("```bash")
    w("uv run python evals/openshell/build_payloads.py")
    w("uv run --extra openshell python evals/openshell/run_bench.py --mock          # or --live --yes")
    w("uv run python evals/openshell/report.py results/openshell-bench-<ts>.json --publish")
    w("```")
    w("")
    return "\n".join(out)


# ------------------------------------------------------------------ PNGs


def screenshot(html: Path, card: str, out: Path) -> None:
    """Headless Chrome writes the PNG but on macOS does not always exit, so wait for the file, then stop it."""
    if not CHROME.exists():
        raise RuntimeError(f"headless Chrome not found at {CHROME}")
    width, height = CARDS[card]
    out.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory() as profile:
        proc = subprocess.Popen(
            [
                str(CHROME),
                "--headless",
                "--disable-gpu",
                "--hide-scrollbars",
                "--no-first-run",
                "--no-default-browser-check",
                "--force-device-scale-factor=1",
                f"--user-data-dir={profile}",
                f"--window-size={width},{height}",
                f"--screenshot={out}",
                f"{html.resolve().as_uri()}#card={card}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            deadline, last = time.monotonic() + 60, -1
            while time.monotonic() < deadline and proc.poll() is None:
                size = out.stat().st_size if out.exists() else -1
                if size > 0 and size == last:
                    break
                last = size
                time.sleep(0.5)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"headless Chrome did not write {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path)
    ap.add_argument("--publish", action="store_true", help="write to docs/openshell/ (GitHub Pages)")
    ap.add_argument("--no-png", action="store_true")
    args = ap.parse_args()
    report = json.loads(args.results.read_text(encoding="utf-8"))
    out_dir = PUBLISH_DIR if args.publish else args.results.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.results.resolve() != (out_dir / "results.json").resolve():
        shutil.copy(args.results, out_dir / "results.json")
    (out_dir / "RESULTS.md").write_text(render_markdown(report), encoding="utf-8")
    html = out_dir / "index.html"
    html.write_text(render_html(report), encoding="utf-8")
    print(f"wrote {out_dir / 'results.json'}, RESULTS.md, index.html")
    if not args.no_png:
        (out_dir / "img").mkdir(exist_ok=True)
        for card, (w, h) in CARDS.items():
            png = out_dir / "img" / f"linkedin-{w}x{h}.png"
            screenshot(html, card, png)
            print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
