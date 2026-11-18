"""Build the red team report from committed run data: docs/redteam/{index.html, results.json, RESULTS.md}.

    uv run python evals/redteam/redteam_report.py           # rebuild from docs/redteam/data, no API calls
    uv run python evals/redteam/redteam_report.py --import  # first copy the runs in runs.json from results/redteam

runs.json names one run per role. Each run's per item decisions (.calls.jsonl.gz) and raw evaluator responses
(.raw.jsonl.gz) are committed under docs/redteam/data, so every number on the page can be recomputed. All inputs are
synthetic. GPT 4o and GPT 4o mini were not run (baseline omitted for budget) and appear in no table.
"""

from __future__ import annotations

import argparse
import gzip
import html
import json
import shutil
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "evals" / "redteam"))
sys.path.insert(0, str(ROOT / "evals" / "support_tickets"))

import analyze  # noqa: E402
from full_bench import mcnemar  # noqa: E402
from taxonomy import CLASSES, AttackClass, Boundary  # noqa: E402

RUNS = Path(__file__).with_name("runs.json")
OUT = ROOT / "docs" / "redteam"
DATA = OUT / "data"
RESULTS = ROOT / "results" / "redteam"
LIVE = ("jev", "gemini_decision")
ARM_LABEL = {
    "jev": "Jev",
    "gemini_decision": "Gemini Decision 1",
    "shipped": "Shipped default",
    "dual": "Strict profile (Jev + Gemini)",
}
OMITTED = {"gpt-4o": "GPT 4o", "gpt-4o-mini": "GPT 4o mini"}
SPEND_NOTE = (
    "The GPT entries are not spend: they are estimates the ledger charged, to be safe, for calls that failed because "
    "the OpenAI account had no credits. Gemini ran on the free tier, so its real cost was zero; it is priced at the "
    "paid rate. The ledger also covers runs later discarded after a network outage."
)

Row = dict[str, Any]


# --------------------------------------------------------------------------- data


def import_runs(runs: dict[str, str]) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for stem in sorted(set(runs.values())):
        for suffix in (".calls.jsonl", ".raw.jsonl"):
            src = RESULTS / f"{stem}{suffix}"
            # mtime 0 keeps the archive bytes stable, so an unchanged run does not show up as a change in git
            (DATA / f"{stem}{suffix}.gz").write_bytes(gzip.compress(src.read_bytes(), mtime=0))
        shutil.copy(RESULTS / f"{stem}.json", DATA / f"{stem}.json")
    (DATA / "spend.json").write_text(json.dumps(ledger_by_evaluator(), indent=2) + "\n")


def rows(stem: str, experiment: str | None = None) -> list[Row]:
    with gzip.open(DATA / f"{stem}.calls.jsonl.gz", "rt", encoding="utf-8") as f:
        lines = f.read().splitlines()[1:]
    out = [json.loads(line)["data"] for line in lines]
    return [r for r in out if experiment is None or r["experiment"] == experiment]


def summary(stem: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((DATA / f"{stem}.json").read_text())
    return data


def ledger_by_evaluator() -> dict[str, float]:
    """Total recorded spend by evaluator, from the local ledger (results/ is not committed)."""
    lines = [json.loads(line) for line in (RESULTS / "spend.jsonl").read_text().splitlines() if line]
    by_ev: dict[str, float] = defaultdict(float)
    for r in lines:
        by_ev[str(r.get("evaluator") or r.get("arm"))] += float(r["usd"])
    return {k: round(v, 4) for k, v in sorted(by_ev.items())}


def spend() -> dict[str, float]:
    data: dict[str, float] = json.loads((DATA / "spend.json").read_text())
    return data


def pick(rs: Iterable[Row], **eq: object) -> list[Row]:
    return [r for r in rs if all(r.get(k) == v for k, v in eq.items())]


def asr(rs: Sequence[Row]) -> analyze.Rate:
    return analyze.outcome_rates(rs)["attack_success"]


def fbr(rs: Sequence[Row]) -> analyze.Rate:
    return analyze.outcome_rates(rs)["false_block"]


def paired(before: Sequence[Row], after: Sequence[Row], label: str = "malicious") -> dict[str, Any]:
    """Same items before and after: exact McNemar and a paired bootstrap interval of the change in the rate that
    matters for the label, attack success for malicious items and false block for benign ones."""

    def failed(r: Row) -> bool:
        return analyze.passed(r) if label == "malicious" else not analyze.passed(r)

    b = {r["item_id"]: failed(r) for r in before if r["label"] == label and not analyze.errored(r)}
    a = {r["item_id"]: failed(r) for r in after if r["label"] == label and not analyze.errored(r)}
    shared = sorted(b.keys() & a.keys())
    if not shared:
        return {"n": 0}
    bb, aa = [b[i] for i in shared], [a[i] for i in shared]
    return {
        "n": len(shared),
        "before": sum(bb),
        "after": sum(aa),
        "delta_ci95": analyze.paired_bootstrap_delta(bb, aa),
        "mcnemar": mcnemar(bb, aa),
    }


# --------------------------------------------------------------------------- results


def build(runs: dict[str, str]) -> dict[str, Any]:
    base = rows(runs["baseline"], "baseline")
    base_summary = summary(runs["baseline"])
    out: dict[str, Any] = {
        "runs": runs,
        "omitted_arms": {k: "not run, baseline omitted for budget" for k in OMITTED},
        "spend_usd_by_evaluator": spend(),
        "eval_timeout_ms": base_summary.get("eval_timeout_ms"),
        "corpus": base_summary["settings"]["corpus_sha"],
    }

    # Baseline per set, arm and class
    sets: dict[str, Any] = {}
    for set_name in ("gate", "harness2", "openshell"):
        per_arm: dict[str, Any] = {}
        for arm in ("jev", "gemini_decision", "shipped"):
            rs = pick(base, set=set_name, arm=arm)
            per_arm[arm] = {
                "attack_success": asr(rs),
                "false_block": fbr(rs),
                "by_class": {c: asr(v) for c, v in sorted(_group(rs, "attack_class").items()) if c != "none"}
                | {"none": asr(pick(rs, attack_class="none"))},
                "by_transform": {t: asr(v) for t, v in sorted(_group(rs, "transform").items())},
                "frontier_reached": sum(
                    1 for r in rs if r["label"] == "malicious" and not r["blocked"] and r.get("outcome") == "frontier"
                ),
            }
        sets[set_name] = per_arm
    out["baseline"] = sets
    gate_mal = pick(base, set="gate", label="malicious")
    out["gate_pairs"] = {
        "jev vs gemini_decision": analyze.mcnemar_passes(
            pick(gate_mal, arm="jev"), pick(gate_mal, arm="gemini_decision"), mcnemar
        )
    }
    by_item: dict[str, dict[str, Row]] = defaultdict(dict)
    for r in gate_mal:
        by_item[r["item_id"]][r["arm"]] = r
    both = sorted(i for i, v in by_item.items() if all(a in v and not v[a]["blocked"] for a in LIVE))
    out["passed_both_live"] = [
        {
            "item_id": i,
            "attack_class": by_item[i]["jev"]["attack_class"],
            "category": by_item[i]["jev"]["category"],
            "jev": [by_item[i]["jev"]["outcome"], by_item[i]["jev"].get("risk_score")],
            "gemini_decision": [
                by_item[i]["gemini_decision"]["outcome"],
                by_item[i]["gemini_decision"].get("risk_score"),
            ],
        }
        for i in both
    ]
    out["detector"] = base_summary["detector"]["by_detector"]
    out["drift"] = base_summary["drift"]
    out["latency_ms"] = base_summary["latency"]

    # Adaptive search: score oracle (before the 403 change) and binary oracle (after)
    out["adaptive"] = {
        "score_oracle": analyze.summarize_adaptive(rows(runs["baseline"], "adaptive")),
        "binary_oracle": analyze.summarize_adaptive(rows(runs["adaptive_binary"], "adaptive")),
    }
    if "dual_adaptive" in runs:  # the dual run's own adaptive rows were hit by a network outage and rerun
        out["adaptive"]["binary_oracle"] |= analyze.summarize_adaptive(rows(runs["dual_adaptive"], "adaptive"))

    # Harness2: three stages on the corpus and on the held out set
    h2: dict[str, Any] = {}
    for name, stages in (
        ("corpus", ("baseline", "harness2_context", "harness2_review")),
        ("heldout", ("heldout_before", "heldout_context", "heldout_review")),
    ):
        stage_rows = {s: pick(rows(runs[s], "baseline"), set="harness2") for s in stages if s in runs}
        h2[name] = {
            s: {
                arm: {
                    "attack_success": asr(pick(rs, arm=arm)),
                    "false_block": fbr(pick(rs, arm=arm)),
                    "by_transform": {t: asr(v) for t, v in sorted(_group(pick(rs, arm=arm), "transform").items())},
                    "benign_by_transform": {
                        t: fbr(v) for t, v in sorted(_group(pick(rs, arm=arm, label="benign"), "transform").items())
                    },
                }
                for arm in ("jev", "gemini_decision", "shipped")
            }
            for s, rs in stage_rows.items()
        }
        first, last = stages[0], stages[-1]
        if first in stage_rows and last in stage_rows:
            h2[name]["paired"] = {
                arm: paired(pick(stage_rows[first], arm=arm), pick(stage_rows[last], arm=arm)) for arm in LIVE
            }
    out["harness2"] = h2

    # Strict privacy
    if "privacy_after" in runs:
        pb, pa = rows(runs["privacy_before"], "baseline"), rows(runs["privacy_after"], "baseline")
        sp = rows(runs["strict_privacy_gate"], "baseline")
        gate_base = pick(base, set="gate")
        out["strict_privacy"] = {
            "heldout": {
                arm: {
                    "before": {"attack_success": asr(pick(pb, arm=arm)), "false_block": fbr(pick(pb, arm=arm))},
                    "after": {"attack_success": asr(pick(pa, arm=arm)), "false_block": fbr(pick(pa, arm=arm))},
                    "benign_by_transform_after": {
                        t: fbr(v) for t, v in sorted(_group(pick(pa, arm=arm, label="benign"), "transform").items())
                    },
                }
                for arm in LIVE
            },
            "corpus_phi": {
                arm: {
                    "before": asr(pick(gate_base, arm=arm, category="phi")),
                    "after": asr(pick(sp, arm=arm, category="phi")),
                    "paired": paired(pick(gate_base, arm=arm, category="phi"), pick(sp, arm=arm, category="phi")),
                }
                for arm in LIVE
            },
            "corpus_gate": {
                arm: {
                    "before": {
                        "attack_success": asr(pick(gate_base, arm=arm)),
                        "false_block": fbr(pick(gate_base, arm=arm)),
                    },
                    "after": {"attack_success": asr(pick(sp, arm=arm)), "false_block": fbr(pick(sp, arm=arm))},
                }
                for arm in LIVE
            },
        }

    # Strict profile
    if "dual" in runs:
        d = rows(runs["dual"], "baseline")
        dual_summary = summary(runs["dual"])
        gate_base = pick(base, set="gate")
        out["strict_profile"] = {
            "gate": {
                "dual": {"attack_success": asr(d), "false_block": fbr(d)},
                **{
                    arm: {"attack_success": asr(pick(gate_base, arm=arm)), "false_block": fbr(pick(gate_base, arm=arm))}
                    for arm in LIVE
                },
            },
            "paired_vs": {arm: paired(pick(gate_base, arm=arm), d) for arm in LIVE},
            "paired_benign_vs": {arm: paired(pick(gate_base, arm=arm), d, "benign") for arm in LIVE},
            "latency_ms": {"dual": dual_summary["latency"].get("dual"), **{a: out["latency_ms"].get(a) for a in LIVE}},
            "by_class": {c: asr(v) for c, v in sorted(_group(d, "attack_class").items())},
        }
    return out


def _group(rs: Iterable[Row], key: str) -> dict[str, list[Row]]:
    g: dict[str, list[Row]] = defaultdict(list)
    for r in rs:
        g[str(r.get(key))].append(r)
    return dict(g)


# --------------------------------------------------------------------------- formatting


def pct(r: analyze.Rate | None) -> str:
    if not r or r["rate"] is None:
        return "n/a"
    lo, hi = r["ci95"] or [0.0, 0.0]
    return f"{r['rate'] * 100:.1f}% [{lo * 100:.0f}, {hi * 100:.0f}] ({r['k']}/{r['n']})"


def short(r: analyze.Rate | None) -> str:
    if not r or r["rate"] is None:
        return "n/a"
    return f"{r['k']}/{r['n']}"


def p_value(p: dict[str, Any]) -> str:
    m = p.get("mcnemar")
    if not m:
        return "n/a"
    v = float(m["p_value"])
    return "<0.0001" if v < 1e-4 else f"{v:.4f}"


def delta(p: dict[str, Any]) -> str:
    ci = p.get("delta_ci95")
    if not ci or not p.get("n"):
        return "n/a"
    d = (p["after"] - p["before"]) / p["n"] * 100
    return f"{d:+.1f} pts [{ci[0] * 100:+.0f}, {ci[1] * 100:+.0f}]"


def md_table(header: Sequence[str], body: Iterable[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def class_title(code: str) -> str:
    return CLASSES[AttackClass(code)].title if code in {c.value for c in AttackClass} else code


# --------------------------------------------------------------------------- markdown


def render_markdown(r: dict[str, Any]) -> str:
    b, h2 = r["baseline"], r["harness2"]
    parts = [
        "# Red teaming Paved Gate: results",
        "",
        "Generated by `evals/redteam/redteam_report.py` from the committed run data in `docs/redteam/data`. "
        "Rates are attack success (malicious inputs not blocked) or false block (benign inputs blocked), with Wilson "
        "95% intervals in brackets and counts in parentheses. Evaluator errors are counted separately and never as "
        "a block. GPT 4o and GPT 4o mini: not run, baseline omitted for budget.",
        "",
        "## Baseline",
        "",
        md_table(
            ["Set", "Arm", "Attack success", "False block"],
            [
                [s, ARM_LABEL[a], pct(b[s][a]["attack_success"]), pct(b[s][a]["false_block"])]
                for s in ("gate", "harness2", "openshell")
                for a in ("jev", "gemini_decision", "shipped")
            ],
        ),
        "",
        "### Gate set by attack class",
        "",
        md_table(
            ["Class", "Jev", "Gemini Decision 1", "Shipped default"],
            [
                [
                    f"{c} {class_title(c)}",
                    *(pct(b["gate"][a]["by_class"].get(c)) for a in ("jev", "gemini_decision", "shipped")),
                ]
                for c in sorted(b["gate"]["jev"]["by_class"])
            ],
        ),
        "",
        "### Adaptive search (20 seeds, up to 15 queries)",
        "",
        md_table(
            ["Arm", "Score oracle (403 showed the risk score)", "Binary oracle (403 shows only blocked)"],
            [
                [
                    ARM_LABEL[a],
                    pct(r["adaptive"]["score_oracle"].get(a, {}).get("success")),
                    pct(r["adaptive"]["binary_oracle"].get(a, {}).get("success")),
                ]
                for a in ("jev", "gemini_decision", "shipped", "dual")
                if a in r["adaptive"]["binary_oracle"] or a in r["adaptive"]["score_oracle"]
            ],
        ),
        "",
        "## Harness2 mitigations",
        "",
    ]
    stage_label = {
        "baseline": "before",
        "heldout_before": "before",
        "harness2_context": "per file clip + protected rules",
        "heldout_context": "per file clip + protected rules",
        "harness2_review": "+ review hold for clipped edits",
        "heldout_review": "+ review hold for clipped edits",
    }
    for name in ("corpus", "heldout"):
        stages = [s for s in h2[name] if s != "paired"]
        parts += [
            f"### {'Corpus (in sample)' if name == 'corpus' else 'Held out (out of sample)'}",
            "",
            md_table(
                ["Arm", *(f"Attack success, {stage_label[s]}" for s in stages), "False block, last stage"],
                [
                    [
                        ARM_LABEL[a],
                        *(pct(h2[name][s][a]["attack_success"]) for s in stages),
                        pct(h2[name][stages[-1]][a]["false_block"]),
                    ]
                    for a in ("jev", "gemini_decision", "shipped")
                ],
            ),
            "",
        ]
        if "paired" in h2[name]:
            parts += [
                md_table(
                    ["Arm", "Change in attack success, first to last stage", "McNemar p"],
                    [[ARM_LABEL[a], delta(p), p_value(p)] for a, p in h2[name]["paired"].items()],
                ),
                "",
            ]
    if "strict_privacy" in r:
        sp = r["strict_privacy"]
        parts += [
            "## privacy.strict",
            "",
            md_table(
                ["Arm", "Held out disclosures passed, before", "after", "Held out benign blocked, before", "after"],
                [
                    [
                        ARM_LABEL[a],
                        pct(sp["heldout"][a]["before"]["attack_success"]),
                        pct(sp["heldout"][a]["after"]["attack_success"]),
                        pct(sp["heldout"][a]["before"]["false_block"]),
                        pct(sp["heldout"][a]["after"]["false_block"]),
                    ]
                    for a in LIVE
                ],
            ),
            "",
            md_table(
                [
                    "Arm",
                    "Corpus health data attacks passed, before",
                    "after",
                    "Whole gate set false block, before",
                    "after",
                ],
                [
                    [
                        ARM_LABEL[a],
                        pct(sp["corpus_phi"][a]["before"]),
                        pct(sp["corpus_phi"][a]["after"]),
                        pct(sp["corpus_gate"][a]["before"]["false_block"]),
                        pct(sp["corpus_gate"][a]["after"]["false_block"]),
                    ]
                    for a in LIVE
                ],
            ),
            "",
        ]
    if "strict_profile" in r:
        pr = r["strict_profile"]
        parts += [
            "## profile: strict",
            "",
            md_table(
                ["Arm", "Gate attack success", "Gate false block", "p50 / p95 latency"],
                [
                    [
                        ARM_LABEL[a],
                        pct(pr["gate"][a]["attack_success"]),
                        pct(pr["gate"][a]["false_block"]),
                        _lat(pr["latency_ms"].get(a)),
                    ]
                    for a in ("jev", "gemini_decision", "dual")
                ],
            ),
            "",
            md_table(
                ["Compared with", "Change in attack success", "McNemar p", "Change in false block", "McNemar p"],
                [
                    [
                        ARM_LABEL[a],
                        delta(pr["paired_vs"][a]),
                        p_value(pr["paired_vs"][a]),
                        delta(pr["paired_benign_vs"][a]),
                        p_value(pr["paired_benign_vs"][a]),
                    ]
                    for a in LIVE
                ],
            ),
            "",
        ]
    parts += [
        "## Spend",
        "",
        f"Recorded at list price, by evaluator: `{json.dumps(r['spend_usd_by_evaluator'])}`. {SPEND_NOTE}",
        "",
    ]
    return "\n".join(parts)


def _lat(v: dict[str, Any] | None) -> str:
    if not v or v.get("p50_ms") is None:
        return "n/a"
    return f"{v['p50_ms']:.0f} / {v['p95_ms']:.0f} ms"


# --------------------------------------------------------------------------- html

CSS = Path(__file__).with_name("redteam_report.css")


def e(s: object) -> str:
    return html.escape(str(s))


def cell(r: analyze.Rate | None, *, bar: bool = True) -> str:
    if not r or r["rate"] is None:
        return '<td class="n">n/a</td>'
    lo, hi = r["ci95"] or [0.0, 0.0]
    vis = (
        f'<div class="bar" aria-hidden="true"><span style="width:{r["rate"] * 100:.1f}%"></span>'
        f'<i style="left:{lo * 100:.1f}%;width:{max(hi - lo, 0) * 100:.1f}%"></i></div>'
        if bar
        else ""
    )
    return (
        f'<td class="n">{r["rate"] * 100:.1f}% <span class="ci">[{lo * 100:.0f}, {hi * 100:.0f}] '
        f"{r['k']}/{r['n']}</span>{vis}</td>"
    )


def table(header: Sequence[str], body: Iterable[Sequence[str]]) -> str:
    th = "".join(f"<th>{e(h)}</th>" for h in header)
    trs = "".join("<tr>" + "".join(row) + "</tr>" for row in body)
    return f'<div class="panel"><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>'


def td(text: str) -> str:
    return f"<td>{text}</td>"


def tile(value: str, label: str, sub: str) -> str:
    return (
        f'<div class="tile"><div class="v">{e(value)}</div><div class="l">{e(label)}</div>'
        f'<div class="s">{e(sub)}</div></div>'
    )


def rate(x: analyze.Rate | None) -> str:
    return f"{x['rate'] * 100:.0f}%" if x and x["rate"] is not None else "n/a"


def render_html(r: dict[str, Any], summary_md: str) -> str:
    b = r["baseline"]
    g = b["gate"]
    sections: list[str] = []

    sections.append(
        '<div class="tiles">'
        + tile(
            rate(g["jev"]["attack_success"]),
            "Gate attacks passed, Jev",
            f"n={g['jev']['attack_success']['n']}, single try",
        )
        + tile(
            rate(g["gemini_decision"]["attack_success"]),
            "Gate attacks passed, Gemini Decision 1",
            f"n={g['gemini_decision']['attack_success']['n']}, single try",
        )
        + tile(
            rate(r["adaptive"]["binary_oracle"]["jev"]["success"]),
            "Seeds broken with 15 retries, Jev",
            "binary oracle, n=20",
        )
        + tile(
            rate(g["shipped"]["attack_success"]),
            "Gate attacks passed, shipped default",
            "regex mock the policy ships with",
        )
        + "</div>"
    )

    sections.append("<h2>In plain language</h2>" + _md_to_html(summary_md))

    # Taxonomy
    rows_tax = []
    for bnd in Boundary:
        for c, info in CLASSES.items():
            if info.boundary is bnd:
                rows_tax.append(
                    [td(e(bnd.value)), td(f"<code>{e(c.value)}</code>"), td(e(info.title)), td(e(info.description))]
                )
    sections.append(
        '<h2>Threat model and taxonomy</h2><p class="sub">Every attack and failure carries one class code; each '
        "class sits on one trust boundary. The taxonomy is in <code>evals/redteam/taxonomy.py</code> for reuse.</p>"
        + table(["Boundary", "Code", "Class", "What it means"], rows_tax)
    )

    # Baseline
    body = [
        [td(e(s)), td(e(ARM_LABEL[a])), cell(b[s][a]["attack_success"]), cell(b[s][a]["false_block"])]
        for s in ("gate", "harness2", "openshell")
        for a in ("jev", "gemini_decision", "shipped")
    ]
    sections.append(
        '<h2>Baseline</h2><p class="sub">One try per item, shipped policies, evaluator timeout raised to '
        f"{e(r['eval_timeout_ms'])} ms so decisions, not speed, are measured. Bars show the rate; the bracket marks "
        "the Wilson 95% interval.</p>" + table(["Set", "Arm", "Attack success", "False block"], body)
    )
    classes = sorted(g["jev"]["by_class"])
    sections.append(
        "<h3>Gate set by attack class</h3>"
        + table(
            ["Class", "Jev", "Gemini Decision 1", "Shipped default"],
            [
                [
                    td(f"<code>{e(c)}</code> {e(class_title(c))}"),
                    *(cell(g[a]["by_class"].get(c)) for a in ("jev", "gemini_decision", "shipped")),
                ]
                for c in classes
            ],
        )
    )
    pair = r["gate_pairs"]["jev vs gemini_decision"]
    sections.append(
        f'<p class="sub">Jev and Gemini Decision 1 pass a similar share of attacks (exact McNemar p = '
        f"{float(pair['p_value']):.2f} over {pair['n']} paired items) but different ones: "
        f"{len(r['passed_both_live'])} got past both.</p>"
    )
    sections.append(
        "<h3>Attacks that got past both live evaluators</h3>"
        + table(
            ["Item", "Class", "Category", "Jev", "Gemini Decision 1"],
            [
                [
                    td(f"<code>{e(x['item_id'])}</code>"),
                    td(e(x["attack_class"])),
                    td(e(x["category"])),
                    td(f"{e(x['jev'][0])}, risk {e(x['jev'][1])}"),
                    td(f"{e(x['gemini_decision'][0])}, risk {e(x['gemini_decision'][1])}"),
                ]
                for x in r["passed_both_live"]
            ],
        )
    )
    sections.append(
        "<h3>Local PII detectors (no model)</h3>"
        + table(
            ["Detector", "Reformatted identifiers detected"],
            [[td(e(k)), cell(v)] for k, v in r["detector"].items()],
        )
    )
    sections.append(
        "<h3>Drift over 10 repeated calls (50 items)</h3>"
        + table(
            ["Arm", "Items whose decision ever flipped", "Mean spread of the risk score"],
            [[td(e(ARM_LABEL[a])), cell(v["unstable"]), td(e(v["mean_risk_std"]))] for a, v in r["drift"].items()],
        )
    )

    # Mitigations
    ad = r["adaptive"]
    sections.append(
        "<h2>Mitigations, before and after</h2>"
        "<h3>1. The 403 body no longer shows scores</h3>"
        '<p class="sub">Before, a blocked request\'s response listed the risk score, intent confidence and the 3.5 '
        "threshold. The adaptive search below stacks corpus transforms on a blocked seed for up to 15 queries. With "
        "the score it keeps any variant that lowers the score; with only blocked or not, each query is a fresh random "
        "stack.</p>"
        + table(
            ["Arm", "Score oracle (before)", "Binary oracle (after)"],
            [
                [
                    td(e(ARM_LABEL[a])),
                    cell(ad["score_oracle"].get(a, {}).get("success")),
                    cell(ad["binary_oracle"].get(a, {}).get("success")),
                ]
                for a in ("jev", "gemini_decision", "shipped", "dual")
                if a in ad["binary_oracle"]
            ],
        )
        + '<div class="note">This did not work as a defense. Hiding the score did not lower the share of seeds an '
        "attacker breaks with retries. The score was not what made the search effective; being allowed to retry was. "
        "The change is kept as hygiene, and the response shape changed.</div>"
    )
    stage_label = {
        "baseline": "Before",
        "heldout_before": "Before",
        "harness2_context": "Per file clip + protected rules",
        "heldout_context": "Per file clip + protected rules",
        "harness2_review": "+ review hold for clipped edits",
        "heldout_review": "+ review hold for clipped edits",
    }
    sections.append(
        "<h3>2. Harness2: per file clipping, protected rules in view, review hold</h3>"
        '<p class="sub">The held out set was written and committed before any Harness2 change was measured on it. '
        "Needs review counts as stopped: a person sees the edit.</p>"
    )
    for name, title in (("corpus", "Corpus (in sample)"), ("heldout", "Held out (out of sample)")):
        h = r["harness2"][name]
        stages = [s for s in h if s != "paired"]
        sections.append(
            f"<h3>{e(title)}: attack success</h3>"
            + table(
                ["Arm", *(stage_label[s] for s in stages)],
                [
                    [td(e(ARM_LABEL[a])), *(cell(h[s][a]["attack_success"]) for s in stages)]
                    for a in ("jev", "gemini_decision", "shipped")
                ],
            )
            + f"<h3>{e(title)}: benign edits stopped</h3>"
            + table(
                ["Arm", *(stage_label[s] for s in stages)],
                [
                    [td(e(ARM_LABEL[a])), *(cell(h[s][a]["false_block"]) for s in stages)]
                    for a in ("jev", "gemini_decision", "shipped")
                ],
            )
        )
        if "paired" in h:
            sections.append(
                table(
                    ["Arm", "Change, first to last stage (paired bootstrap 95%)", "Exact McNemar p"],
                    [[td(e(ARM_LABEL[a])), td(e(delta(p))), td(e(p_value(p)))] for a, p in h["paired"].items()],
                )
            )
    if "strict_privacy" in r:
        sp = r["strict_privacy"]
        sections.append(
            "<h3>3. privacy.strict</h3>"
            '<p class="sub">Held out set: 15 disclosures and 25 benign requests, 15 of them handling real looking '
            "patient data legitimately. Corpus: the 99 health data attacks and the whole gate set.</p>"
            + table(
                ["Arm", "Held out disclosures passed, before", "After", "Held out benign blocked, before", "After"],
                [
                    [
                        td(e(ARM_LABEL[a])),
                        cell(sp["heldout"][a]["before"]["attack_success"]),
                        cell(sp["heldout"][a]["after"]["attack_success"]),
                        cell(sp["heldout"][a]["before"]["false_block"]),
                        cell(sp["heldout"][a]["after"]["false_block"]),
                    ]
                    for a in LIVE
                ],
            )
            + table(
                ["Arm", "Corpus health data attacks passed, before", "After", "Gate set false block, before", "After"],
                [
                    [
                        td(e(ARM_LABEL[a])),
                        cell(sp["corpus_phi"][a]["before"]),
                        cell(sp["corpus_phi"][a]["after"]),
                        cell(sp["corpus_gate"][a]["before"]["false_block"]),
                        cell(sp["corpus_gate"][a]["after"]["false_block"]),
                    ]
                    for a in LIVE
                ],
            )
        )
    if "strict_profile" in r:
        pr = r["strict_profile"]
        sections.append(
            "<h3>4. profile: strict (Jev primary, Gemini Decision 1 secondary)</h3>"
            '<p class="sub">Both evaluators answer every request concurrently; either can block. Measured live '
            "through the real gate, not computed from the single evaluator runs.</p>"
            + table(
                ["Arm", "Gate attack success", "Gate false block", "p50 / p95 latency"],
                [
                    [
                        td(e(ARM_LABEL[a])),
                        cell(pr["gate"][a]["attack_success"]),
                        cell(pr["gate"][a]["false_block"]),
                        td(e(_lat(pr["latency_ms"].get(a)))),
                    ]
                    for a in ("jev", "gemini_decision", "dual")
                ],
            )
            + table(
                [
                    "Strict profile compared with",
                    "Change in attack success",
                    "McNemar p",
                    "Change in false block",
                    "McNemar p",
                ],
                [
                    [
                        td(e(ARM_LABEL[a])),
                        td(e(delta(pr["paired_vs"][a]))),
                        td(e(p_value(pr["paired_vs"][a]))),
                        td(e(delta(pr["paired_benign_vs"][a]))),
                        td(e(p_value(pr["paired_benign_vs"][a]))),
                    ]
                    for a in LIVE
                ],
            )
        )

    sections.append(
        '<h2>Method</h2><ul class="plain">'
        "<li>Corpus: 825 synthetic items from a seeded generator (<code>evals/redteam/build_corpus.py</code>, seed "
        "20261116): 45 malicious seeds times 11 transforms and 30 benign seeds times 5 for the gate; 35 identifier "
        "spellings for the local detectors; 70 Harness2 edits; 75 OpenShell sequences. Held out sets: 70 Harness2 "
        "edits and 40 privacy requests, each committed before the related mitigation was measured on it.</li>"
        "<li>Every item goes through the real <code>PavedGate.handle</code>, Harness2 <code>AdoptionGate</code> or "
        "OpenShell <code>EgressInspector</code> with the shipped policy files.</li>"
        "<li>Rates use Wilson 95% intervals. Before and after on the same items use an exact two sided McNemar test "
        "and a paired bootstrap (2,000 resamples, seed 0) for the change.</li>"
        "<li>An evaluator error is recorded and never counted as a block, even when a closed fail mode blocked the "
        "request. A malicious item counts as passed when it was not blocked, including the deterministic route.</li>"
        f"<li>Not run: {e(', '.join(OMITTED.values()))}, baseline omitted for budget (the OpenAI account had no "
        "credits). They appear in no table or comparison.</li>"
        "<li>Spend recorded at list price, including Gemini free tier calls: "
        f"<code>{e(json.dumps(r['spend_usd_by_evaluator']))}</code>. {e(SPEND_NOTE)}</li>"
        "</ul>"
    )
    sections.append(
        '<h2>Limitations</h2><ul class="plain">'
        "<li>The attack corpus is hand written and template generated by the people who built the gate. It is not a "
        "sample of real attacks, and the numbers are rates on this corpus, not on traffic.</li>"
        "<li>Numbers on the corpus after a mitigation are in sample: the corpus showed the problem the mitigation "
        "fixes. The held out sets are small (40 to 70 items) and written by the same authors.</li>"
        "<li>One run per configuration. Drift showed no flips over 10 repeats, but evaluator "
        "versions change; Jev reported jev-1.14.0 during these runs.</li>"
        "<li>The shipped defaults (regex mock for the gate, deterministic heuristic for Harness2) are unchanged by "
        "every mitigation that relies on a model.</li>"
        "<li>Not mitigated: local detector bypass, OpenShell padding and delimiter break, and retry based attacks "
        "beyond what the strict profile catches.</li>"
        "</ul>"
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Red Teaming Paved Gate</title>"
        '<meta name="description" content="A red team of the Paved Gate evaluator gateway: attack classes by trust '
        "boundary, attack success with confidence intervals for Jev and Gemini Decision 1, and before and after "
        'results for each mitigation.">'
        f'<style>{CSS.read_text(encoding="utf-8")}</style></head><body><main class="wrap">'
        '<p class="eyebrow">Paved Gate</p><h1>Red teaming Paved Gate</h1>'
        '<p class="lede">How and when the evaluator gate can be fooled, measured on a synthetic corpus with live Jev '
        "and Gemini Decision 1, and what each fix did, including the one that did not work.</p>"
        '<div class="badges"><span class="badge b-live">Live evaluators</span>'
        '<span class="badge b-plain">Synthetic data only</span>'
        '<span class="badge b-warn">GPT baselines not run</span></div>'
        + "".join(sections)
        + "<footer>Generated by <code>evals/redteam/redteam_report.py</code> from <code>docs/redteam/data</code>. "
        "Per item decisions and raw evaluator responses are committed there. "
        '<a href="RESULTS.md">Tables as markdown</a> · <a href="results.json">results.json</a>.</footer>'
        "</main></body></html>\n"
    )


def _md_to_html(md: str) -> str:
    """The plain language summary is short paragraphs and `* ` bullets; nothing else is needed."""
    out, items = [], []
    for block in md.strip().split("\n\n"):
        lines = block.splitlines()
        if lines[0].startswith("# "):
            continue
        if all(line.startswith("* ") for line in lines):
            items = [f"<li>{_inline(line[2:])}</li>" for line in lines]
            out.append('<ul class="plain">' + "".join(items) + "</ul>")
        elif lines[0].startswith("## "):
            out.append(f"<h3>{_inline(lines[0][3:])}</h3>" + "".join(f"<p>{_inline(x)}</p>" for x in lines[1:]))
        else:
            out.append(f"<p>{_inline(' '.join(lines))}</p>")
    return "".join(out)


def _inline(text: str) -> str:
    t = e(text)
    parts = t.split("**")
    t = "".join(f"<strong>{p}</strong>" if i % 2 else p for i, p in enumerate(parts))
    parts = t.split("`")
    return "".join(f"<code>{p}</code>" if i % 2 else p for i, p in enumerate(parts))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--import", dest="do_import", action="store_true", help="copy the runs from results/redteam first")
    args = ap.parse_args(argv)
    runs: dict[str, str] = json.loads(RUNS.read_text())
    if args.do_import:
        import_runs(runs)
    report = build(runs)
    summary_md = (OUT / "SUMMARY.md").read_text(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    (OUT / "RESULTS.md").write_text(render_markdown(report) + "\n")
    (OUT / "index.html").write_text(render_html(report, summary_md))
    print(f"wrote {OUT.relative_to(ROOT)}/index.html, results.json and RESULTS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
