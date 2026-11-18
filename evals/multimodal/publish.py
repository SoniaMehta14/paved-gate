"""Write the sanitized multimodal results for the Pages dashboard, from the raw run results.

    uv run python evals/multimodal/publish.py      # writes docs/benchmarks/multimodal-live-<date>.json and .md

Reads results/multimodal/ (cases, the main run and the PDF as PNG run), which git ignores, and keeps only an
allowlist of fields: case IDs, edit names, file hashes, labels, what the records say, each answer with its
probability, latency, tokens and cost. It never copies the audit records, the images or PDFs, file paths or
anything from the request. The export is refused if any of that shows up.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pdf_png  # noqa: E402
import run  # noqa: E402
from jev_image import IMAGES_HEADER  # noqa: E402
from questions import LEGIBLE_FROM_LEVEL  # noqa: E402

DOCS = run.ROOT / "docs" / "benchmarks"
QKEYS = ["synthetic_label", "overall_decision", "rejected_exactly_2", "legibility_ok"]
FORBIDDEN = [
    r"/Users/",
    r"/home/",
    r"/private/",
    r"/tmp/",
    r"results/multimodal",
    r"AIza[0-9A-Za-z_-]{20,}",
    r"sk-[A-Za-z0-9]{16,}",
    r"Bearer ",
    r"data_base64",
    r'\\?"namespace\\?"',
    r'\\?"policy_hash\\?"',
    r"[A-Za-z0-9+/]{200,}",
]
J = dict[str, object]


def answer_cells(answers: J, labels: J) -> J:
    pred, prob = run.predicted(answers), pdf_png.probabilities(answers)
    cells: J = {}
    for q in QKEYS:
        cells[q] = {
            "answer": pred[q],
            "p": round(prob[q], 4),
            "close": abs(prob[q] - 0.5) <= pdf_png.CLOSE,
            "correct": None if q not in labels else pred[q] == labels[q],
        }
    cells["legibility_level"] = pred["legibility_level"]
    return cells


def export() -> J:
    manifest_bytes = (run.CASES / "manifest.json").read_bytes()
    manifest = run.load_manifest()
    calls = run.load_checkpoint(run.OUT / "calls.jsonl", run.settings_hash(manifest_bytes))
    main = run.build_report(manifest, calls)
    rendered = json.loads((pdf_png.OUT / "cases" / "manifest.json").read_text(encoding="utf-8"))
    pp_settings = pdf_png.digest(rendered)
    pp = pdf_png.build_report(manifest, rendered, pp_settings)
    pp_calls = run.load_checkpoint(pdf_png.OUT / "calls.jsonl", pp_settings)

    rows = main["rows"]
    assert isinstance(rows, list)
    main_rows = {str(r["case"]): r for r in rows}
    cases: list[J] = []
    for c in pdf_png.cases_of(manifest):
        labels = c["labels"]
        assert isinstance(labels, dict)
        arms: J = {}
        for arm in run.ARMS:
            rec = calls.get(f"{c['id']}|{arm}")
            if rec is None:
                continue
            answers = rec["answers"]
            assert isinstance(answers, dict)
            arms[arm] = answer_cells(answers, labels) | {
                "latency_ms": rec["latency_ms"],
                "input_tokens": rec["input_tokens"],
            }
        cases.append(
            {
                "id": c["id"],
                "source": c["source"],
                "edits": c["edits"],
                "mime_type": c["mime_type"],
                "sha256": c["sha256"],
                "group": run.group(c),
                "counted": c.get("counted", True),
                "labels": labels,
                "record_says": main_rows[str(c["id"])]["record_says"],
                "arms": arms,
            }
        )

    # Gemini's text tokens for each PDF case, sent as the PDF and as the rendered PNG: the difference is the text
    # Gemini extracted from the PDF itself.
    pdf_text: list[J] = []
    for r in rendered:
        a, b = calls.get(f"{r['id']}|gemini_decision"), pp_calls.get(f"{r['id']}|gemini_png")
        if a and b:
            ta, tb = text_tokens(a), text_tokens(b)
            pdf_text.append({"case": r["id"], "text_tokens_pdf": ta, "text_tokens_png": tb, "difference": ta - tb})

    main_tokens = main["jev_image_tokens_measured"]
    assert isinstance(main_tokens, dict)
    pp_tokens = pp["jev_image_tokens"]
    assert isinstance(pp_tokens, dict)
    sizes = json.loads((run.CASES / "manifest.json").read_text(encoding="utf-8"))["size"]
    return {
        "title": "Multimodal test: synthetic report images and PDFs",
        "run_finished": main["run_finished"],
        "pdf_png_run_finished": pp["run_finished"],
        "models": main["models"],
        "settings": {
            "attempts_per_call": 1,
            "gemini_input_tokens_per_minute": run.GEMINI_TPM,
            "gemini_tier": "free",
            "timeout_s": run.TIMEOUT_S,
            "image_size_px": sizes,
            "pdf_png_size_px": rendered[0]["size"],
            "close_call_margin": pdf_png.CLOSE,
            "legible_from_level": LEGIBLE_FROM_LEVEL,
            "jev_image_header": IMAGES_HEADER,
        },
        "arms": run.ARM_LABEL,
        "calls": main["calls"],
        "counts": with_edited_png(main["counts"], cases),
        "latency_ms": main["latency_ms"],
        "gemini_latency_ms_by_artifact": main["gemini_latency_ms_by_artifact"],
        "gemini_input_tokens_by_modality": main["gemini_input_tokens_by_modality"],
        "jev_image_tokens_measured": {k: main_tokens[k] for k in ("cases", "median", "min", "max")},
        "input_tokens_by_arm": main["input_tokens_by_arm"],
        "cost_usd_at_list_price": main["cost_usd_at_list_price"],
        "price_sources": main["price_sources"],
        "cases": cases,
        "pdf_as_png": {
            "columns": pdf_png.COLUMNS,
            "calls": pp["calls"],
            "counts": pp["counts"],
            "close_calls": pp["close_calls"],
            "rows": pp["rows"],
            "jev_image_tokens": {k: pp_tokens[k] for k in ("median", "min", "max")},
            "gemini_png_tokens_by_modality": pp["gemini_png_tokens_by_modality"],
            "latency_ms": pp["latency_ms"],
            "input_tokens": pp["input_tokens"],
            "cost_usd_at_list_price": pp["cost_usd_at_list_price"],
            "limit_note": pp["limit_note"],
        },
        "gemini_pdf_text_tokens": pdf_text,
    }


def edited_png_counts(cases: list[J]) -> dict[str, dict[str, list[int]]]:
    """Counts for the edited screenshots only: edited cases sent as PNG, leaving out the edited PDF cases."""
    counts: dict[str, dict[str, list[int]]] = {}
    for c in cases:
        if c["group"] != "edited" or c["mime_type"] != "image/png" or not c["counted"]:
            continue
        arms = c["arms"]
        assert isinstance(arms, dict)
        for arm, cells in arms.items():
            for q in QKEYS:
                ok = cells[q]["correct"]
                if ok is None:
                    continue
                n = counts.setdefault(arm, {}).setdefault(q, [0, 0])
                n[0] += int(ok)
                n[1] += 1
    return counts


def with_edited_png(counts: object, cases: list[J]) -> J:
    assert isinstance(counts, dict)
    ordered: J = {}
    for g, v in counts.items():
        ordered[g] = v
        if g == "edited_conflict":
            ordered["edited_png"] = edited_png_counts(cases)
    return ordered


def text_tokens(rec: J) -> int:
    usage = rec["usage"]
    assert isinstance(usage, dict)
    return sum(int(m["tokens"]) for m in usage.get("input_tokens_by_modality", []) if m["modality"] == "text")


def check_clean(text: str) -> None:
    for pattern in FORBIDDEN:
        if re.search(pattern, text):
            raise SystemExit(f"refusing to publish: the export matches {pattern!r}")


def main() -> int:
    data = export()
    data["caveats"] = caveats(data)
    date = str(data["run_finished"])[:10].replace("-", "")
    out = DOCS / f"multimodal-live-{date}.json"
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    check_clean(text)
    md = render_markdown(data)
    check_clean(md)
    out.write_text(text, encoding="utf-8")
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(f"Wrote {out} and {out.with_suffix('.md').name}")
    return 0


# ------------------------------------------------------------------------------------------------ markdown

QNAMES = {
    "synthetic_label": "Synthetic label",
    "overall_decision": "Banner decision",
    "rejected_exactly_2": "Exactly 2 rejected",
    "legibility_ok": "Legible",
}
GROUPS = {
    "unedited": "Unedited cases",
    "edited": "Edited cases, all",
    "edited_conflict": "Edited cases where that question's label differs from what the records say",
    "edited_png": "Edited screenshots only (the edited PNG cases, without the edited PDF cases)",
    "controls": "Legibility controls",
}


def caveats(d: J) -> list[str]:
    pdf = d["gemini_pdf_text_tokens"]
    assert isinstance(pdf, list)
    no_layer = next((p for p in pdf if p["case"] == "pdf_page_image"), None)
    with_layer = [p["difference"] for p in pdf if p["case"] != "pdf_page_image"]
    cases = d["cases"]
    assert isinstance(cases, list)
    n_cases = len(cases)
    return [
        "The text only arm cannot see the image or the PDF. It gets the audit records as JSON and the same "
        "questions, so where an edit makes the artifact disagree with the records it is wrong by construction.",
        "Jev image input is a preview. TypeSafe says it is not yet ready for production and that its request "
        "format, answers and limits can change between releases. It takes PNG and JPEG only, so the PDF cases "
        "reached Jev only as rendered pages.",
        f"{n_cases} cases is far too few to rank anything. Read the counts as what happened on these "
        "synthetic cases, not as a comparison of the evaluators.",
        "Gemini reads a PDF as a page image plus the text extracted from it. With text in the PDF, its text tokens "
        f"were {min(with_layer)} to {max(with_layer)} higher than for the same page sent as a PNG"
        + (f", and {no_layer['difference']} higher for the image only PDF, which has no text" if no_layer else "")
        + ". So the 4 pt PDF's text reached Gemini however small it looked.",
        "The 4 pt text PDF is recorded but not counted: its text stays sharp when zoomed, so its legibility label "
        "is not clear cut.",
        "One attempt per call, no retries. Gemini ran on the free tier, which Google may use to improve its "
        "products, so every artifact is synthetic or already public.",
    ]


def fmt_cell(c: J) -> str:
    ans = c["answer"]
    text = "yes" if ans is True else "no" if ans is False else str(ans)
    return (
        f"{text} {run.num(c['p']):.2f}" + (" ✗" if c["correct"] is False else "") + (" (close)" if c["close"] else "")
    )


def render_markdown(d: J) -> str:
    out = [
        f"# {d['title']}",
        "",
        f"Run finished {d['run_finished']} UTC; the PDF as PNG run finished {d['pdf_png_run_finished']} UTC.",
        "",
        "## Read this first",
        "",
    ]
    notes = d["caveats"]
    assert isinstance(notes, list)
    out += [f"* {c}" for c in notes] + ["", "## Counts against the labels", ""]
    counts = d["counts"]
    arms = d["arms"]
    assert isinstance(counts, dict) and isinstance(arms, dict)
    for g, title in GROUPS.items():
        out += [f"### {title}", "", "| Arm | " + " | ".join(QNAMES.values()) + " |", "| :- |" + " -: |" * 4]
        for arm, name in arms.items():
            cc = counts.get(g, {}).get(arm, {})
            out.append(f"| {name} | " + " | ".join(run.frac(cc.get(q)) for q in QNAMES) + " |")
        out.append("")
    out += [
        "## Every answer",
        "",
        "Each cell: the answer and its probability (yes probability for the truth questions, the chosen option's "
        "probability for the banner, the probability of level 3 or above for legibility). A cross marks a miss "
        "against the label; close marks a probability within 0.1 of 0.5.",
        "",
        "| Case | Group | Question | Label | Records say | " + " | ".join(arms.values()) + " |",
        "| :- | :- | :- | :- | :- |" + " :- |" * len(arms),
    ]
    all_cases = d["cases"]
    assert isinstance(all_cases, list)
    for c in all_cases:
        for q, qn in QNAMES.items():
            lab = c["labels"].get(q, "none")
            said = c["record_says"].get(q, "")
            cells = [fmt_cell(c["arms"][a][q]) if a in c["arms"] else "not sent" for a in arms]
            out.append(f"| {c['id']} | {c['group']} | {qn} | {yn(lab)} | {yn(said)} | " + " | ".join(cells) + " |")
    pp = d["pdf_as_png"]
    assert isinstance(pp, dict)
    cols = pp["columns"]
    out += [
        "",
        "## The 7 PDF cases side by side",
        "",
        "| Case | Question | Label | " + " | ".join(cols.values()) + " |",
        "| :- | :- | :- |" + " :- |" * len(cols),
    ]
    for row in pp["rows"]:
        for q, qn in QNAMES.items():
            cells = [fmt_cell(row["columns"][c][q]) if c in row["columns"] else "not run" for c in cols]
            out.append(f"| {row['case']} | {qn} | {yn(row['labels'].get(q, 'none'))} | " + " | ".join(cells) + " |")
    lat, tok, cost = d["latency_ms"], d["input_tokens_by_arm"], d["cost_usd_at_list_price"]
    assert isinstance(lat, dict) and isinstance(tok, dict) and isinstance(cost, dict)
    jt = d["jev_image_tokens_measured"]
    assert isinstance(jt, dict)
    out += [
        "",
        "## Latency, tokens and cost",
        "",
        "| Arm | Calls | Median ms | Max ms | Input tokens | USD at list price |",
        "| :- | -: | -: | -: | -: | -: |",
    ]
    for arm, name in arms.items():
        out.append(
            f"| {name} | {lat[arm]['n']} | {lat[arm]['median']} | {lat[arm]['max']} | {tok[arm]:,} | {cost[arm]} |"
        )
    out += [
        "",
        f"Jev image tokens per 1200 x 1568 image, measured as the image arm minus the text only arm: "
        f"{jt['median']} on all {jt['cases']} cases. Gemini ran on the free tier, so its calls cost nothing; "
        "the table uses the paid list price.",
        "",
    ]
    return "\n".join(out) + "\n"


def yn(v: object) -> str:
    return "yes" if v is True else "no" if v is False else str(v)


if __name__ == "__main__":
    sys.exit(main())
