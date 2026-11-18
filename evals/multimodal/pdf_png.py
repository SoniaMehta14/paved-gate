"""The 7 PDF cases rendered to PNG, through the Jev image preview and Gemini Decision 1.

    uv run --all-extras --with pypdfium2 --with pillow python evals/multimodal/pdf_png.py --yes
    uv run python evals/multimodal/pdf_png.py --report        # rebuild the report from the checkpoint

Each page is rendered at 1212 x 1568 (the page's own proportions at the full height Jev allows) on white, with
pypdfium2. Both arms get the same PNG, the same audit record text and the same questions as the main run. One
attempt per call, Gemini paced as before, and the run stops at the first error, quota message, blocked input,
warning or image limit refusal. The report puts the 7 cases next to the main run's Gemini with the PDF and Jev
text only. Everything is written to results/multimodal/pdf_png/ (ignored by git).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import io
import json
import os
import statistics
import sys
from collections.abc import Awaitable
from datetime import UTC, datetime
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run  # noqa: E402
from benchmark_evaluators import TokenPacer  # noqa: E402
from build_cases import png_text_chunks  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from jev_image import check_image, evaluate_jev_image  # noqa: E402
from pricing import PRICES  # noqa: E402
from questions import LEGIBLE_FROM_LEVEL, QUESTIONS  # noqa: E402

from paved_gate.evaluator.base import EvaluationResult, MediaPart  # noqa: E402
from paved_gate.evaluator.gemini_decision import GeminiDecisionEvaluator  # noqa: E402

OUT = run.ROOT / "results" / "multimodal" / "pdf_png"
MAIN_CALLS = run.OUT / "calls.jsonl"
HEIGHT = 1568
ARMS = ("gemini_png", "jev_png")
COLUMNS = {
    "gemini_pdf": "Gemini, PDF (main run)",
    "gemini_png": "Gemini, PNG",
    "jev_png": "Jev image preview, PNG",
    "jev_text": "Jev text only (main run)",
}
CLOSE = 0.1  # an answer whose probability is within this of 0.5 is a close call
J = dict[str, object]


def cases_of(manifest: J) -> list[J]:
    cases = manifest["cases"]
    assert isinstance(cases, list)
    return [c for c in cases if isinstance(c, dict)]


def render(manifest: J) -> list[J]:
    """Render page 1 of every PDF case to PNG and write a small manifest of what was rendered."""
    import pypdfium2 as pdfium  # type: ignore[import-not-found]  # only needed here; run with --with pypdfium2

    (OUT / "cases").mkdir(parents=True, exist_ok=True)
    cases = [c for c in cases_of(manifest) if c["mime_type"] == "application/pdf"]
    out: list[J] = []
    for c in cases:
        page = pdfium.PdfDocument(str(run.CASES / str(c["artifact"])))[0]
        _, h = page.get_size()
        image = page.render(scale=HEIGHT / h, fill_color=(255, 255, 255, 255)).to_pil().convert("RGB")
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        data = buf.getvalue()
        if png_text_chunks(data):
            raise SystemExit(f"{c['id']}: rendered PNG carries metadata")
        check_image(MediaPart(mime_type="image/png", data=data, name=f"{c['id']}.png"))  # Jev's documented limits
        path = OUT / "cases" / f"{c['id']}.png"
        path.write_bytes(data)
        out.append(
            {
                "id": c["id"],
                "png": path.name,
                "size": list(image.size),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "pdf_sha256": c["sha256"],
            }
        )
    (OUT / "cases" / "manifest.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def digest(rendered: list[J]) -> str:
    q = json.dumps({k: v.model_dump(mode="json") for k, v in QUESTIONS.items()}, sort_keys=True)
    blob = json.dumps(rendered, sort_keys=True) + q + f"{run.GEMINI_MODEL}|{run.JEV_MODEL}|{run.TIMEOUT_S}"
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


async def run_live(manifest: J, rendered: list[J], settings: str) -> int:
    load_dotenv(run.ROOT / ".env")  # keys from .env; real environment variables take precedence
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    jev_key = os.environ.get("TYPESAFE_API_KEY", "")
    if not gemini_key or not jev_key:
        print("GEMINI_API_KEY and TYPESAFE_API_KEY are both needed", file=sys.stderr)
        return 2
    secrets = [gemini_key, jev_key]
    ckpt = OUT / "calls.jsonl"
    done = run.load_checkpoint(ckpt, settings)
    texts = manifest["record_texts"]
    assert isinstance(texts, dict)
    by_id = {c["id"]: c for c in cases_of(manifest)}
    todo = [(r, a) for r in rendered for a in ARMS if f"{r['id']}|{a}" not in done]
    print(f"{len(done)} calls already in the checkpoint, {len(todo)} to make")
    pacer = TokenPacer(run.GEMINI_TPM)
    gemini = GeminiDecisionEvaluator(
        mode="live", model=run.GEMINI_MODEL, api_key=gemini_key, timeout_s=run.TIMEOUT_S, max_retries=0
    )
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(run.TIMEOUT_S)) as client:
            for n, (r, arm) in enumerate(todo, 1):
                data = (OUT / "cases" / str(r["png"])).read_bytes()
                if hashlib.sha256(data).hexdigest() != r["sha256"]:
                    raise SystemExit(f"{r['id']}: the PNG changed since it was rendered")
                part = MediaPart(mime_type="image/png", data=data, name=str(r["png"]))
                text = str(texts[str(by_id[r["id"]]["text_key"])])
                waited = 0.0
                if arm == "gemini_png":
                    waited = await pacer.wait(pacer.expected("png", run.expected_gemini_tokens(len(text), "image/png")))

                def call(arm: str = arm, part: MediaPart = part, text: str = text) -> Awaitable[EvaluationResult]:
                    if arm == "gemini_png":
                        return gemini.evaluate_media(text, [part], QUESTIONS)
                    return evaluate_jev_image(
                        client, api_key=jev_key, model=run.JEV_MODEL, record_text=text, image=part, questions=QUESTIONS
                    )

                result = await run.guarded(call, secrets, str(r["id"]), arm)
                if arm == "gemini_png" and result.input_tokens is not None:
                    pacer.record("png", result.input_tokens)
                rec = {
                    "settings": settings,
                    "case": r["id"],
                    "arm": arm,
                    "model": result.model,
                    "at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "latency_ms": result.latency_ms,
                    "pacing_wait_s": round(waited, 2),
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "answers": result.raw.get("answers"),
                    "usage": result.raw.get("usage"),
                }
                with ckpt.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, sort_keys=True) + "\n")
                print(f"[{n}/{len(todo)}] {r['id']:22} {arm:12} {result.latency_ms:8.1f} ms  {result.input_tokens}")
    except run.Stop as exc:
        (OUT / "stopped.json").write_text(json.dumps({"reason": str(exc)}, indent=2) + "\n", encoding="utf-8")
        print(f"Stopped: {exc}", file=sys.stderr)
        return 1
    finally:
        await gemini.aclose()
    (OUT / "stopped.json").unlink(missing_ok=True)
    return 0


# ------------------------------------------------------------------------------------------------ report


def probabilities(answers: J) -> dict[str, float]:
    """The probability behind each answer, on a scale where 0.5 is the cutoff.

    Truth: the yes probability. Choice: the probability of the chosen option. Legibility: the probability of
    level 3 or above (levels 3 to 5), next to the expected level the scoring uses.
    """

    def a(k: str) -> J:
        v = answers[k]
        assert isinstance(v, dict)
        return v

    choice = a("overall_decision")
    probs = choice["probabilities"]
    assert isinstance(probs, dict)
    legible = a("legibility")["probabilities"]
    assert isinstance(legible, dict)
    return {
        "synthetic_label": run.num(a("synthetic_label")["noul"]),
        "overall_decision": run.num(probs[choice["choice"]]),
        "rejected_exactly_2": run.num(a("rejected_exactly_2")["noul"]),
        "legibility_ok": sum(run.num(p) for lvl, p in legible.items() if int(lvl) + 1 >= LEGIBLE_FROM_LEVEL),
    }


def build_report(manifest: J, rendered: list[J], settings: str) -> J:
    main = run.load_checkpoint(MAIN_CALLS, run.settings_hash((run.CASES / "manifest.json").read_bytes()))
    new = run.load_checkpoint(OUT / "calls.jsonl", settings)
    by_id = {c["id"]: c for c in cases_of(manifest)}
    source = {
        "gemini_pdf": (main, "gemini_decision"),
        "gemini_png": (new, "gemini_png"),
        "jev_png": (new, "jev_png"),
        "jev_text": (main, "jev_text"),
    }
    qkeys = ["synthetic_label", "overall_decision", "rejected_exactly_2", "legibility_ok"]
    counts: dict[str, dict[str, dict[str, list[int]]]] = {}
    rows: list[J] = []
    close_calls = {col: 0 for col in COLUMNS}
    for r in rendered:
        case = by_id[r["id"]]
        g = run.group(case)
        labels = case["labels"]
        assert isinstance(labels, dict)
        columns: dict[str, dict[str, object]] = {}
        row: J = {"case": r["id"], "group": g, "labels": labels, "columns": columns}
        for col, (calls, arm) in source.items():
            rec = calls.get(f"{r['id']}|{arm}")
            if rec is None:
                continue
            answers = rec["answers"]
            assert isinstance(answers, dict)
            pred, prob = run.predicted(answers), probabilities(answers)
            cells: dict[str, object] = {}
            for q in qkeys:
                close = abs(prob[q] - 0.5) <= CLOSE
                ok = None if q not in labels else pred[q] == labels[q]
                cells[q] = {"answer": pred[q], "p": round(prob[q], 4), "close": close, "correct": ok}
                if close and g != "recorded_only":
                    close_calls[col] += 1
                if ok is not None and g != "recorded_only":
                    c = counts.setdefault(g, {}).setdefault(col, {}).setdefault(q, [0, 0])
                    c[0] += int(ok)
                    c[1] += 1
            cells["legibility_level"] = pred["legibility_level"]
            columns[col] = cells
        rows.append(row)
    jev_tokens = []
    for r in rendered:
        img, txt = new.get(f"{r['id']}|jev_png"), main.get(f"{r['id']}|jev_text")
        if img and txt:
            a, b = int(run.num(img["input_tokens"])), int(run.num(txt["input_tokens"]))
            jev_tokens.append({"case": r["id"], "image_arm": a, "text_arm_main_run": b, "difference": a - b})
    gem_mod: dict[str, list[int]] = {}
    for rec in new.values():
        if rec["arm"] == "gemini_png":
            usage = rec["usage"]
            assert isinstance(usage, dict)
            for m in usage.get("input_tokens_by_modality", []):
                gem_mod.setdefault(m["modality"], []).append(int(m["tokens"]))
    lat = {arm: run.summarise([run.num(c["latency_ms"]) for c in new.values() if c["arm"] == arm]) for arm in ARMS}
    diffs = [int(run.num(t["difference"])) for t in jev_tokens]
    tokens = {arm: sum(int(run.num(c["input_tokens"] or 0)) for c in new.values() if c["arm"] == arm) for arm in ARMS}
    stopped = OUT / "stopped.json"
    limit_note = (
        f"The run stopped: {json.loads(stopped.read_text(encoding='utf-8'))['reason']}"
        if stopped.exists()
        else "Every call returned answers; none was refused for size or format."
    )
    return {
        "title": "PDF cases as PNG: Jev image preview and Gemini Decision 1",
        "limit_note": limit_note,
        "run_finished": max((str(c["at"]) for c in new.values()), default=None),
        "calls": {"made": len(new), "expected": len(rendered) * len(ARMS)},
        "rendered": rendered,
        "counts": counts,
        "close_calls": close_calls,
        "rows": rows,
        "jev_image_tokens": {
            "per_case": jev_tokens,
            "median": statistics.median(diffs) if diffs else None,
            "min": min(diffs, default=None),
            "max": max(diffs, default=None),
        },
        "gemini_png_tokens_by_modality": {
            k: {"calls": len(v), "total": sum(v), "median": statistics.median(v)} for k, v in gem_mod.items()
        },
        "latency_ms": lat,
        "input_tokens": tokens,
        "cost_usd_at_list_price": {
            "gemini_png": round(tokens["gemini_png"] * PRICES["gemini-decision-1"].input_per_mtok / 1e6, 5),
            "jev_png": round(tokens["jev_png"] * PRICES["jev"].input_per_mtok / 1e6, 5),
        },
    }


def cell(c: J | None) -> str:
    if c is None:
        return "not run"
    ans = c["answer"]
    text = "yes" if ans is True else "no" if ans is False else str(ans)
    mark = " ✗" if c["correct"] is False else ""
    close = " (close)" if c["close"] else ""
    return f"{text} {run.num(c['p']):.2f}{mark}{close}"


def render_markdown(r: J) -> str:
    qn = {
        "synthetic_label": "Synthetic label",
        "overall_decision": "Banner decision",
        "rejected_exactly_2": "Exactly 2 rejected",
        "legibility_ok": "Legible",
    }
    calls = r["calls"]
    assert isinstance(calls, dict)
    out = [
        f"# {r['title']}",
        "",
        f"Run finished {r['run_finished']} UTC. {calls['made']} of {calls['expected']} new calls; the PDF and text "
        "only columns are from the main run on 2026-11-18.",
        "",
        "## Read this first",
        "",
        "* Each PDF page was rendered to a 1212 x 1568 PNG on white with pypdfium2, and both new arms got that same "
        "PNG with the same record and questions as the main run. Gemini's PDF column is the main run, where Gemini "
        "read the PDF itself.",
        '* **Jev image input is a preview** ("Dreamer-ready" in TypeSafe\'s docs). Jev does not take PDF, so a '
        "rendered page is the only way to give it these cases.",
        "* **The text only column cannot see the page;** it answers from the record.",
        "* **Seven cases are far too few to rank anything.** Read the counts as what happened here.",
        "* Every number is one call per case and arm, with no retries.",
        "",
        "## Counts against the labels",
        "",
    ]
    counts = r["counts"]
    assert isinstance(counts, dict)
    for g, title in [
        ("unedited", "Unedited (pdf_original)"),
        ("edited", "Edited (4 cases)"),
        ("controls", "Legibility control (pdf_page_image)"),
    ]:
        out += [f"### {title}", "", "| Column | " + " | ".join(qn.values()) + " |", "| :- |" + " -: |" * 4]
        for col, name in COLUMNS.items():
            cc = counts.get(g, {}).get(col, {})
            out.append(f"| {name} | " + " | ".join(run.frac(cc.get(q)) for q in qn) + " |")
        out.append("")
    close = r["close_calls"]
    assert isinstance(close, dict)
    out += [
        "pdf_4pt is recorded below but not counted.",
        "",
        "Close calls (probability within 0.1 of 0.5) on counted cases: "
        + ", ".join(f"{COLUMNS[c]} {n}" for c, n in close.items())
        + ".",
        "",
        "## Every answer with its probability",
        "",
        "Each cell is the answer and the probability behind it: the yes probability for the two truth questions, the "
        "chosen option's probability for the banner, and the probability of level 3 or above for legibility (the "
        "expected level the scoring uses is in brackets). ✗ marks an answer that does not match the label; "
        '"close" marks a probability within 0.1 of 0.5.',
        "",
    ]
    rows = r["rows"]
    assert isinstance(rows, list)
    for row in rows:
        labels = row["labels"]
        out += [
            f"### {row['case']} ({row['group']})",
            "",
            "| Question | Label | " + " | ".join(COLUMNS.values()) + " |",
            "| :- | :- |" + " :- |" * 4,
        ]
        for q, name in qn.items():
            lab = labels.get(q, "none")
            lab_text = "yes" if lab is True else "no" if lab is False else str(lab)
            cells = []
            for col in COLUMNS:
                c = row["columns"].get(col)
                text = cell(c[q] if c else None)
                if q == "legibility_ok" and c:
                    text += f" [{c['legibility_level']}]"
                cells.append(text)
            out.append(f"| {name} | {lab_text} | " + " | ".join(cells) + " |")
        out.append("")
    jt = r["jev_image_tokens"]
    assert isinstance(jt, dict)
    out += [
        "## Jev image tokens",
        "",
        "The Jev image call's input tokens minus the text only arm's tokens for the same record in the main run "
        f"(1212 x 1568 PNG): median {jt['median']}, min {jt['min']}, max {jt['max']}.",
        "",
        "| Case | Image arm | Text only (main run) | Image tokens |",
        "| :- | -: | -: | -: |",
    ]
    for t in jt["per_case"]:
        out.append(f"| {t['case']} | {t['image_arm']} | {t['text_arm_main_run']} | {t['difference']} |")
    gm, lat, tok, cost = (
        r["gemini_png_tokens_by_modality"],
        r["latency_ms"],
        r["input_tokens"],
        r["cost_usd_at_list_price"],
    )
    assert isinstance(gm, dict) and isinstance(lat, dict) and isinstance(tok, dict) and isinstance(cost, dict)
    out += [
        "",
        "Gemini with the PNG, input tokens by modality: "
        + ", ".join(f"{k} {v['total']:,} over {v['calls']} calls (median {v['median']:,.0f})" for k, v in gm.items())
        + ".",
        "",
        "## Latency, tokens and cost of the new calls",
        "",
        "| Arm | Calls | Median ms | Max ms | Input tokens | USD at list price |",
        "| :- | -: | -: | -: | -: | -: |",
        f"| Gemini, PNG | {lat['gemini_png']['n']} | {lat['gemini_png']['median']} | {lat['gemini_png']['max']} | "
        f"{tok['gemini_png']:,} | {cost['gemini_png']:.5f} (free tier: 0) |",
        f"| Jev image preview, PNG | {lat['jev_png']['n']} | {lat['jev_png']['median']} | {lat['jev_png']['max']} | "
        f"{tok['jev_png']:,} | {cost['jev_png']:.5f} |",
        "",
        "## Limit problems",
        "",
        "Every PNG was checked against Jev's documented limits before sending (280 to 1568 px per side, at most "
        "2,458,624 pixels, 16 MiB). " + str(r["limit_note"]),
        "",
    ]
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true", help="confirm the live calls")
    ap.add_argument("--report", action="store_true", help="only rebuild the report from the checkpoint")
    args = ap.parse_args()
    manifest = run.load_manifest()
    rendered_path = OUT / "cases" / "manifest.json"
    if args.report:
        rendered = json.loads(rendered_path.read_text(encoding="utf-8"))
    else:
        if not args.yes:
            print("Live calls need --yes.", file=sys.stderr)
            return 2
        rendered = render(manifest)
        print(f"Rendered {len(rendered)} pages: " + ", ".join(f"{r['id']} {r['size']}" for r in rendered))
    settings = digest(rendered)
    if not args.report:
        code = asyncio.run(run_live(manifest, rendered, settings))
        if code != 0:
            return code
    report = build_report(manifest, rendered, settings)
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (OUT / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"Report: {OUT / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
