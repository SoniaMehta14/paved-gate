"""Full live multimodal test: every case once per arm, scored against labels known by construction.

    uv run --extra gemini python evals/multimodal/run.py --yes        # live calls, resumes from the checkpoint
    uv run python evals/multimodal/run.py --report                    # rebuild the report, no calls

Arms: Gemini Decision 1 with the image or PDF plus the audit records; Jev with the image plus the records
(image input is a TypeSafe preview, PNG only, so image cases only); Jev with the records as text only (it
cannot see the image or PDF). One attempt per call. Gemini is paced to 45,000 input tokens per minute.
The run stops at the first error, quota message, blocked input or warning; finished calls are kept in an
append only checkpoint, so a rerun resumes without repeating them.

Everything is written to results/multimodal/run/ (ignored by git): calls.jsonl, report.json, report.md.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import statistics
import sys
import time
import warnings
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "evals" / "support_tickets"))

from benchmark_evaluators import TokenPacer  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from jev_image import evaluate_jev_image  # noqa: E402
from pricing import PRICES  # noqa: E402
from questions import LEGIBLE_FROM_LEVEL, QUESTIONS  # noqa: E402

from paved_gate.evaluator.base import EvaluationResult, MediaPart  # noqa: E402
from paved_gate.evaluator.gemini_decision import GeminiDecisionEvaluator  # noqa: E402
from paved_gate.evaluator.jev import JevEvaluator  # noqa: E402

CASES = ROOT / "results" / "multimodal" / "cases"
OUT = ROOT / "results" / "multimodal" / "run"
GEMINI_MODEL = "gemini-decision-1"
JEV_MODEL = "jev-1.14.0"
TIMEOUT_S = 15.0
GEMINI_TPM = 45_000
ARMS = ("gemini_decision", "jev_image", "jev_text")
ARM_LABEL = {
    "gemini_decision": "Gemini Decision 1 (image or PDF + records)",
    "jev_image": "Jev 1.14, image preview (image + records)",
    "jev_text": "Jev 1.14, text only (records only, cannot see the artifact)",
}
SEVERITY = {"accept": 0, "needs_review": 1, "reject": 2}

J = dict[str, object]


class Stop(Exception):
    """A call failed or warned; the run stops and the call is not checkpointed."""


class Captured(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(f"log {record.levelname} {record.name}: {record.getMessage()}")


def load_manifest() -> J:
    data = json.loads((CASES / "manifest.json").read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def settings_hash(manifest_bytes: bytes) -> str:
    q = json.dumps({k: v.model_dump(mode="json") for k, v in QUESTIONS.items()}, sort_keys=True)
    h = hashlib.sha256(manifest_bytes + q.encode() + f"{GEMINI_MODEL}|{JEV_MODEL}|{TIMEOUT_S}".encode())
    return h.hexdigest()[:16]


def load_checkpoint(path: Path, digest: str) -> dict[str, J]:
    done: dict[str, J] = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec["settings"] != digest:
            raise SystemExit(f"{path} was written with other settings; move it aside to start fresh")
        done[f"{rec['case']}|{rec['arm']}"] = rec
    return done


def scrub(text: str, secrets: list[str]) -> str:
    for s in secrets:
        if s:
            text = text.replace(s, "[redacted]")
    return text


def expected_gemini_tokens(record_chars: int, mime: str) -> int:
    # From the smoke calls: 0.3635 text tokens per record character plus 508, 1,102 per image, 520 per PDF page.
    return round(0.3635 * record_chars + 508 + (1102 if mime == "image/png" else 520))


async def run_live(manifest: J, digest: str) -> int:
    load_dotenv(ROOT / ".env")  # keys from .env; real environment variables take precedence
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    jev_key = os.environ.get("TYPESAFE_API_KEY", "")
    if not gemini_key or not jev_key:
        print("GEMINI_API_KEY and TYPESAFE_API_KEY are both needed", file=sys.stderr)
        return 2
    secrets = [gemini_key, jev_key]
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = OUT / "calls.jsonl"
    done = load_checkpoint(ckpt, digest)
    texts = manifest["record_texts"]
    assert isinstance(texts, dict)
    cases = manifest["cases"]
    assert isinstance(cases, list)
    todo = [(c, a) for c in cases for a in ARMS if a in c["arms"] and f"{c['id']}|{a}" not in done]
    print(f"{len(done)} calls already in the checkpoint, {len(todo)} to make")

    pacer = TokenPacer(GEMINI_TPM)
    gemini = GeminiDecisionEvaluator(
        mode="live", model=GEMINI_MODEL, api_key=gemini_key, timeout_s=TIMEOUT_S, max_retries=0
    )
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT_S)) as client:
            jev = JevEvaluator(
                mode="live", model=JEV_MODEL, api_key=jev_key, timeout_s=TIMEOUT_S, max_retries=0, http_client=client
            )
            for n, (case, arm) in enumerate(todo, 1):
                part = MediaPart(
                    mime_type=case["mime_type"], data=(CASES / case["artifact"]).read_bytes(), name=case["artifact"]
                )
                if hashlib.sha256(part.data).hexdigest() != case["sha256"]:
                    raise SystemExit(f"{case['id']}: the file changed since the manifest was written")
                text = texts[case["text_key"]]
                waited = 0.0
                kind = f"{case['text_key']}|{case['mime_type']}"
                if arm == "gemini_decision":
                    waited = await pacer.wait(
                        pacer.expected(kind, expected_gemini_tokens(len(text), case["mime_type"]))
                    )

                def call(arm: str = arm, part: MediaPart = part, text: str = text) -> Awaitable[EvaluationResult]:
                    if arm == "gemini_decision":
                        return gemini.evaluate_media(text, [part], QUESTIONS)
                    if arm == "jev_image":
                        return evaluate_jev_image(
                            client, api_key=jev_key, model=JEV_MODEL, record_text=text, image=part, questions=QUESTIONS
                        )
                    return jev.evaluate({"record": text}, QUESTIONS)

                result = await guarded(call, secrets, case["id"], arm)
                if arm == "gemini_decision" and result.input_tokens is not None:
                    pacer.record(kind, result.input_tokens)
                rec = {
                    "settings": digest,
                    "case": case["id"],
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
                print(f"[{n}/{len(todo)}] {case['id']:28} {arm:16} {result.latency_ms:8.1f} ms  {result.input_tokens}")
    except Stop as exc:
        (OUT / "stopped.json").write_text(json.dumps({"reason": str(exc)}, indent=2) + "\n", encoding="utf-8")
        print(f"Stopped: {exc}", file=sys.stderr)
        return 1
    finally:
        await gemini.aclose()
    (OUT / "stopped.json").unlink(missing_ok=True)
    print(f"Gemini pacing waited {pacer.waited_s:.1f} s in this session")
    return 0


async def guarded(
    call: Callable[[], Awaitable[EvaluationResult]], secrets: list[str], case_id: str, arm: str
) -> EvaluationResult:
    handler = Captured()
    logging.getLogger().addHandler(handler)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = await call()
            except Exception as exc:
                raise Stop(scrub(f"{case_id} on {arm}: {type(exc).__name__}: {exc}", secrets)) from exc
    finally:
        logging.getLogger().removeHandler(handler)
    found = [f"{w.category.__name__}: {w.message}" for w in caught] + handler.messages
    if found:
        raise Stop(scrub(f"{case_id} on {arm}: warning: {'; '.join(found)}", secrets))
    if result.raw.get("prompt_feedback"):
        raise Stop(f"{case_id} on {arm}: prompt feedback {json.dumps(result.raw['prompt_feedback'])}")
    return result


# ------------------------------------------------------------------------------------------------ scoring


def record_view(text: str) -> dict[str, object]:
    """What the audit records themselves say, for the same questions (the text only arm's only source)."""
    records = json.loads(text)
    return {
        "synthetic_label": any("synthetic" in str(r.get("namespace", "")).lower() for r in records),
        "overall_decision": max((r["decision"] for r in records), key=SEVERITY.__getitem__),
        "rejected_exactly_2": sum(r["decision"] == "reject" for r in records) == 2,
    }


def predicted(answers: J) -> dict[str, object]:
    def a(k: str) -> J:
        v = answers[k]
        assert isinstance(v, dict)
        return v

    level = float(a("legibility")["score"]) + 1  # type: ignore[arg-type]
    return {
        "synthetic_label": float(a("synthetic_label")["noul"]) >= 0.5,  # type: ignore[arg-type]
        "overall_decision": a("overall_decision")["choice"],
        "rejected_exactly_2": float(a("rejected_exactly_2")["noul"]) >= 0.5,  # type: ignore[arg-type]
        "legibility_ok": level >= LEGIBLE_FROM_LEVEL,
        "legibility_level": round(level, 2),
    }


def group(case: J) -> str:
    if not case.get("counted", True):
        return "recorded_only"
    if case["control"]:
        return "controls"
    return "unedited" if not case["edits"] else "edited"


def num(v: object) -> float:
    assert isinstance(v, int | float), v
    return float(v)


def summarise(values: list[float]) -> dict[str, float | None]:
    return {
        "n": len(values),
        "median": round(statistics.median(values), 1) if values else None,
        "max": round(max(values), 1) if values else None,
    }


def build_report(manifest: J, calls: dict[str, J]) -> J:
    texts = manifest["record_texts"]
    assert isinstance(texts, dict)
    cases = manifest["cases"]
    assert isinstance(cases, list)
    views = {k: record_view(v) for k, v in texts.items()}
    qkeys = ["synthetic_label", "overall_decision", "rejected_exactly_2", "legibility_ok"]
    rows: list[J] = []
    # counts[group][arm][question] = [correct, total]; "edited" also gets "edited_conflict" per question.
    counts: dict[str, dict[str, dict[str, list[int]]]] = {}

    def bump(g: str, arm: str, q: str, ok: bool) -> None:
        c = counts.setdefault(g, {}).setdefault(arm, {}).setdefault(q, [0, 0])
        c[0] += int(ok)
        c[1] += 1

    for case in cases:
        g = group(case)
        view = views[case["text_key"]]
        row: J = {"case": case["id"], "group": g, "labels": case["labels"], "record_says": view, "arms": {}}
        for arm in ARMS:
            rec = calls.get(f"{case['id']}|{arm}")
            if arm not in case["arms"] or rec is None:
                continue
            answers = rec["answers"]
            assert isinstance(answers, dict)
            pred = predicted(answers)
            row["arms"][arm] = pred | {"latency_ms": rec["latency_ms"], "input_tokens": rec["input_tokens"]}  # type: ignore[index]
            if g == "recorded_only":
                continue
            for q in qkeys:
                if q not in case["labels"]:
                    continue
                ok = pred[q] == case["labels"][q]
                bump(g, arm, q, ok)
                if g == "edited" and q in view and view[q] != case["labels"][q]:
                    bump("edited_conflict", arm, q, ok)
        rows.append(row)

    # latency and tokens
    latency = {arm: summarise([num(c["latency_ms"]) for c in calls.values() if c["arm"] == arm]) for arm in ARMS}
    by_mime = {c["id"]: c["mime_type"] for c in cases}
    gemini_latency = {
        mime: summarise(
            [
                num(c["latency_ms"])
                for c in calls.values()
                if c["arm"] == "gemini_decision" and by_mime[c["case"]] == mime
            ]
        )
        for mime in ("image/png", "application/pdf")
    }
    modality: dict[str, list[int]] = {}
    for c in calls.values():
        if c["arm"] != "gemini_decision":
            continue
        usage = c["usage"]
        assert isinstance(usage, dict)
        for m in usage.get("input_tokens_by_modality", []):
            label = f"{m['modality']} ({'PDF' if by_mime[c['case']] == 'application/pdf' else 'PNG'} cases)"
            modality.setdefault(label, []).append(int(m["tokens"]))
    gemini_tokens = {k: {"calls": len(v), "total": sum(v), "median": statistics.median(v)} for k, v in modality.items()}
    image_tokens: list[J] = []
    diffs: list[int] = []
    for case in cases:
        img, txt = calls.get(f"{case['id']}|jev_image"), calls.get(f"{case['id']}|jev_text")
        if img and txt and img["input_tokens"] is not None and txt["input_tokens"] is not None:
            a, b = int(num(img["input_tokens"])), int(num(txt["input_tokens"]))
            diffs.append(a - b)
            image_tokens.append({"case": case["id"], "image_arm": a, "text_arm": b, "difference": a - b})
    tokens_by_arm = {
        arm: sum(int(num(c["input_tokens"] or 0)) for c in calls.values() if c["arm"] == arm) for arm in ARMS
    }
    price = {"gemini_decision": PRICES["gemini-decision-1"], "jev_image": PRICES["jev"], "jev_text": PRICES["jev"]}
    cost = {arm: round(tokens_by_arm[arm] * price[arm].input_per_mtok / 1e6, 5) for arm in ARMS}
    expected_calls = sum(len(c["arms"]) for c in cases)
    return {
        "title": "Live multimodal test: synthetic Harness² reports and a synthetic PDF",
        "run_finished": max((str(c["at"]) for c in calls.values()), default=None),
        "models": {"gemini_decision": GEMINI_MODEL, "jev": JEV_MODEL},
        "calls": {"made": len(calls), "expected": expected_calls},
        "cases": len(cases),
        "counts": counts,
        "latency_ms": latency,
        "gemini_latency_ms_by_artifact": gemini_latency,
        "gemini_input_tokens_by_modality": gemini_tokens,
        "jev_image_tokens_measured": {
            "cases": len(diffs),
            "median": statistics.median(diffs) if diffs else None,
            "min": min(diffs) if diffs else None,
            "max": max(diffs) if diffs else None,
            "per_case": image_tokens,
        },
        "input_tokens_by_arm": tokens_by_arm,
        "cost_usd_at_list_price": cost,
        "price_sources": {arm: price[arm].source for arm in ARMS},
        "rows": rows,
    }


# ------------------------------------------------------------------------------------------------ markdown


def frac(c: list[int] | None) -> str:
    return f"{c[0]} of {c[1]}" if c else "n/a"


def render_markdown(r: J) -> str:
    counts = r["counts"]
    assert isinstance(counts, dict)
    qnames = {
        "synthetic_label": "Synthetic label",
        "overall_decision": "Banner decision",
        "rejected_exactly_2": "Exactly 2 rejected",
        "legibility_ok": "Legible (level 3+)",
    }
    out = [f"# {r['title']}", ""]
    out += [
        f"Run finished {r['run_finished']} UTC. Models: Gemini `{GEMINI_MODEL}`, Jev `{JEV_MODEL}`. "
        f"{r['calls']['made']} of {r['calls']['expected']} calls, {r['cases']} cases.",  # type: ignore[index]
        "",
        "## Read this first",
        "",
        "* **The text only arm cannot see the image or the PDF.** It gets the audit records as JSON and the same "
        "questions, so it answers from the records. Where an edit makes the artifact disagree with the records, it "
        "is wrong by construction; that measures the setup, not Jev.",
        '* **Jev image input is a preview.** TypeSafe labels it "Dreamer-ready", not production ready; the request '
        "format, answers and limits can change between releases. It takes PNG and JPEG only.",
        "* **PDF cases were run on Gemini only** (and on the text only arm, which sees just the record). TypeSafe "
        "does not document PDF input for Jev.",
        "* **24 cases is far too few to rank anything.** Read the counts as what happened on these synthetic cases, "
        "not as a comparison of the evaluators.",
        "* Every artifact is synthetic or already public: three published Harness² report pages re-rendered at "
        "1200 x 1568 px, edited variants of them, and a hand built one page PDF. Gemini ran on the free tier, which "
        "Google may use to improve its products.",
        "* One attempt per call, no retries. Gemini was paced to 45,000 input tokens per minute; Jev was not paced.",
        "",
        "## Accuracy against the labels",
        "",
        "Counts are correct answers of the cases that have a label for that question. Truth answers count as yes at "
        "0.5 or above; the banner decision is the top option; legible means score level 3 or above of 5.",
        "",
    ]
    sections = [
        ("unedited", "Unedited cases (3 report pages and the original PDF)"),
        ("edited", "Edited cases, all"),
        ("edited_conflict", "Edited cases, only where that question's label differs from what the records say"),
        ("controls", "Legibility controls (10 px blur on the 3 reports, image only PDF at about 9 dpi)"),
    ]
    for key, title in sections:
        block = counts.get(key, {})
        assert isinstance(block, dict)
        out += [f"### {title}", "", "| Arm | " + " | ".join(qnames.values()) + " |", "| :- |" + " -: |" * 4]
        for arm in ARMS:
            arm_counts = block.get(arm, {})
            out.append(f"| {ARM_LABEL[arm]} | " + " | ".join(frac(arm_counts.get(q)) for q in qnames) + " |")
        out.append("")
    out += [
        "The 4 pt text PDF was recorded but is not counted: its text stays sharp when zoomed, so its legibility "
        "label is not clear cut. Its answers are in the per case table.",
        "",
        "## Latency",
        "",
        "Wall clock time around one call, sent one at a time. Gemini pacing waits are outside it.",
        "",
        "| Arm | Calls | Median ms | Max ms |",
        "| :- | -: | -: | -: |",
    ]
    lat = r["latency_ms"]
    assert isinstance(lat, dict)
    for arm in ARMS:
        s = lat[arm]
        out.append(f"| {ARM_LABEL[arm]} | {s['n']} | {s['median']} | {s['max']} |")
    glat = r["gemini_latency_ms_by_artifact"]
    assert isinstance(glat, dict)
    out += [
        "",
        f"Gemini by artifact: PNG median {glat['image/png']['median']} ms (max {glat['image/png']['max']}), "
        f"PDF median {glat['application/pdf']['median']} ms (max {glat['application/pdf']['max']}).",
        "",
        "## Tokens",
        "",
        "Gemini input tokens by modality, as the API reports them:",
        "",
        "| Modality | Calls | Total | Median per call |",
        "| :- | -: | -: | -: |",
    ]
    gt = r["gemini_input_tokens_by_modality"]
    assert isinstance(gt, dict)
    for k, v in sorted(gt.items()):
        out.append(f"| {k} | {v['calls']} | {v['total']:,} | {v['median']:,.0f} |")
    jt = r["jev_image_tokens_measured"]
    assert isinstance(jt, dict)
    out += [
        "",
        f"Jev reports only a total. Its image tokens, measured as the image arm's input tokens minus the text only "
        f"arm's on the same case ({jt['cases']} cases; the image arm's state also carries a short reference key): "
        f"median {jt['median']}, min {jt['min']}, max {jt['max']}.",
        "",
        "## Cost",
        "",
        "| Arm | Input tokens | USD at list price |",
        "| :- | -: | -: |",
    ]
    tok, cost = r["input_tokens_by_arm"], r["cost_usd_at_list_price"]
    assert isinstance(tok, dict) and isinstance(cost, dict)
    for arm in ARMS:
        out.append(f"| {ARM_LABEL[arm]} | {tok[arm]:,} | {cost[arm]:.4f} |")
    out += [
        "",
        "Gemini ran on the free tier, so its calls cost nothing; the table uses the paid tier list price. Prices "
        "are from scripts/pricing.py.",
        "",
        "## Per case",
        "",
        "Labels, then each arm's answers: synthetic label / banner decision / exactly 2 rejected / legibility level. "
        "A cross marks an answer that does not match the label; a dash means no label for that question.",
        "",
        "| Case | Group | Labels | Records say | Gemini | Jev image | Jev text |",
        "| :- | :- | :- | :- | :- | :- | :- |",
    ]
    rows = r["rows"]
    assert isinstance(rows, list)
    for row in rows:
        lab = row["labels"]
        out.append(
            f"| {row['case']} | {row['group']} | {fmt_labels(lab)} | {fmt_labels(row['record_says'])} | "
            + " | ".join(
                fmt_pred(row["arms"].get(a), lab) if a in row["arms"] or a != "jev_image" else "PDF not sent"
                for a in ARMS
            )
            + " |"
        )
    return "\n".join(out) + "\n"


def yn(v: object) -> str:
    return "yes" if v is True else "no" if v is False else str(v)


def fmt_labels(lab: J) -> str:
    if not lab:
        return "none"
    parts = [yn(lab.get(q, "-")) for q in ("synthetic_label", "overall_decision", "rejected_exactly_2")]
    if "legibility_ok" in lab:
        parts.append("legible" if lab["legibility_ok"] else "illegible")
    return " / ".join(parts)


def fmt_pred(p: J | None, lab: J) -> str:
    if p is None:
        return "not run"
    parts = []
    for q in ("synthetic_label", "overall_decision", "rejected_exactly_2"):
        mark = " ✗" if q in lab and p[q] != lab[q] else ""
        parts.append(f"{yn(p[q])}{mark}")
    mark = " ✗" if "legibility_ok" in lab and p["legibility_ok"] != lab["legibility_ok"] else ""
    parts.append(f"{p['legibility_level']}{mark}")
    return " / ".join(parts)


def write_report(manifest: J, digest: str) -> None:
    calls = load_checkpoint(OUT / "calls.jsonl", digest)
    report = build_report(manifest, calls)
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (OUT / "report.md").write_text(render_markdown(report), encoding="utf-8")
    print(f"Report: {OUT / 'report.md'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true", help="confirm the live calls")
    ap.add_argument("--report", action="store_true", help="only rebuild the report from the checkpoint")
    args = ap.parse_args()
    manifest_bytes = (CASES / "manifest.json").read_bytes()
    manifest = load_manifest()
    digest = settings_hash(manifest_bytes)
    if not args.report:
        if not args.yes:
            print("Live calls need --yes.", file=sys.stderr)
            return 2
        started = time.perf_counter()
        code = asyncio.run(run_live(manifest, digest))
        print(f"Live section took {time.perf_counter() - started:.0f} s")
        if code != 0:
            return code
    write_report(manifest, digest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
