"""Three live smoke calls that confirm the request formats before the full multimodal test.

    uv run --extra gemini python evals/multimodal/smoke.py --yes

1. Gemini Decision 1 with one image (r1_original) and its audit records.
2. Gemini Decision 1 with the synthetic PDF (pdf_original) and its audit record.
3. Jev, image preview, with the same image and records.

Each call is made once, with no retries. The script stops at the first error, quota message, blocked input
or warning, and writes what it saw. Results go to results/multimodal/smoke/: the raw answers, the full usage
block and latency, never keys, headers or the encoded file.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import warnings
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx
from dotenv import load_dotenv
from pydantic import JsonValue

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jev_image import evaluate_jev_image  # noqa: E402
from questions import QUESTIONS  # noqa: E402

from paved_gate.evaluator.base import EvaluationResult, MediaPart  # noqa: E402
from paved_gate.evaluator.gemini_decision import GeminiDecisionEvaluator  # noqa: E402

CASES = ROOT / "results" / "multimodal" / "cases"
OUT = ROOT / "results" / "multimodal" / "smoke"
GEMINI_MODEL = "gemini-decision-1"
JEV_MODEL = "jev-1.14.0"
TIMEOUT_S = 15.0


class Captured(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(f"log {record.levelname} {record.name}: {record.getMessage()}")


def load_case(case_id: str) -> tuple[MediaPart, str, dict[str, JsonValue]]:
    manifest = json.loads((CASES / "manifest.json").read_text(encoding="utf-8"))
    case = next(c for c in manifest["cases"] if c["id"] == case_id)
    part = MediaPart(mime_type=case["mime_type"], data=(CASES / case["artifact"]).read_bytes(), name=case["artifact"])
    return part, manifest["record_texts"][case["text_key"]], case


def scrub(text: str, secrets: list[str]) -> str:
    for s in secrets:
        if s:
            text = text.replace(s, "[redacted]")
    return text


async def one(
    n: int, arm: str, case_id: str, call: Callable[[MediaPart, str], Awaitable[EvaluationResult]], secrets: list[str]
) -> bool:
    part, record_text, case = load_case(case_id)
    handler = Captured()
    logging.getLogger().addHandler(handler)
    out: dict[str, object] = {
        "call": n,
        "arm": arm,
        "case": case_id,
        "artifact": {"name": part.name, "mime_type": part.mime_type, "bytes": len(part.data), "sha256": case["sha256"]},
        "record_text_chars": len(record_text),
        "questions": list(QUESTIONS),
    }
    ok = True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = await call(part, record_text)
            out |= {
                "model": result.model,
                "latency_ms": result.latency_ms,
                "answers": result.raw.get("answers"),
                "usage": result.raw.get("usage"),
                "other_response_fields": {k: v for k, v in result.raw.items() if k not in ("answers", "usage")},
            }
        except Exception as exc:  # reported as it is; no retry
            out["error"] = scrub(f"{type(exc).__name__}: {exc}", secrets)
            ok = False
    logging.getLogger().removeHandler(handler)
    found = [f"{w.category.__name__}: {w.message}" for w in caught] + handler.messages
    out["warnings"] = [scrub(w, secrets) for w in found]
    feedback = out.get("other_response_fields", {})
    if found or (isinstance(feedback, dict) and feedback.get("prompt_feedback")):
        ok = False
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{n}_{arm}_{case_id}.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    return ok


async def main_async() -> int:
    load_dotenv(ROOT / ".env")  # keys from .env; real environment variables take precedence
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    jev_key = os.environ.get("TYPESAFE_API_KEY", "")
    if not gemini_key or not jev_key:
        print("GEMINI_API_KEY and TYPESAFE_API_KEY are both needed", file=sys.stderr)
        return 2
    secrets = [gemini_key, jev_key]
    gemini = GeminiDecisionEvaluator(
        mode="live", model=GEMINI_MODEL, api_key=gemini_key, timeout_s=TIMEOUT_S, max_retries=0
    )
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(TIMEOUT_S)) as client:
            calls: list[tuple[str, str, Callable[[MediaPart, str], Awaitable[EvaluationResult]]]] = [
                ("gemini_image", "r1_original", lambda p, t: gemini.evaluate_media(t, [p], QUESTIONS)),
                ("gemini_pdf", "pdf_original", lambda p, t: gemini.evaluate_media(t, [p], QUESTIONS)),
                (
                    "jev_image_preview",
                    "r1_original",
                    lambda p, t: evaluate_jev_image(
                        client, api_key=jev_key, model=JEV_MODEL, record_text=t, image=p, questions=QUESTIONS
                    ),
                ),
            ]
            for n, (arm, case_id, call) in enumerate(calls, 1):
                if not await one(n, arm, case_id, call, secrets):
                    print(f"Stopped after call {n} ({arm}): see results/multimodal/smoke/", file=sys.stderr)
                    return 1
    finally:
        await gemini.aclose()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yes", action="store_true", help="confirm the three live calls")
    args = ap.parse_args()
    if not args.yes:
        print("Live calls need --yes.", file=sys.stderr)
        return 2
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())
