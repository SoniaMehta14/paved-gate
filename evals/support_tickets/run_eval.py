"""Support-ticket intent classification eval: Jev vs GPT-4o vs GPT-4o mini.

    uv run python evals/support_tickets/run_eval.py                  # --estimate: cost estimate, NO API calls
    uv run python evals/support_tickets/run_eval.py --mock --open    # Jev mock, GPT arms not run, NO API calls
    uv run python evals/support_tickets/run_eval.py --live --yes     # real API calls (costs money)

Every ticket in the fixed sample (data/eval/, see build_sample.py) is classified into one
of the dataset's intents by each arm, using the same intent list and descriptions:
  - Jev: one typed Choice question over the intents
  - GPT-4o / GPT-4o mini: a JSON-schema-constrained classification prompt

Predictions are compared with the known labels (scikit-learn precision/recall/F1,
accuracy, confusion matrix). Accuracy is ONLY reported for live arms: mock and
not-run arms are shown as "not measured", never simulated or estimated.

Writes results/eval-<timestamp>.json, .html (dashboard) and .predictions.jsonl.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import webbrowser
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))  # reuse the benchmark's arm/cost/report code

from benchmark import OPENAI_NAMES, ArmJson, ArmResult, build_comparison  # noqa: E402
from build_sample import DEFAULT_SAMPLE, Ticket, load_sample  # noqa: E402
from dashboard import render_dashboard  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from intents import INTENTS, INVALID, labels  # noqa: E402
from metrics import Metrics, compute_metrics  # noqa: E402
from pricing import PRICES, cost_usd  # noqa: E402

from paved_gate import ChoiceQuestion, EvaluatorError, JevEvaluator  # noqa: E402
from paved_gate.evaluator.jev import JevAPIError  # noqa: E402

if TYPE_CHECKING:
    from openai.types.shared_params import ResponseFormatJSONSchema

SCHEMA_VERSION = 3
JEV_QUESTION = ChoiceQuestion(
    instructions="Which intent does this customer support message express? Choose exactly one.",
    criteria=dict(INTENTS),
)
# Rough output sizes for the estimate only: {"intent":"..."} for OpenAI; Jev output is free.
EST_OPENAI_OUTPUT_TOKENS = 12

Predict = Callable[[Ticket], Awaitable[tuple[str, int, int]]]  # (label or INVALID, input tokens, output tokens)


class ArmClassification(TypedDict):
    id: str
    label: str
    model: str
    mode: str
    measured: bool
    attempted: int
    api_errors: int
    metrics: Metrics | None
    note: str


@dataclass
class Prediction:
    ticket_id: str
    true: str
    pred: str | None  # None = API error (excluded from metrics)
    latency_ms: float | None


def system_prompt() -> str:
    listing = "\n".join(f"- {k}: {v}" for k, v in INTENTS.items())
    return (
        "You classify customer support messages. Choose exactly one intent from this list and reply "
        'with JSON: {"intent": "<intent>"}.\n\n' + listing
    )


def response_schema() -> ResponseFormatJSONSchema:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "intent_classification",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"intent": {"type": "string", "enum": labels()}},
                "required": ["intent"],
                "additionalProperties": False,
            },
        },
    }


def parse_openai_intent(content: str | None) -> str:
    """The model's label, or INVALID for a refusal, bad JSON, or a label outside the list."""
    if not content:
        return INVALID
    try:
        value = json.loads(content).get("intent")
    except (json.JSONDecodeError, AttributeError):
        return INVALID
    return value if isinstance(value, str) and value in INTENTS else INVALID


# ---------------------------------------------------------------- predictors


def jev_predictor(evaluator: JevEvaluator) -> Predict:
    async def predict(t: Ticket) -> tuple[str, int, int]:
        try:
            result = await evaluator.evaluate(t.text, {"intent": JEV_QUESTION})
        except JevAPIError:
            raise
        except EvaluatorError:
            return INVALID, 0, 0  # e.g. a choice outside the offered options
        answer = result.answers["intent"]
        choice = getattr(answer, "choice", INVALID)
        return (choice if choice in INTENTS else INVALID), result.input_tokens or 0, result.output_tokens or 0

    return predict


def openai_predictor(model: str) -> Predict:
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    system = system_prompt()
    schema = response_schema()

    async def predict(t: Ticket) -> tuple[str, int, int]:
        resp = await client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=50,
            response_format=schema,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": t.text}],
        )
        msg = resp.choices[0].message
        label = INVALID if msg.refusal else parse_openai_intent(msg.content)
        usage = resp.usage
        return label, (usage.prompt_tokens if usage else 0), (usage.completion_tokens if usage else 0)

    return predict


async def run_arm(
    arm: ArmResult, price_key: str, predict: Predict, tickets: list[Ticket], concurrency: int
) -> list[Prediction]:
    sem = asyncio.Semaphore(concurrency)

    async def one(t: Ticket) -> Prediction:
        async with sem:
            started = time.perf_counter()
            try:
                label, in_tok, out_tok = await predict(t)
            except Exception as exc:  # API/network failure: counted, excluded from metrics
                arm.errors += 1
                arm.note = f"last error: {type(exc).__name__}: {str(exc)[:80]}"
                return Prediction(t.id, t.intent, None, None)
            ms = (time.perf_counter() - started) * 1000
        arm.latencies_ms.append(ms)
        arm.input_tokens += in_tok
        arm.output_tokens += out_tok
        c = cost_usd(price_key, in_tok, out_tok)
        if c is not None:
            arm.costs_usd.append(c)
        return Prediction(t.id, t.intent, label, ms)

    return list(await asyncio.gather(*(one(t) for t in tickets)))


def classification_for(arm: ArmResult, preds: list[Prediction]) -> ArmClassification:
    scored = [p for p in preds if p.pred is not None]
    measured = arm.mode == "live" and bool(scored)
    metrics = (
        compute_metrics([p.true for p in scored], [p.pred for p in scored if p.pred is not None], labels(), INVALID)
        if measured
        else None
    )
    if arm.mode == "live":
        note = "" if scored else "no successful requests"
    elif arm.mode == "mock":
        note = "not measured: mock predictions are not real classifications"
    else:
        note = f"not measured ({arm.mode_detail})"
    return {
        "id": arm.id,
        "label": arm.label,
        "model": arm.model,
        "mode": arm.mode,
        "measured": measured,
        "attempted": len(preds),
        "api_errors": sum(1 for p in preds if p.pred is None),
        "metrics": metrics,
        "note": note,
    }


# ---------------------------------------------------------------- estimate


def estimate(tickets: list[Ticket], openai_models: list[str]) -> list[tuple[str, int, int, float]]:
    """(arm, requests, est. input tokens, est. USD) per arm, from characters/4. No API calls."""
    jev_q = json.dumps({"type": "choice", **JEV_QUESTION.model_dump(exclude={"kind"})})
    jev_in = sum((len(t.text) + len(jev_q)) // 4 for t in tickets)
    rows = [("Jev (jev-latest)", len(tickets), jev_in, cost_usd("jev", jev_in, 0) or 0.0)]
    sys_len = len(system_prompt())
    for m in openai_models:
        oa_in = sum((sys_len + len(t.text)) // 4 + 7 for t in tickets)  # +7 chat-format overhead
        out = EST_OPENAI_OUTPUT_TOKENS * len(tickets)
        rows.append((f"{OPENAI_NAMES.get(m, m)} ({m})", len(tickets), oa_in, cost_usd(m, oa_in, out) or 0.0))
    return rows


def print_estimate(rows: list[tuple[str, int, int, float]]) -> None:
    print("\nEstimated cost of a full LIVE run (no API calls were made):")
    for name, n, tok, usd in rows:
        print(f"  {name:32s} {n:>5} requests  ~{tok:>9,} input tokens  ~${usd:.4f}")
    print(f"  {'TOTAL':32s} {'':>5}           {'':>9}               ~${sum(r[3] for r in rows):.4f}")
    print("  Token counts are characters/4 approximations; expect the real figure within about ±30%.")


# ---------------------------------------------------------------- report


def build_report(
    arms: list[ArmResult],
    classes: list[ArmClassification],
    *,
    started_at: datetime,
    manifest: dict[str, object],
    concurrency: int,
) -> dict[str, object]:
    arm_json: list[ArmJson] = [a.to_json() for a in arms]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "classification",
        "title": "Support ticket classification: Jev vs zero-shot frontier models",
        "illustrative": True,
        "run_at": started_at.isoformat(timespec="seconds"),
        "corpus_size": manifest["n"],
        "iterations": 1,
        "policy_hash": f"sha256:{manifest['sample_sha256']}",
        "arms": arm_json,
        "comparison": build_comparison(arm_json),
        "classification": {
            "dataset": {
                k: manifest[k]
                for k in ("dataset", "source_url", "revision", "license", "attribution", "per_intent", "seed", "n")
            },
            "sample_sha256": manifest["sample_sha256"],
            "labels": labels(),
            "invalid_label": INVALID,
            "concurrency": concurrency,
            "arms": classes,
        },
        "prices_per_mtok": {
            m: {"input": p.input_per_mtok, "output": p.output_per_mtok, "source": p.source} for m, p in PRICES.items()
        },
        "caveats": [
            "Bitext tickets are synthetic with clean labels; real support traffic will usually score lower.",
            f"Latency was measured with {concurrency} requests in flight per arm; do not compare it directly "
            "with the sequential latency benchmark.",
            "Accuracy is reported only for live arms. Mock and not-run arms are 'not measured', never simulated.",
            "API errors are counted separately and excluded from accuracy; invalid model outputs count as wrong.",
            "Prices are list prices recorded in scripts/pricing.py with their verification date.",
        ],
    }


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--estimate", action="store_true", help="print the estimated live cost; no API calls (default)")
    mode.add_argument("--mock", action="store_true", help="Jev mock, GPT arms not run; no API calls")
    mode.add_argument("--live", action="store_true", help="real API calls for every arm with a key")
    ap.add_argument("--yes", action="store_true", help="confirm a live run (required with --live)")
    ap.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    ap.add_argument("--openai-model", action="append", help="repeatable (default: gpt-4o and gpt-4o-mini)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--open", action="store_true", help="open the HTML dashboard when done")
    args = ap.parse_args()
    load_dotenv(ROOT / ".env")  # keys from the gitignored .env

    tickets, manifest = load_sample(args.sample)
    openai_models = list(dict.fromkeys(args.openai_model or ["gpt-4o", "gpt-4o-mini"]))
    print(f"Sample: {args.sample.name}  {len(tickets)} tickets, {manifest['intents']} intents, sha256 ok")

    if not (args.mock or args.live):
        print_estimate(estimate(tickets, openai_models))
        print("\nTo run live: add --live --yes")
        return 0
    if args.live and not args.yes:
        print_estimate(estimate(tickets, openai_models))
        print("\nRefusing to make live API calls without --yes.")
        return 2

    started_at = datetime.now(UTC)
    arms: list[ArmResult] = []
    runs: list[tuple[ArmResult, list[Prediction]]] = []

    # Jev arm
    if args.mock:
        jev = JevEvaluator(mode="mock", mock_latency_ms=(60.0, 120.0))
        arm = ArmResult(
            "jev",
            "Jev (typed Choice)",
            "jev-latest",
            "mock",
            "simulated 60-120 ms latency; mock picks are not real",
            "estimated tokens x list price (mock)",
        )
    elif os.environ.get("TYPESAFE_API_KEY"):
        jev = JevEvaluator(mode="live", api_key=os.environ["TYPESAFE_API_KEY"], timeout_s=15.0, max_retries=2)
        arm = ArmResult(
            "jev",
            "Jev (typed Choice)",
            "jev-latest",
            "live",
            "real Jev API, one Choice question per ticket",
            "reported tokens x list price",
        )
    else:
        jev = None
        arm = ArmResult("jev", "Jev (typed Choice)", "jev-latest", "not_run", "TYPESAFE_API_KEY not set")
    arms.append(arm)
    if jev is not None:
        print(f"Running {arm.label} ({arm.mode}) on {len(tickets)} tickets...")
        runs.append((arm, await run_arm(arm, "jev", jev_predictor(jev), tickets, args.concurrency)))
        await jev.aclose()
    else:
        runs.append((arm, []))

    # OpenAI arms
    for i, m in enumerate(openai_models):
        arm_id = "openai" if i == 0 else f"openai-{m}"
        label = f"{OPENAI_NAMES.get(m, m)} zero-shot"
        if args.mock:
            arm = ArmResult(arm_id, label, m, "not_run", "skipped: this run made no API calls")
        elif not os.environ.get("OPENAI_API_KEY"):
            arm = ArmResult(arm_id, label, m, "not_run", "OPENAI_API_KEY not set")
        else:
            arm = ArmResult(
                arm_id, label, m, "live", "real API, JSON-schema classification prompt", "reported tokens x list price"
            )
        arms.append(arm)
        if arm.mode == "live":
            print(f"Running {label} (live) on {len(tickets)} tickets...")
            runs.append((arm, await run_arm(arm, m, openai_predictor(m), tickets, args.concurrency)))
        else:
            runs.append((arm, []))

    classes = [classification_for(a, p) for a, p in runs]
    report = build_report(arms, classes, started_at=started_at, manifest=manifest, concurrency=args.concurrency)

    print()
    for c, a in zip(classes, arms, strict=True):
        cost = a.to_json()["cost_usd"]["run_total"]
        m = c["metrics"]
        acc = f"accuracy {m['accuracy']:.1%} (n={m['n']}), macro F1 {m['macro']['f1']:.3f}" if m else c["note"]
        run_cost = "-" if cost is None else f"${cost:.6f}"
        print(f"  {c['label']:28s} {c['mode']:8s} {acc}  run cost {run_cost}  api errors {c['api_errors']}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"eval-{started_at.strftime('%Y%m%d-%H%M%S')}"
    json_path, html_path = args.out_dir / f"{stem}.json", args.out_dir / f"{stem}.html"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    html_path.write_text(render_dashboard(report), encoding="utf-8")
    with (args.out_dir / f"{stem}.predictions.jsonl").open("w", encoding="utf-8") as f:
        for arm_result, preds in runs:
            for p in preds:
                f.write(json.dumps({"arm": arm_result.id, "mode": arm_result.mode, **p.__dict__}) + "\n")
    print(f"\nResults:   {json_path}\nDashboard: {html_path}")
    if args.open:
        webbrowser.open(html_path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
