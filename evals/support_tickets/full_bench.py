"""Full Bitext benchmark: Jev vs OpenAI baselines (zero-shot and few-shot), three runs each.

    uv run python evals/support_tickets/full_bench.py --estimate             # projected cost; no API calls
    uv run python evals/support_tickets/full_bench.py --smoke                # 50 rows per config (paid)
    uv run python evals/support_tickets/full_bench.py --run                  # the plan below (paid, resumable)
    uv run python evals/support_tickets/full_bench.py --finalize             # CSV + markdown from the raw files

Every configuration is a FastEvaluator (JevEvaluator or OpenAIEvaluator) answering the same ChoiceQuestion: the 27
intents with their one-line descriptions (intents.py). Few-shot configurations add one labelled exemplar per
intent; those 27 tickets are removed from evaluation for every configuration.

Scope (tiered to stay under the budget): Jev and GPT-4o mini on the full evaluation set; GPT-4o on a stratified
subset of 100 per intent; the flagship on a smaller nested stratified subset sized from the smoke test. Every
configuration is also scored on the shared subsets, so the headline table compares like with like.

Raw outputs are appended per call to results/.../raw/<config>/run<k>.jsonl, so a crash or restart loses nothing that
was paid for; --run skips rows already done. A budget guard stops before the total spend would pass the cap.
API keys come from the environment (loaded from the gitignored .env); they are never written anywhere.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import gzip
import json
import math
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from build_sample import (  # noqa: E402  # noqa: E402
    CSV_SHA256,
    DATASET,
    DEFAULT_CACHE,
    LICENSE,
    REVISION,
    SOURCE_URL,
    Ticket,
    ensure_dataset,
    read_rows,
    stratified_sample,
)
from intents import INTENTS, INVALID, labels  # noqa: E402
from metrics import compute_metrics  # noqa: E402
from pricing import PRICES, cost_usd  # noqa: E402

from paved_gate import ChoiceQuestion, EvaluatorError, FastEvaluator, JevEvaluator, OpenAIEvaluator  # noqa: E402
from paved_gate.evaluator.jev import JevAPIError  # noqa: E402
from paved_gate.evaluator.openai_eval import InvalidChoiceError, OpenAIAPIError  # noqa: E402

OUT = ROOT / "evals" / "support_tickets" / "results" / "bitext-full-2026-10-15"
SEED = 20261015
SUBSET_PER_INTENT = 100
DEFAULT_FLAGSHIP_PER_INTENT = 50
SMOKE_ROWS = 50
RUNS = 3
BUDGET_USD = 120.0
# Records without "runner": 2 (written 2026-10-16T02:56Z to 02:58Z) have latencies inflated by a starved event
# loop; their predictions and token usage are valid. Latency is reported from runner 2 records only.
RUNNER_VERSION = 2
# Latency excludes each run's first minute: the runner reads its checkpoint files while the first calls are in
# flight, which inflated tail latency slightly (Jev run 2 p95: 259 ms in the first minute, 208 ms afterwards).
WARMUP_S = 60.0
QUESTION = ChoiceQuestion(
    instructions="Which intent does this customer support message express? Choose exactly one.",
    criteria=dict(INTENTS),
)


# ------------------------------------------------------------------ data


@dataclass(frozen=True)
class Data:
    rows: list[Ticket]
    exemplars: list[Ticket]
    eval_set: list[Ticket]  # all rows minus exemplars, subset rows first
    subset: list[Ticket]
    flagship: list[Ticket]
    smoke: list[Ticket]


def build_data(flagship_per_intent: int) -> Data:
    rows = read_rows(ensure_dataset(DEFAULT_CACHE))
    rng = random.Random(SEED)
    by_intent: dict[str, list[Ticket]] = defaultdict(list)
    for t in rows:
        by_intent[t.intent].append(t)
    exemplars = [rng.choice(sorted(by_intent[i], key=lambda t: t.row_index)) for i in sorted(by_intent)]
    taken = {t.row_index for t in exemplars}
    pool = [t for t in rows if t.row_index not in taken]
    subset = stratified_sample(pool, SUBSET_PER_INTENT, SEED)
    flagship = stratified_sample(subset, flagship_per_intent, SEED + 1)
    smoke = sorted(random.Random(SEED + 2).sample(subset, SMOKE_ROWS), key=lambda t: t.row_index)
    in_subset = {t.row_index for t in subset}
    eval_set = subset + [t for t in pool if t.row_index not in in_subset]  # subset rows first: useful early
    return Data(rows, exemplars, eval_set, subset, flagship, smoke)


# ------------------------------------------------------------------ configurations


@dataclass(frozen=True)
class Config:
    id: str
    label: str
    model: str
    prompt: str  # "zero-shot" | "few-shot" | "typed choice"
    scope: str  # "full" | "subset" | "flagship"
    price_key: str
    temperature: str
    concurrency: int
    make: Callable[[list[Ticket]], FastEvaluator]


def openai_config(
    cid: str, label: str, model: str, prompt: str, scope: str, concurrency: int, reasoning: bool = False
) -> Config:
    def make(exemplars: list[Ticket]) -> FastEvaluator:
        return OpenAIEvaluator(
            model=model,
            temperature=None if reasoning else 0.0,
            reasoning_effort="low" if reasoning else None,
            max_output_tokens=2000 if reasoning else 50,
            examples=[(t.text, t.intent) for t in exemplars] if prompt == "few-shot" else (),
            prompt_cache_key=f"paved-gate-bitext-{cid}" if prompt == "few-shot" else None,
        )

    return Config(
        cid,
        label,
        model,
        prompt,
        scope,
        model,
        "default (1); model rejects 0; reasoning_effort=low" if reasoning else "0",
        concurrency,
        make,
    )


def jev_make(_: list[Ticket]) -> FastEvaluator:
    key = os.environ.get("TYPESAFE_API_KEY")
    return JevEvaluator(mode="live", api_key=key, timeout_s=15.0, max_retries=4)


CONFIGS: list[Config] = [
    Config("jev", "Jev", "jev-latest", "typed choice", "full", "jev", "n/a", 8, jev_make),
    openai_config("gpt-4o-mini-zero", "GPT-4o mini zero-shot", "gpt-4o-mini", "zero-shot", "full", 32),
    openai_config("gpt-4o-mini-few", "GPT-4o mini few-shot", "gpt-4o-mini", "few-shot", "full", 24),
    openai_config("gpt-4o-zero", "GPT-4o zero-shot", "gpt-4o", "zero-shot", "subset", 10),
    openai_config("gpt-4o-few", "GPT-4o few-shot", "gpt-4o", "few-shot", "subset", 6),
    openai_config("gpt-6-astra-zero", "GPT-6 Astra zero-shot", "gpt-6-astra", "zero-shot", "flagship", 12, True),
    openai_config("gpt-6-astra-few", "GPT-6 Astra few-shot", "gpt-6-astra", "few-shot", "flagship", 12, True),
]
BY_ID = {c.id: c for c in CONFIGS}


def scope_rows(cfg: Config, data: Data) -> list[Ticket]:
    return {"full": data.eval_set, "subset": data.subset, "flagship": data.flagship}[cfg.scope]


# ------------------------------------------------------------------ raw records and checkpoints


def raw_path(cfg_id: str, run: str) -> Path:
    return OUT / "raw" / cfg_id / f"{run}.jsonl"


def _parse_lines(lines: Iterable[str]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a line cut short by a killed process; the row is simply redone
    return out


def read_records(path: Path) -> list[dict[str, object]]:
    if path.exists():
        with path.open(encoding="utf-8") as f:
            return _parse_lines(f)
    gz = path.with_suffix(".jsonl.gz")
    if gz.exists():
        with gzip.open(gz, "rt", encoding="utf-8") as f:
            return _parse_lines(f)
    return []


def latest_by_row(records: Iterable[dict[str, object]]) -> dict[int, dict[str, object]]:
    """The last record per row: a retried row (after an API error) replaces its error record."""
    out: dict[int, dict[str, object]] = {}
    for r in records:
        out[int(str(r["row"]))] = r
    return out


def record_cost(r: dict[str, object], price_key: str) -> float:
    return (
        cost_usd(
            price_key,
            int(str(r.get("input_tokens") or 0)),
            int(str(r.get("output_tokens") or 0)),
            int(str(r.get("cached_tokens") or 0)),
        )
        or 0.0
    )


def spent_so_far() -> float:
    total = 0.0
    for cfg in CONFIGS:
        folder = OUT / "raw" / cfg.id
        runs = {path.name.split(".")[0] for path in folder.glob("*.jsonl*")} if folder.exists() else set()
        for run in runs:  # read_records prefers .jsonl and falls back to .jsonl.gz: each run counted once
            total += sum(record_cost(r, cfg.price_key) for r in read_records(raw_path(cfg.id, run)))
        total += sum(record_cost(r, cfg.price_key) for r in read_records(OUT / "smoke" / f"{cfg.id}.jsonl"))
    return total


# ------------------------------------------------------------------ execution


class Budget:
    def __init__(self, cap: float, spent: float) -> None:
        self.cap, self.spent = cap, spent
        self.exceeded = asyncio.Event()

    def add(self, usd: float) -> None:
        self.spent += usd
        if self.spent >= self.cap:
            self.exceeded.set()


class Fatal(Exception):
    pass


class AdaptiveLimit:
    """How many of a configuration's workers may be busy. Halves when many calls needed retries (rate limiting)
    and creeps back up after clean stretches. Workers above the limit idle; nothing is woken in bulk."""

    def __init__(self, start: int) -> None:
        self.max = start
        self.limit = start
        self.recent: list[bool] = []
        self.last_cut = 0.0

    def observe(self, retried: bool) -> None:
        self.recent = [*self.recent, retried][-100:]
        now = time.monotonic()
        if len(self.recent) >= 50 and sum(self.recent) / len(self.recent) > 0.2 and now - self.last_cut > 30:
            self.limit = max(1, self.limit // 2)
            self.last_cut = now
            self.recent = []
        elif len(self.recent) >= 100 and not any(self.recent) and self.limit < self.max:
            self.limit += 1
            self.recent = []


async def watch_loop_lag(stop: asyncio.Event) -> None:
    """Warn if the event loop falls behind: latency measured in a starved loop includes our own queueing."""
    worst = 0.0
    last_report = time.monotonic()
    while not stop.is_set():
        t = time.perf_counter()
        await asyncio.sleep(0.05)
        worst = max(worst, (time.perf_counter() - t - 0.05) * 1000)
        if time.monotonic() - last_report > 60:
            flag = "  WARNING: latency would be inflated" if worst > 50 else ""
            print(f"    event loop lag (max over the last minute): {worst:.1f} ms{flag}", flush=True)
            worst, last_report = 0.0, time.monotonic()


async def classify(evaluator: FastEvaluator, t: Ticket) -> dict[str, object]:
    """One call. Returns the record to append (never raises for per-row failures)."""
    rec: dict[str, object] = {"row": t.row_index, "id": t.id, "true": t.intent, "ts": round(time.time(), 3)}
    started = time.perf_counter()
    try:
        result = await evaluator.evaluate(t.text, {"intent": QUESTION})
        answer = result.answers["intent"]
        rec["pred"] = getattr(answer, "choice", INVALID)
        rec["latency_ms"] = result.latency_ms
        rec["input_tokens"] = result.input_tokens or 0
        rec["output_tokens"] = result.output_tokens or 0
        raw = result.raw
        usage = raw.get("usage") if isinstance(raw, dict) else None
        if isinstance(usage, dict):
            rec["cached_tokens"] = usage.get("cached_tokens", 0)
            rec["reasoning_tokens"] = usage.get("reasoning_tokens", 0)
        if result.evaluator == "openai":
            rec.update({k: raw.get(k) for k in ("content", "refusal", "finish_reason", "attempts", "total_ms")})
            rec["model"] = result.model
            rec["system_fingerprint"] = raw.get("system_fingerprint")
        else:
            rec["content"] = json.dumps({"choice": rec["pred"], "confidence": getattr(answer, "confidence", None)})
            rec["model"] = result.model
            rec["total_ms"] = round((time.perf_counter() - started) * 1000, 2)
    except InvalidChoiceError as exc:  # billed, but not one of the 27 labels: counts as wrong
        rec.update(pred=INVALID, error=str(exc)[:200], input_tokens=exc.input_tokens, output_tokens=exc.output_tokens)
        rec.update({k: exc.raw.get(k) for k in ("content", "refusal", "finish_reason", "attempts", "total_ms")})
        usage = exc.raw.get("usage")
        if isinstance(usage, dict):
            rec["cached_tokens"] = usage.get("cached_tokens", 0)
            rec["reasoning_tokens"] = usage.get("reasoning_tokens", 0)
    except OpenAIAPIError as exc:
        if exc.fatal:
            raise Fatal(f"{exc} ({exc.code})") from exc
        rec.update(pred=None, error=f"{type(exc).__name__}: {str(exc)[:200]}")
    except JevAPIError as exc:
        rec.update(pred=None, error=f"{type(exc).__name__}: {str(exc)[:200]}")
        if exc.status in (401, 402, 403):
            raise Fatal(str(exc)) from exc
    except EvaluatorError as exc:  # Jev returned a choice outside the options, or a malformed answer
        rec.update(pred=INVALID, error=f"{type(exc).__name__}: {str(exc)[:200]}")
    return rec


async def run_task(
    cfg: Config, run: str, tickets: Sequence[Ticket], data: Data, budget: Budget, path: Path, progress: bool = True
) -> None:
    done = {row for row, r in latest_by_row(read_records(path)).items() if r.get("pred") is not None}
    todo = [t for t in tickets if t.row_index not in done]
    if not todo:
        print(f"  {cfg.id} {run}: already complete ({len(tickets)} rows)", flush=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    evaluator = cfg.make(data.exemplars)
    limit = AdaptiveLimit(cfg.concurrency)
    queue: asyncio.Queue[Ticket] = asyncio.Queue()
    for t in todo:
        queue.put_nowait(t)
    n_done = 0
    started = time.monotonic()
    print(f"  {cfg.id} {run}: {len(todo)} rows to do ({len(tickets) - len(todo)} already done)", flush=True)
    with path.open("a", encoding="utf-8") as f:

        async def worker(i: int) -> None:
            nonlocal n_done
            while not budget.exceeded.is_set():
                if i >= limit.limit:  # throttled: this worker sits out until the limit recovers
                    if queue.empty():
                        return
                    await asyncio.sleep(1.0)
                    continue
                try:
                    t = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                rec = await classify(evaluator, t)
                rec["runner"] = RUNNER_VERSION
                limit.observe(int(str(rec.get("attempts") or 1)) > 1 or rec.get("pred") is None)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                budget.add(record_cost(rec, cfg.price_key))
                n_done += 1
                if progress and n_done % 1000 == 0:
                    rate = n_done / max(1e-6, time.monotonic() - started)
                    print(
                        f"    {cfg.id} {run}: {n_done}/{len(todo)}  {rate:.1f}/s  limit {limit.limit}  "
                        f"spent ${budget.spent:.2f}",
                        flush=True,
                    )

        try:
            await asyncio.gather(*(worker(i) for i in range(cfg.concurrency)))
        finally:
            await evaluator.aclose()
    print(
        f"  {cfg.id} {run}: finished {n_done} rows in {time.monotonic() - started:.0f}s; total spent "
        f"${budget.spent:.2f}",
        flush=True,
    )


def per_call_cost(cfg: Config) -> float | None:
    recs = [r for r in read_records(OUT / "smoke" / f"{cfg.id}.jsonl") if r.get("pred") is not None]
    for run in ("run1", "run2", "run3"):
        recs += [r for r in read_records(raw_path(cfg.id, run)) if r.get("pred") is not None]
    if not recs:
        return None
    return sum(record_cost(r, cfg.price_key) for r in recs) / len(recs)


def remaining_rows(cfg: Config, run: str, tickets: Sequence[Ticket]) -> int:
    done = {row for row, r in latest_by_row(read_records(raw_path(cfg.id, run))).items() if r.get("pred") is not None}
    return sum(1 for t in tickets if t.row_index not in done)


def estimate_cost(cfg: Config, rows: int) -> float:
    per = per_call_cost(cfg)
    if per is None:  # before the smoke test: characters/4 tokens, no cache discount
        prefix = len(json.dumps(QUESTION.model_dump())) // 4 + 60
        few = 27 * 30 if cfg.prompt == "few-shot" else 0
        per = cost_usd(cfg.price_key, prefix + few + 20, 10) or 0.0
    return per * rows


# ------------------------------------------------------------------ modes


def print_plan(data: Data, flagship_per_intent: int) -> float:
    print(
        f"Dataset: {DATASET} ({len(data.rows)} rows), exemplars {len(data.exemplars)}, eval set "
        f"{len(data.eval_set)}, subset {len(data.subset)} ({SUBSET_PER_INTENT}/intent), flagship subset "
        f"{len(data.flagship)} ({flagship_per_intent}/intent)"
    )
    total = 0.0
    for cfg in CONFIGS:
        rows = scope_rows(cfg, data)
        left = sum(remaining_rows(cfg, f"run{k}", rows) for k in range(1, RUNS + 1))
        usd = estimate_cost(cfg, left)
        per = per_call_cost(cfg)
        basis = f"measured ${per * 1000:.3f}/1k" if per is not None else "chars/4 estimate"
        print(
            f"  {cfg.id:20s} {cfg.scope:8s} {len(rows):>6} rows x {RUNS}  remaining {left:>6}  ~${usd:7.2f}  ({basis})"
        )
        total += usd
    spent = spent_so_far()
    print(
        f"  spent so far ${spent:.2f}; projected remaining ${total:.2f}; projected total ${spent + total:.2f} "
        f"(cap ${BUDGET_USD:.0f})"
    )
    return total


async def smoke(data: Data) -> None:
    budget = Budget(BUDGET_USD, spent_so_far())
    await asyncio.gather(
        *(run_task(c, "smoke", data.smoke, data, budget, OUT / "smoke" / f"{c.id}.jsonl", False) for c in CONFIGS)
    )
    print("\nSmoke results (50 rows each):")
    for cfg in CONFIGS:
        recs = list(latest_by_row(read_records(OUT / "smoke" / f"{cfg.id}.jsonl")).values())
        ok = [r for r in recs if r.get("pred") is not None]
        valid = sum(1 for r in ok if r["pred"] != INVALID)
        correct = sum(1 for r in ok if r["pred"] == r["true"])
        lat = sorted(float(str(r["latency_ms"])) for r in ok if r.get("latency_ms") is not None)
        first_error = next((r.get("error") for r in recs if r.get("pred") is None or r.get("error")), None)

        def mean(k: str, rows: list[dict[str, object]] = ok) -> float:
            return statistics.fmean(float(str(r.get(k) or 0)) for r in rows) if rows else 0.0

        print(
            f"  {cfg.id:20s} ok {len(ok)}/{len(recs)}  valid {valid}  correct {correct}  in {mean('input_tokens'):.0f} "
            f"cached {mean('cached_tokens'):.0f} out {mean('output_tokens'):.1f} reasoning "
            f"{mean('reasoning_tokens'):.1f}  p50 {lat[len(lat) // 2] if lat else 0:.0f} ms  "
            f"${(per_call_cost(cfg) or 0) * 1000:.3f}/1k  first error {first_error}"
        )


async def run_plan(data: Data) -> None:
    budget = Budget(BUDGET_USD - 1.0, spent_so_far())  # $1 margin for in-flight calls
    stop = asyncio.Event()
    lag = asyncio.create_task(watch_loop_lag(stop))
    try:
        await _run_plan(data, budget)
    finally:
        stop.set()
        await lag


async def _run_plan(data: Data, budget: Budget) -> None:
    for k in range(1, RUNS + 1):
        run = f"run{k}"
        tasks = []
        committed = 0.0  # projected cost of the configurations already launched for this run
        for cfg in CONFIGS:
            rows = scope_rows(cfg, data)
            projected = estimate_cost(cfg, remaining_rows(cfg, run, rows))
            if budget.spent + committed + projected > budget.cap:
                print(
                    f"  SKIP {cfg.id} {run}: projected ${projected:.2f} would pass the ${BUDGET_USD:.0f} cap "
                    f"(spent ${budget.spent:.2f})",
                    flush=True,
                )
                continue
            committed += projected
            tasks.append(run_task(cfg, run, rows, data, budget, raw_path(cfg.id, run)))
        print(f"== {run}: {len(tasks)} configurations, spent so far ${budget.spent:.2f}", flush=True)
        await asyncio.gather(*tasks)
        if budget.exceeded.is_set():
            print(f"Budget cap reached (${budget.spent:.2f}); stopping.", flush=True)
            return


# ------------------------------------------------------------------ finalize


def pct(values: Sequence[float], q: int) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 1)
    return round(statistics.quantiles(values, n=100, method="inclusive")[q - 1], 1)


def scored(cfg: Config, run: str, rows: Sequence[Ticket]) -> tuple[list[dict[str, object]], int]:
    """Records for `rows` (latest per row) that have a prediction, and the number of API errors left."""
    by_row = latest_by_row(read_records(raw_path(cfg.id, run)))
    recs = [by_row[t.row_index] for t in rows if t.row_index in by_row]
    ok = [r for r in recs if r.get("pred") is not None]
    return ok, len(recs) - len(ok)


def mcnemar(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> dict[str, float | int]:
    """Exact two-sided McNemar test on paired outcomes (binomial on the discordant pairs)."""
    b = sum(1 for x, y in zip(a_correct, b_correct, strict=True) if x and not y)
    c = sum(1 for x, y in zip(a_correct, b_correct, strict=True) if y and not x)
    n = b + c
    if n == 0:
        return {"a_only": b, "b_only": c, "p_value": 1.0}
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return {"a_only": b, "b_only": c, "p_value": min(1.0, round(p, 6))}


def run_start(cfg: Config, run: str) -> float:
    """When the runner 2 records of a run began (the latency warm-up window is measured from here)."""
    ts = [
        float(str(r["ts"]))
        for r in read_records(raw_path(cfg.id, run))
        if int(str(r.get("runner") or 1)) >= RUNNER_VERSION
    ]
    return min(ts) if ts else math.inf


def finalize(data: Data, flagship_per_intent: int) -> None:
    scopes = {"full": data.eval_set, "subset": data.subset, "flagship": data.flagship}
    scope_order = ["full", "subset", "flagship"]
    summary: list[dict[str, object]] = []
    per_class: list[dict[str, object]] = []
    preds: dict[tuple[str, str, str], dict[int, str]] = {}
    pooled: dict[str, list[float]] = defaultdict(list)  # latency over all runs, the configuration's own rows
    for cfg in CONFIGS:
        allowed = scope_order[scope_order.index(cfg.scope) :]
        for k in range(1, RUNS + 1):
            run = f"run{k}"
            start = run_start(cfg, run)
            for scope in allowed:
                rows = scopes[scope]
                ok, errors = scored(cfg, run, rows)
                if len(ok) < len(rows):
                    if scope == cfg.scope and ok:
                        print(f"  note: {cfg.id} {run} {scope} incomplete ({len(ok)}/{len(rows)})")
                    if len(ok) < len(rows):
                        continue  # only report complete runs
                y_true = [str(r["true"]) for r in ok]
                y_pred = [str(r["pred"]) for r in ok]
                m = compute_metrics(y_true, y_pred, labels(), INVALID)
                lat = [
                    float(str(r["latency_ms"]))
                    for r in ok
                    if r.get("latency_ms") is not None
                    and int(str(r.get("runner") or 1)) >= RUNNER_VERSION
                    and float(str(r["ts"])) - start >= WARMUP_S
                ]
                if scope == cfg.scope:
                    pooled[cfg.id] += lat
                cost = sum(record_cost(r, cfg.price_key) for r in ok)
                preds[(cfg.id, run, scope)] = {int(str(r["row"])): str(r["pred"]) for r in ok}
                summary.append(
                    {
                        "config": cfg.id,
                        "label": cfg.label,
                        "model": cfg.model,
                        "prompt": cfg.prompt,
                        "temperature": cfg.temperature,
                        "run": k,
                        "scope": scope,
                        "n": m["n"],
                        "accuracy": m["accuracy"],
                        "macro_precision": m["macro"]["precision"],
                        "macro_recall": m["macro"]["recall"],
                        "macro_f1": m["macro"]["f1"],
                        "weighted_precision": m["weighted"]["precision"],
                        "weighted_recall": m["weighted"]["recall"],
                        "weighted_f1": m["weighted"]["f1"],
                        "invalid": m["invalid_predictions"],
                        "api_errors": errors,
                        "latency_p50_ms": pct(lat, 50),
                        "latency_p95_ms": pct(lat, 95),
                        "latency_n": len(lat),
                        "cost_usd": round(cost, 4),
                        "cost_per_1k_usd": round(cost / len(ok) * 1000, 4),
                        "input_tokens": sum(int(str(r.get("input_tokens") or 0)) for r in ok),
                        "cached_tokens": sum(int(str(r.get("cached_tokens") or 0)) for r in ok),
                        "output_tokens": sum(int(str(r.get("output_tokens") or 0)) for r in ok),
                        "concurrency": cfg.concurrency,
                    }
                )
                for pl in m["per_label"]:
                    per_class.append({"config": cfg.id, "run": k, "scope": scope, **pl})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    with (OUT / "per_class.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_class[0]))
        w.writeheader()
        w.writerows(per_class)

    stats = aggregate(summary)
    latency = latency_table(summary, pooled)
    by_cfg = {str(row["config"]): row for row in latency}
    for st in stats:  # latency belongs to the configuration, not to the row set it is scored on
        row = by_cfg[str(st["config"])]
        st["latency_p50_ms"], st["latency_p95_ms"] = row["p50_ms"], row["p95_ms"]
    agreement = run_agreement(preds)
    tests = paired_tests(preds, data)
    spent = spent_so_far()
    manifest = {
        "dataset": DATASET,
        "source_url": SOURCE_URL,
        "revision": REVISION,
        "file_sha256": CSV_SHA256,
        "license": LICENSE,
        "seed": SEED,
        "rows": len(data.rows),
        "exemplar_rows": [t.row_index for t in data.exemplars],
        "subset_rows": sorted(t.row_index for t in data.subset),
        "flagship_rows": sorted(t.row_index for t in data.flagship),
        "subset_per_intent": SUBSET_PER_INTENT,
        "flagship_per_intent": flagship_per_intent,
        "configs": [
            {k: getattr(c, k) for k in ("id", "label", "model", "prompt", "scope", "temperature", "concurrency")}
            for c in CONFIGS
        ],
        "prices_per_mtok": {
            k: {
                "input": p.input_per_mtok,
                "cached_input": p.cached_input_per_mtok,
                "output": p.output_per_mtok,
                "source": p.source,
            }
            for k, p in PRICES.items()
            if k in {c.price_key for c in CONFIGS}
        },
        "question": QUESTION.model_dump(),
        "total_spend_usd": round(spent, 2),
        "finalized_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    (OUT / "results.json").write_text(
        json.dumps(
            {
                "aggregate": stats,
                "latency": latency,
                "run_agreement": agreement,
                "paired_tests": tests,
                "summary": summary,
            },
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    (OUT / "report.md").write_text(
        render_report(stats, agreement, tests, manifest, data, per_class, latency, summary), encoding="utf-8"
    )
    print(f"wrote {OUT.relative_to(ROOT)}/summary.csv, per_class.csv, results.json, manifest.json, report.md")


def latency_table(summary: list[dict[str, object]], pooled: dict[str, list[float]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for cfg in CONFIGS:
        runs = [s for s in summary if s["config"] == cfg.id and s["scope"] == cfg.scope]
        p95s = [float(str(s["latency_p95_ms"])) for s in runs if s["latency_p95_ms"] is not None]
        out.append(
            {
                "config": cfg.id,
                "scope": cfg.scope,
                "n": len(pooled[cfg.id]),
                "p50_ms": pct(pooled[cfg.id], 50),
                "p95_ms": pct(pooled[cfg.id], 95),
                "run_p95_min_ms": min(p95s) if p95s else None,
                "run_p95_max_ms": max(p95s) if p95s else None,
            }
        )
    return out


def aggregate(summary: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for s in summary:
        groups[(str(s["config"]), str(s["scope"]))].append(s)
    out = []
    for (cid, scope), rows in groups.items():
        entry: dict[str, object] = {
            "config": cid,
            "label": BY_ID[cid].label,
            "scope": scope,
            "runs": len(rows),
            "n": rows[0]["n"],
        }
        for k in (
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "latency_p50_ms",
            "latency_p95_ms",
            "cost_per_1k_usd",
        ):
            vals = [float(str(r[k])) for r in rows if r[k] is not None]
            entry[k] = round(statistics.fmean(vals), 4) if vals else None
            entry[k + "_sd"] = round(statistics.stdev(vals), 4) if len(vals) > 1 else None
            entry[k + "_min"] = round(min(vals), 4) if vals else None
            entry[k + "_max"] = round(max(vals), 4) if vals else None
        entry["invalid_total"] = sum(int(str(r["invalid"])) for r in rows)
        out.append(entry)
    return out


def run_agreement(preds: dict[tuple[str, str, str], dict[int, str]]) -> list[dict[str, object]]:
    out = []
    for cfg in CONFIGS:
        runs = [preds.get((cfg.id, f"run{k}", cfg.scope)) for k in range(1, RUNS + 1)]
        have = [r for r in runs if r]
        if len(have) < 2:
            continue
        rows = set.intersection(*(set(r) for r in have))
        same = sum(1 for row in rows if len({r[row] for r in have}) == 1)
        out.append(
            {
                "config": cfg.id,
                "scope": cfg.scope,
                "runs": len(have),
                "rows": len(rows),
                "identical_across_runs": same,
                "rate": round(same / len(rows), 4) if rows else None,
            }
        )
    return out


def paired_tests(preds: dict[tuple[str, str, str], dict[int, str]], data: Data) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    truth = {t.row_index: t.intent for t in data.rows}
    for scope in ("subset", "flagship"):
        jev = preds.get(("jev", "run1", scope))
        if not jev:
            continue
        for cfg in CONFIGS:
            if cfg.id == "jev":
                continue
            other = preds.get((cfg.id, "run1", scope))
            if not other:
                continue
            rows = sorted(set(jev) & set(other))
            t = mcnemar([jev[r] == truth[r] for r in rows], [other[r] == truth[r] for r in rows])
            out.append(
                {
                    "scope": scope,
                    "jev_vs": cfg.id,
                    "n": len(rows),
                    "jev_only_correct": t["a_only"],
                    "other_only_correct": t["b_only"],
                    "p_value": t["p_value"],
                }
            )
    return out


def fmt_p(v: object) -> str:
    x = float(str(v))
    return "< 0.000001" if x < 1e-6 else f"{x:.6f}".rstrip("0").rstrip(".") if x < 0.001 else f"{x:.3f}"


def fmt(v: object, pct_: bool = False, sd: object = None) -> str:
    if v is None:
        return "n/a"
    x = float(str(v))
    s = f"{x * 100:.2f}%" if pct_ else (f"{x:.0f}" if x >= 100 else f"{x:.1f}")
    if sd is not None:
        s += f" ± {float(str(sd)) * 100:.2f}" if pct_ else f" ± {float(str(sd)):.0f}"
    return s


def where_jev_loses(stats: list[dict[str, object]], tests: list[dict[str, object]]) -> str:
    """Plain statements of every configuration that beat Jev on accuracy, on the shared scopes."""
    lines = ["## Where Jev loses", ""]
    p_by = {(str(t["scope"]), str(t["jev_vs"])): t for t in tests}
    for scope in ("subset", "flagship"):
        rows = {str(s["config"]): s for s in stats if s["scope"] == scope}
        jev = rows.get("jev")
        if not jev:
            continue
        for cid, s in sorted(rows.items(), key=lambda kv: -float(str(kv[1]["accuracy"]))):
            if cid == "jev" or float(str(s["accuracy"])) <= float(str(jev["accuracy"])):
                continue
            gap = (float(str(s["accuracy"])) - float(str(jev["accuracy"]))) * 100
            t = p_by.get((scope, cid))
            sig = (
                f"paired McNemar p {'' if fmt_p(t['p_value']).startswith('<') else '= '}{fmt_p(t['p_value'])} on run 1"
                if t
                else "no paired test"
            )
            cost_ratio = float(str(s["cost_per_1k_usd"])) / float(str(jev["cost_per_1k_usd"]))
            lat_j, lat_s = jev["latency_p50_ms"], s["latency_p50_ms"]
            lat = (
                f", with {float(str(lat_s)) / float(str(lat_j)):.0f}x Jev's p50 latency"
                if lat_j is not None and lat_s is not None
                else ""
            )
            lines.append(
                f"- {BY_ID[cid].label} beats Jev by {gap:.2f} accuracy points on the {scope} "
                f"({sig}), at {cost_ratio:.0f}x Jev's cost per classification{lat}."
            )
    lines.append("")
    return "\n".join(lines)


def intent_gaps(per_class: list[dict[str, object]], stats: list[dict[str, object]]) -> str:
    """Per intent F1 on the subset: Jev against the most accurate configuration that ran on the whole subset."""
    subset = [s for s in stats if s["scope"] == "subset" and s["config"] != "jev"]
    if not subset:
        return ""
    best = max(subset, key=lambda s: float(str(s["accuracy"])))
    best_id = str(best["config"])
    f1: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in per_class:
        if r["scope"] == "subset" and r["config"] in ("jev", best_id):
            f1[(str(r["config"]), str(r["label"]))].append(float(str(r["f1"])))
    gaps = sorted(
        (
            (statistics.fmean(f1[(best_id, lab)]) - statistics.fmean(f1[("jev", lab)]), lab)
            for lab in labels()
            if f1.get(("jev", lab)) and f1.get((best_id, lab))
        ),
        reverse=True,
    )
    out = [
        f"## Intents where Jev trails most (subset, mean F1 over runs, against {BY_ID[best_id].label})",
        "",
        f"| Intent | Jev F1 | {BY_ID[best_id].label} F1 | Gap (points) |",
        "|---|---|---|---|",
    ]
    for gap, lab in gaps[:8]:
        out.append(
            f"| {lab} | {statistics.fmean(f1[('jev', lab)]) * 100:.1f} | "
            f"{statistics.fmean(f1[(best_id, lab)]) * 100:.1f} | {gap * 100:+.1f} |"
        )
    ahead = [lab for gap, lab in gaps if gap < 0]
    out.append("")
    out.append(f"Jev has the higher F1 on {len(ahead)} of 27 intents: {', '.join(ahead) or 'none'}.")
    out.append("")
    return "\n".join(out)


def render_report(
    stats: list[dict[str, object]],
    agreement: list[dict[str, object]],
    tests: list[dict[str, object]],
    manifest: dict[str, object],
    data: Data,
    per_class: list[dict[str, object]],
    latency: list[dict[str, object]],
    summary: list[dict[str, object]],
) -> str:
    lines = ["# Bitext support ticket intents: Jev vs OpenAI baselines", ""]
    w = lines.append
    w(
        f"Dataset: [{DATASET}]({SOURCE_URL}) at revision `{REVISION[:12]}` ({len(data.rows)} rows, 27 intents, "
        f"{LICENSE}). Few-shot exemplars: 27 (one per intent), removed from evaluation for every configuration. "
        f"Total API spend: **${manifest['total_spend_usd']}**. Generated by `evals/support_tickets/full_bench.py`."
    )
    w("")
    titles = {
        "full": f"Full evaluation set ({len(data.eval_set)} tickets)",
        "subset": f"Stratified subset ({len(data.subset)} tickets, {SUBSET_PER_INTENT} per intent)",
        "flagship": f"Flagship subset ({len(data.flagship)} tickets, nested in the subset)",
    }
    for scope in ("subset", "flagship", "full"):
        rows = [s for s in stats if s["scope"] == scope]
        if not rows:
            continue
        rows.sort(key=lambda s: -float(str(s["accuracy"] or 0)))
        w(f"## {titles[scope]}")
        w("")
        w("Mean ± standard deviation over the runs (percentage points for the metrics).")
        w("")
        w(
            "| Configuration | Runs | Accuracy | Macro precision | Macro recall | Macro F1 "
            "| p50 ms | p95 ms | $ per 1k |"
        )
        w("|---|---|---|---|---|---|---|---|---|")
        for s in rows:
            w(
                f"| {s['label']} | {s['runs']} | {fmt(s['accuracy'], True, s['accuracy_sd'])} | "
                f"{fmt(s['macro_precision'], True, s['macro_precision_sd'])} | "
                f"{fmt(s['macro_recall'], True, s['macro_recall_sd'])} | {fmt(s['macro_f1'], True, s['macro_f1_sd'])} "
                f"| {fmt(s['latency_p50_ms'])} | {fmt(s['latency_p95_ms'])} | ${float(str(s['cost_per_1k_usd'])):.3f} |"
            )
        w("")
    if agreement:
        w("## Run to run agreement")
        w("")
        w("| Configuration | Scope | Runs | Rows | Identical prediction in every run |")
        w("|---|---|---|---|---|")
        for a in agreement:
            w(
                f"| {BY_ID[str(a['config'])].label} | {a['scope']} | {a['runs']} | {a['rows']} | "
                f"{fmt(a['rate'], True)} |"
            )
        w("")
    if tests:
        w("## Paired tests against Jev (run 1, exact McNemar)")
        w("")
        w("| Scope | Jev vs | n | Only Jev correct | Only the other correct | p value |")
        w("|---|---|---|---|---|---|")
        for t in tests:
            w(
                f"| {t['scope']} | {BY_ID[str(t['jev_vs'])].label} | {t['n']} | {t['jev_only_correct']} | "
                f"{t['other_only_correct']} | {fmt_p(t['p_value'])} |"
            )
        w("")
    w("## Latency (all runs pooled, each configuration's own rows)")
    w("")
    w("| Configuration | Scope | Calls timed | p50 ms | p95 ms | p95 range across runs |")
    w("|---|---|---|---|---|---|")
    for row in latency:
        w(
            f"| {BY_ID[str(row['config'])].label} | {row['scope']} | {row['n']} | {fmt(row['p50_ms'])} | "
            f"{fmt(row['p95_ms'])} | {fmt(row['run_p95_min_ms'])} to {fmt(row['run_p95_max_ms'])} |"
        )
    w("")
    w("## Spend by configuration")
    w("")
    w("| Configuration | Scope | Calls | Spend | Input tokens | Cached share |")
    w("|---|---|---|---|---|---|")
    for cfg in CONFIGS:
        runs = [x for x in summary if x["config"] == cfg.id and x["scope"] == cfg.scope]
        calls = sum(int(str(x["n"])) for x in runs)
        spend = sum(float(str(x["cost_usd"])) for x in runs)
        tokens = sum(int(str(x["input_tokens"])) for x in runs)
        cached = sum(int(str(x["cached_tokens"])) for x in runs)
        w(
            f"| {cfg.label} | {cfg.scope} | {calls:,} | ${spend:.2f} | {tokens:,} | "
            f"{cached / tokens * 100 if tokens else 0:.0f}% |"
        )
    w("")
    w(
        f"Total spend including the 50 row smoke test and calls that ended in an API error: "
        f"${manifest['total_spend_usd']}."
    )
    w("")
    w(where_jev_loses(stats, tests))
    w(intent_gaps(per_class, stats))
    w("## Setup")
    w("")
    for c in CONFIGS:
        w(
            f"- **{c.label}**: `{c.model}`, {c.prompt}, temperature {c.temperature}, scope {c.scope}, "
            f"concurrency up to {c.concurrency}."
        )
    w("")
    prices = manifest["prices_per_mtok"]
    assert isinstance(prices, dict)
    for k, p in prices.items():
        w(
            f"- Price `{k}`: input ${p['input']}/M, cached input "
            f"{'$' + str(p['cached_input']) + '/M' if p['cached_input'] is not None else 'n/a'}, output "
            f"${p['output']}/M ({p['source']})."
        )
    w("")
    w("## Files and reproduction")
    w("")
    w(
        "- `raw/<config>/run<k>.jsonl.gz`: one record per call (row index, ticket id, true and predicted label, "
        "the raw model reply, token usage including cached and reasoning tokens, latency, attempts, errors). "
        "Ticket text is not repeated; it comes from the pinned dataset."
    )
    w(
        "- `summary.csv` (one row per configuration, run and scope), `per_class.csv` (per intent precision, recall, F1 "
        "and support), `results.json` (aggregates, latency, agreement, paired tests) and `manifest.json` "
        "(dataset hash, seeds, exemplar, subset and flagship row indexes, prices)."
    )
    w(
        "- Reproduce: `uv run python evals/support_tickets/full_bench.py --estimate`, then `--smoke`, `--run "
        "--flagship-per-intent 30` and `--finalize --flagship-per-intent 30`. Keys are read from the environment "
        "(`OPENAI_API_KEY`, `TYPESAFE_API_KEY`)."
    )
    w("")
    w("## Caveats")
    w("")
    w("- Bitext tickets are synthetic with clean labels; real support traffic will usually score lower.")
    w(
        "- Latency is the HTTP latency of the attempt that succeeded, measured with several requests in flight "
        "from one Mac; it depends on concurrency, the network and the provider's load at the time. It is computed "
        "from runner 2 records only (the first 8,447 records were timed by an event loop starved by a runner bug; "
        "their predictions and costs are kept) and excludes each run's first minute, while the runner was still "
        "reading its checkpoint files. Otherwise the runner's event loop fell behind by up to about 50 to 100 ms "
        "at its worst moment in a minute, so treat latency differences of a few milliseconds as noise."
    )
    w(
        "- Cost is reported tokens times list price, with OpenAI's cached input rate where the response reported "
        "cached tokens. Jev bills input only."
    )
    w(
        "- GPT-6 Astra is a reasoning model that rejects temperature 0; it ran at its default temperature (1) with "
        "reasoning_effort=low, so its run to run variance is not comparable with the temperature 0 baselines."
    )
    w(
        "- Invalid outputs (not one of the 27 labels) count as wrong. Rows that still had an API error after "
        "retries are left out, and a run is only reported when it is complete."
    )
    w("")
    return "\n".join(lines)


def gzip_raw() -> None:
    for path in (OUT / "raw").rglob("*.jsonl"):
        with path.open("rb") as src, gzip.open(path.with_suffix(".jsonl.gz"), "wb", compresslevel=9) as dst:
            dst.write(src.read())
        path.unlink()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--estimate", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--finalize", action="store_true")
    ap.add_argument("--flagship-per-intent", type=int, default=DEFAULT_FLAGSHIP_PER_INTENT)
    ap.add_argument("--gzip", action="store_true", help="with --finalize: compress the raw files")
    args = ap.parse_args()

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")  # keys into this process's environment only
    data = build_data(args.flagship_per_intent)
    if args.estimate:
        print_plan(data, args.flagship_per_intent)
        return 0
    if args.finalize:
        finalize(data, args.flagship_per_intent)
        if args.gzip:
            gzip_raw()
        return 0
    projected = print_plan(data, args.flagship_per_intent)
    if not args.smoke and spent_so_far() + projected > BUDGET_USD:
        print("Projected total is over the cap: lower --flagship-per-intent; the guard will also skip configurations.")
    try:
        asyncio.run(smoke(data) if args.smoke else run_plan(data))
    except Fatal as exc:
        print(f"STOPPED: {exc}")
        return 3
    print(f"Total spent so far: ${spent_so_far():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
