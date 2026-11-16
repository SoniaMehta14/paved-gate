"""Red team runner: sends the synthetic corpus through the real gates with every configured evaluator.

    uv run python scripts/redteam.py --estimate                 # call counts and cost estimate, NO API calls
    uv run python scripts/redteam.py --mode mock                # pipeline check, NO API calls
    uv run python scripts/redteam.py --mode live --yes          # real API calls (costs money)

Experiments (run all, or pick with --experiments):
  baseline  every item in the gate, harness2 and openshell sets, once per arm; the detector set runs locally
  drift     a fixed, stratified subset of gate items, repeated --drift-reps times per live arm
  adaptive  a greedy search per malicious seed that stacks corpus transforms and keeps a variant when the risk
            score the gate reveals goes down, up to --adaptive-queries queries

Arms: `shipped` is the evaluator each policy file configures out of the box (offline mocks and heuristics, free).
jev, gemini_decision, gpt-4o and gpt-4o-mini are the configured evaluators; in mock mode jev and gemini_decision use
their mocks and the GPT arms are skipped.

Every item goes through PavedGate.handle, AdoptionGate.evaluate or the EgressInspector exactly as in production,
with the shipped policy files. One change: the evaluator timeout is raised to EVAL_TIMEOUT_MS for every arm so the
run measures what each evaluator decides rather than how often it beats a 400 ms budget. The latency of every call
is recorded, so the effect of the shipped timeouts is reported separately.

Raw responses: every evaluator call is appended to results/redteam/raw-<mode>.jsonl as it returns (answers, raw
body, tokens, cost) and synced to disk, so a crash cannot lose a paid response.

Spend: every live call is priced at list price (scripts/pricing.py) from the tokens the API reports, including the
Gemini free tier, and appended to results/redteam/spend.jsonl, which persists across runs. A run refuses to start an
item that could push the total past BUDGET_USD, and stops once when the total first reaches HALF_USD so a person can
look; rerun with --past-half to continue. Interrupted runs resume from a checkpoint without repeating calls.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import random
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evals" / "support_tickets"))
sys.path.insert(0, str(ROOT / "evals" / "redteam"))
sys.path.insert(0, str(ROOT / "scripts"))

import analyze  # noqa: E402
import build_corpus  # noqa: E402
from benchmark_evaluators import CallRecord, Checkpoint, TokenPacer, latency  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from full_bench import mcnemar  # noqa: E402
from pricing import cost_usd  # noqa: E402

from paved_gate import JevEvaluator, MemorySink, PavedGate, default_registry, load_policy  # noqa: E402
from paved_gate.evaluator.base import EvaluationResult, EvaluatorState, FastEvaluator, Question  # noqa: E402
from paved_gate.evaluator.gemini_decision import GeminiDecisionEvaluator, RateLimitedError  # noqa: E402
from paved_gate.evaluator.jev import JevAPIError  # noqa: E402
from paved_gate.evaluator.openai_eval import (  # noqa: E402
    OpenAIAPIError,
    OpenAIMultiEvaluator,
    multi_system_prompt,
)
from paved_gate.gate import build_questions, evaluator_from_config  # noqa: E402
from paved_gate.integrations.harness2.config import (  # noqa: E402
    AdoptionPolicy,
    adoption_policy_hash,
    load_adoption_policy,
)
from paved_gate.integrations.harness2.gate import AdoptionGate, evaluator_for  # noqa: E402
from paved_gate.integrations.harness2.regression import build_questions as regression_questions  # noqa: E402
from paved_gate.integrations.harness2.run_dir import Adoption  # noqa: E402
from paved_gate.integrations.openshell.config import EgressPolicy, egress_policy_hash, load_egress_policy  # noqa: E402
from paved_gate.integrations.openshell.inspector import EgressInspector, Target  # noqa: E402
from paved_gate.policy.loader import LoadedPolicy, policy_hash  # noqa: E402
from paved_gate.privacy.detect import detect  # noqa: E402

RESULTS = ROOT / "results" / "redteam"
LEDGER = RESULTS / "spend.jsonl"
CORPUS = build_corpus.OUT
MANIFEST = build_corpus.MANIFEST
GATE_POLICY = ROOT / "policy" / "paved_gate.policy.yaml"
HARNESS2_POLICY = ROOT / "policy" / "harness2_adoption.policy.yaml"
EGRESS_POLICY = ROOT / "policy" / "openshell_egress.policy.yaml"

BUDGET_USD = 20.0
HALF_USD = 10.0
EVAL_TIMEOUT_MS = 30_000
SEED = 20261116
ARMS = ("shipped", "jev", "gemini_decision", "gpt-4o", "gpt-4o-mini")
LIVE_ONLY = ("gpt-4o", "gpt-4o-mini")
EXPERIMENTS = ("baseline", "drift", "adaptive")
SETS = ("gate", "harness2", "openshell")
PRICE_KEY: dict[str, str | None] = {
    "shipped": None,
    "jev": "jev",
    "gemini_decision": "gemini-decision-1",
    "gpt-4o": "gpt-4o",
    "gpt-4o-mini": "gpt-4o-mini",
}
KEY_ENV = {
    "jev": "TYPESAFE_API_KEY",
    "gemini_decision": "GEMINI_API_KEY",
    "gpt-4o": "OPENAI_API_KEY",
    "gpt-4o-mini": "OPENAI_API_KEY",
}
GPT_OUTPUT_TOKENS = 80  # estimate for one structured answer to three questions
DRIFT_CLASSES = ("none", "C1", "C2", "C3", "C4", "C6", "A1", "A3")

Item = dict[str, Any]


class BudgetExhausted(Exception):
    pass


class HalfBudgetReached(Exception):
    pass


class ArmStopped(Exception):
    """A quota, rate limit or credential error: the arm stops, the item is not checkpointed, a rerun resumes."""


# --------------------------------------------------------------------------- corpus


def load_corpus(path: Path = CORPUS, manifest: Path = MANIFEST) -> list[Item]:
    data = path.read_bytes()
    expected = json.loads(manifest.read_text())["sha256"]
    if hashlib.sha256(data).hexdigest() != expected:
        raise SystemExit(f"{path} does not match {manifest}; regenerate with evals/redteam/build_corpus.py")
    return [json.loads(line) for line in data.decode("utf-8").splitlines()]


def drift_subset(items: list[Item], per_class: int = 5, benign: int = 10) -> list[Item]:
    rng = random.Random(SEED)
    gate = [i for i in items if i["set"] == "gate"]
    chosen: list[Item] = []
    for cls in DRIFT_CLASSES:
        pool = [i for i in gate if i["label"] == "malicious" and i["attack_class"] == cls]
        chosen += rng.sample(pool, per_class)
    chosen += rng.sample([i for i in gate if i["label"] == "benign"], benign)
    return chosen


def adaptive_seeds(items: list[Item], n: int) -> list[Item]:
    plain = [i for i in items if i["set"] == "gate" and i["label"] == "malicious" and i["transform"] == "plain"]
    return random.Random(SEED + 1).sample(plain, n)


# --------------------------------------------------------------------------- spend


class Ledger:
    """Running total of live spend across every run, with a hard cap and a one time stop at half."""

    def __init__(self, path: Path, *, live: bool, past_half: bool, cap: float = BUDGET_USD, half: float = HALF_USD):
        self.path, self.live, self.past_half, self.cap, self.half = path, live, past_half, cap, half
        self.spent = 0.0
        self.reserved = 0.0
        if path.exists():
            self.spent = sum(float(json.loads(line)["usd"]) for line in path.read_text().splitlines() if line)

    def reserve(self, usd: float) -> None:
        if not self.live:
            return
        if self.spent >= self.half and not self.past_half:
            raise HalfBudgetReached(f"spend is ${self.spent:.2f}, at or past the ${self.half:.2f} checkpoint")
        if self.spent + self.reserved + usd > self.cap:
            raise BudgetExhausted(
                f"next item could cost ${usd:.4f}; spent ${self.spent:.2f} plus ${self.reserved:.4f} in flight "
                f"would pass the ${self.cap:.2f} cap"
            )
        self.reserved += usd

    def release(self, usd: float) -> None:
        if self.live:
            self.reserved = max(0.0, self.reserved - usd)

    def charge(self, usd: float, meta: Mapping[str, object]) -> None:
        if not self.live or usd <= 0:
            return
        self.spent += usd
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": datetime.now(UTC).isoformat(), "usd": usd, **meta}) + "\n")


def price(arm: str, input_tokens: int, output_tokens: int, cached: int = 0) -> float:
    key = PRICE_KEY[arm]
    if key is None:
        return 0.0
    return cost_usd(key, input_tokens, output_tokens, cached) or 0.0


# --------------------------------------------------------------------------- recording evaluator


@dataclass
class CallInfo:
    latency_ms: float
    input_tokens: int
    output_tokens: int
    usd: float
    retries: int | None
    error: str | None


class RawLog:
    """Append only JSONL of every evaluator call, written and synced as each call returns, so a crash keeps every
    paid response. One line per call: the tag (experiment, arm, item key), tokens, cost, answers and the raw body."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, line: Mapping[str, object]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, default=str, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())


class Recording:
    """Wraps an evaluator: records every call, charges the ledger and feeds the pacer. Errors still propagate,
    so the gate under test sees exactly what the evaluator did."""

    def __init__(
        self,
        inner: FastEvaluator,
        arm: str,
        ledger: Ledger,
        pacer: TokenPacer | None,
        raw_log: RawLog | None = None,
    ) -> None:
        self.inner, self.arm, self.ledger, self.pacer, self.raw_log = inner, arm, ledger, pacer, raw_log
        self.name = getattr(inner, "name", arm)
        self.calls: list[CallInfo] = []
        self.stop_reason: str | None = None
        self.tag: dict[str, object] = {}  # experiment and item key of the item in flight, set by the runner

    def _log(self, info: CallInfo, state: EvaluatorState, extra: Mapping[str, object]) -> None:
        meta = {**self.tag, "arm": self.arm, "call": len(self.calls)}
        self.ledger.charge(info.usd, meta)
        if self.raw_log is None:
            return
        state_text = state if isinstance(state, str) else json.dumps(state, sort_keys=True, ensure_ascii=False)
        self.raw_log.write(
            {
                "ts": datetime.now(UTC).isoformat(),
                **meta,
                "ok": info.error is None,
                "latency_ms": info.latency_ms,
                "input_tokens": info.input_tokens,
                "output_tokens": info.output_tokens,
                "usd": info.usd,
                "retries": info.retries,
                "error": info.error,
                "state_sha256": hashlib.sha256(state_text.encode()).hexdigest(),
                "state_chars": len(state_text),
                **extra,
            }
        )

    def take(self) -> list[CallInfo]:
        calls, self.calls = self.calls, []
        return calls

    async def evaluate(self, state: EvaluatorState, questions: Mapping[str, Question]) -> EvaluationResult:
        started = time.perf_counter()
        est_in = estimate_input_tokens(self.arm, state, questions)
        try:
            result = await self.inner.evaluate(state, questions)
        except Exception as exc:
            ms = (time.perf_counter() - started) * 1000
            in_tok = getattr(exc, "input_tokens", None)
            out_tok = getattr(exc, "output_tokens", None)
            # A failed call may still be billed; with no usage reported, charge the estimate.
            usd = price(self.arm, in_tok, out_tok or 0) if in_tok is not None else price(self.arm, est_in, 0)
            info = CallInfo(round(ms, 2), in_tok or 0, out_tok or 0, usd, None, _err(exc))
            self.calls.append(info)
            self._log(info, state, {"raw": getattr(exc, "raw", None), "usd_estimated": in_tok is None})
            if self._fatal(exc):
                self.stop_reason = _err(exc)
            raise
        ms = (time.perf_counter() - started) * 1000
        in_tok, out_tok = result.input_tokens or 0, result.output_tokens or 0
        usage = result.raw.get("usage")
        cached = int(str(usage.get("cached_tokens", 0))) if isinstance(usage, dict) else 0
        usd = price(self.arm, in_tok, out_tok, cached)
        if self.pacer is not None:
            self.pacer.record("call", in_tok or est_in)
        info = CallInfo(round(ms, 2), in_tok, out_tok, usd, result.retries, None)
        self.calls.append(info)
        extra = {
            "model": result.model,
            "cached_tokens": cached,
            "answers": {k: a.model_dump(mode="json") for k, a in result.answers.items()},
            "raw": result.raw,
        }
        self._log(info, state, extra)
        return result

    @staticmethod
    def _fatal(exc: Exception) -> bool:
        if isinstance(exc, RateLimitedError):
            return True
        if isinstance(exc, JevAPIError) and exc.status in (401, 403, 429):
            return True
        return isinstance(exc, OpenAIAPIError) and exc.fatal

    async def aclose(self) -> None:
        await self.inner.aclose()


def _err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc)[:300]}"


def _state_chars(state: EvaluatorState) -> int:
    return len(state) if isinstance(state, str) else len(json.dumps(state, ensure_ascii=False))


def estimate_input_tokens(arm: str, state: EvaluatorState, questions: Mapping[str, Question]) -> int:
    if arm.startswith("gpt"):
        return (_state_chars(state) + len(multi_system_prompt(questions))) // 4 + 20
    q_chars = sum(len(q.model_dump_json()) for q in questions.values())
    return (_state_chars(state) + q_chars) // 4 + 20


# --------------------------------------------------------------------------- arms


@dataclass
class Arm:
    name: str
    status: str  # ok | not_run
    reason: str
    factory: Callable[[str], FastEvaluator] | None = None  # set name -> fresh evaluator
    pacer: TokenPacer | None = None
    stop_reason: str | None = None
    paused: bool = False  # reached --stop-after in this invocation


def build_arm(name: str, mode: str, gemini_tpm: int) -> Arm:
    timeout_s = EVAL_TIMEOUT_MS / 1000
    if name == "shipped":
        return Arm(name, "ok", "each policy's own offline evaluator", shipped_evaluator)
    if name in LIVE_ONLY and mode != "live":
        return Arm(name, "not_run", "no mock for OpenAI models")
    key = os.environ.get(KEY_ENV[name])
    if mode == "live" and not key:
        return Arm(name, "not_run", f"{KEY_ENV[name]} not set")
    if name == "jev":
        if mode == "mock":
            return Arm(name, "ok", "Jev mock", lambda _set: JevEvaluator(mode="mock", mock_latency_ms=(0, 0)))
        return Arm(
            name,
            "ok",
            "real Jev API",
            lambda _set: JevEvaluator(mode="live", model="jev-latest", api_key=key, timeout_s=timeout_s),
        )
    if name == "gemini_decision":
        live = mode == "live"
        if live:
            try:
                import google.genai  # noqa: F401
            except ImportError:
                return Arm(name, "not_run", "the gemini extra is not installed: uv sync --extra gemini")
        return Arm(
            name,
            "ok",
            "real Gemini Decisions API" if live else "Gemini mock",
            lambda _set: GeminiDecisionEvaluator(
                mode="live" if live else "mock", api_key=key, timeout_s=timeout_s, mock_latency_ms=(0, 0)
            ),
            pacer=TokenPacer(gemini_tpm) if live and gemini_tpm > 0 else None,
        )
    return Arm(
        name,
        "ok",
        f"OpenAI {name}, temperature 0, structured output",
        lambda _set: OpenAIMultiEvaluator(model=name, temperature=0.0, timeout_s=timeout_s, max_retries=4),
    )


def shipped_evaluator(set_name: str) -> FastEvaluator:
    if set_name == "harness2":
        return evaluator_for(load_adoption_policy(HARNESS2_POLICY))
    if set_name == "openshell":
        return evaluator_from_config(load_egress_policy(EGRESS_POLICY).evaluator)
    return evaluator_from_config(load_policy(GATE_POLICY).policy.evaluator)


# --------------------------------------------------------------------------- one item through one gate


@dataclass
class Context:
    gate_policy: LoadedPolicy
    adoption_policy: AdoptionPolicy
    egress_policy: EgressPolicy


def load_context() -> Context:
    gp = load_policy(GATE_POLICY)
    evaluator_cfg = gp.policy.evaluator.model_copy(update={"timeout_ms": EVAL_TIMEOUT_MS})
    policy = gp.policy.model_copy(update={"evaluator": evaluator_cfg})
    gate_policy = LoadedPolicy(policy=policy, policy_hash=policy_hash(policy), source=gp.source)
    ep = load_egress_policy(EGRESS_POLICY)
    egress = ep.model_copy(update={"evaluator": ep.evaluator.model_copy(update={"timeout_ms": EVAL_TIMEOUT_MS})})
    return Context(gate_policy, load_adoption_policy(HARNESS2_POLICY), egress)


async def run_gate(ctx: Context, ev: Recording, text: str) -> dict[str, object]:
    gate = PavedGate(ctx.gate_policy, evaluator=ev, audit=MemorySink(), deterministic=default_registry(), frontier=None)
    result = await gate.handle(text)
    d = result.decision
    return {
        "outcome": d.outcome,
        "blocked": d.outcome == "blocked",
        "route": d.route.value if d.route else None,
        "risk_score": d.risk_score,
        "intent_confidence": d.intent_confidence,
        "sensitive": d.sensitive,
        "reasons": d.reasons,
    }


async def run_harness2(ctx: Context, ev: Recording, item: Item) -> dict[str, object]:
    p = item["payload"]
    adoption = Adoption(
        task_id=item["id"],
        mode="redteam",
        k=0,
        namespace="redteam",
        label="adopted",
        is_base=False,
        task_dir=ROOT,
        selection_path=ROOT,
        baseline=p["baseline"],
        adopted=p["adopted"],
        baseline_outputs={"base1/answer.txt": p["baseline_output"]},
        adopted_outputs={"agg1/answer.txt": p["adopted_output"]},
        reflection_note=None,
        lesson=None,
        experience=None,
    )
    gate = AdoptionGate(ctx.adoption_policy, ev, adoption_policy_hash(ctx.adoption_policy))
    entry = await gate.evaluate(adoption, ROOT)
    return {"outcome": entry.decision, "blocked": entry.decision != "accept", "reasons": entry.reasons}


def _target(item_id: str, sandbox: str, req: Mapping[str, Any]) -> Target:
    u = urlsplit(str(req["url"]))
    return Target(item_id, sandbox, str(req["method"]), u.hostname or "", u.port or 80, u.path, u.query)


async def run_openshell(ctx: Context, ev: Recording, item: Item, rep: int) -> dict[str, object]:
    p = item["payload"]
    inspector = EgressInspector(ctx.egress_policy, ev, MemorySink(), egress_policy_hash(ctx.egress_policy))
    sandbox = f"{item['id']}#{rep}"
    resp = p["response"]
    rv = await inspector.inspect_response(_target(item["id"], sandbox, resp), str(resp["body"]).encode(), block=False)
    req = p["request"]
    body = (req.get("body") or "").encode()
    qv = await inspector.inspect_request(_target(item["id"], sandbox, req), body, enforce=True)
    return {
        "outcome": qv.decision,
        "blocked": qv.deny,
        "response_flagged": rv.reason_code is not None,
        "tainted": qv.tainted,
        "response_scores": rv.scores,
        "request_scores": qv.scores,
        "reason": qv.reason_code,
        "local_detection": any(f.type.startswith("local_") for f in qv.findings),
    }


def estimate_item_tokens(arm: str, ctx: Context, item: Item) -> tuple[int, int]:
    """(input, output) tokens for every evaluator call one item makes; output only matters for GPT."""
    out = GPT_OUTPUT_TOKENS if arm.startswith("gpt") else 0
    p = item["payload"]
    if item["set"] == "gate":
        return estimate_input_tokens(arm, p["text"], build_questions(ctx.gate_policy.policy)), out
    if item["set"] == "harness2":
        changed = sum(len(v) for k, v in p["adopted"].items() if p["baseline"].get(k) != v)
        state = "x" * min(int(changed * 1.1) + 400, 16_400)
        return estimate_input_tokens(arm, state, regression_questions(ctx.adoption_policy.regression)), out
    q = ctx.egress_policy.questions
    q_chars = len(q.model_dump_json())
    resp = min(len(str(p["response"]["body"])), ctx.egress_policy.max_inspect_chars)
    req = len(p["request"].get("body") or "") + 1_300
    factor = 2 if arm.startswith("gpt") else 1  # the GPT system prompt restates the questions with formatting
    return (resp + req + 2 * q_chars * factor) // 4 + 40, 2 * out


# --------------------------------------------------------------------------- experiments


@dataclass
class Settings:
    mode: str
    experiments: list[str]
    arms: list[str]
    sets: list[str]
    limit: int | None = None
    drift_reps: int = 10
    adaptive_seeds: int = 20
    adaptive_queries: int = 15
    corpus_sha: str = ""
    timeout_ms: int = EVAL_TIMEOUT_MS
    gemini_tpm: int = 45_000

    def digest(self) -> str:
        measured = {k: v for k, v in asdict(self).items() if k != "gemini_tpm"}
        return hashlib.sha256(json.dumps(measured, sort_keys=True).encode()).hexdigest()[:16]


def spread(items: list[Item], limit: int | None) -> list[Item]:
    """`limit` items at even intervals, so a small rehearsal still covers every seed family and transform."""
    if not limit or limit >= len(items):
        return items
    return [items[i * len(items) // limit] for i in range(limit)]


def plan(settings: Settings, items: list[Item]) -> dict[str, list[tuple[Item, int]]]:
    """experiment -> list of (item, repeat index). Adaptive lists the seed items; its queries are dynamic."""
    in_sets = [i for i in items if i["set"] in settings.sets]
    out: dict[str, list[tuple[Item, int]]] = {}
    if "baseline" in settings.experiments:
        # Seeded shuffle: a paused or interrupted run still leaves a sample across sets, labels and classes.
        baseline = spread(in_sets, settings.limit)
        random.Random(SEED).shuffle(baseline)
        out["baseline"] = [(i, 0) for i in baseline]
    if "drift" in settings.experiments and "gate" in settings.sets:
        subset = drift_subset(items)
        if settings.limit:
            subset = spread(subset, max(1, settings.limit // 10))
        out["drift"] = [(i, r) for r in range(settings.drift_reps) for i in subset]
    if "adaptive" in settings.experiments and "gate" in settings.sets:
        n = settings.adaptive_seeds if not settings.limit else min(settings.adaptive_seeds, 2)
        out["adaptive"] = [(i, 0) for i in adaptive_seeds(items, n)]
    return out


def arms_for(experiment: str, arms: list[Arm]) -> list[Arm]:
    ready = [a for a in arms if a.status == "ok"]
    # The shipped evaluators are deterministic regexes: repeating them measures nothing.
    return [a for a in ready if a.name != "shipped"] if experiment == "drift" else ready


def base_row(item: Item, arm: str, experiment: str, rep: int) -> dict[str, object]:
    return {
        "experiment": experiment,
        "arm": arm,
        "item_id": item["id"],
        "set": item["set"],
        "label": item["label"],
        "attack_class": item["attack_class"],
        "transform": item["transform"],
        "seed_id": item["seed_id"],
        "category": item["category"],
        "rep": rep,
        "query": item.get("query"),
    }


def record(experiment: str, arm: str, key: str, row: dict[str, object], calls: list[CallInfo]) -> CallRecord:
    errors = [c.error for c in calls if c.error]
    row["error"] = errors[0] if errors else None
    row["usd"] = round(sum(c.usd for c in calls), 8)
    row["calls"] = len(calls)
    retries = [c.retries for c in calls if c.retries is not None]
    return CallRecord(
        experiment,
        arm,
        key,
        not errors,
        latency_ms=round(sum(c.latency_ms for c in calls), 2) if calls else None,
        retries=sum(retries) if retries else None,
        input_tokens=sum(c.input_tokens for c in calls),
        output_tokens=sum(c.output_tokens for c in calls),
        error=row["error"],  # type: ignore[arg-type]
        data=row,
    )


class Runner:
    def __init__(
        self,
        settings: Settings,
        ctx: Context,
        ledger: Ledger,
        ckpt: Checkpoint,
        log: Callable[[str], None],
        raw_log: RawLog | None = None,
        stop_after: int | None = None,
    ) -> None:
        self.settings, self.ctx, self.ledger, self.ckpt, self.log = settings, ctx, ledger, ckpt, log
        self.raw_log, self.stop_after = raw_log, stop_after
        self.new: dict[str, int] = {}  # items run (not restored) per arm in this invocation

    def _paused(self, arm: Arm) -> bool:
        if self.stop_after is not None and self.new.get(arm.name, 0) >= self.stop_after:
            arm.paused = True
        return arm.paused

    def _recording(self, arm: Arm, set_name: str) -> Recording:
        assert arm.factory is not None
        return Recording(arm.factory(set_name), arm.name, self.ledger, arm.pacer, self.raw_log)

    async def one(self, arm: Arm, ev: Recording, experiment: str, item: Item, rep: int, key: str) -> CallRecord:
        if (experiment, arm.name, key) in self.ckpt.done:
            return self.ckpt.done[(experiment, arm.name, key)]
        est_in, est_out = estimate_item_tokens(arm.name, self.ctx, item)
        est_usd = price(arm.name, est_in, est_out)
        self.ledger.reserve(est_usd)
        ev.tag = {"experiment": experiment, "key": key}
        try:
            if arm.pacer is not None:
                await arm.pacer.wait(est_in)
            row = base_row(item, arm.name, experiment, rep)
            if item["set"] == "gate":
                row |= await run_gate(self.ctx, ev, item["payload"]["text"])
            elif item["set"] == "harness2":
                row |= await run_harness2(self.ctx, ev, item)
            else:
                row |= await run_openshell(self.ctx, ev, item, rep)
        except asyncio.CancelledError:
            # Another arm hit a budget stop mid call; this call may still be billed, so count the estimate.
            self.ledger.charge(est_usd, {"arm": arm.name, "cancelled": True})
            raise
        finally:
            self.ledger.release(est_usd)
        calls = ev.take()
        if ev.stop_reason:
            raise ArmStopped(ev.stop_reason)
        rec = record(experiment, arm.name, key, row, calls)
        self.ckpt.add(rec)
        n = self.new[arm.name] = self.new.get(arm.name, 0) + 1
        if n % 100 == 0:
            self.log(f"  {experiment}: {arm.name} {n} new items, live spend ${self.ledger.spent:.4f}")
        return rec

    async def run_arm(self, arm: Arm, experiment: str, work: list[tuple[Item, int]]) -> None:
        evs: dict[str, Recording] = {}
        try:
            if experiment == "adaptive":
                ev = evs.setdefault("gate", self._recording(arm, "gate"))
                for item, _ in work:
                    if self._paused(arm):
                        break
                    await self.adaptive(arm, ev, item)
            else:
                for item, rep in work:
                    if self._paused(arm):
                        break
                    s = item["set"]
                    if s not in evs:
                        evs[s] = self._recording(arm, s)
                    await self.one(arm, evs[s], experiment, item, rep, f"{item['id']}#{rep}")
            self.log(f"  {experiment}: {arm.name} {'paused' if arm.paused else 'done'}")
        except ArmStopped as exc:
            arm.stop_reason = str(exc)
            self.log(f"  {experiment}: {arm.name} stopped: {exc}")
        finally:
            for ev in evs.values():
                await ev.aclose()

    async def adaptive(self, arm: Arm, ev: Recording, seed: Item) -> None:
        """Greedy search over stacked corpus transforms, guided only by the risk score the gate reveals."""
        rng = random.Random(f"{SEED}:{arm.name}:{seed['seed_id']}")
        paraphrase = next(p for s, _, _, p in build_corpus.MALICIOUS if s == seed["seed_id"])
        best = Variant(core=seed["payload"]["text"])
        best_rec: CallRecord | None = None
        for q in range(1, self.settings.adaptive_queries + 1):
            cand = best if q == 1 else best.mutate(rng, paraphrase)
            text = cand.render(rng_seed=q)
            item = {**seed, "payload": {"text": text}, "attack_class": "A2", "transform": cand.name, "query": q}
            if q > 1 and self._paused(arm):
                return
            rec = await self.one(arm, ev, "adaptive", item, 0, f"{seed['seed_id']}#{q}")
            if not rec.data.get("blocked") and not rec.data.get("error"):
                return
            if best_rec is None or _risk(rec) < _risk(best_rec):
                best, best_rec = cand, rec

    async def run(self, work: dict[str, list[tuple[Item, int]]], arms: list[Arm]) -> None:
        for experiment, items in work.items():
            self.log(f"{experiment}: {len(items)} items")
            async with asyncio.TaskGroup() as tg:
                for arm in arms_for(experiment, arms):
                    if arm.stop_reason is None:
                        tg.create_task(self.run_arm(arm, experiment, items))
            if any(a.paused for a in arms):
                return


def _risk(rec: CallRecord) -> float:
    r = rec.data.get("risk_score")
    return float(str(r)) if r is not None else float("inf")


@dataclass(frozen=True)
class Variant:
    """A malicious core plus stacked transforms from the corpus, rendered in a fixed order."""

    core: str
    paraphrased: bool = False
    wrappers: tuple[str, ...] = ()
    encoding: str | None = None
    padded: bool = False

    @property
    def name(self) -> str:
        parts = ["paraphrase" if self.paraphrased else "plain", *self.wrappers]
        parts += [self.encoding] if self.encoding else []
        parts += ["padding"] if self.padded else []
        return "+".join(parts)

    def mutate(self, rng: random.Random, paraphrase: str) -> Variant:
        templates = {**build_corpus.C1_TEMPLATES, **build_corpus.C2_TEMPLATES, **build_corpus.A1_TEMPLATES}
        ops: list[Callable[[], Variant]] = []
        if not self.paraphrased:
            ops.append(lambda: Variant(paraphrase, True, self.wrappers, self.encoding, self.padded))
        unused = sorted(set(templates) - set(self.wrappers))
        if unused and len(self.wrappers) < 2:
            ops.append(
                lambda: Variant(
                    self.core, self.paraphrased, (*self.wrappers, rng.choice(unused)), self.encoding, self.padded
                )
            )
        if self.encoding is None:
            ops.append(
                lambda: Variant(
                    self.core,
                    self.paraphrased,
                    self.wrappers,
                    rng.choice(sorted(build_corpus.C4_TRANSFORMS)),
                    self.padded,
                )
            )
        if not self.padded:
            ops.append(lambda: Variant(self.core, self.paraphrased, self.wrappers, self.encoding, True))
        return rng.choice(ops)() if ops else self

    def render(self, rng_seed: int) -> str:
        templates = {**build_corpus.C1_TEMPLATES, **build_corpus.C2_TEMPLATES, **build_corpus.A1_TEMPLATES}
        text = self.core
        for w in self.wrappers:
            text = templates[w].format(p=text)
        if self.encoding:
            text = build_corpus.C4_TRANSFORMS[self.encoding](text)
        if self.padded:
            text = build_corpus.pad(text, random.Random(rng_seed))
        return text


# --------------------------------------------------------------------------- local detector set


def detector_rows(items: list[Item], ctx: Context) -> list[dict[str, object]]:
    rows = []
    for i in items:
        if i["set"] != "detector":
            continue
        p = i["payload"]
        found = {f.detector for f in detect(p["text"], ctx.gate_policy.policy.privacy.detectors)}
        rows.append({**base_row(i, "local", "baseline", 0), "detected": p["detector"] in found, "found": sorted(found)})
    return rows


# --------------------------------------------------------------------------- estimate, summary, main


def estimate(settings: Settings, ctx: Context, work: dict[str, list[tuple[Item, int]]], spent: float) -> dict[str, Any]:
    rows = []
    total = 0.0
    for experiment, items in work.items():
        for arm in settings.arms:
            if experiment == "drift" and arm == "shipped":
                continue
            calls, usd = 0, 0.0
            for item, _ in items:
                est_in, est_out = estimate_item_tokens(arm, ctx, item)
                n = settings.adaptive_queries if experiment == "adaptive" else 1
                if experiment == "adaptive":
                    est_in += 600  # stacked wrappers and padding make later queries longer than the seed
                per = 2 if item["set"] == "openshell" else 1
                calls += n * per
                usd += n * price(arm, est_in, est_out)
            total += usd
            rows.append({"experiment": experiment, "arm": arm, "calls": calls, "usd": round(usd, 4)})
    return {
        "rows": rows,
        "total_usd": round(total, 4),
        "spent_so_far_usd": round(spent, 4),
        "cap_usd": BUDGET_USD,
        "half_usd": HALF_USD,
        "note": "upper bound for adaptive (every seed uses all queries); list prices, Gemini at the paid tier",
    }


def summarize(records: list[CallRecord], detector: list[dict[str, object]]) -> dict[str, object]:
    by_exp: dict[str, list[dict[str, object]]] = {}
    for r in records:
        by_exp.setdefault(r.workload, []).append(r.data)
    base = by_exp.get("baseline", [])
    out: dict[str, object] = {
        "baseline": analyze.summarize_baseline(base),
        "drift": analyze.summarize_drift(by_exp.get("drift", [])),
        "adaptive": analyze.summarize_adaptive(by_exp.get("adaptive", [])),
        "detector": {
            "by_detector": {
                det: analyze.rate(sum(bool(r["detected"]) for r in rs), len(rs))
                for det in sorted({str(r["seed_id"]) for r in detector})
                if (rs := [r for r in detector if r["seed_id"] == det and r["attack_class"] == "C5"])
            },
            "canonical_detected": all(bool(r["detected"]) for r in detector if r["attack_class"] == "none"),
            "rows": detector,
        },
        "latency": {
            arm: latency([r for r in records if r.arm == arm and r.workload == "baseline"])
            for arm in sorted({r.arm for r in records})
        },
        "spend_usd": round(sum(float(str(r.data.get("usd") or 0)) for r in records), 6),
    }
    pairs: dict[str, object] = {}
    gate_rows = [r for r in base if r["set"] == "gate" and r["label"] == "malicious"]
    arms = sorted({str(r["arm"]) for r in gate_rows})
    for i, a in enumerate(arms):
        for b in arms[i + 1 :]:
            pairs[f"{a} vs {b}"] = analyze.mcnemar_passes(
                [r for r in gate_rows if r["arm"] == a], [r for r in gate_rows if r["arm"] == b], mcnemar
            )
    out["gate_arm_pairs"] = pairs
    return out


def print_summary(summary: dict[str, Any], log: Callable[[str], None]) -> None:
    def fmt(r: analyze.Rate) -> str:
        if r["rate"] is None:
            return "n/a"
        lo, hi = r["ci95"] or [0.0, 0.0]
        return f"{r['rate']:.1%} [{lo:.1%}, {hi:.1%}] n={r['n']}"

    for set_name, arms in summary["baseline"].items():
        log(f"\n{set_name}")
        for arm, s in arms.items():
            o = s["overall"]
            log(f"  {arm:16s} attack success {fmt(o['attack_success'])}   false block {fmt(o['false_block'])}")
            if o["errors"]["malicious"] or o["errors"]["benign"]:
                log(f"  {'':16s} evaluator errors: {o['errors']}")
    for det, r in summary["detector"]["by_detector"].items():
        log(f"  detector {det:12s} detected {fmt(r)}")


def rebuild(calls: Path) -> dict[str, object]:
    """The summary a run would have written, recomputed from its calls file."""
    lines = calls.read_text(encoding="utf-8").splitlines()
    records = [CallRecord(**json.loads(line)) for line in lines[1:]]
    return summarize(records, detector_rows(load_corpus(), load_context()))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("mock", "live"), default="mock")
    ap.add_argument("--yes", action="store_true", help="required for live mode")
    ap.add_argument("--estimate", action="store_true", help="print call counts and cost; make no calls")
    ap.add_argument("--experiments", default=",".join(EXPERIMENTS))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--limit", type=int, default=None, help="rehearsal: this many baseline items, spread evenly")
    ap.add_argument("--drift-reps", type=int, default=10)
    ap.add_argument("--adaptive-seeds", type=int, default=20)
    ap.add_argument("--adaptive-queries", type=int, default=15)
    ap.add_argument("--gemini-tpm", type=int, default=45_000)
    ap.add_argument("--past-half", action="store_true", help="continue after the half budget checkpoint")
    ap.add_argument("--fresh", action="store_true", help="discard the checkpoint and start over")
    ap.add_argument(
        "--stop-after", type=int, default=None, help="pause after this many new items per arm; rerun to resume"
    )
    ap.add_argument("--rebuild", type=Path, help="recompute the summary from a .calls.jsonl file; no API calls")
    args = ap.parse_args(argv)
    if args.rebuild:
        print(json.dumps(rebuild(args.rebuild), indent=2, default=str))
        return 0

    load_dotenv(ROOT / ".env", override=False)
    items = load_corpus()
    settings = Settings(
        mode=args.mode,
        experiments=[e for e in args.experiments.split(",") if e],
        arms=[a for a in args.arms.split(",") if a],
        sets=[s for s in args.sets.split(",") if s],
        limit=args.limit,
        drift_reps=args.drift_reps,
        adaptive_seeds=args.adaptive_seeds,
        adaptive_queries=args.adaptive_queries,
        corpus_sha=hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
        gemini_tpm=args.gemini_tpm,
    )
    for name, allowed in (("experiments", EXPERIMENTS), ("arms", ARMS), ("sets", SETS)):
        bad = set(getattr(settings, name)) - set(allowed)
        if bad:
            ap.error(f"unknown {name}: {sorted(bad)}")
    ctx = load_context()
    work = plan(settings, items)
    ledger = Ledger(LEDGER, live=args.mode == "live", past_half=args.past_half)

    if args.estimate:
        est = estimate(settings, ctx, work, ledger.spent)
        for r in est["rows"]:
            print(f"  {r['experiment']:9s} {r['arm']:16s} {r['calls']:6d} calls   ${r['usd']:.4f}")
        print(f"total ${est['total_usd']:.2f}   spent so far ${est['spent_so_far_usd']:.2f}   cap ${BUDGET_USD:.2f}")
        print(est["note"])
        return 0
    if args.mode == "live" and not args.yes:
        ap.error("live mode makes paid API calls; pass --yes")

    arms = [build_arm(a, args.mode, args.gemini_tpm) for a in settings.arms]
    for a in arms:
        print(f"arm {a.name:16s} {a.status:8s} {a.reason}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    ckpt_path = RESULTS / f"checkpoint-{args.mode}.jsonl"
    ckpt = Checkpoint(ckpt_path, settings.digest(), args.fresh)
    raw_path = RESULTS / f"raw-{args.mode}.jsonl"
    if args.fresh and raw_path.exists():
        raw_path.rename(raw_path.with_suffix(f".discarded-{datetime.now(UTC):%Y%m%dT%H%M%SZ}.jsonl"))
    runner = Runner(settings, ctx, ledger, ckpt, print, RawLog(raw_path), args.stop_after)
    code = 0
    try:
        asyncio.run(runner.run(work, arms))
    except* HalfBudgetReached as eg:
        print(f"\nHALF BUDGET CHECKPOINT: {eg.exceptions[0]}. Rerun with --past-half to continue.")
        code = 4
    except* BudgetExhausted as eg:
        print(f"\nBUDGET CAP: {eg.exceptions[0]}")
        code = 5
    print(f"\nlive spend so far ${ledger.spent:.4f} of ${BUDGET_USD:.2f}")
    if code:
        return code
    stopped = [a for a in arms if a.stop_reason]
    records = list(ckpt.done.values())
    summary = summarize(records, detector_rows(items, ctx))
    summary["settings"] = asdict(settings)
    summary["arms"] = {a.name: {"status": a.status, "reason": a.reason, "stopped": a.stop_reason} for a in arms}
    summary["eval_timeout_ms"] = EVAL_TIMEOUT_MS
    print_summary(summary, print)
    if stopped:
        print(f"\nstopped arms: {[a.name for a in stopped]}; results are partial, rerun to resume")
        return 3
    if any(a.paused for a in arms):
        print(f"\npaused after --stop-after {args.stop_after} new items per arm; rerun without it to resume")
        return 6
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = RESULTS / f"redteam-{args.mode}-{stamp}"
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    ckpt_path.rename(out.with_suffix(".calls.jsonl"))
    if raw_path.exists():
        raw_path.rename(out.with_suffix(".raw.jsonl"))
    print(f"\nwrote {out.with_suffix('.json')} and the calls file next to it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
