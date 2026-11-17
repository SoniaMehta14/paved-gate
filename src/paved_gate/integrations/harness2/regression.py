"""Regression check: ask any FastEvaluator whether an adopted harness edit looks worse than its baseline."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass

from pydantic import JsonValue

from paved_gate.evaluator.base import (
    EvaluationResult,
    EvaluatorError,
    FastEvaluator,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    TruthAnswer,
    TruthQuestion,
)
from paved_gate.integrations.harness2.config import RegressionConfig
from paved_gate.integrations.harness2.diff import FileChange
from paved_gate.integrations.harness2.run_dir import Tree

TRUNCATED = "\n[truncated]\n"
PROTECTED_RULES_CHARS = 8_000  # total budget for the protected rule files shown alongside the diff


@dataclass(frozen=True)
class RegressionOutcome:
    regression: float | None  # 1 based
    constraint_removal: float | None
    passed: bool | None  # None when the evaluator failed
    reasons: list[str]
    used_outputs: bool
    evaluation: EvaluationResult | None
    error: str | None
    latency_ms: float
    clipped: tuple[str, ...] = ()  # changed files the evaluator saw only in part


def build_questions(cfg: RegressionConfig) -> dict[str, Question]:
    return {
        "regression": ScoreQuestion(instructions=cfg.regression.instructions, criteria=cfg.regression.criteria),
        "constraint_removal": TruthQuestion(instructions=cfg.constraint_removal.instructions),
    }


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + TRUNCATED


def _by_run(outputs: Tree, limit: int) -> dict[str, JsonValue]:
    """{run label: that run's non empty output files as one text}, from `label/path` keys."""
    runs: dict[str, list[str]] = {}
    for key, text in sorted(outputs.items()):
        label, _, path = key.partition("/")
        parts = runs.setdefault(label, [])
        if text.strip():
            parts.append(f"### {path}\n{text}")
    return {label: _clip("\n".join(parts), limit) for label, parts in runs.items()}


def fair_shares(lengths: list[int], limit: int) -> list[int]:
    """Split `limit` characters across texts so short ones are kept whole and long ones share what is left."""
    shares = [0] * len(lengths)
    remaining = limit
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    for k, i in enumerate(order):
        shares[i] = min(lengths[i], remaining // (len(order) - k))
        remaining -= shares[i]
    return shares


def clip_per_file(texts: list[tuple[str, str]], limit: int) -> str:
    """Join (path, text) pairs within `limit` characters, clipping each file to a fair share rather than the
    joined text, so one large file cannot push every later file out of view. A clipped file says so."""
    shares = fair_shares([len(t) for _, t in texts], limit)
    out = []
    for (path, text), share in zip(texts, shares, strict=True):
        if share < len(text):
            text = text[:share] + f"\n[truncated: {len(text) - share} of {len(text)} characters of {path} not shown]\n"
        out.append(text)
    return "".join(out)


def clipped_paths(changes: list[FileChange], limit: int) -> tuple[str, ...]:
    """Paths whose diff clip_per_file cuts at this limit."""
    shares = fair_shares([len(c.unified) for c in changes], limit)
    return tuple(c.path for c, share in zip(changes, shares, strict=True) if share < len(c.unified))


def build_state(
    cfg: RegressionConfig,
    changes: list[FileChange],
    baseline_outputs: Tree,
    adopted_outputs: Tree,
    protected_rules: Mapping[str, str] | None = None,
) -> dict[str, JsonValue]:
    """The evaluator's input. `protected_rules` are the baseline text of protected rule files: they are usually
    unchanged, so they are absent from the diff, yet an edit elsewhere can contradict them."""
    state: dict[str, JsonValue] = {"diff": clip_per_file([(c.path, c.unified) for c in changes], cfg.max_diff_chars)}
    if protected_rules:
        rules = clip_per_file(
            [(p, f"### {p}\n{t}\n") for p, t in sorted(protected_rules.items())], PROTECTED_RULES_CHARS
        )
        state["protected_rules"] = rules
    # With baseline outputs on record, an adopted run that produced nothing is itself evidence.
    if baseline_outputs:
        state["baseline_outputs"] = _by_run(baseline_outputs, cfg.max_output_chars)
        state["adopted_outputs"] = _by_run(adopted_outputs, cfg.max_output_chars)
    return state


async def run_regression(
    evaluator: FastEvaluator,
    cfg: RegressionConfig,
    changes: list[FileChange],
    baseline_outputs: Tree,
    adopted_outputs: Tree,
    protected_rules: Mapping[str, str] | None = None,
) -> RegressionOutcome:
    state = build_state(cfg, changes, baseline_outputs, adopted_outputs, protected_rules)
    used_outputs = "baseline_outputs" in state
    started = time.perf_counter()
    try:
        evaluation = await evaluator.evaluate(state, build_questions(cfg))
        reg, con = evaluation.answers["regression"], evaluation.answers["constraint_removal"]
        if not isinstance(reg, ScoreAnswer) or not isinstance(con, TruthAnswer):
            raise EvaluatorError("evaluator returned the wrong answer types")
    except Exception as exc:  # evaluator failure is handled by fail_mode, like the main gate
        ms = round((time.perf_counter() - started) * 1000, 2)
        error = f"{type(exc).__name__}: {exc}"
        return RegressionOutcome(None, None, None, [f"regression check failed: {error}"], used_outputs, None, error, ms)

    ms = round((time.perf_counter() - started) * 1000, 2)
    level = round(reg.score + 1, 2)
    reasons: list[str] = []
    if level > cfg.regression.threshold:
        reasons.append(f"regression={level} (threshold {cfg.regression.threshold})")
    if con.probability > cfg.constraint_removal.threshold:
        reasons.append(f"constraint_removal={con.probability:.2f} (threshold {cfg.constraint_removal.threshold})")
    passed = not reasons
    clipped = clipped_paths(changes, cfg.max_diff_chars)
    if clipped:
        reasons.append(f"diff clipped for the evaluator ({', '.join(clipped)}): an unseen change cannot be accepted")
    return RegressionOutcome(level, con.probability, passed, reasons, used_outputs, evaluation, None, ms, clipped)
