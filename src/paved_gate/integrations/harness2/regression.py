"""Regression check: ask any FastEvaluator whether an adopted harness edit looks worse than its baseline."""

from __future__ import annotations

import time
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


def build_questions(cfg: RegressionConfig) -> dict[str, Question]:
    return {
        "regression": ScoreQuestion(instructions=cfg.regression.instructions, criteria=cfg.regression.criteria),
        "constraint_removal": TruthQuestion(instructions=cfg.constraint_removal.instructions),
    }


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + TRUNCATED


def _by_run(outputs: Tree, limit: int) -> dict[str, JsonValue]:
    """{run label: that run's output files as one text}, from `label/path` keys."""
    runs: dict[str, list[str]] = {}
    for key, text in sorted(outputs.items()):
        label, _, path = key.partition("/")
        runs.setdefault(label, []).append(f"### {path}\n{text}")
    return {label: _clip("\n".join(parts), limit) for label, parts in runs.items()}


def build_state(
    cfg: RegressionConfig, changes: list[FileChange], baseline_outputs: Tree, adopted_outputs: Tree
) -> dict[str, JsonValue]:
    state: dict[str, JsonValue] = {"diff": _clip("".join(c.unified for c in changes), cfg.max_diff_chars)}
    if baseline_outputs and adopted_outputs:
        state["baseline_outputs"] = _by_run(baseline_outputs, cfg.max_output_chars)
        state["adopted_outputs"] = _by_run(adopted_outputs, cfg.max_output_chars)
    return state


async def run_regression(
    evaluator: FastEvaluator,
    cfg: RegressionConfig,
    changes: list[FileChange],
    baseline_outputs: Tree,
    adopted_outputs: Tree,
) -> RegressionOutcome:
    state = build_state(cfg, changes, baseline_outputs, adopted_outputs)
    used_outputs = "adopted_outputs" in state
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
    return RegressionOutcome(level, con.probability, not reasons, reasons, used_outputs, evaluation, None, ms)
