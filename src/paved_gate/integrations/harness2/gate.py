"""The adoption gate: may a Harness² composed harness replace its baseline?"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from paved_gate.audit.models import (
    AdoptionComponentChange,
    AdoptionFileResult,
    AdoptionRegression,
    AdoptionTriggerLink,
    EvaluatorInfo,
    HarnessAdoptionEntry,
    LatencyInfo,
)
from paved_gate.evaluator.base import FastEvaluator
from paved_gate.gate import evaluator_from_config
from paved_gate.integrations.harness2.config import AdoptionPolicy, adoption_policy_hash, load_adoption_policy
from paved_gate.integrations.harness2.deterministic import HarnessRegressionHeuristic
from paved_gate.integrations.harness2.diff import FileChange, diff_trees
from paved_gate.integrations.harness2.policy_check import FileResult, check_policy
from paved_gate.integrations.harness2.regression import RegressionOutcome, run_regression
from paved_gate.integrations.harness2.run_dir import Adoption, Skipped, discover, tree_sha256

Decision = Literal["accept", "reject", "needs_review"]
UNMAPPED = "(unmapped)"


def evaluator_for(policy: AdoptionPolicy) -> FastEvaluator:
    if policy.evaluator.provider == "heuristic":
        return HarnessRegressionHeuristic()
    return evaluator_from_config(policy.evaluator)


def _components(results: list[FileResult]) -> list[AdoptionComponentChange]:
    grouped: dict[str, list[FileChange]] = {}
    for r in results:
        grouped.setdefault(r.component or UNMAPPED, []).append(r.change)
    return [
        AdoptionComponentChange(
            component=name,
            added=[c.path for c in changes if c.kind == "added"],
            removed=[c.path for c in changes if c.kind == "removed"],
            modified=[c.path for c in changes if c.kind == "modified"],
            lines_added=sum(c.lines_added for c in changes),
            lines_removed=sum(c.lines_removed for c in changes),
        )
        for name, changes in grouped.items()
    ]


def _decide(policy: AdoptionPolicy, results: list[FileResult], reg: RegressionOutcome | None) -> Decision:
    if any(r.result == "reject" for r in results):
        return "reject"
    if reg is not None and reg.passed is False:
        return "reject"
    if reg is not None and reg.passed is None and policy.evaluator.fail_mode == "closed":
        return "reject"
    if any(r.result == "review" for r in results) or (reg is not None and reg.passed is None):
        return "needs_review"
    return "accept"


def _rel(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


@dataclass
class AdoptionGate:
    policy: AdoptionPolicy
    evaluator: FastEvaluator
    policy_hash: str

    async def evaluate(self, adoption: Adoption, root: Path) -> HarnessAdoptionEntry:
        started = time.perf_counter()
        changes = diff_trees(adoption.baseline, adoption.adopted)
        check = check_policy(self.policy, changes, adoption.adopted)
        reg = (
            await run_regression(
                self.evaluator,
                self.policy.regression,
                changes,
                adoption.baseline_outputs,
                adoption.adopted_outputs,
            )
            if changes
            else None
        )
        decision = _decide(self.policy, check.files, reg)

        reasons = [r.reason for r in check.files if r.result != "ok"]
        if reg is not None:
            reasons += reg.reasons
        if not changes:
            reasons.append("adopted harness is identical to the baseline")
        elif decision == "accept":
            reasons.append(f"{len(changes)} file(s) changed, all allowed by policy; regression check passed")

        cfg = self.policy.regression
        return HarnessAdoptionEntry(
            task_id=adoption.task_id,
            mode=adoption.mode,
            k=adoption.k,
            namespace=adoption.namespace,
            adopted_label=adoption.label,
            adopted_is_base=adoption.is_base,
            policy_name=self.policy.name,
            policy_hash=self.policy_hash,
            decision=decision,
            reasons=reasons,
            components=_components(check.files),
            files=[
                AdoptionFileResult(
                    path=r.change.path,
                    component=r.component,
                    change=r.change.kind,
                    access=r.access,
                    source=r.source,
                    result=r.result,
                    reason=r.reason,
                )
                for r in check.files
            ],
            skill_trigger_lines=self.policy.skill_trigger_lines,
            skill_trigger_file=self.policy.skill_trigger_format.file,
            skill_trigger_note=check.trigger_note,
            skill_triggers=[
                AdoptionTriggerLink(
                    line=t.line,
                    skill=t.skill,
                    skill_file=t.skill_file,
                    skill_added=t.skill_added,
                    skill_policy_ok=t.skill_policy_ok,
                    regression_passed=reg.passed if reg is not None else None,
                    accepted=t.accepted,
                    reason=t.reason,
                )
                for t in check.triggers
            ],
            regression=AdoptionRegression(
                ran=reg is not None,
                passed=reg.passed if reg is not None else None,
                regression=reg.regression if reg is not None else None,
                regression_threshold=cfg.regression.threshold,
                constraint_removal=reg.constraint_removal if reg is not None else None,
                constraint_removal_threshold=cfg.constraint_removal.threshold,
                used_outputs=reg.used_outputs if reg is not None else False,
                raw_evaluator_response=reg.evaluation.raw if reg is not None and reg.evaluation else None,
                error=reg.error if reg is not None else None,
                fail_mode=self.policy.evaluator.fail_mode,
            ),
            evaluator=(
                EvaluatorInfo(name=reg.evaluation.evaluator, model=reg.evaluation.model, mode=reg.evaluation.mode)
                if reg is not None and reg.evaluation
                else None
            ),
            latency=LatencyInfo(
                evaluator_ms=reg.latency_ms if reg is not None else None,
                gate_ms=round((time.perf_counter() - started) * 1000, 2),
            ),
            baseline_sha256=tree_sha256(adoption.baseline),
            adopted_sha256=tree_sha256(adoption.adopted),
            selection=_rel(adoption.selection_path, root) or "",
            reflection_note=_rel(adoption.reflection_note, root),
            lesson=_rel(adoption.lesson, root),
            experience=adoption.experience,
        )

    async def aclose(self) -> None:
        await self.evaluator.aclose()


def build_gate(policy: AdoptionPolicy | str | Path, *, evaluator: FastEvaluator | None = None) -> AdoptionGate:
    p = policy if isinstance(policy, AdoptionPolicy) else load_adoption_policy(policy)
    return AdoptionGate(policy=p, evaluator=evaluator or evaluator_for(p), policy_hash=adoption_policy_hash(p))


async def run_gate(
    root: Path, policy: AdoptionPolicy, *, evaluator: FastEvaluator | None = None
) -> tuple[list[HarnessAdoptionEntry], list[Skipped]]:
    """Every adoption under a Harness² output directory, through one gate. Raises RunDirError."""
    found = discover(root)
    gate = build_gate(policy, evaluator=evaluator)
    entries: list[HarnessAdoptionEntry] = []
    try:
        for item in found:
            if isinstance(item, Adoption):
                entries.append(await gate.evaluate(item, root))
    finally:
        await gate.aclose()
    return entries, [item for item in found if isinstance(item, Skipped)]
