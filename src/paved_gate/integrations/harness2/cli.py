"""Run the Harness² edit adoption gate over a Harness² output directory.

    python -m paved_gate.integrations.harness2 ../harness2/runs/demo
    python -m paved_gate.integrations.harness2 ../harness2/runs/demo --policy examples/harness2/demo.policy.yaml

Writes one JSON audit record per adoption to --out and prints one line per adoption.
Exit code 0 when every adoption is accepted, 1 otherwise.
API keys (TYPESAFE_API_KEY for a live Jev evaluator) are read from the environment.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from paved_gate.audit.models import HarnessAdoptionEntry
from paved_gate.integrations.harness2.config import AdoptionPolicy, load_adoption_policy
from paved_gate.integrations.harness2.gate import run_gate
from paved_gate.integrations.harness2.run_dir import RunDirError, Skipped

LABELS = {"accept": "ACCEPT", "reject": "REJECT", "needs_review": "REVIEW"}


def _record_name(entry: HarnessAdoptionEntry) -> str:
    return f"{entry.namespace}__{entry.task_id.replace('/', '__')}__{entry.adopted_label}.json"


def with_evaluator(policy: AdoptionPolicy, provider: str | None, mode: str | None) -> AdoptionPolicy:
    update = {k: v for k, v in (("provider", provider), ("mode", mode)) if v}
    if not update:
        return policy
    return policy.model_copy(update={"evaluator": policy.evaluator.model_copy(update=update)})


def write_records(entries: list[HarnessAdoptionEntry], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        (out / _record_name(entry)).write_text(entry.model_dump_json(indent=2) + "\n", encoding="utf-8")


def print_summary(entries: list[HarnessAdoptionEntry], skipped: list[Skipped], root: Path, out: Path) -> None:
    for entry in entries:
        first = entry.reasons[0] if entry.reasons else ""
        more = f" (+{len(entry.reasons) - 1} more)" if len(entry.reasons) > 1 else ""
        print(f"  {LABELS[entry.decision]:<7} {entry.task_id} [{entry.mode}] {entry.adopted_label}: {first}{more}")
    for item in skipped:
        print(f"  SKIP    {item.selection_path.relative_to(root)}: {item.reason}")
    counts = {d: sum(e.decision == d for e in entries) for d in LABELS}
    print(
        f"{counts['accept']} accepted, {counts['reject']} rejected, {counts['needs_review']} need review, "
        f"{len(skipped)} skipped. Audit records: {out}"
    )


async def _run(root: Path, policy_path: str, out: Path, provider: str | None, mode: str | None) -> int:
    policy = with_evaluator(load_adoption_policy(policy_path), provider, mode)
    print(f"Harness² adoption gate: policy {policy.name}, evaluator {policy.evaluator.provider}, run {root}")
    try:
        entries, skipped = await run_gate(root, policy)
    except RunDirError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    write_records(entries, out)
    print_summary(entries, skipped, root, out)
    return 0 if entries and all(e.decision == "accept" for e in entries) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="Harness² output directory, e.g. ../harness2/runs/demo")
    ap.add_argument("--policy", default="policy/harness2_adoption.policy.yaml")
    ap.add_argument("--out", type=Path, default=Path("logs/harness2"), help="directory for the JSON audit records")
    ap.add_argument("--evaluator", choices=["heuristic", "jev"], help="override the policy's evaluator provider")
    ap.add_argument("--mode", choices=["mock", "live"], help="override the policy's evaluator mode (jev only)")
    args = ap.parse_args()
    return asyncio.run(_run(args.run_dir, args.policy, args.out, args.evaluator, args.mode))
