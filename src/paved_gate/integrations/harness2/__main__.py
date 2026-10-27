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
from paved_gate.integrations.harness2.config import load_adoption_policy
from paved_gate.integrations.harness2.gate import build_gate
from paved_gate.integrations.harness2.run_dir import Adoption, RunDirError, discover

LABELS = {"accept": "ACCEPT", "reject": "REJECT", "needs_review": "REVIEW"}


def _record_name(entry: HarnessAdoptionEntry) -> str:
    return f"{entry.namespace}__{entry.task_id.replace('/', '__')}__{entry.adopted_label}.json"


async def _run(root: Path, policy_path: str, out: Path, provider: str | None, mode: str | None) -> int:
    policy = load_adoption_policy(policy_path)
    if provider or mode:
        update = {k: v for k, v in (("provider", provider), ("mode", mode)) if v}
        policy = policy.model_copy(update={"evaluator": policy.evaluator.model_copy(update=update)})
    try:
        found = discover(root)
    except RunDirError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    gate = build_gate(policy)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Harness² adoption gate: policy {policy.name}, evaluator {policy.evaluator.provider}, run {root}")
    counts = {"accept": 0, "reject": 0, "needs_review": 0, "skipped": 0}
    try:
        for item in found:
            if not isinstance(item, Adoption):
                counts["skipped"] += 1
                print(f"  SKIP    {item.selection_path.relative_to(root)}: {item.reason}")
                continue
            entry = await gate.evaluate(item, root)
            counts[entry.decision] += 1
            (out / _record_name(entry)).write_text(entry.model_dump_json(indent=2) + "\n", encoding="utf-8")
            first = entry.reasons[0] if entry.reasons else ""
            more = f" (+{len(entry.reasons) - 1} more)" if len(entry.reasons) > 1 else ""
            print(f"  {LABELS[entry.decision]:<7} {item.task_id} [{item.mode}] {item.label}: {first}{more}")
    finally:
        await gate.aclose()
    print(
        f"{counts['accept']} accepted, {counts['reject']} rejected, {counts['needs_review']} need review, "
        f"{counts['skipped']} skipped. Audit records: {out}"
    )
    adopted = counts["accept"] + counts["reject"] + counts["needs_review"]
    return 0 if adopted and counts["accept"] == adopted else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path, help="Harness² output directory, e.g. ../harness2/runs/demo")
    ap.add_argument("--policy", default="policy/harness2_adoption.policy.yaml")
    ap.add_argument("--out", type=Path, default=Path("logs/harness2"), help="directory for the JSON audit records")
    ap.add_argument("--evaluator", choices=["heuristic", "jev"], help="override the policy's evaluator provider")
    ap.add_argument("--mode", choices=["mock", "live"], help="override the policy's evaluator mode (jev only)")
    args = ap.parse_args()
    return asyncio.run(_run(args.run_dir, args.policy, args.out, args.evaluator, args.mode))


if __name__ == "__main__":
    sys.exit(main())
