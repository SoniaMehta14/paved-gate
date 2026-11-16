"""Check an image or PDF, plus optional text, against the multimodal gate policy in one evaluator call.

    python -m paved_gate.integrations.multimodal_gate docs/harness2/1_synthetic_fixture.png \\
        --expect overall_decision=reject
    python -m paved_gate.integrations.multimodal_gate tests/fixtures/multimodal/synthetic_policy.pdf --mode live

Prints the decision (allow, review or block) and the reason for each failed check, and appends a JSON audit
record to --audit. Exit code 0 on allow, 1 otherwise. GEMINI_API_KEY is read from the environment in live mode.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from paved_gate.integrations.multimodal_gate.config import load_multimodal_policy
from paved_gate.integrations.multimodal_gate.gate import MultimodalGateError, build_gate


async def _run(args: argparse.Namespace) -> int:
    policy = load_multimodal_policy(args.policy)
    evaluator = policy.evaluator
    if args.evaluator:
        evaluator = evaluator.with_provider(args.evaluator)
    if args.mode:
        evaluator = evaluator.model_copy(update={"mode": args.mode})
    policy = policy.model_copy(update={"evaluator": evaluator})
    text = args.text_file.read_text(encoding="utf-8") if args.text_file else args.text
    expect = dict(pair.split("=", 1) for pair in args.expect)
    try:
        gate = build_gate(policy)
    except MultimodalGateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        entry = await gate.evaluate(args.artifact, text, expect)
    except MultimodalGateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        await gate.aclose()
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    with args.audit.open("a", encoding="utf-8") as f:
        f.write(entry.model_dump_json() + "\n")
    mode = entry.evaluator.mode if entry.evaluator else policy.evaluator.mode
    print(f"{entry.decision.upper()}  {entry.artifact.name} ({entry.artifact.mime_type}, {mode})")
    for reason in entry.reasons:
        print(f"  {reason}")
    if mode == "mock":
        print("  (mock evaluator: keyword heuristics, not a judgment of the file)")
    return 0 if entry.decision == "allow" else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("artifact", type=Path, help="a PNG, JPEG, WebP or PDF file")
    ap.add_argument("--text", default="", help="text state sent with the file")
    ap.add_argument("--text-file", type=Path, help="read the text state from a file")
    ap.add_argument("--expect", action="append", default=[], metavar="CHECK=OPTION", help="expected choice")
    ap.add_argument("--policy", default="policy/multimodal_gate.policy.yaml")
    ap.add_argument("--evaluator", choices=["gemini_decision", "jev", "heuristic"], help="override the provider")
    ap.add_argument("--mode", choices=["mock", "live"], help="override the evaluator mode")
    ap.add_argument("--audit", type=Path, default=Path("logs/multimodal_gate.audit.jsonl"))
    args = ap.parse_args()
    if any("=" not in pair for pair in args.expect):
        ap.error("--expect takes CHECK=OPTION")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
