#!/usr/bin/env bash
# Build three adoption gate reports into results/harness2/, ready to open and screenshot:
#   default_policy.html    the Harness² demo output under the default policy
#   example_policy.html    the same demo output under examples/harness2/demo.policy.yaml
#   synthetic_fixture.html a synthetic guardrail fixture under the default policy, labelled synthetic
#
# Needs the Harness² demo output at ../harness2/runs/demo (see the README quickstart),
# or pass its location: examples/harness2/reports.sh path/to/harness2/runs/demo
set -euo pipefail

DEMO="${1:-../harness2/runs/demo}"
OUT="results/harness2"
REPORT=(uv run python -m paved_gate.integrations.harness2.report)

"${REPORT[@]}" "$DEMO" --policy policy/harness2_adoption.policy.yaml \
  --records "$OUT/records/default_policy" --html "$OUT/default_policy.html" \
  --title "Harness² demo under the default policy" \
  --subtitle "The demo editor changes scripts/summarize.py. Scripts are protected, so the gate rejects every adoption."

"${REPORT[@]}" "$DEMO" --policy examples/harness2/demo.policy.yaml \
  --records "$OUT/records/example_policy" --html "$OUT/example_policy.html" \
  --title "Harness² demo under the example policy" \
  --subtitle "A path rule makes scripts/summarize.py editable, so the same edits are accepted."

uv run python examples/harness2/synthetic_run.py "$OUT/synthetic_run"
"${REPORT[@]}" "$OUT/synthetic_run" --policy policy/harness2_adoption.policy.yaml --synthetic \
  --records "$OUT/records/synthetic_fixture" --html "$OUT/synthetic_fixture.html" \
  --title "Synthetic guardrail fixture under the default policy" \
  --subtitle "Hand written edits that touch guardrails, skills, subagents and the system prompt, to show every kind of decision."

echo "Reports in $OUT/"
