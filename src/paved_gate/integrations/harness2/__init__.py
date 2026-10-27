"""Paved Gate as an edit adoption gate for Harness² (google-research/harness2).

Harness² rewrites an agent's harness and adopts the composed harness without ground truth
verification. This gate sits between the composed harness and its adoption: it diffs the harness
per component, checks the edit against a YAML adoption policy, runs a regression check through any
FastEvaluator, and decides accept, reject or needs_review with a JSON audit record.

Unofficial integration; Harness² is not a Google supported product.

    python -m paved_gate.integrations.harness2 ../harness2/runs/demo --policy policy/harness2_adoption.policy.yaml
"""

from paved_gate.integrations.harness2.config import AdoptionPolicy, adoption_policy_hash, load_adoption_policy
from paved_gate.integrations.harness2.deterministic import HarnessRegressionHeuristic
from paved_gate.integrations.harness2.gate import AdoptionGate, build_gate
from paved_gate.integrations.harness2.run_dir import Adoption, RunDirError, Skipped, discover

__all__ = [
    "Adoption",
    "AdoptionGate",
    "AdoptionPolicy",
    "HarnessRegressionHeuristic",
    "RunDirError",
    "Skipped",
    "adoption_policy_hash",
    "build_gate",
    "discover",
    "load_adoption_policy",
]
