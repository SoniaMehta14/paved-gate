"""Paved Gate as an NVIDIA OpenShell supervisor middleware.

OpenShell builds the jail: which hosts, ports and binaries a sandbox may reach. Paved Gate reads the mail:
what is inside the requests OpenShell already allowed, and the tool responses coming back to the agent.

Requires the `openshell` extra (`grpcio`, `protobuf`). Implements `openshell.middleware.v1` as of
OpenShell v0.1.2 (see proto/PROVENANCE.md).

    python -m paved_gate.integrations.openshell --policy policy/openshell_egress.policy.yaml --bind 127.0.0.1:50051
"""

from paved_gate.integrations.openshell.config import EgressPolicy, egress_policy_hash, load_egress_policy
from paved_gate.integrations.openshell.inspector import EgressInspector, Finding, Target, Verdict
from paved_gate.integrations.openshell.server import build_inspector, build_server

__all__ = [
    "EgressInspector",
    "EgressPolicy",
    "Finding",
    "Target",
    "Verdict",
    "build_inspector",
    "build_server",
    "egress_policy_hash",
    "load_egress_policy",
]
