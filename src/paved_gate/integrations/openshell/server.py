"""Build and run the Paved Gate supervisor-middleware gRPC server."""

from __future__ import annotations

from pathlib import Path

import grpc

from paved_gate.audit.sink import AuditSink, sink_from_config
from paved_gate.evaluator.base import FastEvaluator
from paved_gate.gate import evaluator_from_config
from paved_gate.integrations.openshell._gen import supervisor_middleware_pb2_grpc as pbg
from paved_gate.integrations.openshell.config import EgressPolicy, egress_policy_hash, load_egress_policy
from paved_gate.integrations.openshell.inspector import EgressInspector
from paved_gate.integrations.openshell.service import ResponsePreReturnService, SupervisorMiddlewareService

VERSION = "0.1.0"
# OpenShell sends request bodies up to 4 MiB; gRPC's default receive limit is 4 MiB (the docs say raise it).
MAX_MESSAGE_BYTES = 5 * 1024 * 1024


def build_inspector(
    policy: EgressPolicy | str | Path,
    *,
    evaluator: FastEvaluator | None = None,
    audit: AuditSink | None = None,
) -> EgressInspector:
    p = policy if isinstance(policy, EgressPolicy) else load_egress_policy(policy)
    return EgressInspector(
        policy=p,
        evaluator=evaluator or evaluator_from_config(p.evaluator),
        audit=audit or sink_from_config(p.audit),
        policy_hash=egress_policy_hash(p),
    )


def build_server(inspector: EgressInspector, bind: str) -> tuple[grpc.aio.Server, int]:
    """Returns (server, bound port). Use bind "127.0.0.1:0" for an ephemeral port in tests."""
    server = grpc.aio.server(
        options=[
            ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
            ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
        ]
    )
    pbg.add_SupervisorMiddlewareServicer_to_server(  # type: ignore[no-untyped-call]
        SupervisorMiddlewareService(inspector, VERSION), server
    )
    pbg.add_HttpResponsePreReturnServicer_to_server(  # type: ignore[no-untyped-call]
        ResponsePreReturnService(inspector, default_mode=inspector.policy.response_mode), server
    )
    port = server.add_insecure_port(bind)
    return server, port
