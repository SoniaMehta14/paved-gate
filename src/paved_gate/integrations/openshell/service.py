"""gRPC services implementing OpenShell's supervisor-middleware contract (openshell.middleware.v1, v0.1.2)."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping

import grpc
from google.protobuf import struct_pb2

from paved_gate.integrations.openshell._gen import extension_pb2 as ext
from paved_gate.integrations.openshell._gen import supervisor_middleware_pb2 as pb
from paved_gate.integrations.openshell._gen import supervisor_middleware_pb2_grpc as pbg
from paved_gate.integrations.openshell.config import CONFIG_KEYS
from paved_gate.integrations.openshell.inspector import EgressInspector, Finding, Target, Verdict

PROTOCOL_MAJOR, PROTOCOL_MINOR = 1, 0  # openshell-core extension_protocol.rs @ v0.1.2
CONTRACT_CAPABILITY = "openshell.supervisor-middleware.contract"
IMPLEMENTATION_NAME = "paved-gate/openshell-egress"
MAX_PAYLOAD_BYTES = 1024 * 1024
TEXT_TYPES = ("text/", "application/json", "application/xml", "application/x-www-form-urlencoded", "+json", "+xml")


def peer_metadata(version: str) -> ext.PeerMetadata:
    return ext.PeerMetadata(
        protocol_version=ext.ProtocolVersion(major=PROTOCOL_MAJOR, minor=PROTOCOL_MINOR),
        implementation_name=IMPLEMENTATION_NAME,
        implementation_version=version,
        supported_capabilities=[CONTRACT_CAPABILITY],
        required_capabilities=[CONTRACT_CAPABILITY],
    )


def parse_config(config: struct_pb2.Struct | None) -> tuple[dict[str, str], str | None]:
    """Per-sandbox overrides from the OpenShell policy. Returns (values, error)."""
    values: dict[str, str] = {}
    if config is None:
        return values, None
    for key, value in config.fields.items():
        allowed = CONFIG_KEYS.get(key)
        if allowed is None:
            return {}, f"unknown config key {key!r}; allowed: {sorted(CONFIG_KEYS)}"
        if value.WhichOneof("kind") != "string_value" or value.string_value not in allowed:
            return {}, f"config.{key} must be one of {list(allowed)}"
        values[key] = value.string_value
    return values, None


def _pb_findings(findings: tuple[Finding, ...]) -> list[pb.Finding]:
    return [
        pb.Finding(type=f.type, label=f.label, count=f.count, confidence=f.confidence, severity=f.severity)
        for f in findings
    ]


def _metadata(v: Verdict) -> Mapping[str, str]:
    md = {
        "paved_gate.decision": v.decision,
        "paved_gate.total_ms": f"{v.total_ms:.2f}",
        "paved_gate.tainted": str(v.tainted).lower(),
    }
    if v.evaluator_ms is not None:
        md["paved_gate.evaluator_ms"] = f"{v.evaluator_ms:.2f}"
    md.update({f"paved_gate.score.{k}": f"{s}" for k, s in v.scores.items()})
    return md


def _target(context: pb.RequestContext, target: pb.HttpRequestTarget) -> Target:
    return Target(
        request_id=context.request_id,
        sandbox_id=context.sandbox_id,
        method=target.method,
        host=target.host,
        port=target.port,
        path=target.path,
        query=target.query,
    )


class SupervisorMiddlewareService(pbg.SupervisorMiddlewareServicer):
    def __init__(self, inspector: EgressInspector, version: str) -> None:
        self.inspector = inspector
        self.version = version

    async def Describe(
        self, request: pb.MiddlewareDescribeRequest, context: grpc.aio.ServicerContext
    ) -> pb.MiddlewareManifest:
        gateway = request.gateway
        if gateway.HasField("protocol_version") and gateway.protocol_version.major != PROTOCOL_MAJOR:
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION, f"unsupported protocol major {gateway.protocol_version.major}"
            )
        missing = set(gateway.required_capabilities) - {CONTRACT_CAPABILITY}
        if missing:
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION, f"unsupported required capabilities {sorted(missing)}"
            )
        return pb.MiddlewareManifest(
            name=IMPLEMENTATION_NAME,
            bindings=[
                pb.MiddlewareBinding(
                    operation=pb.SUPERVISOR_MIDDLEWARE_OPERATION_HTTP_REQUEST,
                    phase=pb.SUPERVISOR_MIDDLEWARE_PHASE_PRE_CREDENTIALS,
                    max_payload_bytes=MAX_PAYLOAD_BYTES,
                ),
                pb.MiddlewareBinding(
                    operation=pb.SUPERVISOR_MIDDLEWARE_OPERATION_HTTP_RESPONSE,
                    phase=pb.SUPERVISOR_MIDDLEWARE_PHASE_PRE_RETURN,
                    max_payload_bytes=MAX_PAYLOAD_BYTES,
                ),
            ],
            extension=peer_metadata(self.version),
        )

    async def ValidateConfig(
        self, request: pb.ValidateConfigRequest, context: grpc.aio.ServicerContext
    ) -> pb.ValidateConfigResponse:
        _, error = parse_config(request.config if request.HasField("config") else None)
        return pb.ValidateConfigResponse(valid=error is None, reason=error or "")

    async def EvaluateHttpRequest(
        self, request: pb.HttpRequestEvaluation, context: grpc.aio.ServicerContext
    ) -> pb.HttpRequestResult:
        if request.phase != pb.SUPERVISOR_MIDDLEWARE_PHASE_PRE_CREDENTIALS:
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT, "paved-gate only evaluates requests at PRE_CREDENTIALS"
            )
        cfg, error = parse_config(request.config if request.HasField("config") else None)
        if error:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, error)
        verdict = await self.inspector.inspect_request(
            _target(request.context, request.target), request.body, enforce=cfg.get("mode", "enforce") == "enforce"
        )
        return pb.HttpRequestResult(
            decision=pb.DECISION_DENY if verdict.deny else pb.DECISION_ALLOW,
            reason=(verdict.error or verdict.reason_code or "")[:4000],
            reason_code=verdict.reason_code or "",
            findings=_pb_findings(verdict.findings),
            metadata=_metadata(verdict),
        )

    async def EvaluateWebSocketSession(
        self, request_iterator: AsyncIterator[pb.WebSocketSessionEvent], context: grpc.aio.ServicerContext
    ) -> AsyncIterator[pb.WebSocketSessionEventResult]:
        # Not advertised in Describe; OpenShell should never call it.
        await context.abort(grpc.StatusCode.UNIMPLEMENTED, "paved-gate does not inspect WebSocket traffic")
        yield pb.WebSocketSessionEventResult()  # pragma: no cover


def _is_text(headers: list[pb.HttpHeader]) -> bool:
    ctype = next((h.value.lower() for h in headers if h.name == "content-type"), "")
    return any(t in ctype for t in TEXT_TYPES)


class ResponsePreReturnService(pbg.HttpResponsePreReturnServicer):
    """Scores tool/API responses for prompt injection before OpenShell returns them to the agent."""

    def __init__(self, inspector: EgressInspector, *, default_mode: str) -> None:
        self.inspector = inspector
        self.default_mode = default_mode

    async def Evaluate(
        self, request_iterator: AsyncIterator[pb.HttpResponseEvent], context: grpc.aio.ServicerContext
    ) -> AsyncIterator[pb.HttpResponseEventResult]:
        target: Target | None = None
        block = self.default_mode == "block"
        buffer = bytearray()
        async for event in request_iterator:
            kind = event.WhichOneof("event")
            if kind == "preflight":
                pf = event.preflight
                cfg, _ = parse_config(pf.config if pf.HasField("config") else None)
                block = cfg.get("response_mode", self.default_mode) == "block"
                target = _target(pf.context, pf.target)
                if pb.HTTP_RESPONSE_BODY_MODE_WHOLE_BODY_BYTES in pf.permitted_body_modes and _is_text(
                    list(pf.headers)
                ):
                    yield pb.HttpResponseEventResult(
                        preflight_result=pb.HttpResponsePreflightResult(
                            inspect=pb.HttpResponsePreflightInspect(
                                body_mode=pb.HTTP_RESPONSE_BODY_MODE_WHOLE_BODY_BYTES
                            )
                        )
                    )
                else:
                    yield pb.HttpResponseEventResult(
                        preflight_result=pb.HttpResponsePreflightResult(
                            skip=pb.HttpResponsePreflightSkip(),
                            findings=[
                                pb.Finding(
                                    type="response_not_inspected",
                                    label="binary, encoded or oversized response",
                                    count=1,
                                    confidence="high",
                                    severity="low",
                                )
                            ],
                        )
                    )
                    return  # skip ends this stage: OpenShell sends no body events for it
            elif kind == "body":
                unit = event.body
                buffer.extend(unit.data)
                if not unit.end_of_stream or target is None:
                    yield pb.HttpResponseEventResult(
                        body_result=pb.HttpResponseBodyResult(
                            sequence=unit.sequence, pass_through=pb.HttpResponseBodyPassThrough()
                        )
                    )
                    continue
                verdict = await self.inspector.inspect_response(target, bytes(buffer), block=block)
                result = pb.HttpResponseBodyResult(
                    sequence=unit.sequence,
                    reason_code=verdict.reason_code or "",
                    findings=_pb_findings(verdict.findings),
                    metadata=_metadata(verdict),
                )
                if verdict.deny:
                    result.block_delivery.SetInParent()
                else:
                    result.pass_through.SetInParent()
                yield pb.HttpResponseEventResult(body_result=result)
            elif kind == "trailers":
                yield pb.HttpResponseEventResult(trailers_result=pb.HttpResponseTrailersResult())
            # session_end: no result
