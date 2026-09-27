"""ASGI middleware for FastAPI / Starlette (requires the `fastapi` extra).

Gated paths are evaluated before the app sees them:
  blocked       -> 403, the app is never called
  deterministic -> 200 with the handler output, the app is never called
  frontier      -> 200 with the frontier response if the gate has a frontier handler;
                   otherwise the request continues to the app with the GateResult on
                   `request.state.paved_gate`
"""

from __future__ import annotations

import json
from collections.abc import Callable, Collection

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from paved_gate.gate import PavedGate
from paved_gate.types import Blocked, Deterministic, GateResult

InputExtractor = Callable[[bytes], str]
_MAX_BODY_BYTES = 1_000_000


def json_field(field: str = "input") -> InputExtractor:
    def extract(body: bytes) -> str:
        try:
            data = json.loads(body or b"{}")
        except json.JSONDecodeError as exc:
            raise ValueError("request body must be JSON") from exc
        value = data.get(field) if isinstance(data, dict) else None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"request body must include a non-empty string field {field!r}")
        return value

    return extract


def _gate_headers(result: GateResult) -> dict[str, str]:
    return {
        "x-paved-gate-decision": result.decision.outcome,
        "x-paved-gate-latency-ms": f"{result.decision.gate_ms:.1f}",
        "x-paved-gate-request-id": result.decision.request_id,
    }


class PavedGateMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        gate: PavedGate,
        paths: Collection[str],
        extract_input: InputExtractor | None = None,
    ) -> None:
        self.app = app
        self.gate = gate
        self.paths = set(paths)
        self.extract_input = extract_input or json_field("input")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] not in self.paths:
            await self.app(scope, receive, send)
            return

        body = await self._read_body(receive)
        if body is None:
            await JSONResponse({"error": "payload_too_large"}, status_code=413)(scope, receive, send)
            return
        try:
            text = self.extract_input(body)
        except ValueError as exc:
            await JSONResponse({"error": "bad_request", "detail": str(exc)}, status_code=400)(scope, receive, send)
            return

        headers = dict(scope.get("headers") or [])
        request_id = headers.get(b"x-request-id", b"").decode() or None
        result = await self.gate.handle(text, request_id=request_id)
        out_headers = _gate_headers(result)

        if isinstance(result, Blocked):
            response = JSONResponse(
                {
                    "error": "policy_violation",
                    "request_id": result.decision.request_id,
                    "reasons": result.decision.reasons,
                },
                status_code=403,
                headers=out_headers,
            )
        elif isinstance(result, Deterministic):
            response = JSONResponse(
                {"request_id": result.decision.request_id, "handler": result.handler, "result": result.output},
                headers=out_headers,
            )
        elif result.response is not None:
            response = JSONResponse(
                {"request_id": result.decision.request_id, "frontier": result.response.model_dump(mode="json")},
                headers=out_headers,
            )
        else:
            scope.setdefault("state", {})["paved_gate"] = result
            await self.app(scope, _replay(body), _with_headers(send, out_headers))
            return
        await response(scope, receive, send)

    @staticmethod
    async def _read_body(receive: Receive) -> bytes | None:
        chunks: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > _MAX_BODY_BYTES:
                return None
            chunks.append(chunk)
            if not message.get("more_body", False):
                return b"".join(chunks)


def _replay(body: bytes) -> Receive:
    sent = False

    async def receive() -> Message:
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return receive


def _with_headers(send: Send, extra: dict[str, str]) -> Send:
    async def wrapped(message: Message) -> None:
        if message["type"] == "http.response.start":
            message = {
                **message,
                "headers": [*message.get("headers", []), *((k.encode(), v.encode()) for k, v in extra.items())],
            }
        await send(message)

    return wrapped
