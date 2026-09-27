from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import JsonValue

from paved_gate.types import FrontierResponse, GateState


@runtime_checkable
class DeterministicHandler(Protocol):
    """Returns None when the input is not something this handler can compute."""

    name: str

    def try_run(self, text: str) -> dict[str, JsonValue] | None: ...


@runtime_checkable
class FrontierHandler(Protocol):
    provider: str

    async def respond(self, state: GateState) -> FrontierResponse: ...
