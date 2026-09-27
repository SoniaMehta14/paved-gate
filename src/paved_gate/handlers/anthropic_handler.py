"""FRONTIER_AGENT_REQUIRED passthrough to Claude (requires the `anthropic` extra)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from paved_gate.types import FrontierResponse, GateState

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic

_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicFrontierHandler:
    """Sends the original (unmasked) request to Claude.

    Server-side refusal fallbacks are on by default: if the model declines on
    policy grounds, the API re-runs the request on a fallback model within the
    same call. A final `stop_reason == "refusal"` means the whole chain declined.
    """

    provider = "anthropic"

    def __init__(
        self,
        *,
        model: str = "claude-opus-5",
        max_tokens: int = 16000,
        system: str | None = None,
        use_fallbacks: bool = True,
        workspace_id: str | None = None,
        client: AsyncAnthropic | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.system = system
        self.use_fallbacks = use_fallbacks
        # Needed only for API keys not scoped to a workspace (the API then requires the header).
        self.workspace_id = workspace_id or os.environ.get("ANTHROPIC_WORKSPACE_ID") or None
        self._client = client

    def _get_client(self) -> AsyncAnthropic:
        if self._client is None:
            from anthropic import AsyncAnthropic

            headers = {"anthropic-workspace-id": self.workspace_id} if self.workspace_id else None
            self._client = AsyncAnthropic(default_headers=headers)
        return self._client

    async def respond(self, state: GateState) -> FrontierResponse:
        from anthropic import omit

        client = self._get_client()
        message = await client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            thinking={"type": "adaptive"},
            messages=[{"role": "user", "content": state.input}],
            system=self.system or omit,
            betas=[_FALLBACK_BETA] if self.use_fallbacks else omit,
            fallbacks="default" if self.use_fallbacks else omit,
        )
        text = ""
        if message.stop_reason != "refusal":
            text = "".join(block.text for block in message.content if block.type == "text")
        return FrontierResponse(
            provider=self.provider,
            model=message.model,
            text=text,
            stop_reason=message.stop_reason,
            input_tokens=message.usage.input_tokens,
            output_tokens=message.usage.output_tokens,
        )
