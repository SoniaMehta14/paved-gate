"""FRONTIER_AGENT_REQUIRED passthrough to OpenAI (requires the `openai` extra)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from paved_gate.types import FrontierResponse, GateState

if TYPE_CHECKING:
    from openai import AsyncOpenAI


class OpenAIFrontierHandler:
    provider = "openai"

    def __init__(
        self,
        *,
        model: str = "gpt-4o",
        max_tokens: int = 4096,
        system: str | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.system = system
        self._client = client

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI()
        return self._client

    async def respond(self, state: GateState) -> FrontierResponse:
        client = self._get_client()
        messages: list[dict[str, str]] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": state.input})
        completion = await client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages,  # type: ignore[arg-type]
        )
        choice = completion.choices[0]
        return FrontierResponse(
            provider=self.provider,
            model=completion.model,
            text=choice.message.content or "",
            stop_reason=choice.finish_reason,
            input_tokens=completion.usage.prompt_tokens if completion.usage else None,
            output_tokens=completion.usage.completion_tokens if completion.usage else None,
        )
