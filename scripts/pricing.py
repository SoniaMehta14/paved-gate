"""List prices, USD per million tokens. Each entry records where and when it was verified. Prices change.

Edit these values to match your contract; the benchmark only multiplies them by the
token counts each provider reports.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float
    source: str
    cached_input_per_mtok: float | None = None  # None: cached input billed at the input rate


PRICES: dict[str, Price] = {
    # TypeSafe list price: input billed, output and cached input free.
    "jev": Price(0.042, 0.0, "TypeSafe list price, verified 2026-09-27 at docs.typesafe.ai/models"),
    "claude-opus-5": Price(5.00, 25.00, "Anthropic list price, verified 2026-09-27 at claude.com/pricing"),
    "claude-sonnet-5": Price(2.00, 10.00, "Anthropic list price, verified 2026-09-27 at claude.com/pricing"),
    "claude-haiku-4-5": Price(1.00, 5.00, "Anthropic list price, verified 2026-09-27 at claude.com/pricing"),
    "gpt-4o": Price(
        2.50, 10.00, "OpenAI list price, verified 2026-10-15 at developers.openai.com/api/docs/pricing", 1.25
    ),
    "gpt-4o-mini": Price(
        0.15, 0.60, "OpenAI list price, verified 2026-10-15 at developers.openai.com/api/docs/pricing", 0.075
    ),
    # OpenAI's flagship on 2026-10-15 ("our flagship model for complex reasoning and coding",
    # developers.openai.com/api/docs/models). A reasoning model: reasoning tokens are billed as output.
    "gpt-6-astra": Price(
        10.00, 50.00, "OpenAI list price, verified 2026-10-15 at developers.openai.com/api/docs/pricing", 1.00
    ),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float | None:
    """`input_tokens` includes `cached_tokens`, as OpenAI reports it; cached tokens use the cached rate if any."""
    price = PRICES.get(model)
    if price is None:
        return None
    cached_rate = price.input_per_mtok if price.cached_input_per_mtok is None else price.cached_input_per_mtok
    uncached = input_tokens - cached_tokens
    return (
        uncached * price.input_per_mtok + cached_tokens * cached_rate + output_tokens * price.output_per_mtok
    ) / 1_000_000
