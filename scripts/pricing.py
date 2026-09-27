"""Illustrative list prices, USD per million tokens. SNAPSHOT: 2026-09. Prices change.

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


PRICES: dict[str, Price] = {
    # TypeSafe list price: input billed, output and cached input free.
    "jev": Price(0.042, 0.0, "TypeSafe Jev list price (2026-09)"),
    "claude-opus-5": Price(5.00, 25.00, "Anthropic list price (2026-09)"),
    "claude-sonnet-5": Price(2.00, 10.00, "Anthropic list price (2026-09)"),
    "claude-haiku-4-5": Price(1.00, 5.00, "Anthropic list price (2026-09)"),
    "gpt-4o": Price(2.50, 10.00, "OpenAI list price, last verified by the author; confirm before quoting"),
    "gpt-4o-mini": Price(0.15, 0.60, "OpenAI list price, last verified by the author; confirm before quoting"),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    price = PRICES.get(model)
    if price is None:
        return None
    return (input_tokens * price.input_per_mtok + output_tokens * price.output_per_mtok) / 1_000_000
