"""Paved Gate demo server.

    uv run python examples/server.py

Runs with no API keys: Jev in mock mode, and FRONTIER_AGENT_REQUIRED requests fall
through to the app's own /v1/agent handler (an echo). Set ANTHROPIC_API_KEY to have
the gate forward those requests to Claude instead.

    curl -s localhost:8000/v1/agent -H 'content-type: application/json' \
         -d '{"input": "What is (17.5 * 4) - 3^2?"}'
"""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request

from paved_gate import Frontier, JsonlFileSink, paved_gate
from paved_gate.middleware.fastapi import PavedGateMiddleware

ROOT = Path(__file__).resolve().parent.parent
POLICY = ROOT / "policy" / "paved_gate.policy.yaml"

load_dotenv(ROOT / ".env")  # keys from .env; real environment variables take precedence
use_claude = bool(os.environ.get("ANTHROPIC_API_KEY"))
gate = paved_gate(
    POLICY,
    audit=JsonlFileSink(ROOT / "logs" / "paved_gate.audit.jsonl"),
    frontier="from_policy" if use_claude else None,
)

app = FastAPI(title="Paved Gate demo")
app.add_middleware(PavedGateMiddleware, gate=gate, paths={"/v1/agent"})


@app.post("/v1/agent")
async def agent(request: Request) -> dict[str, object]:
    # Only FRONTIER_AGENT_REQUIRED requests reach here, and only when the gate has
    # no frontier handler. This is where your own agent / orchestrator would run.
    result: Frontier = request.state.paved_gate
    return {
        "request_id": result.decision.request_id,
        "handled_by": "app-agent (demo echo)",
        "echo": result.state.input,
        "gate": result.decision.model_dump(mode="json"),
    }


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok", "policy_hash": gate.loaded.policy_hash}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8000")))
