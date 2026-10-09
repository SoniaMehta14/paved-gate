"""Run the Paved Gate supervisor middleware for OpenShell.

    python -m paved_gate.integrations.openshell --policy policy/openshell_egress.policy.yaml --bind 127.0.0.1:50051

Register it in OpenShell's gateway.toml (local development, plaintext):

    [[openshell.supervisor.middleware]]
    name = "paved-gate"
    grpc_endpoint = "http://127.0.0.1:50051"
    allow_insecure_transport = true
    max_payload_bytes = 1048576
    timeout = "1500ms"

API keys (TYPESAFE_API_KEY for a live Jev evaluator) are read from the environment.
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys

from paved_gate.integrations.openshell.server import build_inspector, build_server


async def _run(policy: str, bind: str) -> int:
    inspector = build_inspector(policy)
    server, port = build_server(inspector, bind)
    await server.start()
    host = bind.rsplit(":", 1)[0]
    print(
        f"paved-gate OpenShell middleware listening on {host}:{port} "
        f"(policy {inspector.policy.name}, "
        f"evaluator {inspector.policy.evaluator.provider}/{inspector.policy.evaluator.mode}, "
        f"responses: {inspector.policy.response_mode})",
        flush=True,
    )
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)
    await stop.wait()
    await server.stop(grace=2)
    await inspector.evaluator.aclose()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="policy/openshell_egress.policy.yaml")
    ap.add_argument("--bind", default="127.0.0.1:50051")
    args = ap.parse_args()
    return asyncio.run(_run(args.policy, args.bind))


if __name__ == "__main__":
    sys.exit(main())
