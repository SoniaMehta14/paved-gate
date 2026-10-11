"""Scripted benchmark agent. Runs inside an OpenShell sandbox: python3 /app/bench_agent.py <plan>

Reads /app/plan.json (written by evals/openshell/run_bench.py), runs the isolation probes and then every step in
order, and prints one JSON line per probe and per step. No LLM. Python standard library only.
"""

from __future__ import annotations

import json
import socket
import sys
import time
import urllib.error
import urllib.request

PLAN = "/app/plan.json"
MAX_RETRIES = 2


def emit(obj: dict[str, object]) -> None:
    print(json.dumps(obj), flush=True)


def classify_os_error(e: BaseException) -> str:
    if isinstance(e, PermissionError) or getattr(e, "errno", None) == 13:
        return "blocked"
    if isinstance(e, ConnectionRefusedError):
        return "refused"
    if isinstance(e, socket.gaierror):
        return "unresolved"
    if isinstance(e, TimeoutError | socket.timeout):
        return "timeout"
    return "error"


def http(method: str, url: str, body: str | None, tag: str) -> dict[str, object]:
    data = body.encode("utf-8") if body is not None else None
    headers = {"content-type": "application/json", "x-bench-tag": tag}
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    started = time.perf_counter()
    out: dict[str, object]
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            r.read()
            out = {"status": r.status}
    except urllib.error.HTTPError as e:
        raw = e.read(2000).decode("utf-8", "replace")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {}
        denial = {k: parsed.get(k) for k in ("error", "reason_code", "layer", "middleware")} if parsed else None
        out = {"status": e.code, "denial": denial}
    except urllib.error.URLError as e:
        reason = e.reason if isinstance(e.reason, BaseException) else OSError(str(e.reason))
        out = {"status": None, "error": f"{type(reason).__name__}: {reason}", "kind": classify_os_error(reason)}
    except OSError as e:
        out = {"status": None, "error": f"{type(e).__name__}: {e}", "kind": classify_os_error(e)}
    out["ms"] = round((time.perf_counter() - started) * 1000, 2)
    return out


def probe(p: dict[str, object]) -> dict[str, object]:
    if p["kind"] == "tcp":
        try:
            with socket.create_connection((str(p["host"]), int(str(p["port"]))), timeout=5):
                return {"result": "reached"}
        except OSError as e:
            return {"result": classify_os_error(e), "detail": f"{type(e).__name__}: {e}"[:160]}
    r = http("GET", str(p["url"]), None, f"probe-{p['name']}")
    if r.get("status") is None:
        return {"result": r.get("kind", "error"), "detail": str(r.get("error"))[:160]}
    denial = r.get("denial")
    if r["status"] == 403 and isinstance(denial, dict) and denial.get("error") == "policy_denied":
        return {"result": "blocked", "detail": "403 policy_denied"}
    return {"result": "reached", "detail": f"HTTP {r['status']}"}


def main() -> None:
    which = sys.argv[1]
    with open(PLAN, encoding="utf-8") as f:
        plan = json.load(f)[which]
    emit({"ready": True})
    for p in plan["probes"]:
        emit({"probe": p["name"], **probe(p)})
    for s in plan["steps"]:
        failed: list[str] = []
        r = http(s["method"], s["url"], s["body"], s["tag"])
        while r.get("status") is None and r.get("kind") in ("error", "timeout") and len(failed) < MAX_RETRIES:
            failed.append(str(r.get("error")))  # transport failure inside the sandbox path, not a policy decision
            time.sleep(0.5)
            r = http(s["method"], s["url"], s["body"], s["tag"])
        emit({"step": s["tag"], **r, "failed_attempts": failed})
        time.sleep(0.03)
    emit({"done": True})


if __name__ == "__main__":
    main()
