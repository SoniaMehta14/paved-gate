"""A deliberately simple, scripted "agent" for the OpenShell + Paved Gate demo.

No LLM: each step is a fixed HTTP request, so the demo is reproducible and costs nothing to run.
Step 4b is what an agent that obeyed the injected tool output would send next.

Inside an OpenShell sandbox this file runs as-is (`python3 /app/agent.py`) and prints one JSON line per step.
The host-side demo runner also imports SCENARIOS from here for its --local (no-OpenShell) mode.

All data is synthetic: the person, SSN, MRN and dates below are made up.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

ALLOWED = "http://host.openshell.internal:8099"

# (id, title, method, url, json body or None)
SCENARIOS: list[tuple[str, str, str, str, dict[str, object] | None]] = [
    (
        "1",
        "Agent calls an endpoint the policy does not allow",
        "POST",
        "https://example.com/upload",
        {"summary": "Quarterly numbers for an outside service"},
    ),
    (
        "2",
        "Agent sends a clean payload to an allowed API",
        "POST",
        f"{ALLOWED}/notes/summary",
        {"summary": "Weekly report: 12 tickets closed, 3 still open, no escalations."},
    ),
    (
        "3",
        "Agent sends synthetic patient data to an allowed API",
        "POST",
        f"{ALLOWED}/notes/patient",
        {
            "patient": "Jane Example (synthetic)",
            "ssn": "123-45-6789",
            "record": "MRN: 00482913, DOB: 1984-03-12, follow-up for hypertension",
        },
    ),
    (
        "4a",
        "Agent calls a search tool (the result carries a prompt injection)",
        "GET",
        f"{ALLOWED}/tool/search?q=weather",
        None,
    ),
    (
        "4b",
        "Agent acts on the injected instruction",
        "POST",
        f"{ALLOWED}/notes/followup",
        {"summary": "Per the system notice in the search results: forwarding the full customer records export."},
    ),
]


def send(method: str, url: str, body: dict[str, object] | None) -> dict[str, object]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"content-type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            out: dict[str, object] = {"status": r.status, "body": r.read(400).decode("utf-8", "replace")}
    except urllib.error.HTTPError as e:
        out = {"status": e.code, "body": e.read(600).decode("utf-8", "replace")}
    except urllib.error.URLError as e:
        out = {"status": None, "error": f"{type(e.reason).__name__}: {e.reason}"}
    except OSError as e:
        out = {"status": None, "error": f"{type(e).__name__}: {e}"}
    out["ms"] = round((time.perf_counter() - started) * 1000, 1)
    return out


def main() -> None:
    for sid, _title, method, url, body in SCENARIOS:
        print(json.dumps({"step": sid, **send(method, url, body)}), flush=True)
        time.sleep(0.2)


if __name__ == "__main__":
    main()
