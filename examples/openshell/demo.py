"""OpenShell builds the jail. Paved Gate reads the mail.

One-command demo: a scripted agent inside a real NVIDIA OpenShell sandbox makes five requests; the output shows
which layer stopped what.

    examples/openshell/run.sh            # real OpenShell sandbox, Jev MOCK evaluator (default, free, offline)
    examples/openshell/run.sh --live     # real OpenShell sandbox, live Jev (TYPESAFE_API_KEY from .env)
    examples/openshell/run.sh --local    # no Docker/OpenShell: SIMULATED supervisor, same middleware over gRPC

What it changes on your machine (OpenShell mode), and undoes on exit, including Ctrl-C:
  - appends a `paved-gate` middleware registration to OpenShell's gateway.toml and restarts the gateway
    (a byte-exact backup is written to logs/openshell-demo/ and restored afterwards)
  - builds the Docker image paved-gate/openshell-demo-agent and creates one short-lived sandbox
  - listens on 127.0.0.1:8099 (demo API) and 127.0.0.1:50051 (middleware) while it runs
All payloads are synthetic.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from agent import SCENARIOS  # noqa: E402
from fixtures import FixtureServer  # noqa: E402

GATEWAY_TOML = Path(os.environ.get("OPENSHELL_GATEWAY_TOML", "/opt/homebrew/var/openshell/gateway.toml"))
DOCKER_SETTINGS = Path.home() / "Library/Group Containers/group.com.docker/settings-store.json"
IMAGE = "paved-gate/openshell-demo-agent:latest"
API_PORT, MW_PORT = 8099, 50051
STATE = ROOT / "logs" / "openshell-demo"
REGISTRATION = f"""
# --- added by examples/openshell/demo.py (removed automatically when the demo exits) ---
[[openshell.supervisor.middleware]]
name = "paved-gate"
grpc_endpoint = "http://127.0.0.1:{MW_PORT}"
allow_insecure_transport = true   # local demo only: plaintext on loopback
max_payload_bytes = 1048576
timeout = "1500ms"
"""


# ---------------------------------------------------------------- terminal styling


class Style:
    def __init__(self, enabled: bool) -> None:
        self.on = enabled

    def _c(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.on else text

    def bold(self, t: str) -> str:
        return self._c("1", t)

    def dim(self, t: str) -> str:
        return self._c("2", t)

    def green(self, t: str) -> str:
        return self._c("32", t)

    def red(self, t: str) -> str:
        return self._c("31", t)

    def yellow(self, t: str) -> str:
        return self._c("33", t)

    def cyan(self, t: str) -> str:
        return self._c("36", t)


S = Style(sys.stdout.isatty())


def say(text: str = "", pace: float = 0.0) -> None:
    print(text, flush=True)
    if pace:
        time.sleep(pace)


def banner(runtime: str, evaluator: str, pace: float) -> None:
    line = "─" * 62
    say(S.cyan(f"╭{line}╮"))
    say(S.cyan("│") + S.bold("   OpenShell builds the jail.  Paved Gate reads the mail.".ljust(62)) + S.cyan("│"))
    say(S.cyan(f"╰{line}╯"))
    say(f"  {S.dim('Runtime  ')} {runtime}")
    say(f"  {S.dim('Evaluator')} {evaluator}")
    say(f"  {S.dim('Policy   ')} deny by default; the agent may reach only host.openshell.internal:{API_PORT}")
    say(f"  {S.dim('         ')} every allowed request and tool response goes through Paved Gate")
    say(f"  {S.dim('Data     ')} synthetic only", pace)
    say()


# ---------------------------------------------------------------- helpers


def run(cmd: list[str], **kw: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)  # type: ignore[call-overload,no-any-return]


def port_free(port: int) -> bool:
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # match the servers: TIME_WAIT is not "in use"
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def fail(msg: str) -> int:
    say(S.red(f"✗ {msg}"))
    return 2


def wait_connected(timeout: float = 30.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if "Connected" in run(["openshell", "status"]).stdout:
            return True
        time.sleep(0.5)
    return False


def restart_gateway() -> bool:
    if shutil.which("brew") and "openshell" in run(["brew", "services", "list"]).stdout:
        run(["brew", "services", "restart", "openshell"])
        return wait_connected()
    return False


def start_middleware(mode: str, audit: Path, env: dict[str, str]) -> subprocess.Popen[str]:
    STATE.mkdir(parents=True, exist_ok=True)
    log = (STATE / "middleware.log").open("w")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "paved_gate.integrations.openshell",
            "--policy",
            str(ROOT / "policy/openshell_egress.policy.yaml"),
            "--bind",
            f"127.0.0.1:{MW_PORT}",
            "--mode",
            mode,
            "--audit",
            str(audit),
        ],
        cwd=ROOT,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for _ in range(60):
        if "listening" in (STATE / "middleware.log").read_text():
            return proc
        if proc.poll() is not None:
            break
        time.sleep(0.25)
    raise RuntimeError("Paved Gate middleware did not start; see logs/openshell-demo/middleware.log")


def fmt_ms(v: object) -> str:
    if not isinstance(v, int | float):
        return "? ms"
    return "<1 ms" if v < 1 else f"{v:.0f} ms"


def fmt_scores(scores: object) -> str:
    if not isinstance(scores, dict):
        return ""
    parts = []
    if "exfiltration" in scores:
        parts.append(f"exfiltration risk {float(scores['exfiltration']):.1f}/5")
    if "sensitive_probability" in scores:
        parts.append(f"PII/PHI {float(scores['sensitive_probability']):.0%}")
    if "injection" in scores:
        parts.append(f"injection {float(scores['injection']):.1f}/5")
    return ", ".join(parts)


def read_audit(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ---------------------------------------------------------------- --local: simulated supervisor


async def local_agent_run() -> list[dict[str, object]]:
    """Replays SCENARIOS through a simulated OpenShell supervisor that calls the same middleware over gRPC."""
    import grpc
    from google.protobuf import struct_pb2

    from paved_gate.integrations.openshell._gen import supervisor_middleware_pb2 as pb
    from paved_gate.integrations.openshell._gen import supervisor_middleware_pb2_grpc as pbg

    out: list[dict[str, object]] = []
    async with grpc.aio.insecure_channel(f"127.0.0.1:{MW_PORT}") as ch:
        mw = pbg.SupervisorMiddlewareStub(ch)  # type: ignore[no-untyped-call]
        resp = pbg.HttpResponsePreReturnStub(ch)  # type: ignore[no-untyped-call]
        for sid, _t, method, url, body in SCENARIOS:
            started = time.perf_counter()
            u = urllib.parse.urlsplit(url)
            port = u.port or (443 if u.scheme == "https" else 80)
            if not (u.hostname == "host.openshell.internal" and port == API_PORT):
                out.append(
                    {
                        "step": sid,
                        "status": None,
                        "ms": 0.0,
                        "error": "PermissionError: [Errno 13] Permission denied (SIMULATED OpenShell policy)",
                    }
                )
                continue
            raw = json.dumps(body).encode() if body is not None else b""
            ctx = pb.RequestContext(request_id=str(uuid.uuid4()), sandbox_id="local-demo")
            target = pb.HttpRequestTarget(
                scheme="http", host=u.hostname, port=port, method=method, path=u.path, query=u.query
            )
            r = await mw.EvaluateHttpRequest(
                pb.HttpRequestEvaluation(
                    phase=pb.SUPERVISOR_MIDDLEWARE_PHASE_PRE_CREDENTIALS,
                    context=ctx,
                    config=struct_pb2.Struct(),
                    target=target,
                    body=raw,
                    middleware_name="paved-gate",
                )
            )
            if r.decision == pb.DECISION_DENY:
                out.append(
                    {
                        "step": sid,
                        "status": 403,
                        "ms": round((time.perf_counter() - started) * 1000, 1),
                        "body": json.dumps({"error": "middleware_denied", "reason_code": r.reason_code}),
                    }
                )
                continue
            local_url = f"http://127.0.0.1:{port}{u.path}" + (f"?{u.query}" if u.query else "")
            req = urllib.request.Request(
                local_url, data=raw or None, method=method, headers={"content-type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10) as up:
                status, payload, ctype = up.status, up.read(), up.headers.get("content-type", "")

            async def events(
                ctx: pb.RequestContext, target: pb.HttpRequestTarget, status: int, payload: bytes, ctype: str
            ) -> AsyncIterator[pb.HttpResponseEvent]:
                yield pb.HttpResponseEvent(
                    preflight=pb.HttpResponsePreflight(
                        context=ctx,
                        target=target,
                        status_code=status,
                        headers=[pb.HttpHeader(name="content-type", value=ctype)],
                        middleware_name="paved-gate",
                        max_payload_bytes=1 << 20,
                        permitted_body_modes=[
                            pb.HTTP_RESPONSE_BODY_MODE_HEADERS_ONLY,
                            pb.HTTP_RESPONSE_BODY_MODE_WHOLE_BODY_BYTES,
                        ],
                    )
                )
                yield pb.HttpResponseEvent(body=pb.HttpResponseBodyUnit(sequence=1, data=payload, end_of_stream=True))

            results = [x async for x in resp.Evaluate(events(ctx, target, status, payload, ctype))]
            blocked = any(
                x.WhichOneof("result") == "body_result" and x.body_result.WhichOneof("action") == "block_delivery"
                for x in results
            )
            out.append(
                {
                    "step": sid,
                    "status": 403 if blocked else status,
                    "ms": round((time.perf_counter() - started) * 1000, 1),
                    "body": json.dumps({"error": "middleware_denied"})
                    if blocked
                    else payload[:400].decode("utf-8", "replace"),
                }
            )
    return out


# ---------------------------------------------------------------- narrative


def narrate(
    steps: list[dict[str, object]],
    audit: list[dict[str, object]],
    received: list[tuple[str, int]],
    mode: str,
    local: bool,
    pace: float,
) -> None:
    by_step = {str(s["step"]): s for s in steps}
    jail = "OpenShell" if not local else "OpenShell (SIMULATED)"
    tag_mock = S.yellow(" [mock evaluator]") if mode == "mock" else ""
    eval_ms: list[float] = []
    tally = {"jail_blocked": 0, "gate_denied": 0, "gate_flagged": 0}

    def entries(direction: str, path: str) -> list[dict[str, object]]:
        return [a for a in audit if a.get("direction") == direction and a.get("path") == path]

    for sid, title, method, url, _body in SCENARIOS:
        u = urllib.parse.urlsplit(url)
        say(S.bold(f"  ── {sid} ─ {title}"))
        say(S.dim(f"       {method} {u.hostname}{':' + str(u.port) if u.port else ''}{u.path}"))
        res = by_step.get(sid, {"status": None, "error": "no result from agent"})
        status, err = res.get("status"), str(res.get("error") or "")
        body = str(res.get("body") or "")
        try:
            denial = json.loads(body) if status == 403 else {}
        except json.JSONDecodeError:
            denial = {}
        req = (entries("request", u.path) or [{}])[-1]
        rsp = (entries("response", u.path) or [{}])[-1]
        for e in (req, rsp):
            lat = e.get("latency")
            if isinstance(lat, dict) and lat.get("evaluator_ms") is not None:
                eval_ms.append(float(lat["evaluator_ms"]))

        if status is None:
            why = "host is not in the sandbox policy (connection refused)" if "Errno 13" in err else err[:70]
            say(f"       JAIL  {jail:<22}{S.red('✗ BLOCKED'.ljust(14))}{why}")
            say(
                f"       MAIL  {'Paved Gate':<22}{S.dim('· not reached'.ljust(14))}{S.dim('the jail stopped it first')}"
            )
            tally["jail_blocked"] += 1
        elif status == 403 and denial.get("error") == "policy_denied":
            say(f"       JAIL  {jail:<22}{S.red('✗ BLOCKED'.ljust(14))}request rule (L7) denied it")
            tally["jail_blocked"] += 1
        else:
            say(f"       JAIL  {jail:<22}{S.green('✓ allowed'.ljust(14))}host and port are in the policy")
            if status == 403 and denial.get("error") == "middleware_denied":
                code = str(denial.get("reason_code") or req.get("reason_code") or "")
                raw_findings = req.get("findings")
                found = raw_findings if isinstance(raw_findings, list) else []
                findings = ", ".join(str(f).removeprefix("local_") for f in found if str(f).startswith("local_"))
                lat = req.get("latency") if isinstance(req.get("latency"), dict) else {}
                assert isinstance(lat, dict)
                if findings:
                    detail = f"{code} · {findings} found locally, no evaluator call · {fmt_ms(lat.get('gate_ms'))}"
                else:
                    detail = f"{code} · {fmt_scores(req.get('scores'))} · {fmt_ms(lat.get('evaluator_ms'))}{tag_mock}"
                say(f"       MAIL  {'Paved Gate':<22}{S.red('✗ DENIED'.ljust(14))}{detail}")
                tally["gate_denied"] += 1
            else:
                lat = req.get("latency") if isinstance(req.get("latency"), dict) else {}
                assert isinstance(lat, dict)
                checked = f"request checked in {fmt_ms(lat.get('evaluator_ms'))}{tag_mock}"
                say(f"       MAIL  {'Paved Gate':<22}{S.green('✓ passed'.ljust(14))}{checked}")
                if rsp.get("decision") in ("flag", "deny"):
                    verdict = (
                        S.yellow("⚑ FLAGGED".ljust(14))
                        if rsp.get("decision") == "flag"
                        else S.red("✗ BLOCKED".ljust(14))
                    )
                    scores = rsp.get("scores") or {}
                    inj = fmt_scores(scores)
                    label = "Paved Gate (response)"
                    why = f"prompt injection in the tool result · {inj}{tag_mock}"
                    say(f"       MAIL  {label:<22}{verdict}{why}")
                    say(S.dim(" " * 49 + "sandbox is now tainted: its next requests get a stricter check"))
                    tally["gate_flagged"] += 1
        say("", pace)

    arrived = ", ".join(p for p, _ in received) or "nothing"
    say(S.bold("  ── Result"))
    say(
        f"       {jail} blocked {tally['jail_blocked']} · Paved Gate denied {tally['gate_denied']} · "
        f"flagged {tally['gate_flagged']} · upstream actually received: {S.bold(arrived)}"
    )
    if eval_ms:
        p50 = statistics.median(eval_ms)
        note = "simulated mock latency, not a measurement" if mode == "mock" else "live Jev, this run"
        say(f"       Paved Gate evaluator time per checked call: p50 {p50:.0f} ms, max {max(eval_ms):.0f} ms ({note})")
    if mode == "mock":
        say(S.yellow("       Mock evaluator: keyword heuristics. Run with --live for real Jev judgments."))
    say()


# ---------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--live", action="store_true", help="use live Jev (TYPESAFE_API_KEY from .env); default is the mock"
    )
    ap.add_argument("--local", action="store_true", help="no Docker/OpenShell: SIMULATED supervisor, same middleware")
    ap.add_argument("--pace", type=float, default=0.8, help="seconds to pause between scenarios (for screen capture)")
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    if args.no_color:
        S.on = False
    mode = "live" if args.live else "mock"
    env = dict(os.environ)
    if args.live:
        from dotenv import dotenv_values

        env.update({k: v for k, v in dotenv_values(ROOT / ".env").items() if v and k not in os.environ})
        if not env.get("TYPESAFE_API_KEY"):
            return fail("--live needs TYPESAFE_API_KEY in .env (see .env.example)")
    evaluator = (
        (S.green("Jev LIVE") + " · jev-latest via api.typesafe.ai")
        if args.live
        else (S.yellow("Jev MOCK") + S.dim(" · keyword heuristics, simulated latency: not a real model judgment"))
    )

    for port in (API_PORT, MW_PORT):
        if not port_free(port):
            return fail(f"port {port} on 127.0.0.1 is in use; stop whatever is using it and retry")

    backup: bytes | None = None
    STATE.mkdir(parents=True, exist_ok=True)
    audit = STATE / f"audit-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.jsonl"

    if not args.local:
        if not shutil.which("docker") or run(["docker", "info"]).returncode != 0:
            return fail("Docker is not running (OpenShell on macOS needs Docker Desktop 28+); or use --local")
        if sys.platform == "darwin":
            try:
                if not json.loads(DOCKER_SETTINGS.read_text()).get("HostNetworkingEnabled"):
                    return fail("enable Docker Desktop host networking (Settings → Resources → Network); see README")
            except (OSError, json.JSONDecodeError):
                say(S.yellow("! could not read Docker Desktop settings; assuming host networking is enabled"))
        if not shutil.which("openshell") or not wait_connected(5):
            return fail("OpenShell gateway not connected (`openshell status`); see docs/openshell/FRICTION_LOG.md")
        if not GATEWAY_TOML.exists():
            return fail(f"gateway config not found at {GATEWAY_TOML}; set OPENSHELL_GATEWAY_TOML")
        version = run(["openshell", "--version"]).stdout.strip().removeprefix("openshell ")
        runtime = f"NVIDIA OpenShell {version} · real sandbox (Docker Desktop)"
    else:
        runtime = S.yellow("SIMULATED OpenShell") + S.dim(
            " · no sandbox; host allowlist + the same gRPC middleware calls"
        )

    fixtures = FixtureServer(API_PORT)
    fixtures.start()
    mw: subprocess.Popen[str] | None = None

    def restore(*_: object) -> None:
        nonlocal backup
        if backup is not None:
            GATEWAY_TOML.write_bytes(backup)
            backup = None
            say(S.dim("  restoring OpenShell gateway config and restarting the gateway…"))
            restart_gateway()

    def on_signal(signum: int, _frame: FrameType | None) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, on_signal)
    try:
        banner(runtime, evaluator, args.pace)
        say(S.dim("  starting Paved Gate middleware and the demo API…"))
        mw = start_middleware(mode, audit, env)
        if args.local:
            steps = asyncio.run(local_agent_run())
        else:
            backup = GATEWAY_TOML.read_bytes()
            (STATE / "gateway.toml.backup").write_bytes(backup)
            GATEWAY_TOML.write_bytes(backup + REGISTRATION.encode())
            say(S.dim("  registering it with the OpenShell gateway (backup in logs/openshell-demo/)…"))
            if not restart_gateway():
                raise RuntimeError("OpenShell gateway did not come back after registering the middleware")
            say(S.dim("  building the agent image and creating a sandbox…"))
            build = run(["docker", "build", "-q", "-t", IMAGE, str(HERE)])
            if build.returncode != 0:
                raise RuntimeError(f"docker build failed: {build.stderr[-400:]}")
            name = f"pg-demo-{int(time.time())}"
            proc = run(
                [
                    "openshell",
                    "sandbox",
                    "create",
                    "--name",
                    name,
                    "--no-keep",
                    "--from",
                    IMAGE,
                    "--policy",
                    str(HERE / "sandbox-policy.yaml"),
                    "--",
                    "python3",
                    "/app/agent.py",
                ],
                timeout=300,
            )
            steps = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith('{"step"')]
            if not steps:
                raise RuntimeError(f"sandbox produced no results:\n{(proc.stdout + proc.stderr)[-800:]}")
        say()
        time.sleep(0.3)  # let the middleware flush its audit lines
        narrate(steps, read_audit(audit), fixtures.received, mode, args.local, args.pace)
        say(S.dim(f"  audit log: {audit.relative_to(ROOT)}  (hashes and scores only, no payload content)"))
        return 0
    except KeyboardInterrupt:
        say(S.yellow("\n  interrupted"))
        return 130
    except Exception as exc:
        return fail(str(exc))
    finally:
        restore()
        if mw is not None:
            mw.terminate()
            try:
                mw.wait(timeout=5)
            except subprocess.TimeoutExpired:
                mw.kill()
        fixtures.stop()


if __name__ == "__main__":
    sys.exit(main())
