# OpenShell builds the jail. Paved Gate reads the mail.

A one-command demo of two layers of agent safety working together:

- **[NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell)** (containment). The agent runs in a sandbox with
  deny-by-default network, filesystem and process policy.
- **Paved Gate** (verification). It runs as an OpenShell
  [supervisor middleware](https://docs.nvidia.com/openshell/extensibility/supervisor-middleware.md) and reads what
  is inside the requests OpenShell already allowed, and the tool responses coming back to the agent.

```bash
examples/openshell/run.sh            # real OpenShell sandbox, Jev MOCK evaluator (default: free, offline)
examples/openshell/run.sh --live     # real OpenShell sandbox, live Jev (needs TYPESAFE_API_KEY in .env)
examples/openshell/run.sh --local    # no Docker or OpenShell: SIMULATED supervisor, same middleware over gRPC
```

`--pace 1.5` slows the output for screen recordings, and `--no-color` turns off colour.

## The five requests

A scripted agent (no LLM, so the demo is reproducible and costs nothing) makes these requests from inside the
sandbox. All data is synthetic.

| # | What the agent does | Jail (OpenShell) | Mail (Paved Gate) |
|---|---|---|---|
| 1 | POST to `example.com`, which is not in the policy | **blocked** | never reached |
| 2 | Clean summary to the allowed API | allowed | passed |
| 3 | Synthetic patient record (fake name, SSN, MRN, DOB) to the allowed API | allowed | **denied** by local detectors, with no evaluator call, so nothing leaves the Mac |
| 4a | Calls a search tool whose result hides an instruction for the agent | allowed | **flagged** at the response hook; the sandbox is marked tainted |
| 4b | Sends what the injected instruction asked for | allowed | **denied** as injection follow-through |

The run ends by listing what the upstream API actually received, normally only request 2.

**What "mock" means:** the default evaluator is the Jev mock, a set of keyword heuristics with simulated latency. It
is labelled `[mock evaluator]` everywhere it was used. Its scores are not model judgments, and in a tainted sandbox
it treats *every* follow-up request as risky. Use `--live` to see real Jev decisions; the live false-positive rate is
measured in Phase 4.

## Requirements (OpenShell mode)

Tested on macOS 26 (Apple Silicon) with Docker Desktop 4.88.1 (engine 29.8.2) and OpenShell 0.1.2. Every step,
error and workaround is in [docs/openshell/FRICTION_LOG.md](../../docs/openshell/FRICTION_LOG.md).

1. **Docker Desktop 28 or later**: `brew install --cask docker-desktop` (asks for your admin password), then open it
   once.
2. **Docker Desktop host networking enabled**: Settings → Resources → Network → *Enable host networking*. Without
   it, OpenShell 0.1.2 sandboxes either fail to start
   ([NVIDIA/OpenShell#3880](https://github.com/NVIDIA/OpenShell/issues/3880)) or can't reach services on your Mac.
   See the trade-off below.
3. **OpenShell 0.1.2**: `curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh -o install.sh`,
   read it, then `OPENSHELL_VERSION=v0.1.2 sh install.sh`. The installer sets up a Homebrew service that **starts at
   login**; stop it with `brew services stop openshell`.
4. `uv sync --extra openshell`

### The host-networking trade-off

Host networking is a Docker-wide setting, not an OpenShell one. We checked what it changes:

- **The sandbox boundary does not change.** The sandbox's own container still runs with `network_mode=none`. Only
  OpenShell's trusted supervisor container uses host networking, and every connection from the agent still passes
  its policy check. With a policy allowing only `host.openshell.internal:8099`, the sandbox could not reach other
  ports on the Mac, OpenShell's own gateway port, public hosts or public IPs, Docker's host IP, or the Mac's
  loopback. Details are in the friction log, entries 27–28.
- **The machine does change.** Any container *you* start with `--network host` can now reach services your Mac binds
  to `127.0.0.1` (databases, dev servers, local admin UIs). Only run host-network containers you trust, and consider
  turning the setting off when you're done.

## What the demo changes, and undoes on exit

- It appends a `paved-gate` middleware registration to OpenShell's `gateway.toml` (Homebrew default:
  `$(brew --prefix)/var/openshell/gateway.toml`; override with `OPENSHELL_GATEWAY_TOML`) and restarts the gateway.
  A byte-exact backup goes to `logs/openshell-demo/gateway.toml.backup`, and the original is restored when the demo
  exits, including on Ctrl-C and errors.
- It builds the image `paved-gate/openshell-demo-agent` and creates one sandbox that deletes itself (`--no-keep`).
- While it runs, it listens on `127.0.0.1:8099` (the demo API) and `127.0.0.1:50051` (the middleware, plaintext on
  loopback for local use only).
- It writes an audit log to `logs/openshell-demo/` containing hashes, sizes, finding types and scores, never payload
  content.

## Files

| File | Role |
|---|---|
| `run.sh` | the one command |
| `demo.py` | orchestration and narration |
| `agent.py` | the scripted agent; runs inside the sandbox |
| `fixtures.py` | the demo API and the poisoned search tool |
| `sandbox-policy.yaml` | the OpenShell policy (jail plus middleware selection) |
| `Dockerfile` | the sandbox workload image |
| [`../../policy/openshell_egress.policy.yaml`](../../policy/openshell_egress.policy.yaml) | Paved Gate's rubric and thresholds |
