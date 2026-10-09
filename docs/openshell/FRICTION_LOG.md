# OpenShell friction log

A factual record of every step, error, unclear doc and workaround met while installing and running
[NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell) on a Mac, as preparation for the Paved Gate
supervisor-middleware integration. Entries are in time order (UTC). Each entry notes whether a claim
was **verified** (observed or read in a primary source, with its URL) or is an **assumption**.

## Environment

| Item | Value |
|---|---|
| Host | macOS 26.6.2, Apple Silicon (arm64), 16 GB RAM |
| Package manager | Homebrew 6.0.17 |
| Container runtime at start | none: no Docker Desktop, Podman or Colima installed |
| OpenShell docs version read | v0.1.2 ("latest") |

## Log

### 2026-10-09T13:14Z · Discovery (documentation only)

1. **The architecture page doesn't mention the extension points.** The
   [architecture overview](https://docs.nvidia.com/openshell/latest/about/architecture) describes the gateway,
   sandbox, supervisor and compute runtime, but says nothing about supervisor middleware or gateway interceptors.
   These are documented only under [Extensibility](https://docs.nvidia.com/openshell/extensibility/overview.md),
   which I found through the docs index (`llms.txt`), not by navigating from the architecture page. *Verified.*
   - Impact: someone evaluating whether OpenShell supports third-party content inspection could conclude from
     the architecture page that it doesn't.

2. **A docs link returns 404.** `https://docs.nvidia.com/openshell/sandboxes/inference-routing` (returned by web
   search as "Inference Routing") gives HTTP 404, rechecked at 2026-10-09T13:14Z. The current page is
   [how-it-works/inference](https://docs.nvidia.com/openshell/how-it-works/inference.md). *Verified.*
   - Workaround: use the `llms.txt` index to find current page paths.

3. **The "privacy router" in marketing doesn't match the docs.** The
   [OpenShell product page](https://perspectives.nvidia.com/nvidia-openshell/) says the privacy router "keeps
   sensitive context on-device with local open models and routes to frontier models … only when policy allows".
   The [inference docs](https://docs.nvidia.com/openshell/how-it-works/inference.md) and
   [NemoClaw inference options](https://docs.nvidia.com/nemoclaw/latest/inference/inference-options.html)
   describe a route chosen in configuration (in NemoClaw, during onboarding), with no per-request, sensitivity-based
   routing and no inspection of prompt content. The issue that planned sensitivity routing,
   [#1043 "Privacy Guard"](https://github.com/NVIDIA/OpenShell/issues/1043), was closed as *not planned* on
   2026-09-21. *Verified (docs and issue); the gap between them is my reading.*

4. **The middleware API is explicitly unstable.** The
   [supervisor middleware page](https://docs.nvidia.com/openshell/extensibility/supervisor-middleware.md) says
   "The middleware API is still evolving." The proto package is `openshell.middleware.v1`, but no compatibility
   promise or deprecation policy is documented. *Verified.*
   - Workaround for this integration: vendor the `.proto` pinned to a release tag.

5. **The only reference middleware is labelled unsafe.** NVIDIA's example
   [`examples/supervisor-middleware-content-guard`](https://github.com/NVIDIA/OpenShell/tree/main/examples/supervisor-middleware-content-guard)
   (Rust) matches literal strings and "must not be used as a security control". There is no Python example, and
   no example of a model-backed evaluator. *Verified.*

### 2026-10-09T13:14Z – 13:16Z · Install Docker Desktop (prerequisite, not OpenShell itself)

OpenShell on macOS requires Docker Desktop or Engine 28.0 or later
([support matrix](https://docs.nvidia.com/openshell/about/support-matrix.md)). This Mac had no container runtime.

6. `brew install --cask docker` resolves to the cask **`docker-desktop` 4.88.1,237512**. The old name is now an
   alias. *Verified* (`brew info --cask docker-desktop`).
7. **A non-interactive install fails because it needs sudo.** The cask moves `Docker.app` into place, then runs
   `sudo mkdir -p /usr/local/bin` (that directory doesn't exist on a fresh Apple Silicon Mac, where Homebrew lives
   in `/opt/homebrew`). In a shell with no terminal this fails with
   `sudo: a terminal is required to read the password`, and Homebrew rolls back the whole install. *Verified*
   (exit 1, full log kept locally).
   - Not an OpenShell bug, but it affects anyone scripting OpenShell setup on a new Mac: the documented
     prerequisite can't be installed unattended.
   - Workaround: run `brew install --cask docker-desktop` in an interactive terminal, enter the admin password,
     then open Docker.app once to accept the Docker Subscription Service Agreement.
   - Also note: the default Homebrew output ends with a long list of `Removing:` cache-cleanup lines that pushes
     the error off-screen. Setting `HOMEBREW_NO_INSTALL_CLEANUP=1` surfaced it.

### 2026-10-09T13:23Z · Docker Desktop installed (by the user, interactively)

8. The user ran `brew install --cask docker-desktop` in Terminal, entered the admin password and accepted the
   Docker agreement. Result: **Docker 29.8.2** (client and server), Linux VM `linux/arm64`, 10 CPUs, about 7.7 GiB
   memory. This meets the 28.0 minimum. *Verified* (`docker version`, `docker info`).

### 2026-10-09T13:24Z · Install OpenShell v0.1.2 (about 15 s)

9. **Pinned install works.** I reviewed the installer before running it (`install.sh` from `main`, sha256
   `15fa05f2ccdb3f51…`, 1,632 lines). It supports `OPENSHELL_VERSION=<tag>` and checks a published sha256
   checksum file. Ran `OPENSHELL_VERSION=v0.1.2 sh install.sh`: exit 0, 13:23:59Z → 13:24:14Z. *Verified.*
   - Doc gap: the [README](https://github.com/NVIDIA/OpenShell) shows only the unpinned
     `curl … main/install.sh | sh`. Pinning with `OPENSHELL_VERSION` is documented only in the script's own
     `--help` text.
10. **On Apple Silicon the installer uses Homebrew**, not a standalone binary. It downloads the release formula
    `openshell.rb`, creates a *local* tap `nvidia/openshell` with `brew tap-new`, installs `openshell` 0.1.2
    (161.9 MB, 10 files) and starts a `brew services` LaunchAgent (`sh.brew.openshell`) that **starts at login**.
    *Verified.*
    - The background-service-at-login behaviour isn't mentioned in the README quick start. Users may not expect a
      persistent service; stop it with `brew services stop openshell`.
11. **Install-time warnings** (none fatal):
    - `Warning: tap-new is a developer command…` (the installer's local-tap approach)
    - `Warning: Calling post_install is deprecated! Use post_install_steps instead.` (from the OpenShell formula)
    - `This is a Tier 2 configuration` (Homebrew's notice, triggered by an outdated Xcode on this machine, so
      environmental, not OpenShell)

    *Verified* (install log). The formula's deprecated `post_install` will need updating before Homebrew drops it.
12. The installer registered the local gateway automatically (`openshell gateway add https://localhost:17670
    --local`). `openshell status` reports **Connected, Authenticated (mTLS transport), Version 0.1.2**. *Verified.*

### 2026-10-09T13:24Z · First sandbox fails to start (`ControlSupervisorStartFailed`)

13. **The first `openshell sandbox create` on a fresh macOS install fails.** `openshell status` had just reported
    Connected. 13 s after `openshell sandbox create --name pg-smoke --no-keep -- sh -c '…'`:
    ```
    × sandbox entered error phase while provisioning:
      ControlSupervisorStartFailed: Docker supervisor exited before becoming ready
      … Startup configuration fetch failed after 5 attempts: failed to connect to OpenShell server
    ```
    *Verified.* Cause: the Homebrew gateway binds `127.0.0.1:17670`. The Docker driver's default `grpc_endpoint` is
    `https://127.0.0.1:<port>` and the supervisor runs with `network_mode: host`, which under Docker Desktop is the
    Linux VM, not the Mac. This is the open, triaged bug
    **[NVIDIA/OpenShell#3880](https://github.com/NVIDIA/OpenShell/issues/3880)**. It was filed for WSL 2, and a
    comment from 2026-10-01 reproduces it on macOS 26.6.2 with OpenShell 0.1.2, the same as this machine.
    - Nothing in the docs mentions it: the [support matrix](https://docs.nvidia.com/openshell/about/support-matrix.md)
      lists macOS with Docker Desktop as *Supported*, and the docs index has no troubleshooting page for this error.
    - Workaround applied (from #3880): add to `/opt/homebrew/var/openshell/gateway.toml`
      ```toml
      [openshell.drivers.docker]
      grpc_endpoint = "https://host.docker.internal:17670"
      ```
      then `brew services restart openshell`. Sandboxes then start in under 1 s. *Verified* (13:26Z).
    - I first misread a `curl` test from a container (exit 56) as "unreachable". It was actually the gateway
      rejecting a connection without a client certificate, as mutual TLS requires. The TLS handshake to
      `host.docker.internal` succeeded.

### 2026-10-09T13:26Z · Default image and default egress behaviour

14. **The default sandbox image has no `curl` or `python3`.** It is `nvcr.io/nvidia/base/ubuntu:24.04`, so even a
    first egress check needs a custom image (`--from <image>`). `openshell sandbox template list` returns "No sandbox
    templates found". *Verified.* Workaround: a two-line `python:3.12-slim` Dockerfile.
15. **A denied connection looks like an OS error, not a policy decision.** Under the default policy, Python's
    `urllib` gets `URLError: [Errno 13] Permission denied` for both `http://` and `https://example.com`. The
    [network rules docs](https://docs.nvidia.com/openshell/how-it-works/policies/network-rules.md) say "an HTTP
    client receives an OpenShell `policy_denied` response". That appears to apply only to request-stage (L7)
    denials, while a connection-stage denial surfaces as `EACCES`. *Verified* (observed); the interpretation is mine.
    An agent, or a developer, seeing "Permission denied" has no hint that network policy was the cause.
16. **TLS trust works without changes to the image.** The supervisor sets `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`,
    `CURL_CA_BUNDLE`, `GIT_SSL_CAINFO` and `NODE_EXTRA_CA_CERTS` to its generated CA bundle, so Python's standard
    library trusts OpenShell's TLS interception. *Verified.*

### 2026-10-09T13:27Z – 13:29Z · Reaching a service on the Mac from a sandbox (not yet working)

17. **`host.openshell.internal` does not resolve inside the sandbox** (`[Errno -5] No address associated with
    hostname`), even with a policy allowing it. The same happens with `host.docker.internal`, and with
    `allowed_ips: [192.168.65.0/24 plus the matching Docker Desktop IPv6 range]` added for Docker Desktop's private addresses (private
    destinations are blocked unless listed in `allowed_ips`, per the network rules docs). *Verified.*
18. **The real reason is only in the sandbox log.** `openshell logs <sandbox>` shows
    `NET:REFUSE [MED] DENIED host.docker.internal [reason:policy_dns_trusted_gateway_unavailable]`. That reason code
    isn't documented. The source explains it:
    [`policy_dns/mod.rs`](https://github.com/NVIDIA/OpenShell/blob/v0.1.2/crates/openshell-supervisor-network/src/policy_dns/mod.rs)
    refuses the reserved host-gateway aliases unless a *trusted host gateway* exists, and
    [`proxy.rs` `detect_trusted_host_gateway()`](https://github.com/NVIDIA/OpenShell/blob/v0.1.2/crates/openshell-supervisor-network/src/proxy.rs)
    reads it from a `host.openshell.internal` line in the supervisor container's `/etc/hosts`. *Verified* (v0.1.2
    source).
19. **The #3880 workaround causes this.**
    [`openshell-driver-docker/src/lib.rs`](https://github.com/NVIDIA/OpenShell/blob/v0.1.2/crates/openshell-driver-docker/src/lib.rs)
    (`docker_supervisor_host_address`, around lines 4999–5009) injects the `host.openshell.internal` and
    `host.docker.internal` extra-host entries **only when `grpc_endpoint` is an IPv4 literal or `localhost`**. With
    `grpc_endpoint = "https://host.docker.internal:17670"` it returns `None`, the supervisor gets `ExtraHosts=[]`
    (confirmed with `docker inspect`), and every host alias is refused. Using the VM's host address
    (`https://192.168.65.254:17670`) instead would inject the aliases, but the gateway's generated server certificate
    has no IP SAN for it (SANs include `host.docker.internal`, `host.openshell.internal`, `127.0.0.1` and `::1`), so
    TLS verification would fail. *Verified* (source, `docker inspect`, `openssl x509`).
    - Net effect on macOS with Docker Desktop: **either sandboxes don't start (default), or they start but can't
      reach any service on the host (#3880 workaround).** That's the setup the docs' own middleware example uses
      (`grpc_endpoint = "http://host.openshell.internal:50051"`).
    - Next step: enable Docker Desktop host networking (Settings → Resources → Network) and revert `grpc_endpoint`
      to the default. A 2026-10-02 comment on #3880 reports that host networking alone makes sandboxes start on macOS.

### 2026-10-09T13:40Z · Fix: enable Docker Desktop host networking

20. **Host networking fixes both problems, and needs no gateway changes.** The user enabled Docker Desktop
    *Settings → Resources → Network → Enable host networking* and restarted Docker
    (`settings-store.json`: `"HostNetworkingEnabled": true`). I then **removed** the #3880 workaround, restoring
    the default `gateway.toml`, and restarted the gateway. *Verified* (13:40Z):
    - a host-network container reaches the gateway on `127.0.0.1:17670`
    - a sandbox starts in 0.66 s
    - with a policy allowing `host.openshell.internal:8099`, the sandbox gets HTTP 200 from a test server on the
      Mac (also visible in the server's log)
    - `https://example.com` is still blocked (`EACCES`)

    This matches the 2026-10-02 comment on [#3880](https://github.com/NVIDIA/OpenShell/issues/3880). The support
    matrix doesn't mention that host networking is effectively required on macOS.

### 2026-10-09T13:41Z – 13:42Z · Supervisor middleware smoke test (Python, gRPC)

NVIDIA's reference middleware is Rust and needs `cargo` (plus `zig` for Linux cross-builds). Neither was installed,
so I wrote a minimal throwaway **Python** `grpc.aio` middleware against the v0.1.2 protos instead (scratch only, not
committed). It denies any request body containing a marker string.

21. **Generating stubs works:** `proto/supervisor_middleware.proto` plus its import `proto/extension.proto` at tag
    `v0.1.2`, compiled with `grpcio-tools` 1.76.0. *Verified.*
22. **The negotiation values aren't in the proto or the docs.** `MiddlewareManifest.extension` (`PeerMetadata`) is
    "Required for negotiation", but the version and capability to declare are found only in Rust:
    [`openshell-core/src/extension_protocol.rs`](https://github.com/NVIDIA/OpenShell/blob/v0.1.2/crates/openshell-core/src/extension_protocol.rs)
    gives `protocol_version {major: 1, minor: 0}` and the capability `openshell.supervisor-middleware.contract`
    (both supported and required). *Verified* (source). A non-Rust implementer has to read the Rust source to get
    past `Describe`.
23. **Registration and policy work as documented.** I added to `gateway.toml`:
    ```toml
    [[openshell.supervisor.middleware]]
    name = "pg-smoke"
    grpc_endpoint = "http://127.0.0.1:50051"
    allow_insecure_transport = true
    max_payload_bytes = 262144
    timeout = "1500ms"
    ```
    and to the sandbox policy: `network_middlewares: {smoke: {middleware: pg-smoke, endpoints: {include:
    ["host.openshell.internal"]}}}`. After the gateway restart it called `Describe`, and `ValidateConfig` was called
    when the sandbox was created. *Verified.* `127.0.0.1` works for both the gateway and the supervisors because of
    host networking (entry 20).
24. **End-to-end allow and deny both work.** From the sandbox:
    - A clean POST was allowed and reached upstream (the server logged `POST /upload`), 12 ms round trip.
    - A POST containing the marker got **HTTP 403** and never reached upstream. The agent receives this JSON:
      ```json
      {"binary":"/usr/local/bin/python3.12","detail":"Request rejected by configured middleware",
       "error":"middleware_denied","host":"host.openshell.internal","layer":"l7","method":"POST",
       "middleware":"smoke","path":"/upload","policy":"fixture_api","port":8099,"reason_code":"pg_smoke_marker"}
      ```
      The middleware's free-text `reason` isn't relayed; the `reason_code` is. *Verified.* This is a clear,
      structured denial, much more helpful than the connection-stage `EACCES` (entry 15).
25. **`RequestContext.originating_process` was empty** in every `EvaluateHttpRequest` the middleware received
    (`binary=""`), although the gateway knew the binary: it appears in the 403 body above, and the policy matched on
    `binaries`. The proto says the field is set "when available". *Verified* (observed); why it's empty is unknown.
    A middleware can't make decisions based on the process.
26. **Clean-up:** the smoke middleware and test server are stopped, `gateway.toml` is back to the install default,
    and there are no sandboxes. Docker Desktop host networking stays **on**, which the integration needs.

### 2026-10-09T13:44Z · Isolation check with Docker Desktop host networking enabled

27. **Host networking does not weaken sandbox isolation.** It is a Docker-wide setting, so after enabling it I
    checked that the sandbox boundary is unchanged. One sandbox ran a policy allowing **only**
    `host.openshell.internal:8099`, with two servers on the Mac: `:8099` (allowed) and `:8098` (not allowed). Probes
    from inside the sandbox (Python `urllib` and raw `socket`):

    | Probe from inside the sandbox | Result |
    |---|---|
    | `http://host.openshell.internal:8099/` (allowed) | **reached**, HTTP 200 |
    | `http://host.openshell.internal:8098/` (port not in policy) | blocked, `EACCES` |
    | `host.openshell.internal:17670` (the gateway's own port) | blocked, `EACCES` |
    | `https://example.com/` (public host, not in policy) | blocked, `EACCES` |
    | `1.1.1.1:443` (public IP, raw TCP) | blocked, `EACCES` |
    | `127.0.0.1:8098` and `127.0.0.1:17670` (raw loopback) | `ECONNREFUSED`: the sandbox's **own** loopback, not the Mac's |
    | `192.168.65.254:8098` (Docker Desktop host IP, raw TCP) | blocked, `EACCES` |
    | `host.docker.internal:8098` | does not resolve |

    The servers' access logs agree: `:8099` logged the sandbox's request, while `:8098` logged only my own check
    from the Mac. *Verified* (13:44Z).

    **Why:** `docker inspect` shows the **workload** container with `NetworkMode=none`, `Privileged=false`, no added
    capabilities and no published ports. Its only interface is `lo` plus the kernel's default unconfigured tunnel
    devices, with no `eth0`. Only the trusted **supervisor** container uses `NetworkMode=host`, and every workload
    connection goes through it and its policy check. This matches the driver source, which requires the workload's
    "outer fence" to be `network_mode=none`
    ([`openshell-driver-docker/src/lib.rs`](https://github.com/NVIDIA/OpenShell/blob/v0.1.2/crates/openshell-driver-docker/src/lib.rs),
    around lines 3008–3015). *Verified.*

    **Host-level side effect, outside OpenShell:** with host networking on, *any* container started with
    `--network host` on this Mac (not only OpenShell's supervisor) can reach services the Mac binds to `127.0.0.1`.
    That widens trust in the Docker VM as a whole, not the sandbox boundary. Only run host-network containers you
    trust; OpenShell sandbox workloads are not host-network containers. *Assumption* based on Docker Desktop's
    documented behaviour, consistent with the probe results above.

28. **Default deny holds:** every destination not named in the policy was refused (public hostnames, public IPs,
    other host ports, Docker's host IP and alias), and the connection-stage denial stays `EACCES` (entry 15).
    *Verified.*

29. **Minor CLI friction:** `openshell sandbox create --upload <file> -- <command>` errors with "the argument
    '--upload <UPLOAD>' cannot be used with '[COMMAND]...'". You can't upload a script and run it in one step.
    Workaround: pass the script inline (`python3 -c "$(cat probe.py)"`) or upload into a kept sandbox and then use
    `sandbox exec`. *Verified.*

## Status at end of Phase 1 (2026-10-09T13:42Z)

| Component | Version / state |
|---|---|
| Docker Desktop | 4.88.1 (engine 29.8.2), **host networking enabled** |
| OpenShell | 0.1.2 via the Homebrew local tap, LaunchAgent `sh.brew.openshell`, gateway `https://localhost:17670` (mTLS) |
| Sandbox start | works (under 1 s) with host networking and the default gateway config |
| Sandbox → host service | works via `host.openshell.internal`, **only on ports the policy allows** (entry 27) |
| Default-deny egress | holds under host networking (entries 27–28) |
| Supervisor middleware (Python gRPC) | works: `Describe`, `ValidateConfig` and `EvaluateHttpRequest` with allow and deny |
