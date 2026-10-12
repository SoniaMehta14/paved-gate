# Feedback for NVIDIA OpenShell

Notes from building [Paved Gate](../../README.md#nvidia-openshell-integration) as an OpenShell supervisor middleware on
macOS, Apple Silicon, with Docker Desktop. Every item comes from the timestamped
[friction log](FRICTION_LOG.md), which has the commands, outputs and source links.

## Already raised upstream

- **Docker Desktop: sandboxes either fail to start or cannot reach host services.** Data point on
  [#3880](https://github.com/NVIDIA/OpenShell/issues/3880#issuecomment-6084784746). NVIDIA opened
  [#4403](https://github.com/NVIDIA/OpenShell/issues/4403) and the fix [#4404](https://github.com/NVIDIA/OpenShell/pull/4404);
  test results on this Mac, with the unpatched base as a control:
  [#4404 comment](https://github.com/NVIDIA/OpenShell/pull/4404#issuecomment-6115080249) (friction log entries 13 to 20,
  27 and 41).
- **The base capability that `Describe` must return is no longer in the docs.**
  [#4378](https://github.com/NVIDIA/OpenShell/issues/4378) (entries 22 and 33). **Fixed** by
  [#4405](https://github.com/NVIDIA/OpenShell/pull/4405), merged 2026-10-12: a base capability table and example
  manifest in the docs, the values in each proto, and clearer negotiation errors
  ([reply confirming it](https://github.com/NVIDIA/OpenShell/issues/4378#issuecomment-6132126928)).
- **A file baked into the image fails with a bare "Permission denied".** Comment on
  [#4270](https://github.com/NVIDIA/OpenShell/issues/4270#issuecomment-6085162496), which asks for filesystem denial
  events (entry 34).

## Draft issues (not posted)

### 1. `generate-certs --output-dir` silently replaces the CLI's credentials for the default gateway

**Environment**
* OpenShell 0.1.2 (Homebrew) on macOS 26.6.2, Apple Silicon, Docker Desktop 4.88.1
* Reproduced with a gateway built from `dca78bd` (#4404, one commit on `main` at `eeba0e7`); `certgen.rs` has the same
  copy at v0.1.2

**Steps to reproduce**
1. Install OpenShell with Homebrew, so the CLI talks to the local gateway with the bundle in
   `~/.config/openshell/gateways/openshell/mtls/`.
2. Generate a PKI for a second, separate gateway:
   `openshell-gateway generate-certs --output-dir /tmp/second-gateway/tls --server-san host.openshell.internal`
3. Run `openshell status`.

**Expected**
The PEMs are written to the output directory and nothing else changes.

**Actual**
`ca.crt`, `tls.crt` and `tls.key` in `~/.config/openshell/gateways/openshell/mtls/` are replaced with the new client
bundle, and the CLI can no longer reach the existing gateway: `invalid peer certificate: BadSignature`. Local mode
always copies the client materials for "CLI auto-discovery"
([`certgen.rs` at v0.1.2](https://github.com/NVIDIA/OpenShell/blob/v0.1.2/crates/openshell-server/src/certgen.rs),
module comment and the `store_pki_bundle("openshell", …)` call). The copy is described in the source, but
`--help` says only "Write PEMs to a filesystem directory instead of Kubernetes Secrets".

**Suggestion**
* Skip the copy, or warn, when the CLI bundle already exists and was signed by a different CA.
* Mention the copy in `--help`, or add a flag such as `--no-cli-bundle`.

Related: [#2481](https://github.com/NVIDIA/OpenShell/issues/2481) and
[#2560](https://github.com/NVIDIA/OpenShell/issues/2560) report other commands that overwrite this bundle through the
same `store_pki_bundle` function. Friction log entry 42.

### 2. Registering or changing a supervisor middleware requires a gateway restart

**Environment**
* OpenShell 0.1.2 (Homebrew) on macOS 26.6.2, Apple Silicon, Docker Desktop 4.88.1

**Steps to reproduce**
1. Add a `[[openshell.supervisor.middleware]]` entry to `gateway.toml`.
2. Create a sandbox whose policy selects it in `network_middlewares`.

**Expected**
A way to register a middleware without restarting the gateway, for example through the CLI or the API, as the
`network_policies` and `network_middlewares` policy sections already update live
([policy schema](https://docs.nvidia.com/openshell/how-it-works/policies/schema.md)).

**Actual**
The registration is read only at startup. The
[configure page](https://github.com/NVIDIA/OpenShell/blob/v0.1.2/docs/extensibility/supervisor-middleware/configure.mdx)
says to "restart the gateway after changing registrations". A demo that adds a middleware and then removes it costs
two gateway restarts, each about 2 to 3 seconds on this Mac.

**Suggestion**
* Reload middleware registrations on a signal or through an admin RPC, running `Describe` and the capability
  negotiation before accepting traffic, as at startup.
* Or allow a sandbox scoped registration that is created and removed with the sandbox.

Friction log entry 36.

### 3. State the 19 character sandbox name limit in the CLI

**Environment**
* OpenShell 0.1.2 (Homebrew) on macOS 26.6.2, Apple Silicon

**Steps to reproduce**
1. `openshell sandbox create --name pg-bench-u-1791572940 -- echo hi` (21 characters)

**Expected**
The limit is visible before the request: in `openshell sandbox create --help`, or as a check in the CLI.

**Actual**
The gateway rejects it with `Client specified an invalid argument … name exceeds maximum length (21 > 19)`. The
help text describes `--name` only as "Optional sandbox name (auto-generated when omitted)". The limit is deliberate:
three names joined by `--` must fit one 63 character DNS label
([`grpc/mod.rs` at v0.1.2](https://github.com/NVIDIA/OpenShell/blob/v0.1.2/crates/openshell-server/src/grpc/mod.rs),
`MAX_ROUTABLE_NAME_LEN`), and [#2651](https://github.com/NVIDIA/OpenShell/issues/2651) shows integrations tripping on
it before.

**Suggestion**
* Add the limit and its reason to the `--name` help text and the sandbox naming docs.
* Validate the length in the CLI so the error appears before any request is sent.

Friction log entry 37.

### 4. Comment for [#4269](https://github.com/NVIDIA/OpenShell/issues/4269) (`RequestContext.originating_process`)

A data point from a Python middleware on OpenShell 0.1.2 (Homebrew, macOS 26.6.2, Docker driver):
`originating_process` was empty in every `EvaluateHttpRequest` the middleware received, although the gateway knew the
binary. The same request's `middleware_denied` response to the sandbox named it
(`"binary":"/usr/local/bin/python3.12"`), and the network policy matched on `binaries`. The proto marks the field as set
"when available"
([`supervisor_middleware.proto` at v0.1.2](https://github.com/NVIDIA/OpenShell/blob/v0.1.2/proto/supervisor_middleware.proto)),
so a middleware cannot tell whether it is missing by design or by accident. Filling it, or documenting when it is
empty, would let a middleware combine caller and content in one decision, as #4269 describes.

Friction log entries 24 and 25.

## Not drafted

- **Intermittent transport failures** (entry 38): a few requests in each 430 request run failed with
  `RemoteDisconnected` or a timeout and succeeded on retry, with or without a middleware. The cause is unknown, and
  the test API on the Mac was a simple Python server, so this needs isolating before it is worth an issue.
- **`PRE_RETURN` runs on every forwarded response** (entry 39): working as configured; worth knowing for cost.
