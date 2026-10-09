# Vendored OpenShell protos

| File | Source | sha256 |
|---|---|---|
| `supervisor_middleware.proto` | https://github.com/NVIDIA/OpenShell/blob/v0.1.2/proto/supervisor_middleware.proto | `bee96ac940303d8a7530e6e2283a98077bb428a05ef610b51d018ea18377bebd` |
| `extension.proto` | https://github.com/NVIDIA/OpenShell/blob/v0.1.2/proto/extension.proto | `60f28ee9dc846d41152f4db7e82a439c78d024786efaceb464900d1d1ad129ad` |

Copied verbatim from tag **v0.1.2** (released 2026-09-28), Apache-2.0, © NVIDIA CORPORATION & AFFILIATES.
Upstream notes that "the middleware API is still evolving"
(https://docs.nvidia.com/openshell/extensibility/supervisor-middleware.md), so the version is pinned
deliberately. Regenerate the Python stubs with `sh scripts/gen_openshell_stubs.sh`.

Negotiation values used by the middleware (from
https://github.com/NVIDIA/OpenShell/blob/v0.1.2/crates/openshell-core/src/extension_protocol.rs, not documented
elsewhere): protocol version `1.0`, capability `openshell.supervisor-middleware.contract`.
