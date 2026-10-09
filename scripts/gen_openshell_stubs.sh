#!/bin/sh
# Regenerate the Python gRPC stubs for the vendored OpenShell supervisor-middleware protos.
#
#   sh scripts/gen_openshell_stubs.sh
#
# The protos in src/paved_gate/integrations/openshell/proto/ are copied verbatim from
# github.com/NVIDIA/OpenShell at the tag in PROVENANCE.md. Upstream says "the middleware API is
# still evolving", so upgrade deliberately: copy the new protos, update PROVENANCE.md, rerun this.
set -eu
cd "$(dirname "$0")/.."
PKG=src/paved_gate/integrations/openshell
OUT=$PKG/_gen
rm -f "$OUT"/*_pb2*.py "$OUT"/*_pb2*.pyi
uv run python -m grpc_tools.protoc -I "$PKG/proto" \
  --python_out="$OUT" --grpc_python_out="$OUT" --pyi_out="$OUT" \
  "$PKG/proto/extension.proto" "$PKG/proto/supervisor_middleware.proto"
# protoc emits top-level imports ("import extension_pb2"); make them package-relative.
for f in "$OUT"/*_pb2*.py "$OUT"/*_pb2*.pyi; do
  sed -i.bak -E 's/^import (extension_pb2|supervisor_middleware_pb2) as /from . import \1 as /' "$f"
  rm -f "$f.bak"
done
echo "regenerated stubs in $OUT"
