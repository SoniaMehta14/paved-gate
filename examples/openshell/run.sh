#!/bin/sh
# OpenShell builds the jail. Paved Gate reads the mail.
#   examples/openshell/run.sh            real OpenShell sandbox, mock evaluator (default)
#   examples/openshell/run.sh --live     real OpenShell sandbox, live Jev (TYPESAFE_API_KEY in .env)
#   examples/openshell/run.sh --local    no Docker/OpenShell: simulated supervisor
set -eu
cd "$(dirname "$0")/../.."
uv sync --quiet --inexact --extra openshell
# macOS: iCloud Drive can mark files in .venv as hidden, and Python skips hidden .pth files, which breaks the
# editable install ("No module named 'paved_gate'"). Clear the flag after syncing, then run without re-syncing.
if [ "$(uname)" = Darwin ]; then chflags -R nohidden .venv 2>/dev/null || true; fi
exec uv run --no-sync python examples/openshell/demo.py "$@"
