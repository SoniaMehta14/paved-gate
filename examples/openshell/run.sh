#!/bin/sh
# OpenShell builds the jail. Paved Gate reads the mail.
#   examples/openshell/run.sh            real OpenShell sandbox, mock evaluator (default)
#   examples/openshell/run.sh --live     real OpenShell sandbox, live Jev (TYPESAFE_API_KEY in .env)
#   examples/openshell/run.sh --local    no Docker/OpenShell: simulated supervisor
set -eu
cd "$(dirname "$0")/../.."
exec uv run --extra openshell python examples/openshell/demo.py "$@"
