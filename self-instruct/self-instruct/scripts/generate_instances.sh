#!/usr/bin/env bash
set -euo pipefail
workspace_root="$(cd -- "$(dirname -- "$0")/../../.." && pwd)"
exec python3 "$workspace_root/run_self_instruct.py" answer "$@"
