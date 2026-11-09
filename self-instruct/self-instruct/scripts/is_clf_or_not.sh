#!/usr/bin/env bash
set -euo pipefail
printf "%s\n" "This legacy GPT-3 stage is disabled. Use run_self_instruct.py; task labels and schemas are explicit, and student training runs separately." >&2
exit 2
