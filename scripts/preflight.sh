#!/usr/bin/env bash
# mimo26f-afd preflight (doctor). Read-only. Never mutates the fleet.
set -euo pipefail
note() { printf '%s\n' "$*"; }
fail=0

note "=== mimo26f-afd preflight ==="
note "host: $(hostname)  time: $(TZ=Australia/Sydney date '+%Y-%m-%d %H:%M:%S AEST')"

if command -v nvidia-smi >/dev/null 2>&1; then
  note "--- nvidia-smi -L ---"
  nvidia-smi -L || fail=1
  note "--- compute caps ---"
  nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader || fail=1
else
  note "MISSING: nvidia-smi (ok on non-GPU hosts / CI)"
fi

if command -v findmnt >/dev/null 2>&1; then
  note "--- scratch guard (never build on /mnt/scratch) ---"
  findmnt -T /mnt/scratch >/dev/null 2>&1 && note "NOTE: /mnt/scratch is mounted — do not build there" || note "/mnt/scratch: not mounted (good)"
fi

note "--- disk (repo parent) ---"
df -h "${PWD%/*}" 2>/dev/null || true

note "--- concurrent CUDA owners (informational) ---"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || note "no nvidia-smi"

if [[ "$fail" -ne 0 ]]; then
  note "RESULT: FAIL"
  exit 1
fi
note "RESULT: PASS (informational doctor; not a promote gate)"
