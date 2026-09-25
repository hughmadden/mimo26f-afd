#!/usr/bin/env bash
# Shared attention preflight. Source only; no actions occur until called.
# GPU0 only. This is a snapshot, not a cross-family reservation: the builder
# still coordinates future launches. Never kill owners or modify services.

m26_guard_service() {
  # Parse argv, not a substring in a flattened command line. A script or -c
  # argument mentioning a service is not the service's Python -m invocation.
  (( $# >= 3 )) || return 1
  local exe="${1##*/}"
  [[ "$exe" =~ ^python(3(\.[0-9]+)?)?$ ]] || return 1
  shift
  while (( $# )); do
    case "$1" in -u|-B|-E|-I|-s|-S|-O|-OO|-q) shift ;; *) break ;; esac
  done
  (( $# >= 2 )) && [[ "$1" == -m ]] || return 1
  case "$2" in
    service.cotenant_a.cotenant_a|service.cotenant_b.cotenant_b) printf '%s\n' "$2" ;;
    *) return 1 ;;
  esac
}

m26_guard_read_argv() {
  M26_OWNER_ARGV=()
  [[ -r "/proc/$1/cmdline" ]] || return 1
  mapfile -d '' -t M26_OWNER_ARGV < "/proc/$1/cmdline" || return 1
  (( ${#M26_OWNER_ARGV[@]} > 0 ))
}

m26_gpu_guard() {
  # CUDA device0 must refer to the same physical GPU queried below.
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "$CUDA_VISIBLE_DEVICES" != 0 ]]; then
    echo 'RESULT: REFUSE GPU guard requires default CUDA visibility or device0' >&2
    return 3
  fi
  local owners raw pid module free
  if ! owners=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits); then
    echo 'RESULT: REFUSE cannot enumerate CUDA owners' >&2; return 3
  fi
  while IFS= read -r raw; do
    [[ -n "${raw//[[:space:]]/}" ]] || continue
    if [[ ! "$raw" =~ ^[[:space:]]*([1-9][0-9]{0,9})[[:space:]]*$ ]]; then
      echo 'RESULT: REFUSE malformed CUDA owner response' >&2; return 3
    fi
    pid="${BASH_REMATCH[1]}"
    if ! m26_guard_read_argv "$pid"; then
      echo "RESULT: REFUSE cannot identify CUDA owner pid=$pid" >&2; return 3
    fi
    if ! module=$(m26_guard_service "${M26_OWNER_ARGV[@]}"); then
      # Do not log arbitrary argv: an unrelated process may contain secrets.
      echo "RESULT: REFUSE concurrent CUDA owner pid=$pid (not an approved resident module)" >&2
      return 3
    fi
    echo "GPU_OWNER allow pid=$pid module=$module"
  done <<< "$owners"
  if ! free=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits); then
    echo 'RESULT: REFUSE cannot read free GPU memory' >&2; return 3
  fi
  if [[ ! "$free" =~ ^[[:space:]]*([0-9]{1,9})[[:space:]]*$ ]]; then
    echo 'RESULT: REFUSE malformed GPU memory response' >&2; return 3
  fi
  free=$((10#${BASH_REMATCH[1]}))
  if (( free < 4096 )); then
    echo "RESULT: REFUSE free_MiB=$free below reserve_MiB=4096" >&2; return 3
  fi
  echo "GPU_GUARD PASS device=0 free_MiB=$free reserve_MiB=4096 owners=resident-only-or-none"
}
