#!/usr/bin/env bash
# Sole entry: scripts/dev.sh test attn-bench c5-inspect / c5-reference.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
source "$REPO/crates/mimo26-attn/tests/gpu/gpu_guard.sh"
[[ "${HOST:-local}" == coordinator ]] || { echo 'RESULT: REFUSE C5 is coordinator-only'; exit 3; }
mode="${1:-}"
case "$mode" in inspect|import) ;; run) [[ "${MIMO26_ATTN_BUILDER_WINDOW:-0}" == 1 ]] || exit 3;; *) exit 3;; esac
export TZ=Australia/Sydney
python3 "$REPO/crates/mimo26-attn/tests/gpu/c5_reference.py" --selftest
root="${MIMO26F_BUILD_ROOT:-$REPO/target/mimo26f-builds}"
case "$(realpath -m "$root")" in /mnt/scratch|/mnt/scratch/*) exit 3;; esac
mkdir -p "$root"
slot=$(mktemp -d "$root/attn-c5.XXXXXX")
receipt="$REPO/runs/20260923-i4/attn/$(basename "$slot")"
mkdir -p "$receipt"
main() {
  date '+%F %T %Z'
  echo "COMMAND HOST=coordinator scripts/dev.sh test attn-bench C5 mode=$mode"
  printf 'SOURCE '; git -C "$REPO" rev-parse HEAD || return
  stage=$(ssh coordinator 'mkdir -p /var/tmp/mimo26f-attn; mktemp -d /var/tmp/mimo26f-attn/mimo26f-c5-XXXXXXXX') || return
  [[ "$stage" =~ ^/var/tmp/mimo26f-attn/mimo26f-c5-[A-Za-z0-9]+$ ]] || return 3
  rsync -a "$REPO/crates/mimo26-attn/tests/gpu/c5_inspect.py" "coordinator:$stage/c5_inventory.py" || return
  rsync -a "$REPO/crates/mimo26-attn/tests/gpu/c5_reference.py" "coordinator:$stage/mimo26f-c5-reference.py" || return
  {
    declare -f m26_guard_service m26_guard_read_argv m26_gpu_guard
    cat <<'REMOTE'
set -euo pipefail
stage="$1"; mode="$2"
export TZ=Australia/Sydney
date '+%F %T %Z'
docker image ls --no-trunc --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.CreatedAt}}' --filter 'reference=afd-native-vllm:*'
# Immutable image inspected before the one-shot run; no moving-tag selection.
image=sha256:28f2e44cce67d891d93a563d4a21511bf92d29d4f92958098aa17ef05c0e1a92
if ! docker image inspect "$image" --format 'C5_IMAGE {{.Id}} tags={{json .RepoTags}} created={{.Created}}'; then
  echo 'C5_SKIP inspected local image absent; no pull attempted'; exit 0
fi
name="mimo26f-attn-c5-${stage##*-}"
trap 'docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
args=(-e NVIDIA_VISIBLE_DEVICES=void -e CUDA_VISIBLE_DEVICES=-1)
script=c5_inventory.py; script_args=()
if [[ "$mode" == import ]]; then script=mimo26f-c5-reference.py; script_args=(--import-probe); fi
if [[ "$mode" == run ]]; then
  exec 8>/var/tmp/mimo26f-attn/.gpu.lock; flock 8
  nvidia-smi; m26_gpu_guard
  args=(--gpus device=0 -e CUDA_VISIBLE_DEVICES=0 -e CUDA_DISABLE_PTX_JIT=1
    -e FLASHINFER_DISABLE_JIT=1 -e TORCH_COMPILE_DISABLE=1 -e HF_HUB_OFFLINE=1 -e HOME=/tmp)
  script=mimo26f-c5-reference.py
fi
sha256sum "$stage/$script"
timeout --signal=TERM --kill-after=5s 180s docker run --rm --pull=never --name "$name" \
  --network none --read-only --tmpfs /tmp:rw,exec,size=2g -e TZ=Australia/Sydney "${args[@]}" \
  --mount "type=bind,src=$stage,dst=/work,readonly" --entrypoint python3 "$image" "/work/$script" "${script_args[@]}"
REMOTE
  } | ssh coordinator bash -s -- "$stage" "$mode"
}
set +e
main 2>&1 | tee "$slot/receipt.log"
statuses=("${PIPESTATUS[@]}")
set -e
rc="${statuses[0]}"
(( statuses[1] == 0 )) || rc="${statuses[1]}"
cp "$slot/receipt.log" "$receipt/RESULT.md"
echo "C5_RECEIPT $receipt/RESULT.md"
exit "$rc"
