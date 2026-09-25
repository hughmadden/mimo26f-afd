#!/usr/bin/env bash
# R12 exact retained binaries: no nvcc, no kernel edits, no new timing gate.
set -euo pipefail
mode="${1:?--selftest|--run|--remote}"
selection() {
  case "$1:$2" in B1:1) echo '36 3' ;; B1:8) echo '918 3' ;; B2:2) echo '216 4' ;; B2:3) echo '384 4' ;; B2:4) echo '552 4' ;; B2:5) echo '720 4' ;; B2:8) echo '1224 4' ;; *) return 2 ;; esac
}
validate_log() {
  local family="$1" log="$2" prefix final
  if [[ "$family" == B1 ]]; then prefix='B1 BENCH ROW '; final='B1 BENCH FINAL verdict=6;'; else prefix='BENCH ROW residents=256 '; final='BENCH FINAL residents=256 verdict=STOP;'; fi
  [[ "$(grep -c "^$prefix" "$log" || true)" == 8 ]] || return 2
  grep -q "^$final" "$log" || return 2
  if grep -Eq '==ERROR==|CUDA_FAILURE|ORACLE_MISMATCH|INPUT_FAILURE' "$log"; then return 2; fi
}
if [[ "$mode" == --selftest ]]; then
  root="${2:?unique staging}"; mkdir "$root/frozen-control"
  # Enumerate the old schedule independently: correctness, M1 negative,
  # ten warmups, 31 samples; validation does not launch kernels.
  for family in B1 B2; do
    k=3; [[ "$family" != B2 ]] || k=4
    cursor=0
    for m in {1..8}; do
      ((cursor+=k)); if [[ "$m" == 1 ]]; then ((cursor+=k)); fi
      ((cursor+=10*k))
      case "$family:$m" in B1:1|B1:8|B2:2|B2:3|B2:4|B2:5|B2:8)
        read -r skip count <<<"$(selection "$family" "$m")"
        [[ "$skip" == "$cursor" && "$count" == "$k" ]] ;;
      esac
      ((cursor+=31*k))
    done
    log="$root/frozen-control/$family.log"
    if [[ "$family" == B1 ]]; then prefix='B1 BENCH ROW '; final='B1 BENCH FINAL verdict=6;'; else prefix='BENCH ROW residents=256 '; final='BENCH FINAL residents=256 verdict=STOP;'; fi
    for m in {1..8}; do echo "${prefix}M=$m"; done >"$log"
    echo "$final" >>"$log"; validate_log "$family" "$log"
    echo '==ERROR== counter failure' >>"$log"
    got=0; validate_log "$family" "$log" || got=$?; [[ "$got" == 2 ]]
  done
  got=0; selection B1 4 >/dev/null || got=$?; [[ "$got" == 2 ]]
  echo 'HOST PASS frozen selectors: exact warmed sample-zero offsets, seven cases; counter errors and unsupported selectors refused; no GPU'
  exit 0
fi
if [[ "$mode" == --remote ]]; then
  ROOT="${2:?scratch}"; SOURCE="${3:?revision}"; cases="${4:-r12}"
  case "$cases" in r12) pairs=(B1:1 B1:8 B2:4 B2:5) ;; r13-f2) pairs=(B2:2 B2:3) ;; r14-model) pairs=(B2:8) ;; *) exit 2 ;; esac
  [[ "$ROOT" == /var/tmp/mimo26f-kernel/profile-frozen-* && -d "$ROOT" ]] || exit 2
  export TZ=Australia/Sydney LC_ALL=C MIMO26_BUILDER_GPU=1
  mkdir "$ROOT/receipts"; exec > >(tee "$ROOT/receipts/cell.log") 2>&1
  exec 9>/var/tmp/mimo26f-kernel/.cell.lock; flock -n 9 || exit 2
  printf 'time=%s source=%s exact-retained-binaries=yes timing-gate=NO\n' "$(date --iso-8601=seconds)" "$SOURCE"
  sha256sum -c "$ROOT/SOURCE.sha256"
  guard() {
    local info owners available uuid
    info="$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader)"
    [[ "$info" != *$'\n'* && "$info" == *GB10* ]] || return 2
    uuid="${info%%,*}"; export CUDA_VISIBLE_DEVICES="${uuid//[[:space:]]/}"
    owners="$(nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader)"
    [[ -z "${owners//[[:space:]]/}" ]] || { echo "Refuse CUDA owners: $owners"; return 2; }
    available="$(free -b | awk '$1=="Mem:" {print $7}')"
    [[ "$available" =~ ^[0-9]+$ && "$available" -ge 8589934592 ]] || return 2
    printf '%s %s MemAvailable=%s CUDA owners=none\n' "$(date --iso-8601=seconds)" "$info" "$available" | tee -a "$ROOT/receipts/guards.log"
  }
  guard
  NCU=/usr/local/cuda/bin/ncu; [[ -x "$NCU" ]] || exit 2
  "$NCU" --version >"$ROOT/receipts/ncu-version.log"
  "$NCU" --help >"$ROOT/receipts/ncu-help.txt"
  for pair in "${pairs[@]}"; do
    family="${pair%:*}"; m="${pair#*:}"; read -r skip count <<<"$(selection "$family" "$m")"
    if [[ "$family" == B1 ]]; then
      frozen=/var/tmp/mimo26f-kernel/unpack-SQAnjz6z
      driver="$frozen/mimo26f-b1-primitive"
      args=(--bench-connected /var/tmp/models/MiMo-V2.6-Flash-RL "$frozen/oracle")
      filter='regex:.*(connected_fc1|quantize_kernel|connected_fc2_fp8).*'
    else
      frozen=/var/tmp/mimo26f-kernel/unpack-rGUNIJiC
      driver="$frozen/mimo26f-expert-proof"
      args=(bench /var/tmp/models/MiMo-V2.6-Flash-RL "$frozen/fixture.json" "$frozen" 256)
      filter='regex:.*(gemm|activate).*'
    fi
    [[ -x "$driver" ]] || { echo 'Retained frozen binary missing; no rebuild fallback'; exit 2; }
    sha256sum -c "$frozen/SOURCE.sha256" >"$ROOT/receipts/$family-m$m-frozen-source.log"
    sha256sum "$driver" >"$ROOT/receipts/$family-m$m-binary-before.txt"
    guard; stem="$ROOT/profile-$family-m$m"
    cmd=(sudo -n MIMO26_BUILDER_GPU=1 "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" TZ=Australia/Sydney LC_ALL=C "$NCU" --set full --replay-mode kernel --clock-control none --cache-control none --check-exit-code no --kernel-name "$filter" --launch-skip "$skip" --launch-count "$count" --export "$stem" "$driver" "${args[@]}")
    printf 'FROZEN PROFILE family=%s M=%s skip=%s count=%s sample=0 after=10-warmups\n' "$family" "$m" "$skip" "$count"
    printf '%q ' "${cmd[@]}"; printf '\n'
    rc=0; timeout 120s "${cmd[@]}" >"$ROOT/receipts/profile-$family-m$m.log" 2>&1 || rc=$?
    guard; echo "NCU_EXIT family=$family M=$m rc=$rc"
    [[ "$rc" == 0 ]] || exit "$rc"
    validate_log "$family" "$ROOT/receipts/profile-$family-m$m.log"
    [[ -s "$stem.ncu-rep" ]] || exit 2
    sha256sum "$driver" >"$ROOT/receipts/$family-m$m-binary-after.txt"
    cmp "$ROOT/receipts/$family-m$m-binary-before.txt" "$ROOT/receipts/$family-m$m-binary-after.txt"
    "$NCU" --import "$stem.ncu-rep" --page raw --csv >"$ROOT/receipts/profile-$family-m$m.csv"
    "$NCU" --import "$stem.ncu-rep" --page details >"$ROOT/receipts/profile-$family-m$m-details.txt"
    if [[ "$cases" == r13-f2 || "$cases" == r14-model ]]; then
      "$NCU" --import "$stem.ncu-rep" --page source --print-source cuda,sass --csv --metrics smsp__pcsamp_warps_issue_stalled_long_scoreboard,smsp__pcsamp_warps_issue_stalled_short_scoreboard,smsp__pcsamp_warps_issue_stalled_wait,smsp__pcsamp_warps_issue_stalled_barrier,smsp__pcsamp_warps_issue_stalled_math_pipe_throttle,smsp__pcsamp_warps_issue_stalled_not_selected >"$ROOT/receipts/profile-$family-m$m-source.csv"
    fi
    sha256sum "$stem.ncu-rep" >>"$ROOT/receipts/profile-binaries-sha256.txt"
  done
  echo "FROZEN CAPTURE PASS: ${#pairs[@]} cases ($cases), original binaries unchanged; all printed timings are profiler-contaminated diagnostics, NEVER replacement bandwidth gates"
  exit 0
fi
case "$mode" in --run) cases=r12 ;; --run-r13-f2) cases=r13-f2 ;; --run-r14-model) cases=r14-model ;; *) exit 2 ;; esac
[[ "${HOST:-}" == spark1 && "${MIMO26_BUILDER_GPU:-0}" == 1 ]] || exit 2
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
[[ -z "$(git -C "$REPO" status --porcelain -- crates/mimo26-expert crates/mimo26-repack)" ]] || { echo 'Commit expert sources first'; exit 2; }
SOURCE="$(git -C "$REPO" rev-parse HEAD)"
STAGING="$(mktemp -d "${MIMO26F_BUILD_ROOT:?dev.sh required}/expert-frozen-profile.XXXXXX")"
RECEIPTS="$REPO/runs/20260923-i4/expert/spark1-$(TZ=Australia/Sydney date +%Y%m%d-%H%M%S)-${SOURCE:0:12}-$(basename "$STAGING")"
mkdir "$RECEIPTS"; exec > >(tee "$RECEIPTS/local.log") 2>&1
printf 'source=%s staging=%s receipts=%s\n' "$SOURCE" "$STAGING" "$RECEIPTS"
REMOTE="$(ssh -o BatchMode=yes -o ConnectTimeout=10 spark1 'mkdir -p /var/tmp/mimo26f-kernel && mktemp -d /var/tmp/mimo26f-kernel/profile-frozen-XXXXXXXX')"
[[ "$REMOTE" =~ ^/var/tmp/mimo26f-kernel/profile-frozen-[[:alnum:]]{8}$ ]] || exit 2
printf 'remote=%s\n' "$REMOTE" >"$RECEIPTS/remote.txt"
cp "$REPO/crates/mimo26-expert/tests/gpu/frozen_profile.sh" "$STAGING/"
sha256sum "$STAGING/frozen_profile.sh" | sed "s|$STAGING/|$REMOTE/|" >"$STAGING/SOURCE.sha256"
rsync -a --checksum "$STAGING/frozen_profile.sh" "$STAGING/SOURCE.sha256" "spark1:$REMOTE/"
rc=0; ssh -o BatchMode=yes spark1 "timeout 540s bash $REMOTE/frozen_profile.sh --remote $REMOTE $SOURCE $cases" || rc=$?
retrieval=0; rsync -a --safe-links "spark1:$REMOTE/receipts/" "$RECEIPTS/" || retrieval=$?
echo "remote_exit=$rc retrieve_exit=$retrieval"
[[ "$retrieval" == 0 ]] || exit "$retrieval"
[[ "$rc" == 0 ]] || exit "$rc"
mkdir "$STAGING/ncu"
rsync -a --safe-links --include='profile-*.ncu-rep' --exclude='*' "spark1:$REMOTE/" "$STAGING/ncu/"
analysis=0
for raw in "$RECEIPTS/"profile-*-m[1-8].csv; do
  family=B1; [[ "$(basename "$raw")" != profile-B2-* ]] || family=B2
  python3 -B "$REPO/crates/mimo26-expert/tests/gpu/ncu_report.py" --family "$family" "$raw" >"${raw%.csv}-summary.json" || analysis=$?
done
echo "FROZEN ANALYSIS exit=$analysis; missing metrics remain explicit"
exit "$analysis"
