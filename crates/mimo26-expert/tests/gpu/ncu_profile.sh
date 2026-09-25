#!/usr/bin/env bash
# Sourced by the guarded native Spark cell. Never installs tools or changes clocks.
validate_ncu_capture() {
  local family="$1" m="$2" rc="$3" log="$4" report="$5"
  case "$family:$m" in B1:1|B1:8|B2:4|B2:5) ;; *) return 2 ;; esac
  [[ "$rc" == 0 ]] || return "$rc"
  [[ -s "$report" && -f "$log" ]] || return 2
  [[ "$(grep -Fc "PROFILE PASS family=$family M=$m " "$log" || true)" == 1 ]] || return 2
  if grep -Eq '==ERROR==|ERR_NVGPUCTRPERM|No kernels were profiled|^B1 BENCH ROW |^BENCH ROW ' "$log"; then return 2; fi
}
profile_ncu() {
  local family="$1" m="$2"; shift 2
  local ncu=/usr/local/cuda/bin/ncu rc
  [[ -x "$ncu" ]] || ncu="$(command -v ncu || true)"
  [[ -n "$ncu" && -x "$ncu" ]] || { echo 'NCU unavailable; no installation attempted'; return 2; }
  "$ncu" --version >"$ROOT/receipts/ncu-version.log"
  guard
  local -a runner=("$ncu")
  if [[ "${MIMO26_NCU_SUDO:-0}" == 1 ]]; then
    runner=(sudo -n MIMO26_BUILDER_GPU=1 "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" TZ=Australia/Sydney LC_ALL=C "$ncu")
  fi
  if [[ ! -f "$ROOT/receipts/ncu-metrics.txt" ]]; then
    timeout 20s "${runner[@]}" --query-metrics --query-metrics-mode all >"$ROOT/receipts/ncu-metrics.txt" 2>&1 || return "$?"
  fi
  local stem="$ROOT/profile-$family-m$m"
  printf 'PROFILE COMMAND: '; printf '%q ' "${runner[@]}" --set full --profile-from-start off --replay-mode kernel --clock-control none --cache-control none --export "$stem" "$@"; printf '\n'
  if timeout 120s "${runner[@]}" --set full --profile-from-start off --replay-mode kernel --clock-control none --cache-control none --export "$stem" "$@" >"$ROOT/receipts/profile-$family-m$m.log" 2>&1; then rc=0; else rc=$?; fi
  guard
  printf 'PROFILE EXIT family=%s M=%s rc=%s\n' "$family" "$m" "$rc"
  validate_ncu_capture "$family" "$m" "$rc" "$ROOT/receipts/profile-$family-m$m.log" "$stem.ncu-rep" || return "$?"
  "$ncu" --import "$stem.ncu-rep" --page raw --csv >"$ROOT/receipts/profile-$family-m$m.csv"
  "$ncu" --import "$stem.ncu-rep" --page details >"$ROOT/receipts/profile-$family-m$m-details.txt"
  sha256sum "$stem.ncu-rep" >>"$ROOT/receipts/profile-binaries-sha256.txt"
}
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  set -euo pipefail
  [[ "${1:-}" == --selftest && -d "${2:-}" ]] || exit 2
  root="$2/ncu-control"; mkdir "$root"
  printf 'mock report, NOT a real NCU capture\n' >"$root/report"
  for pair in B1:1 B1:8 B2:4 B2:5; do
    family="${pair%:*}"; m="${pair#*:}"
    printf 'PROFILE PASS family=%s M=%s kernels=mock\n' "$family" "$m" >"$root/good"
    validate_ncu_capture "$family" "$m" 0 "$root/good" "$root/report"
    for rc in 1 3 4 124; do
      got=0; validate_ncu_capture "$family" "$m" "$rc" "$root/good" "$root/report" || got=$?
      [[ "$got" == "$rc" ]]
    done
    for suffix in '==ERROR== counter failure' 'No kernels were profiled' 'BENCH ROW forbidden timing' 'B1 BENCH ROW forbidden timing' "PROFILE PASS family=$family M=$m duplicate"; do
      printf 'PROFILE PASS family=%s M=%s kernels=mock\n%s\n' "$family" "$m" "$suffix" >"$root/bad"
      got=0; validate_ncu_capture "$family" "$m" 0 "$root/bad" "$root/report" || got=$?
      [[ "$got" == 2 ]]
    done
    for missing in report log; do
      log="$root/good"; report="$root/report"; [[ "$missing" != log ]] || log="$root/missing"; [[ "$missing" != report ]] || report="$root/missing"
      got=0; validate_ncu_capture "$family" "$m" 0 "$log" "$report" || got=$?; [[ "$got" == 2 ]]
    done
  done
  got=0; validate_ncu_capture B1 4 0 "$root/good" "$root/report" || got=$?; [[ "$got" == 2 ]]
  echo 'HOST PASS NCU capture validator: four pairs, process exits, missing/duplicate markers, counter errors, timing contamination and missing artifacts; no profiler/GPU launch'
fi
