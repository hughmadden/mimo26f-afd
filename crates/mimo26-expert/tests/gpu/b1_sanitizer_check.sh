#!/usr/bin/env bash
# CPU-tested, fail-closed sanitizer result contract. No CUDA in --selftest.
set -euo pipefail
validate_b1_sanitizer() {
  local tool="$1" log="$2" rc="$3" summary width m scale
  local scope="${4:-compute}" expected_m="${5:-8}" rank
  [[ "$scope" == compute || "$scope" == connected ]] || return 2
  case "$tool" in memcheck|initcheck|synccheck) summary='========= ERROR SUMMARY: 0 errors' ;;
    racecheck) summary='========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)' ;;
    *) return 2 ;; esac
  if [[ "$rc" != 0 ]]; then echo "SANITIZER FAIL tool=$tool process_exit=$rc"; return "$rc"; fi
  [[ -f "$log" ]] || return 4
  [[ "$(grep -Fxc "$summary" "$log" || true)" == 1 ]] || { echo "SANITIZER FAIL tool=$tool missing/dirty summary"; return 4; }
  grep -Fxq '========= COMPUTE-SANITIZER' "$log" || return 4
  if grep -Eiq '^ORACLE_MISMATCH |^CUDA_ERROR |^CUDA_FAILURE |^INPUT_FAILURE |^HARNESS_ERROR |^========= (error:|fatal:|warning:)' "$log"; then return 4; fi
  if [[ "$scope" == connected ]]; then
    [[ "$expected_m" == 1 || "$expected_m" == 8 ]] || return 2
    if [[ "$tool" == initcheck ]]; then
      [[ "$(grep -Fxc 'INITCHECK PAYLOAD initial_poison=no redzones=yes' "$log" || true)" == 1 ]] || return 4
    fi
    [[ "$(grep -c '^PASS B1 connected execution ' "$log" || true)" == 4 ]] || return 4
    for rank in 0 1 2 3; do
      [[ "$(grep -Fxc "PASS B1 connected execution rank=$rank M=$expected_m flag=0; comparison pending, not numerical qualification" "$log" || true)" == 1 ]] || return 4
    done
    echo "SANITIZER PASS tool=$tool connected-v1 M=$expected_m all four ranks; independent comparison still required"
    return 0
  fi
  for width in 64 128; do for m in 1 2 4 8 16 64; do
    [[ "$(grep -Fc "PASS B1 compute width=$width M=$m flag=0 bad=0 " "$log" || true)" == 1 ]] || return 4
    [[ "$(grep -Fc "PASS B1 rank width=$width M=$m flag=0 bad=0 " "$log" || true)" == 1 ]] || return 4
  done; done
  for scale in 0 1 253 254 255; do
    [[ "$(grep -Fxc "PASS B1 exceptional weight_scale=$scale bypass=0 bad=0/128; conforming decoded-weight FMA" "$log" || true)" == 1 ]] || return 4
  done
  echo "SANITIZER PASS tool=$tool all 12 slice cases, 12 rank cases and 5 scale cases; compute-only scope"
}
selftest_b1_sanitizer() {
  local root="$1" tool variant width m scale rc count=0
  for tool in memcheck initcheck synccheck racecheck; do
    for variant in good dirty no-summary no-header missing-case diagnostic warning duplicate-summary duplicate-case missing-rank duplicate-rank; do
      {
        [[ "$variant" == no-header ]] || echo '========= COMPUTE-SANITIZER'
        for width in 64 128; do for m in 1 2 4 8 16 64; do
          [[ "$variant" != missing-case || "$width:$m" != 128:64 ]] || continue
          echo "PASS B1 compute width=$width M=$m flag=0 bad=0 fixture"
        done; done
        for width in 64 128; do for m in 1 2 4 8 16 64; do
          [[ "$variant" != missing-rank || "$width:$m" != 128:64 ]] || continue
          echo "PASS B1 rank width=$width M=$m flag=0 bad=0 fixture"
        done; done
        [[ "$variant" != duplicate-rank ]] || echo 'PASS B1 rank width=64 M=1 flag=0 bad=0 fixture'
        for scale in 0 1 253 254 255; do echo "PASS B1 exceptional weight_scale=$scale bypass=0 bad=0/128; conforming decoded-weight FMA"; done
        if [[ "$variant" != no-summary ]]; then
          if [[ "$tool" == racecheck ]]; then
            if [[ "$variant" == dirty ]]; then echo '========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 1 warnings)'
            else echo '========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)'; fi
          elif [[ "$variant" == dirty ]]; then echo '========= ERROR SUMMARY: 1 errors'
          else echo '========= ERROR SUMMARY: 0 errors'; fi
        fi
        [[ "$variant" != diagnostic ]] || echo 'ORACLE_MISMATCH extra failing case'
        [[ "$variant" != warning ]] || echo '========= Warning: incomplete instrumentation'
        [[ "$variant" != duplicate-case ]] || echo 'PASS B1 compute width=64 M=1 flag=0 bad=0 fixture'
        if [[ "$variant" == duplicate-summary ]]; then
          if [[ "$tool" == racecheck ]]; then echo '========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)'
          else echo '========= ERROR SUMMARY: 0 errors'; fi
        fi
        :
      } >"$root/$tool-$variant.log"
      if validate_b1_sanitizer "$tool" "$root/$tool-$variant.log" 0; then rc=0; else rc=$?; fi
      if [[ "$variant" == good ]]; then [[ "$rc" == 0 ]] || return 1
      else [[ "$rc" == 4 ]] || return 1; ((count+=1)); fi
    done
    for rc in 3 4 124; do
      local got=0
      validate_b1_sanitizer "$tool" "$root/$tool-good.log" "$rc" || got=$?
      [[ "$got" == "$rc" ]] || return 1
      ((count+=1))
    done
  done
  local bad_tool=0
  validate_b1_sanitizer unknown "$root/memcheck-good.log" 0 || bad_tool=$?
  [[ "$bad_tool" == 2 ]] || return 1
  for tool in memcheck initcheck synccheck racecheck; do for m in 1 8; do
    for variant in good dirty missing-rank duplicate-rank wrong-m naive warning no-summary duplicate-summary no-init-mode; do
      [[ "$variant" != no-init-mode || "$tool" == initcheck ]] || continue
      local log="$root/connected-$tool-$m-$variant.log" rank
      {
        echo '========= COMPUTE-SANITIZER'
        [[ "$tool" != initcheck || "$variant" == no-init-mode ]] || echo 'INITCHECK PAYLOAD initial_poison=no redzones=yes'
        for rank in 0 1 2 3; do
          [[ "$variant:$rank" != missing-rank:3 ]] || continue
          local row_m="$m" flag=0
          [[ "$variant" != wrong-m ]] || row_m=4
          [[ "$variant" != naive ]] || flag=1
          echo "PASS B1 connected execution rank=$rank M=$row_m flag=$flag; comparison pending, not numerical qualification"
        done
        [[ "$variant" != duplicate-rank ]] || echo "PASS B1 connected execution rank=0 M=$m flag=0; comparison pending, not numerical qualification"
        [[ "$variant" != warning ]] || echo '========= Warning: incomplete instrumentation'
        local summary='========= ERROR SUMMARY: 0 errors'
        [[ "$tool" != racecheck ]] || summary='========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)'
        [[ "$variant" != dirty ]] || summary='========= ERROR SUMMARY: 1 errors'
        [[ "$variant" == no-summary ]] || echo "$summary"
        [[ "$variant" != duplicate-summary ]] || echo "$summary"
        :
      } >"$log"
      local got=0
      validate_b1_sanitizer "$tool" "$log" 0 connected "$m" || got=$?
      if [[ "$variant" == good ]]; then [[ "$got" == 0 ]] || return 1
      else [[ "$got" == 4 ]] || return 1; ((count+=1)); fi
    done
  done; done
  echo "HOST PASS sanitizer validator: 4 tools, $count failure controls, unknown-tool refusal; compute and connected scopes; no tool/GPU launched"
}
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  [[ "${1:-}" == --selftest && -d "${2:-}" ]] || exit 2
  selftest_b1_sanitizer "$2"
fi
