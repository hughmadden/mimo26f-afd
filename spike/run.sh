#!/usr/bin/env bash
# spike/run.sh — I1 spike entry point, exec'd by `scripts/dev.sh spike`.
# Subcommands:
#   tests          run the P-101 negative suite against the CORRECT impl
#   tests --naive  run the same suite against the NAIVE impl (must FAIL)
#   audit          P-102 fail-loud name audit over the real shard index (the coordinator, read-only)
#   slices         P-102 per-layer stream + slice sanity vs oracle shapes (the coordinator, read-only)
#   cpu            topology-C CPU/tiny loop sanity (P-103)
#   needle         I3-I1b 4K needle on the dev host's 4090 (D7-local; CHUNK=512 SEG=512 defaults); --probe = dry footprint; --memtrace = 2-layer live-tensor trace
#   help
# All runs APPEND to logs under runs/<date>-i1-spike/ (needle: runs/20260923-i3/) — never replace.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${MIMO26_SPIKE_PY:-$HOME/.venvs/mimo26f-spike/bin/python}"
RUNS="$ROOT/runs/20260923-i1-spike"
mkdir -p "$RUNS"

sub="${1:-tests}"
shift || true

case "$sub" in
  tests)
    # Both runs APPEND to their log (never replace): per-test naive-fail lines
    # accumulate across reruns (lead, 2026-09-22).  No -x/--maxfail: every
    # negative must record its own naive failure line.
    if [[ "${1:-}" == "--naive" ]]; then
      LOG="$RUNS/pytest-naive.log"
      echo "== spike tests (NAIVE impl — these MUST fail) [$(TZ=Australia/Sydney date '+%Y-%m-%d %H:%M AEST')] -> $LOG" | tee -a "$LOG"
      MIMO26_SPIKE_NAIVE=1 "$PY" -m pytest "$ROOT/spike/tests" -q 2>&1 | tee -a "$LOG"
      t=${PIPESTATUS[0]}
      if [[ $t -eq 0 ]]; then
        # Fail-loud (bash-24 chain audit, lead 2026-09-23): UNEXPECTED-PASS means
        # the negatives lost detection power — as severe as a red test -> exit 1.
        echo "RESULT: UNEXPECTED-PASS (naive impl passed the negatives!)"
        exit 1
      fi
      echo "RESULT: FAIL-AS-EXPECTED (naive impl caught by negatives)"
    else
      LOG="$RUNS/pytest-correct.log"
      echo "== spike tests (CORRECT impl — these MUST pass) [$(TZ=Australia/Sydney date '+%Y-%m-%d %H:%M AEST')] -> $LOG" | tee -a "$LOG"
      "$PY" -m pytest "$ROOT/spike/tests" -q 2>&1 | tee -a "$LOG"
      t=${PIPESTATUS[0]}
      if [[ $t -ne 0 ]]; then
        # Fail-loud (bash-24 chain audit, lead 2026-09-23): `cmd | tee && PASS ||
        # FAIL` swallowed the pytest exit (pipeline status = tee's) and the cell
        # exited 0 on red.  RESULT line format kept; the cell now exits non-zero.
        echo "RESULT: FAIL"
        exit 1
      fi
      echo "RESULT: PASS"
    fi
    ;;
  audit)
    # P-102: fail-loud name audit over the real shard index (read-only, on
    # The coordinator where the weights mount lives).  No weights are moved.
    LOG="$RUNS/name-audit.log"
    echo "== spike name audit (read-only index on the coordinator) [$(TZ=Australia/Sydney date '+%Y-%m-%d %H:%M AEST')] -> $LOG" | tee -a "$LOG"
    ssh coordinator "python3 -" <<'PYEOF' 2>&1 | tee -a "$LOG"
import collections, json, re, sys
p = "/srv/models/XiaomiMiMo/MiMo-V2.6-Flash-RL/model.safetensors.index.json"
d = json.load(open(p)); wm = d["weight_map"]
EXPERT_RE = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight(_scale)?$")
def canonical(n):
    if n.startswith("model.mtp."): return "mtp." + n[len("model."):]
    if n.startswith("model."): return n[len("model."):]
    return n
kinds = collections.Counter(); bad = []
for n in wm:
    c = canonical(n)
    if EXPERT_RE.match(n): kinds["expert"] += 1
    elif n.startswith("model.mtp."):
        kinds["mtp"] += 1
        if not c.startswith("mtp."): bad.append(n + ": mtp root lost")
    elif "dflash" in n or "draft" in n: kinds["dflash"] += 1
    elif n.startswith(("audio_encoder.", "model.audio_encoder.", "visual.", "model.visual.",
                       "speech_embeddings.", "model.speech_embeddings.")):
        kinds["audio/visual/speech"] += 1  # expected non-backbone, dropped by loader.py:43
    elif re.match(r"^layers\.\d+\.", c) or c in ("embed_tokens.weight", "norm.weight", "lm_head.weight"):
        kinds["backbone"] += 1
    else:
        kinds["UNCLASSIFIED"] += 1; bad.append(n)
qkv = [n for n in wm if re.match(r"^model\.layers\.\d+\.self_attn\.qkv_proj\.weight$", n)]
print(f"tensors={len(wm)} shards={len(set(wm.values()))} qkv={len(qkv)} kinds={dict(kinds)}")
if bad or len(qkv) != 48:
    print("RESULT: FAIL audit:", bad[:8], "qkv_count", len(qkv)); sys.exit(1)
print("RESULT: PASS name audit clean")
PYEOF
    ;;
  slices)
    # P-102: stream ONLY layers 0 (GA) and 1 (SWA) fused-QKV byte ranges from
    # their shard (read-only pread), verify oracle shapes + ckpt_tp=4 split +
    # per-shard scale-grid trim indices.  Nothing written, nothing moved.
    LOG="$RUNS/real-slices.log"
    echo "== spike real slice load (read-only, per-layer stream on the coordinator) [$(TZ=Australia/Sydney date '+%Y-%m-%d %H:%M AEST')] -> $LOG" | tee -a "$LOG"
    ssh coordinator "python3 -" <<'PYEOF' 2>&1 | tee -a "$LOG"
import array, json, struct, sys
BASE = "/srv/models/XiaomiMiMo/MiMo-V2.6-Flash-RL"
idx = json.load(open(f"{BASE}/model.safetensors.index.json"))["weight_map"]
# oracle geometry (AGENTS.md §3 attn_dims): QK 192 / V 128; 4-way TP every layer
ORACLE = {
    0: dict(kind="GA", rows=13568, segs=(12288, 768, 512), per=(3072, 192, 128), grid_rows=108, per_grid=27),
    1: dict(kind="SWA", rows=14848, segs=(12288, 1536, 1024), per=(3072, 384, 256), grid_rows=116, per_grid=29),
}
HIDDEN = 4096
ok = True
for lay, exp in ORACLE.items():
    wn = f"model.layers.{lay}.self_attn.qkv_proj.weight"
    sn = f"model.layers.{lay}.self_attn.qkv_proj.weight_scale_inv"
    shard = idx[wn]
    with open(f"{BASE}/{shard}", "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(hlen))
        wi, si = meta[wn], meta[sn]
        lo, hi = wi["data_offsets"]
        f.seek(8 + hlen + lo)
        sample = f.read(min(hi - lo, HIDDEN * 4))  # 4 sample rows only: stream, don't load
        slo, shi = si["data_offsets"]
        f.seek(8 + hlen + slo)
        grid_bytes = f.read(shi - slo)  # whole grid is tiny (108x32 floats)
    grid = array.array("f"); grid.frombytes(grid_bytes)
    wshape, sshape = wi["shape"], si["shape"]
    per = exp["per"]
    pad = exp["per_grid"] * 128 - sum(per)
    max_idx = (sum(per) - 1) // 128
    checks = [
        ("rows", wshape[0] == exp["rows"]),
        ("hidden", wshape[1] == HIDDEN),
        ("grid", sshape == [exp["grid_rows"], HIDDEN // 128]),
        ("co_shard", idx.get(sn) == shard),
        ("segs_sum", sum(exp["segs"]) == exp["rows"]),
        ("ckpt_tp4", all(s % 4 == 0 for s in exp["segs"]) and per == tuple(s // 4 for s in exp["segs"])),
        ("per_grid", exp["per_grid"] == exp["grid_rows"] // 4),
        ("sample_stream", len(sample) == HIDDEN * 4),
        ("grid_count", len(grid) == exp["grid_rows"] * (HIDDEN // 128)),
        ("local_trim", max_idx <= exp["per_grid"] - 1),
        ("pad_rows_T2", pad in (0, 64)),
    ]
    bad = [k for k, v in checks if not v]
    print(f"layer {lay} ({exp['kind']}) shard={shard} w={wshape} s={sshape} per_shard={per} "
          f"scale_row<={max_idx}/{exp['per_grid']-1} pad={pad} -> {'PASS' if not bad else 'FAIL ' + str(bad)}")
    ok &= not bad
print("RESULT: PASS slice sanity vs oracle shapes" if ok else "RESULT: FAIL slice sanity")
sys.exit(0 if ok else 1)
PYEOF
    ;;
  real)
    # P-103 topology A: stage this package's CODE to the coordinator (weights never move)
    # and run the bare greedy loop there against the read-only weights.
    LOG="$RUNS/real-transcripts.log"
    echo "== spike real loop (topology A, the coordinator 5090) [$(TZ=Australia/Sydney date '+%Y-%m-%d %H:%M AEST')] -> $LOG" | tee -a "$LOG"
    ARGS="${*:-}"
    # UNIQUE per-run staging dir (lead retro: fixed paths make duplicate
    # launches clobber each other); CODE only — weights stay at /srv/models.
    STAGE=$(ssh coordinator 'mktemp -d /var/afd/mimo26f-stage-XXXXXX') || exit 1
    echo "stage: $STAGE" | tee -a "$LOG"
    tar -C "$ROOT" -cf - spike \
      | ssh coordinator "cd '$STAGE' && tar -xf - \
          && export HOME=/var/afd XDG_CACHE_HOME=/var/afd/mimo26f-cache HF_HOME=/var/afd/mimo26f-cache \
          && mkdir -p /var/afd/mimo26f-cache \
          && PYTHONPATH='$STAGE' /var/afd/mimo26f-venv/bin/python -m spike.real_loop $ARGS" \
        2>&1 | tee -a "$LOG"
    ;;
  cpu)
    LOG="$RUNS/cpu-tiny-loop.log"
    echo "== spike cpu tiny loop (topology C) -> $LOG"
    # -m (not script path): running spike/twin_loop.py directly puts spike/ on
    # sys.path and breaks `from spike import ...`.
    (cd "$ROOT" && PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m spike.twin_loop --tiny "$@") 2>&1 | tee -a "$LOG"
    ;;
  needle)
    # I3-I1b: 4K long-window needle on the dev host's RTX 4090 — D7-LOCAL ONLY (never
    # The coordinator's 5090, never the 4 Sparks).  `needle --probe` = dry device
    # footprint probe (no run, no torch).  Weights read-only at the dev host copy.
    LOG="$ROOT/runs/20260923-i3/needle.log"
    mkdir -p "$ROOT/runs/20260923-i3"
    export MIMO26_WEIGHTS_DIR="${MIMO26_WEIGHTS_DIR:-$HOME/models/XiaomiMiMo/MiMo-V2.6-Flash-RL}"
    # query-chunked prefill (I1b delta: the 4090 is tenant-adjusted -> ~18.3 GiB
    # free; chunk=512 peak ~10.6 GiB vs ~17.7 GiB unchunked).  0 = legacy path.
    export MIMO26_SPIKE_CHUNK="${MIMO26_SPIKE_CHUNK:-512}"
    # I1b fix (2): segmentwise prefill — KV-accumulating 512-token segments cap
    # all T-shaped transients (equality pin exists; 0 = monolithic legacy).
    export MIMO26_SPIKE_SEG="${MIMO26_SPIKE_SEG:-512}"
    # fire rows #1/#2: reserved-but-unallocated = fragmentation (symptom).
    # expandable_segments grows segments in place — KEPT for rows #3+.
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    echo "== spike needle (dev-host 4090, D7-local) [$(TZ=Australia/Sydney date '+%Y-%m-%d %H:%M AEST')] -> $LOG" | tee -a "$LOG"
    if [[ "${1:-}" == "--probe" ]]; then
      (cd "$ROOT" && PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m spike.footprint_probe) 2>&1 | tee -a "$LOG"
      exit "${PIPESTATUS[0]}"   # fail-loud: NON-FIT (1) must stop the caller's chain
    else
      "$PY" -c 'import torch' 2>/dev/null \
        || { echo "RESULT: FAIL — torch missing in $PY (set MIMO26_SPIKE_PY to a torch env)" | tee -a "$LOG"; exit 1; }
      (cd "$ROOT" && PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m spike.needle_prompt --run "$@") 2>&1 | tee -a "$LOG"
      exit "${PIPESTATUS[0]}"   # fail-loud: RESULT: FAIL (exit 1) must stop the caller's chain
    fi
    ;;
  needle-chat)
    # I4-X1a (ADVISOR-I4 §3.0): the same 4K needle through chat_template.jinja
    # with the EOS set honoured (T28).  dev-host 4090, D7-local only.  Expect
    # 'KESTREL-41' then EOS; any REWARD:/role-less assistant marker/immediate EOS = red flag.
    LOG="$ROOT/runs/20260923-i4/x1a-chat-needle.log"
    mkdir -p "$ROOT/runs/20260923-i4"
    export MIMO26_WEIGHTS_DIR="${MIMO26_WEIGHTS_DIR:-$HOME/models/XiaomiMiMo/MiMo-V2.6-Flash-RL}"
    export MIMO26_SPIKE_CHUNK="${MIMO26_SPIKE_CHUNK:-512}"
    export MIMO26_SPIKE_SEG="${MIMO26_SPIKE_SEG:-512}"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    echo "== spike needle-chat X1a (dev-host 4090, D7-local) [$(TZ=Australia/Sydney date '+%Y-%m-%d %H:%M AEST')] -> $LOG" | tee -a "$LOG"
    "$PY" -c 'import torch, jinja2' 2>/dev/null \
      || { echo "RESULT: FAIL — torch/jinja2 missing in $PY (set MIMO26_SPIKE_PY to the mimo26f-torch env)" | tee -a "$LOG"; exit 1; }
    (cd "$ROOT" && PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m spike.needle_prompt --run --chat "$@") 2>&1 | tee -a "$LOG"
    exit "${PIPESTATUS[0]}"   # fail-loud: RESULT: FAIL (exit 1) must stop the caller's chain
    ;;
  help|-h|--help)
    sed -n '2,10p' "$0"
    ;;
  *)
    echo "RESULT: FAIL"
    echo "spike/run.sh: unknown subcommand '$sub' (tests | tests --naive | audit | slices | cpu | help)"
    exit 2
    ;;
esac
