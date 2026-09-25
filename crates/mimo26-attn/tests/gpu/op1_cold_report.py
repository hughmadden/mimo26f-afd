"""Strict OP1 cold-prefill validator and tok/s summary. Exit 0 valid, 2 invalid."""
import argparse
import json
import math
from pathlib import Path
import re
import statistics

from op1_schedule import COLD_PREFILL, D7_BARS, load_manifest


def fields(line):
    pairs = re.findall(r"([A-Za-z_][A-Za-z_0-9]*)=([^ ]+)", line)
    if len({k for k, _ in pairs}) != len(pairs):
        raise ValueError("duplicate field")
    return dict(pairs)


def expect(d, **wanted):
    for key, value in wanted.items():
        if d.get(key) != str(value):
            raise ValueError(f"wrong {key}: {d.get(key)} != {value}")


def bounded(value, lo=0, hi=2e-5, positive=False):
    x = float(value)
    if not math.isfinite(x) or x < lo or x > hi or (positive and x == 0):
        raise ValueError("nonfinite/out-of-range number")
    return x


def pairs(t, s, window=0):
    return sum(min(s - t + i + 1, window) if window else (s - t + i + 1) for i in range(t))


def useful_flops(t, s, window=0):
    return 2.0 * 64 * (192 + 128) * pairs(t, s, window)


def p1_mma_flops(t, s, nkv, window, residual):
    qt = 64 // (64 // nkv)
    padded = 0
    for base in range(0, t, qt):
        hi = s - t + min(t, base + qt)
        lo = max(0, s - t + base - window + 1) if window else 0
        padded += ((hi + 15) // 16 - lo // 16) * 16
    return 2.0 * 64 * nkv * ((3 if residual else 1) * 192 + 2 * 128) * padded


def parse(text):
    if re.search(r"RESULT: (FAIL|REFUSE|INCOMPLETE)|\bPROXY\b", text):
        raise ValueError("failed/incomplete/proxy receipt")
    identity = re.findall(r"^IDENTITY (.+)$", text, re.M)
    if len(identity) != 1 or not re.search(r"^[0-9a-f]{64}  .*mimo26f-attn-bench-sm120$", text, re.M):
        raise ValueError("missing/duplicate device or binary identity")
    ident = fields(identity[0])
    expect(ident, arch="sm_120", sms=170, baked_arch="sm_120", baked_sms=170, label="TARGET")
    if "NVIDIA GeForce RTX 5090" not in identity[0] or not re.fullmatch(r"[0-9a-f]{12}", ident.get("source", "")):
        raise ValueError("wrong GPU/dirty source")
    if not re.search(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d AE[SD]T$", text, re.M) or "AOT: PASS architecture, SM-count, and baked kernel launch/readback" not in text:
        raise ValueError("missing Sydney/AOT evidence")
    manifest = load_manifest()
    points, samples, chunks, results = {}, {}, {}, {}
    began = ended = False
    for line in text.splitlines():
        if line.startswith("OP1_COLD_BEGIN "):
            if began or ended:
                raise ValueError("invalid begin")
            expect(fields(line), d7_sha=manifest["inputs"]["d7_sha256"], points=4, kinds=2, modes=2,
                   T=2048, batch=8, scope="attention-cold-prefill-multichunk-P1", gate="UNSET",
                   boundary="core-post-rope-prescaled-kv", Q_values="BF16-exact")
            began = True
        elif line.startswith("OP1_COLD_COMPLETE "):
            if not began or ended or len(samples) != 1484:
                raise ValueError("incomplete completion")
            expect(fields(line), points=4, kinds=2, modes=2, chunks=106, samples=1484, full_ref_chunks=20, gate="UNSET")
            ended = True
        elif line.startswith("OP1_COLD_"):
            if not began or ended:
                raise ValueError("row outside begin/end")
            tag = line.split()[0]; d = fields(line)
            S = int(d["S"]); kind = d["kind"]
            if S not in COLD_PREFILL or kind not in ("ga", "swa"):
                raise ValueError("unknown point")
            nkv, window = (4, 0) if kind == "ga" else (8, 128)
            if tag == "OP1_COLD_POINT":
                if (S, kind) in points:
                    raise ValueError("duplicate point")
                expect(d, nkv=nkv, window=window, chunks=S // 2048)
                points[(S, kind)] = d
            elif tag == "OP1_COLD_CHUNK":
                i = int(d["chunk"]); n_chunks = S // 2048
                if not (0 <= i < n_chunks) or (S, kind, i) in chunks:
                    raise ValueError("invalid chunk")
                full_ref = (i == 0) or (i == n_chunks // 2) or (i == n_chunks - 1)
                expect(d, S_c=2048 * (i + 1), full_ref=1 if full_ref else 0,
                       checked=16777216 if full_ref else 0, coordinates=3, query_exact="PASS", reference_finite="PASS")
                if abs(float(d["useful_flops"]) - useful_flops(2048, 2048 * (i + 1), window)) > 1.0:
                    raise ValueError("wrong useful flops")
                if abs(float(d["executed_f32"]) - p1_mma_flops(2048, 2048 * (i + 1), nkv, window, True)) > 1.0:
                    raise ValueError("wrong f32 executed flops")
                if abs(float(d["executed_bf16"]) - p1_mma_flops(2048, 2048 * (i + 1), nkv, window, False)) > 1.0:
                    raise ValueError("wrong bf16 executed flops")
                if full_ref: bounded(d["max_error"])
                else: expect(d, max_error="0")
                bounded(d["max_coordinate_diff"])
                chunks[(S, kind, i)] = d
            elif tag == "OP1_COLD_SAMPLE":
                i, mode, index = int(d["chunk"]), d["mode"], int(d["index"])
                if mode not in ("f32q", "bf16q") or index not in range(7):
                    raise ValueError("wrong sample")
                key = (S, kind, i, mode, index)
                if key in samples or (index and (S, kind, i, mode, index - 1) not in samples):
                    raise ValueError("duplicate/ordered sample")
                expect(d, finite="PASS"); bounded(d["coord"])
                samples[key] = bounded(d["ms"], positive=True, hi=100000)
            elif tag == "OP1_COLD_RESULT":
                mode = d["mode"]
                if mode not in ("f32q", "bf16q") or (S, kind, mode) in results:
                    raise ValueError("wrong result")
                medians = [statistics.median(samples[(S, kind, i, mode, j)] for j in range(7)) for i in range(S // 2048)]
                ttft = sum(medians)
                if not math.isclose(float(d["ttft_ms"]), ttft, rel_tol=1e-6, abs_tol=1e-6):
                    raise ValueError("wrong ttft")
                if not math.isclose(float(d["tok_s"]), S / (ttft * 1e-3), rel_tol=1e-6, abs_tol=1e-6):
                    raise ValueError("wrong tok_s")
                results[(S, kind, mode)] = {"ttft_ms": ttft, "tok_s": S / (ttft * 1e-3)}
            else:
                raise ValueError("unknown OP1 cold record")
    if not ended or len(points) != 8 or len(chunks) != 106 or len(results) != 16:
        raise ValueError("incomplete")
    # 48-layer attention slice = 9 GA + 39 SWA (exact geometry arithmetic).
    slices = {}
    for mode in ("f32q", "bf16q"):
        rows = {}
        for S in COLD_PREFILL:
            ttft = 9 * results[(S, "ga", mode)]["ttft_ms"] + 39 * results[(S, "swa", mode)]["ttft_ms"]
            rows[str(S)] = {"ttft_ms": ttft, "attention_tok_s": S / (ttft * 1e-3),
                            "d7_full_model_tok_s": D7_BARS["cold_prefill_tok_s"][str(S)]}
        slices[mode] = rows
    return {"status": "VALID", "scope": "attention-cold-prefill-multichunk-P1", "gate": "UNSET",
            "source": ident["source"], "d7_sha256": manifest["inputs"]["d7_sha256"],
            "chunks": 106, "samples": 1484,
            "per_kind": {f"{S}-{k}-{m}": v for (S, k, m), v in results.items()},
            "attention_slice": slices,
            "note": "attention-only tok/s vs D7 full-model bar; 48-layer slice is 9 GA + 39 SWA arithmetic"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    try:
        result = parse(args.receipt.read_text())
    except (ValueError, KeyError, TypeError) as exc:
        print(f"INCOMPLETE/INVALID: {exc}"); return 2
    print(json.dumps(result, indent=2, allow_nan=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
