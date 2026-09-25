#!/usr/bin/env python3
"""Review-only quantization counterexamples, NOT a CUDA/tensor-core emulator.

FP64 exp/sums isolate representation loss. Two keys, one representative Q head,
QK192/V128; every V column is identical. Values embed in real GQA geometry by
replicating heads. The closed-form reference is independent of the softmax code.
No goldens, tolerances, dispatches, or kernels are changed by this study.
"""
import argparse
import json
import math
import struct
from datetime import datetime
from zoneinfo import ZoneInfo

TOLERANCE = 2e-5  # Existing benchmark absolute bound; not a new policy.


def f32(value):
    if not math.isfinite(value):
        raise ValueError("nonfinite probe input")
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bf16(value):
    bits = struct.unpack("<I", struct.pack("<f", f32(value)))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    rounded = struct.unpack("<f", struct.pack("<I", bits))[0]
    if not math.isfinite(rounded):
        raise ValueError("BF16 rounding overflow")
    return rounded


def split_bf16(value):
    value = f32(value)  # Residual must come from the actual FP32 operand.
    high = bf16(value)
    low = bf16(f32(value - high))
    return high + low


def triple_bf16(value):
    value = f32(value)
    high = bf16(value)
    residual = f32(value - high)
    low = bf16(residual)
    tail = bf16(f32(residual - low))
    return high + low + tail


def scaled_fp16_probability(value):
    if not 0 <= value <= 1:
        raise ValueError("unnormalized probability outside [0,1]")
    scale = 2**15
    return struct.unpack("<e", struct.pack("<e", f32(value) * scale))[0] / scale


def attention(q, keys, q_cast=lambda x: x, p_cast=lambda x: x):
    # All other Q/K dimensions are zero. V is +1 then -1 in all 128 columns.
    scores = [math.fsum(q_cast(x) * y for x, y in zip(q, key)) / math.sqrt(192)
              for key in keys]
    maximum = max(scores)
    probabilities = [math.exp(score - maximum) for score in scores]
    # As in R1, the denominator uses full probabilities, not rounded P.
    return (p_cast(probabilities[0]) - p_cast(probabilities[1])) / math.fsum(probabilities)


def study():
    cases = [
        ("fp32_q_cancellation", [1 + 2**-8, 1], [[448, -448], [-448, 448]],
         math.tanh(448 * 2**-8 / math.sqrt(192))),
        ("bf16_exact_q_probability", [1, 0], [[1, 0], [0, 0]],
         math.tanh(1 / (2 * math.sqrt(192)))),
        ("fp32_q_residual_tail", [1 + 2**-9 + 2**-17, 1 + 2**-9],
         [[448, -448], [-448, 448]], math.tanh(448 * 2**-17 / math.sqrt(192))),
    ]
    variants = [
        ("round_q_only", bf16, lambda x: x),
        ("round_p_bf16_only", lambda x: x, bf16),
        ("round_p_scaled_fp16_only", lambda x: x, scaled_fp16_probability),
        ("split_q_split_p_bf16", split_bf16, split_bf16),
        ("triple_q_split_p_bf16", triple_bf16, split_bf16),
    ]
    rows = []
    for name, q, keys, analytic in cases:
        reference = attention(q, keys)
        if abs(reference - analytic) > 1e-14:
            raise AssertionError("reference disagrees with independent tanh identity")
        results = []
        for variant, q_cast, p_cast in variants:
            got = attention(q, keys, q_cast, p_cast)
            error = abs(got - analytic)
            results.append(dict(variant=variant, output=got, max_abs=error,
                                within_existing_bound=error <= TOLERANCE))
        rows.append(dict(case=name, q_nonzero_prefix=q, k_nonzero_prefixes=keys,
                         analytic_reference=analytic, variants=results))
    return rows


def selftest():
    checks = 0

    def check(ok, name):
        nonlocal checks
        if not ok:
            raise AssertionError(name)
        checks += 1

    check(bf16(1) == 1, "exact BF16 input")
    check(bf16(1 + 2**-8) == 1, "positive even tie")
    check(bf16(-(1 + 2**-8)) == -1, "negative even tie")
    check(bf16(1 + 3 * 2**-8) == 1 + 2**-6, "odd tie rounds to even")
    check(bf16(448) == 448, "E4M3 maximum exactly BF16")
    check(split_bf16(1 + 2**-8) == 1 + 2**-8, "Q residual restores constructed input")
    check(scaled_fp16_probability(1) == 1, "probability maximum does not overflow")
    check(scaled_fp16_probability(0) == 0, "zero probability")
    check(scaled_fp16_probability(math.exp(-20)) > 0, "scaling preserves small probability")
    for bad in (float("nan"), float("inf"), -float("inf")):
        try:
            bf16(bad)
        except ValueError:
            checks += 1
        else:
            raise AssertionError("nonfinite input accepted")
    for bad in (-0.1, 1.1):
        try:
            scaled_fp16_probability(bad)
        except ValueError:
            checks += 1
        else:
            raise AssertionError("invalid probability accepted")
    rows = study()
    check(len(rows) == 3, "all three closed-form references checked")
    first = {r["variant"]: r for r in rows[0]["variants"]}
    second = {r["variant"]: r for r in rows[1]["variants"]}
    check(not first["round_q_only"]["within_existing_bound"], "Q-only rounding counterexample")
    check(second["round_q_only"]["max_abs"] < 1e-14, "native BF16 Q control")
    check(not second["round_p_bf16_only"]["within_existing_bound"], "BF16 P counterexample")
    check(not second["round_p_scaled_fp16_only"]["within_existing_bound"], "scaled FP16 P counterexample")
    check(all(row["variants"][3]["within_existing_bound"] for row in rows[:2]),
          "residual candidate passes the original two constructions only")
    check(not rows[2]["variants"][3]["within_existing_bound"],
          "two BF16 Q components lose the constructed FP32 tail")
    check(all(row["variants"][4]["within_existing_bound"] for row in rows),
          "Q tail correction passes these three constructions only")
    check(triple_bf16(1 + 2**-9 + 2**-17) == f32(1 + 2**-9 + 2**-17),
          "tail restores the constructed query operand")
    samples = [struct.unpack("<f", struct.pack("<I", 0x3E000000 + (i * 7919) % 0x2800000))[0]
               for i in range(4096)]
    check(all(triple_bf16(sign*x) == sign*x for x in samples for sign in (-1, 1)),
          "8192 sampled normal FP32 operands reconstruct numerically")
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true", help="host assertions only")
    args = parser.parse_args()
    count = selftest()
    print(f"RESULT: PASS precision-probe-selftest {count}/{count}")
    if args.selftest:
        raise SystemExit(0)
    print(json.dumps(dict(
        schema="mimo26-attn-precision-probe-v1",
        evidence="CPU_QUANTIZATION_STUDY_ONLY",
        promotion="NONE; neither GPU qualification nor a replacement for oracle goldens",
        generated_at_sydney=datetime.now(ZoneInfo("Australia/Sydney")).isoformat(timespec="seconds"),
        accumulation="FP64 ideal exp/sums; does not emulate FP32 MMA reduction order",
        logical_shape=dict(T=1, S=2, representative_q_heads=1, QK=192, V=128),
        kv="finite, exactly E4M3-unit representable; V columns all +1 then -1",
        unchanged_benchmark_absolute_bound=TOLERANCE,
        analytic_references_checked="3/3",
        cells=study()), indent=2, allow_nan=False))
