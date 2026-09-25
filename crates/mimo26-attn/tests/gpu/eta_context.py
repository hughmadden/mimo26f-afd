#!/usr/bin/env python3
"""N5 context accounting. Prefill rows are models, NEVER full-context measurements."""
import argparse
import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo
from bench_report import parse_receipt, require, InvalidReceipt, visible_pairs
from eta_report import validate


def prefill_work(t, s, residual):
    require(0 < t <= s and t % 4 == 0, "GA M64 packs four query positions")
    # Four KV-head CTAs per group of four queries, each with M64/N64 tiles.
    padded_keys = sum(((s-t+q+4+63)//64)*64 for q in range(0, t, 4))
    executed = 4*2*64*((3 if residual else 1)*192+2*128)*padded_keys
    useful = 40960*visible_pairs(t, s, 0)
    return useful, executed


def report(decode_text, eta_text):
    decode = parse_receipt(decode_text, ["decode-1m"])
    eta = validate(eta_text)
    require(decode["label"] == "TARGET", "decode must be measured on target")
    d = next(r for r in decode["cells"] if r["cell"] == "decode-1m")
    require(d["kernel"] == "tc-q3-p2-d01", "expected D0.1 context anchor")
    models = []
    # State both interpretations of '128K prefill'; do not hide the query count.
    for t in (2048, 131072):
        for mode in ("f32q", "bf16q"):
            micro = next(r for r in eta["rows"] if (r["M"], r["N"], r["mode"]) == (64, 64, mode))
            useful, executed = prefill_work(t, 131072, mode == "f32q")
            rate = micro["executed_TFLOPS"]
            milliseconds = executed/(rate*1e9)
            useful_rate = useful/(milliseconds*1e9)
            models.append(dict(evidence="MODEL_NOT_MEASURED", family="GA", Hq=64, Hkv=4,
                               QK=192, V=128, M=64, N=64, T=t, S=131072, mode=mode,
                               useful_flops=useful, padded_executed_mma_flops=executed,
                               padded_work_factor=executed/useful,
                               source_micro_executed_TFLOPS=rate,
                               source_micro_eta=micro["eta"],
                               mechanical_rate_transplant_ms=milliseconds,
                               planning_3_to_5x_ms=[3*milliseconds, 5*milliseconds],
                               implied_useful_TFLOPS=useful_rate,
                               gate=("125.7 executed TFLOPS" if mode == "f32q" else "100 useful TFLOPS"),
                               verdict="PASS" if (rate >= 125.7 if mode == "f32q" else useful_rate >= 100) else "MISS"))
    return dict(schema="mimo26-attn-n5-context-v1",
                recorded_at_sydney=datetime.datetime.now(ZoneInfo("Australia/Sydney")).isoformat(),
                decode_source=decode["source"], micro_source=eta["source"],
                measured_decode_1m=dict(evidence="MEASURED", median_ms=d["median_ms"],
                                       executed_TFLOPS=d["executed_mma_TFLOPS"],
                                       eta=d["executed_mma_TFLOPS"]/209.5,
                                       effective_KV_GBs=d["effective_KV_GBs"], verdict=d["verdict"]),
                R7_ideal_decode_model=dict(compute_ms=0.889, bandwidth_ms=0.895,
                                          evidence="PAPER_MODEL_NOT_MEASURED", status="CO_BOUND"),
                prefill_128k_models=models,
                limitations=["No 128K-context prefill GPU execution is claimed.",
                             "M64/N64 is an accounting candidate, not a frozen P1 geometry.",
                             "Measured micro inputs are resident/repeated; streaming, online rescale, masks and sink boundaries were not qualified.",
                             "Rates are mechanically transplanted to useful causal/padded MMA work; the 3-5x range is a planning allowance, not a confidence interval.",
                             "BF16-Q is post-RoPE RNE; K/V are unit E4M3, cached V prescaled. No f32q-oracle or X1c promotion."])


def selftest():
    u, e = prefill_work(64, 64, True)
    assert u == 85196800
    assert e == 436207616
    assert prefill_work(64, 64, False)[1] == 234881024
    for t in (2048, 131072):
        uf, ef = prefill_work(t, 131072, True)
        ub, eb = prefill_work(t, 131072, False)
        assert uf == ub and ef*7 == eb*13
        assert ef >= 2.6*uf and eb >= 1.4*ub
    for t, s in ((0, 64), (3, 64), (128, 64)):
        try:
            prefill_work(t, s, True)
        except InvalidReceipt:
            pass
        else:
            raise AssertionError("invalid geometry accepted")
    print("RESULT: PASS eta-context-selftest 10/10 (accounting only, no GPU)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--decode", type=Path)
    parser.add_argument("--eta", type=Path)
    args = parser.parse_args()
    if args.selftest:
        selftest()
    else:
        require(args.decode and args.eta, "both receipts required")
        print(json.dumps(report(args.decode.read_text(), args.eta.read_text()), indent=2))
