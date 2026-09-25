"""CPU-only R13 model pin from retained full-set NCU CSV; never a timing gate."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import ncu_report

STORED = 855638016
PAYLOAD = 256 * 3 * 512 * 4096 // 2
COUNTERS = {
    "warp_instructions": "smsp__inst_executed.sum",
    "thread_instructions": "smsp__thread_inst_executed.sum",
    "sm_frequency": "sm__cycles_elapsed.avg.per_second",
}
ncu_report.OPTIONAL.update(COUNTERS)


def count(metric):
    factors = {"inst": 1, "Kinst": 1000, "Minst": 1e6, "Ginst": 1e9, "Tinst": 1e12}
    value = ncu_report.number(metric["value"])
    if metric["unit"] not in factors or value is None or value <= 0:
        raise ValueError("invalid instruction value/unit: " + str(metric))
    return value * factors[metric["unit"]]


def analyze(text):
    raw = ncu_report.parse(text, "B2")
    ids = [int(k["id"]) for k in raw["kernels"]]
    if ids != sorted(set(ids)):
        raise ValueError("duplicate or reordered launch IDs")
    out = []
    for role, kernel in zip(("gate", "up", "activate", "down"), raw["kernels"]):
        metrics = kernel["metrics"]
        warp = count(metrics["warp_instructions"])
        tile = None
        if role != "activate":
            match = re.search(r"gemm<([^>]+)>", kernel["kernel"])
            if not match:
                raise ValueError("GEMM identity missing")
            args = [int(x.strip()) for x in re.sub(r"\([^)]*\)", "", match[1]).split(",")]
            if len(args) not in (2,3) or args[1] != 1 or (len(args)==3 and args[2]!=0):
                raise ValueError("model pin requires correct, single-CTA-stream B2")
            tile = args[0]
        elif "activate" not in kernel["kernel"]:
            raise ValueError("activation position mismatch")
        out.append({"role": role, "kernel": kernel["kernel"], "tile": tile,
                    "warp_instructions": warp, "warp_instruction_raw": metrics["warp_instructions"],
                    "thread_instructions": count(metrics["thread_instructions"]) if "thread_instructions" in metrics else None,
                    "sm_frequency_raw": metrics.get("sm_frequency"),
                    "counter_anomalies": kernel["counter_anomalies"]})
    tiles = {r["tile"] for r in out if r["tile"] is not None}
    if len(tiles) != 1:
        raise ValueError("mixed tile specializations")
    tile = tiles.pop()
    total = sum(r["warp_instructions"] for r in out if r["role"] != "activate")
    slots_stored = total * 32 / STORED
    slots_payload = total * 32 / PAYLOAD
    predicted = 33 + 4 * tile
    return {"rows": out, "stored_bytes": STORED, "payload_bytes": PAYLOAD,
            "gemm_warp_instructions": total,
            "warp32_slots_per_stored_byte": slots_stored,
            "warp32_slots_per_payload_byte": slots_payload,
            "all_four_kernel_slots_per_stored_byte": sum(r["warp_instructions"] for r in out)*32/STORED,
            "mimo_predicted_instructions_per_byte": predicted,
            "F_a_within_15pct_stored_basis": abs(slots_stored/predicted-1)<=.15,
            "F_a_within_15pct_payload_basis": abs(slots_payload/predicted-1)<=.15,
            "raw_completeness_gaps": raw["missing"],
            "note": "smsp instruction sum counts executed warp instructions; x32 is a fully-populated-warp normalization, not measured thread instructions, predicate-on useful instructions or an issued-instruction counter. Static model is compared on both byte bases. Replay durations are not bandwidth gates."}


def selftest():
    assert PAYLOAD == 805306368 and STORED/PAYLOAD == 1.0625
    assert count({"value":"1.5", "unit":"Ginst"}) == 1500000000
    for metric in ({"value":"nan","unit":"inst"}, {"value":"1","unit":"warp"}, {"value":"0","unit":"inst"}):
        try:
            count(metric)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid count accepted")
    text = '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
    for i in range(4):
        name = "activate" if i==2 else "gemm<4, 1>"
        text += f'"{i}","{name}","smsp__inst_executed.sum","inst","10"\n'
    result = analyze(text)
    assert result["gemm_warp_instructions"] == 30 and result["warp32_slots_per_stored_byte"] == 960/STORED
    try:
        analyze(text.replace("smsp__inst_executed.sum", "smsp__other.sum",1))
    except KeyError:
        pass
    else:
        raise AssertionError("missing warp counter accepted")
    for broken in (text.replace("gemm<4, 1>","gemm<4, 0>"),
                   text.replace("gemm<4, 1>","gemm<4, 1, 1>"),
                   text.replace("gemm<4, 1>","gemm<8, 1>",1),
                   text.replace('"0",','"1",',1), text.replace('"0",','"4",',1)):
        try:
            analyze(broken)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid launch/variant accepted")
    print("HOST PASS instruction model: warp x32, activation exclusion, two byte bases, unit/missing-counter/variant/launch-order negatives")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--output", type=Path)
    p.add_argument("--sources", nargs="*", type=Path, default=[])
    p.add_argument("inputs", nargs="*", type=Path)
    args = p.parse_args()
    if args.selftest:
        selftest()
    else:
        if not args.inputs or args.output is None:
            p.error("inputs and --output required")
        results = []
        for path in args.inputs:
            data = path.read_bytes()
            result = analyze(data.decode())
            result.update(input=str(path), sha256=hashlib.sha256(data).hexdigest())
            results.append(result)
        sources = []
        if args.sources:
            import ncu_pc_delta
            ncu_pc_delta.selftest()
            for path in args.sources:
                data = path.read_bytes()
                kernels = ncu_pc_delta.parse_source(data.decode())
                if not path.name.endswith("-source.csv"):
                    raise ValueError("source CSV must have its matching aggregate sibling")
                aggregate_path = path.with_name(path.name.removesuffix("-source.csv")+".csv")
                aggregate_data = aggregate_path.read_bytes()
                aggregate = ncu_report.parse(aggregate_data.decode(), "B2")
                normalize = lambda name: re.sub(r"\s+|\(int\)|\(bool\)", "", name)
                for k, a in zip(kernels, aggregate["kernels"]):
                    if normalize(k["function"]) != normalize(a["kernel"]):
                        raise ValueError("source/aggregate kernel mismatch")
                    ncu_pc_delta.match_totals(k["selected_sample_totals"], a["diagnostic_metrics"])
                sources.append({"input":str(path), "sha256":hashlib.sha256(data).hexdigest(),
                                "aggregate_input":str(aggregate_path), "aggregate_sha256":hashlib.sha256(aggregate_data).hexdigest(),
                                "verified_sample_totals":24,
                                "kernels":[{key:k[key] for key in ("role","function","static_shared_load_mnemonics","static_opcode_counts")} for k in kernels]})
        args.output.write_text(json.dumps({"instruction_counts":results,"static_source_counts":sources},indent=2)+"\n")
        for r in results:
            print(f"{r['input']}: GEMM warp-inst={r['gemm_warp_instructions']:.0f}, slots/stored-B={r['warp32_slots_per_stored_byte']:.6f}, slots/payload-B={r['warp32_slots_per_payload_byte']:.6f}, prediction={r['mimo_predicted_instructions_per_byte']}, F-a(stored)={r['F_a_within_15pct_stored_basis']}")
        print("Model artifact:",args.output)
