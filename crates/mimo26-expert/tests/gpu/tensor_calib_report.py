"""GB10 tensor-core calibration receipt audit. Not a ceiling certification.

Adds the A1 R2 / A2 F5 clock pin: the same-run clock64 SM-cycle deltas back out
the true sustained clock, cross-checked against the nvidia-smi sample.
"""
import argparse
from collections import defaultdict
import json
import math
import re
from pathlib import Path

ARMS = ["bf16f32", "f16f32", "f16f16", "fp8x4"]
SMS = 48
# Single-wave saturated config for the clock64 reconciliation: grid 192 = 4
# CTAs/SM (well under occupancy), 128 threads, K=8, N=8192. All blocks resident
# in one wave, so the per-block clock64 delta is the true per-SM kernel duration.
MEASURED = (192, 128, 8192)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def fields(line):
    pairs = [w.split("=", 1) for w in line.split() if "=" in w]
    need(len({k for k, _ in pairs}) == len(pairs), "duplicate field")
    return dict(pairs)


def _clock_of(line):
    """First CSV column is clocks.sm, e.g. '208 MHz' or '2476 MHz'."""
    parts = [p.strip() for p in line.split(",")]
    if not parts:
        return None
    m = re.match(r"(\d+)", parts[0])
    return int(m.group(1)) if m else None


def clocks(path):
    vals = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        v = _clock_of(line)
        if v is not None:
            vals.append(v)
    return vals


def selftest():
    good = ("M0_SAMPLE arm=bf16f32 K=8 threads=128 grid=192 N=8192 executed_flops=1000 "
            "median_ms=1.0 min_ms=0.9 max_ms=1.1 tflops=1.0 max_sm_cycles=1000 "
            "flop_per_sm_cycle=512.0 clock_mhz=2000\n")
    for bad in [good.replace("arm=bf16f32", "arm=zzz"),
                good.replace("median_ms=1.0", "median_ms=nan"),
                good.replace("flop_per_sm_cycle=512.0", "flop_per_sm_cycle=0"),
                good.replace("max_sm_cycles=1000", "max_sm_cycles=0"),
                good.replace("tflops=1.0", "tflops=nan")]:
        try:
            audit_text(bad)
        except (ValueError, KeyError):
            pass
        else:
            raise AssertionError("unpowered calib negative")
    need(_clock_of("208 MHz, 3003 MHz, 5.30 W, 47") == 208, "clock idle parse")
    need(_clock_of("2476 MHz, 3003 MHz, 54.30 W, 59") == 2476, "clock boost parse")
    need(_clock_of("no-clocks-line") is None, "clock parse miss")
    print("TENSOR CALIB HOST PASS receipt negatives")


def audit_text(text):
    samples = {}  # (arm, K, grid, threads, N) -> field dict
    arms = []
    latencies = defaultdict(list)
    begin = None
    for line in text.splitlines():
        if line.startswith("M0_BEGIN "):
            begin = fields(line)
        elif line.startswith("M0_ARM "):
            arms.append(fields(line))
        elif line.startswith("M0_SAMPLE "):
            f = fields(line)
            key = (f["arm"], int(f["K"]), int(f["grid"]), int(f["threads"]), int(f["N"]))
            need(key not in samples, "duplicate sample")
            samples[key] = f
        elif line.startswith("M0_LATENCY "):
            latencies[fields(line)["arm"]].append(int(fields(line)["cycles"]))
        elif line.startswith("RESULT: FAIL"):
            raise ValueError("native known-answer/mutation failure")
    need(begin is not None and begin["sms"] == "48" and begin["arms"] == "4", "calibration header")
    need([a["arm"] for a in arms] == ARMS, "four arms required")
    for arm in ARMS:
        ks = sorted({k for (a, k, _, _, _) in samples if a == arm})
        need(ks == [1, 2, 4, 8], "K sweep 1,2,4,8")
        need(len(latencies[arm]) == 4, "four latency lengths")
        for (a, k, _g, _t, _n), f in samples.items():
            if a != arm:
                continue
            for v in ("median_ms", "min_ms", "max_ms", "tflops", "flop_per_sm_cycle"):
                need(math.isfinite(float(f[v])) and float(f[v]) > 0, f"nonpositive {arm} K{k} {v}")
            need(int(f["executed_flops"]) > 0, "FLOP count")
            need(int(f["max_sm_cycles"]) > 0, "cycle count")
    best = {arm: max((float(f["tflops"]), f) for (a, _k, _g, _t, _n), f in samples.items() if a == arm)
            for arm in ARMS}
    measured = {arm: samples[(arm, 8) + MEASURED] for arm in ARMS}
    return dict(begin=begin, samples=samples, latencies=latencies, best=best, measured=measured)


def receipt(root):
    status = fields((Path(root) / "status.txt").read_text())
    need(status == dict(native_exit="0", tee_exit="0"), "native/retrieval failure")
    result = audit_text((Path(root) / "calib.log").read_text())
    during = clocks(Path(root) / "clocks-during.txt")
    after = clocks(Path(root) / "clocks-after.txt")
    # Sustained under-load clock from nvidia-smi (coarse): peak during the run.
    actual = during or after
    need(actual, "no actual clock samples")
    actual_mhz = float(max(actual))
    need(actual_mhz > 0, "nonpositive actual clock")
    nominal_khz = int(result["begin"]["nominal_clock_khz"])
    rows = {}
    for arm in ARMS:
        tflops, f = result["best"][arm]
        m = result["measured"][arm]
        per_sm_actual = float(f["executed_flops"]) / (SMS * float(f["median_ms"]) * 1e-3 * actual_mhz * 1e6)
        per_sm_measured = float(m["flop_per_sm_cycle"])  # clock64 delta, ground truth
        # true_clock = flop/s / (SMS * flop/SM/measured-cycle), then MHz.
        true_clock_mhz = float(m["tflops"]) * 1e12 / (SMS * per_sm_measured) / 1e6
        rows[arm] = dict(best_K=int(f["K"]), executed_flops=int(f["executed_flops"]),
                         median_ms=float(f["median_ms"]), sustained_TFLOPS=tflops,
                         per_SM_FLOP_per_actual_cycle=per_sm_actual,
                         per_SM_FLOP_per_measured_cycle=per_sm_measured,
                         clock64_true_clock_mhz=true_clock_mhz)
    true_clocks = [v["clock64_true_clock_mhz"] for v in rows.values()]
    pin = sum(true_clocks) / len(true_clocks)
    return dict(source=None, device=result["begin"]["device"], nominal_clock_khz=nominal_khz,
                nvidia_smi_clock_mhz=actual_mhz, nvidia_smi_samples=len(actual),
                clock64_true_clock_mhz=pin,
                clock64_span_mhz=[min(true_clocks), max(true_clocks)],
                rows=rows,
                clock_pin_note=("same-run clock64 per-block SM-cycle deltas at the single-wave "
                                "config (192/128/K8); flop_per_sm_measured_cycle is the work count "
                                "per SM cycle, clock64_true_clock_mhz back-calculated from event-time "
                                "TFLOPS; cross-checked against the nvidia-smi sample"),
                scope="gb10-tensor-core-calibration; independent chains; known-answer+mutation; not a ceiling certification",
                no_promotion=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("receipt", type=Path, nargs="?")
    p.add_argument("output", type=Path, nargs="?")
    a = p.parse_args()
    selftest()
    if not a.selftest:
        if a.receipt is None or a.output is None:
            p.error("receipt/output required")
        result = receipt(a.receipt)
        with a.output.open("x") as f:
            json.dump(result, f, indent=2, allow_nan=False)
            f.write("\n")
        print("TENSOR CALIB PASS " + json.dumps({
            "sustained_TFLOPS": {k: round(v["sustained_TFLOPS"], 2) for k, v in result["rows"].items()},
            "clock64_true_clock_mhz": round(result["clock64_true_clock_mhz"], 1),
            "nvidia_smi_clock_mhz": result["nvidia_smi_clock_mhz"],
        }))
