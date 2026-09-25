"""Read NCU raw CSV without converting diagnostic durations into bandwidth gates."""
import argparse
import csv
import io
import json
import math
from pathlib import Path

REQUIRED = {
    "occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "eligible_warps": "smsp__warps_eligible.avg.per_cycle_active",
    "dram_throughput_pct": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "l2_hit_pct": "lts__t_sector_hit_rate.pct",
}
OPTIONAL = {
    "issue_slots_busy_pct": "sm__issue_active.avg.pct_of_peak_sustained_elapsed",
    "scheduler_issue_slots_busy_pct": "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "tensor_pipe_active_pct": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "local_spilling_requests": "derived__local_spilling_requests",
    "local_spilling_requests_pct": "derived__local_spilling_requests_pct",
    "warp_cycles_per_issue": "smsp__average_warp_latency_per_inst_issued.ratio",
    "l2_throughput_pct": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "compute_throughput_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "duration": "gpu__time_duration.sum", "dram_bytes_per_second": "dram__bytes.sum.per_second",
    "registers_per_thread": "launch__registers_per_thread",
    "static_shared": "launch__shared_mem_per_block_static", "dynamic_shared": "launch__shared_mem_per_block_dynamic",
    "register_block_limit": "launch__occupancy_limit_registers",
    "shared_block_limit": "launch__occupancy_limit_shared_mem",
    "warp_block_limit": "launch__occupancy_limit_warps",
}


def number(value):
    try: result = float(value.replace(",", ""))
    except ValueError: return None
    return result if math.isfinite(result) else None


def parse(text, family):
    if family not in ("B1", "B2"): raise ValueError("family")
    lines = list(csv.reader(io.StringIO(text)))
    headers = [i for i, row in enumerate(lines) if "Kernel Name" in row and "ID" in row]
    if len(headers) != 1: raise ValueError("one raw CSV header required")
    start = headers[0]; header = lines[start]; groups = {}
    if len(set(header)) != len(header): raise ValueError("duplicate CSV column")
    if "Metric Name" not in header:
        # NCU 2025.3 --page raw exports one kernel per row and a units row.
        if len(lines) <= start+1 or len(lines[start+1]) != len(header): raise ValueError("missing wide units row")
        units = lines[start+1]
        if units[header.index("ID")] or units[header.index("Kernel Name")]: raise ValueError("invalid wide units row")
        expanded = [["ID", "Kernel Name", "Metric Name", "Metric Unit", "Metric Value"]]
        for fields in lines[start+2:]:
            if not fields: continue
            if len(fields) != len(header): raise ValueError("truncated wide row")
            for i, metric in enumerate(header):
                if "__" in metric:
                    expanded.append([fields[header.index("ID")], fields[header.index("Kernel Name")], metric, units[i], fields[i]])
        lines = expanded; start = 0; header = lines[0]
    for fields in lines[start+1:]:
        if not fields or (len(fields) == 1 and fields[0].startswith("==PROF==")): continue
        if len(fields) != len(header): raise ValueError("truncated CSV row")
        row = dict(zip(header, fields)); key = (row["ID"], row["Kernel Name"])
        if not key[0] or not key[1] or not row["Metric Name"]: raise ValueError("empty metric identity")
        metrics = groups.setdefault(key, {})
        value = dict(value=row["Metric Value"], unit=row["Metric Unit"])
        old = metrics.setdefault(row["Metric Name"], value)
        if old != value: raise ValueError("conflicting duplicate metric")
    expected = 3 if family == "B1" else 4
    if len(groups) != expected: raise ValueError(f"expected {expected} captured kernels, got {len(groups)}")
    kernels = []; missing = []
    for (kid, name), metrics in groups.items():
        selected = {}
        for label, metric in (REQUIRED | OPTIONAL).items():
            if metric in metrics: selected[label] = metrics[metric]
            if label in REQUIRED and (metric not in metrics or number(metrics[metric]["value"]) is None): missing.append(f"{kid}:{label}")
        extra = ["issue_slots_busy_pct", "registers_per_thread", "local_spilling_requests"]
        if family == "B1": extra.append("tensor_pipe_active_pct")
        for label in extra:
            if label not in selected or number(selected[label]["value"]) is None: missing.append(f"{kid}:{label}")
        stalls = [dict(metric=k, **v) for k, v in metrics.items()
                  if "warp_issue_stalled_" in k and k.endswith("_per_warp_active.pct") and number(v["value"]) is not None]
        if not stalls:
            stalls = [dict(metric=k, **v) for k, v in metrics.items()
                      if k.startswith("smsp__average_warps_issue_stalled_") and k.endswith("_per_issue_active.ratio") and number(v["value"]) is not None]
        stalls.sort(key=lambda v: number(v["value"]), reverse=True)
        if not stalls: missing.append(f"{kid}:stall_reasons")
        diagnostic = {k: v for k, v in metrics.items() if "stalled_" in k or "tensor" in k or "spilling" in k or ("issue_active" in k and "stalled" not in k) or k.startswith(("dram__", "sys__", "c2clink__"))}
        anomalies = [label for label, value in selected.items() if label.endswith("_pct") and number(value["value"]) is not None and not 0 <= number(value["value"]) <= 100]
        kernels.append(dict(id=kid, kernel=name, metrics=selected, stalls=stalls, diagnostic_metrics=diagnostic,
                            counter_anomalies=anomalies, note="raw units preserved; out-of-range replay counters are flagged, never clamped"))
    return dict(family=family, complete=not missing, missing=missing, kernels=kernels,
                scope="NCU full-set diagnostic counters; kernel replay, cache-control none, clock-control none; NOT a replacement bandwidth receipt")


def selftest():
    field_names = [*REQUIRED.values(), *(OPTIONAL[k] for k in ("issue_slots_busy_pct", "registers_per_thread", "local_spilling_requests", "tensor_pipe_active_pct"))]
    def fixture(count=3, omit=None, conflict=False):
        output = io.StringIO(); writer = csv.writer(output)
        writer.writerow(["ID", "Kernel Name", "Metric Name", "Metric Unit", "Metric Value"])
        for kid in range(count):
            for name in [*field_names, "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct"]:
                if name != omit: writer.writerow([kid, f"kernel<{kid}>", name, "%", "12.5"])
            if conflict: writer.writerow([kid, f"kernel<{kid}>", REQUIRED["occupancy_pct"], "%", "9"])
        return output.getvalue()
    assert parse(fixture(), "B1")["complete"] and parse(fixture(4), "B2")["complete"]
    assert not parse(fixture(omit=REQUIRED["eligible_warps"]), "B1")["complete"]
    for label in ("issue_slots_busy_pct", "registers_per_thread", "local_spilling_requests", "tensor_pipe_active_pct"):
        result = parse(fixture(omit=OPTIONAL[label]), "B1")
        assert not result["complete"] and f"0:{label}" in result["missing"]
    for text in ("", fixture(2), fixture(conflict=True), fixture()+"truncated,row\n", fixture()+fixture()):
        try: parse(text, "B1")
        except ValueError: pass
        else: raise AssertionError("malformed profile accepted")
    wide = io.StringIO(); writer = csv.writer(wide)
    names = [*field_names, "smsp__warp_issue_stalled_barrier_per_warp_active.pct"]
    writer.writerow(["ID", "Kernel Name", *names]); writer.writerow(["", "", *(["%"]*len(names))])
    for kid in range(3): writer.writerow([kid, f"kernel<{kid}>", *(["12.5"]*len(names))])
    assert parse(wide.getvalue(), "B1")["complete"]
    ratio = wide.getvalue().replace("smsp__warp_issue_stalled_barrier_per_warp_active.pct", "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio")
    assert parse(ratio, "B1")["complete"]
    assert parse(ratio.replace("12.5", "137"), "B1")["kernels"][0]["counter_anomalies"]
    for text in (wide.getvalue().replace('""', 'bad', 1), "\n".join(wide.getvalue().splitlines()[::2])):
        # Missing units and truncated wide data must not acquire a green result.
        if text == wide.getvalue(): continue
        try: parse(text, "B1")
        except ValueError: pass
        else: raise AssertionError("invalid wide CSV accepted")
    assert number("1,234.5") == 1234.5 and number("n/a") is None and number("nan") is None
    print("HOST PASS NCU raw parser: kernel counts, missing metrics, duplicate/header/truncation refusals, explicit units; CPU only")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--brief", action="store_true")
    parser.add_argument("--family", choices=("B1", "B2")); parser.add_argument("csv", type=Path, nargs="?")
    args = parser.parse_args()
    if args.selftest: selftest()
    else:
        if args.csv is None or args.family is None: parser.error("family and CSV required")
        try:
            result = parse(args.csv.read_text(), args.family)
            if args.brief:
                for kernel in result["kernels"]:
                    kernel.pop("diagnostic_metrics"); kernel["stalls"] = kernel["stalls"][:3]
                labels = ("occupancy_pct", "eligible_warps", "issue_slots_busy_pct", "l2_hit_pct", "registers_per_thread", "local_spilling_requests", "tensor_pipe_active_pct")
                result["compact_rows"] = ["id=" + k["id"] + " " + "; ".join(label + "=" + k["metrics"].get(label, {}).get("value", "UNAVAILABLE") for label in labels) + "; stalls=" + ", ".join(s["metric"].removeprefix("smsp__average_warps_issue_stalled_").removesuffix("_per_issue_active.ratio") + ":" + s["value"] for s in k["stalls"]) for k in result["kernels"]]
            print(json.dumps(result, indent=2))
            raise SystemExit(0 if result["complete"] else 2)
        except (ValueError, KeyError, OSError) as error: parser.exit(2, f"PROFILE HARNESS ERROR: {error}\n")
