"""Summarize the M1 P1 full-set NCU profile; does not run a GPU or certify timing."""
import argparse
import csv
import json
import math
import re
from pathlib import Path

# Counters the M1 spec requires, grouped by purpose. Availability is reported.
REQUIRED = {
    "time": ["gpu__time_duration.sum", "sm__cycles_elapsed.avg", "sm__cycles_elapsed.max"],
    "residency": ["sm__warps_active.avg.pct_of_peak_sustained_active", "launch__occupancy_limit_warps",
                  "launch__waves_per_multiprocessor", "launch__registers_per_thread",
                  "launch__occupancy_limit_blocks"],
    "issue": ["sm__issue_active.avg.pct_of_peak_sustained_elapsed",
              "smsp__warps_eligible.avg.per_cycle_active",
              "smsp__issue_active.avg.pct_of_peak_sustained_active"],
    "pipes": ["sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
              "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed",
              "sm__inst_executed_pipe_xu.avg.pct_of_peak_sustained_active",
              "sm__pipe_lsu_cycles_active.avg.pct_of_peak_sustained_elapsed"],
    "stalls": ["smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
               "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
               "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
               "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
               "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
               "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio"],
    "shared": ["l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
               "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
               "l1tex__data_pipe_lsu_wavefronts_mem_shared.sum",
               "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum"],
    "traffic": ["dram__bytes.sum", "dram__throughput.avg.pct_of_peak_sustained_elapsed",
                "lts__t_bytes.sum", "lts__t_sector_hit_rate.pct",
                "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum"],
}


def summarize(raw_path):
    raw = raw_path.read_text()
    rows = list(csv.reader(raw.splitlines()))
    if len(rows) != 3 or len({len(r) for r in rows}) != 1:
        raise ValueError("expected one kernel, wide header/units/value NCU CSV")
    names, units, values = rows
    data = dict(zip(names, values))
    if len(data) != len(names):
        raise ValueError("duplicate metric names")
    kernel = data.get("Kernel Name", "")
    m = re.search(r"prefill_tc<([01])>", kernel)
    if not m:
        raise ValueError(f"not a prefill_tc launch: {kernel}")
    if data.get("Grid Size") != "(512, 4, 1)" or data.get("Block Size") != "(256, 1, 1)":
        raise ValueError(f"not the P1 T2048/S128K GA launch: grid={data.get('Grid Size')} block={data.get('Block Size')}")
    metrics = {}
    for group, keys in REQUIRED.items():
        metrics[group] = {k: (float(data[k]) if k in data and data[k] and data[k] != "nan"
                              and math.isfinite(float(data[k])) else None)
                          for k in keys}
    available = sum(1 for g in metrics.values() for v in g.values() if v is not None)
    missing = [k for g in metrics.values() for k, v in g.items() if v is None]
    return {"kernel": kernel, "grid": data["Grid Size"], "block": data["Block Size"],
            "mode": "bf16q" if m.group(1) == "0" else "f32q",
            "metrics": metrics, "available": available, "missing": missing}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("receipt", type=Path)
    args = ap.parse_args()
    master = (args.receipt / "RESULT.md").read_text()
    if "P1_COMPLETE modes=2" not in master or "P1_UNINSTRUMENTED_END" not in master:
        raise SystemExit("INCOMPLETE/INVALID: missing uninstrumented P1 checks/timing")
    result = {}
    for mode, tag in (("f32q", "true"), ("bf16q", "false")):
        raw = args.receipt / mode / "ncu-raw" / "RESULT.md"
        if not raw.exists():
            raise SystemExit(f"INCOMPLETE/INVALID: missing {mode} raw receipt")
        try:
            result[mode] = summarize(raw)
        except ValueError as exc:
            raise SystemExit(f"INCOMPLETE/INVALID: {mode} {exc}")
    print(json.dumps({"status": "VALID", "scope": "P1-T2048-S128K-full-set-NCU", "gate": "UNSET",
                      "note": "profile events are not gate evidence; uninstrumented timing retained separately",
                      "modes": result}, indent=2))


if __name__ == "__main__":
    main()
