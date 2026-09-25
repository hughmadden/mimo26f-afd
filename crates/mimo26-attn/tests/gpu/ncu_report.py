#!/usr/bin/env python3
"""Summarize an existing NCU text export; does not run a GPU or certify timing."""
import argparse
import csv
import json
import math

REQUIRED = (
    "gpu__time_duration.sum", "dram__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__issue_active.avg.pct_of_peak_sustained_elapsed", "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__warps_eligible.avg.per_cycle_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_elapsed", "lts__t_sector_hit_rate.pct",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
)
from pathlib import Path
import re


def summarize(raw, source):
    rows = list(csv.reader(raw.splitlines()))
    if len(rows) != 3 or len({len(r) for r in rows}) != 1:
        raise ValueError("expected one kernel, wide header/units/value NCU CSV")
    names, units, values = rows
    data = dict(zip(names, values))
    if len(data) != len(names):
        raise ValueError("duplicate metric names")
    for key in REQUIRED:
        if key not in data or not data[key] or not math.isfinite(float(data[key])):
            raise ValueError(f"missing/nonfinite required counter: {key}")
    if not re.search(r"decode_pipe<8(?:,\s*(?:true|1))?>", data["Kernel Name"]) or data["Grid Size"] != "(85, 4, 1)" or data["Block Size"] != "(256, 1, 1)":
        raise ValueError("not the requested FP32-Q pipe P85/w8 launch")
    patterns = (
        r"^gpu__time_duration.sum$", r"^dram__(bytes|throughput|cycles_active.avg.pct)",
        r"^sm(sp)?__.*(issue_active|warps_active|warps_eligible|pipe_tensor_cycles_active|pipe_alu.*cycles_active).*",
        r"^smsp__average_warps_issue_stalled", r"^lts__t_sector_hit_rate.pct$",
        r"^l1tex__data_bank_conflicts_pipe_lsu_mem_shared", r"^launch__(occupancy_limit|registers_per_thread|shared_mem)",
    )
    metrics = {n: {"value": v, "unit": u} for n, u, v in zip(names, units, values)
               if any(re.search(p, n) for p in patterns) and v not in ("", "nan")}
    lines = source.splitlines()
    spans = [(m.start(), m.end()) for m in re.finditer(r"-+", lines[0])]
    # NCU splits long column labels across several aligned header rows.
    first = next(i for i, line in enumerate(lines) if line.startswith("0x"))
    header = next(i for i, line in enumerate(lines) if line.startswith("Address"))
    columns = [re.sub(r"\s+", "", "".join(line[a:b] for line in lines[header:first])) for a, b in spans]
    instructions = []
    for line in lines[first:]:
        if line.startswith("0x"):
            cells = [line[a:b].strip() for a, b in spans]
            instructions.append(dict(zip(columns, cells)))
    if not instructions:
        raise ValueError("missing source counters")
    base = int(instructions[0][columns[0]], 16)
    top = {}
    for key in ("stall_wait", "stall_barrier", "stall_short_sb", "stall_long_sb", "L1WavefrontsSharedExcessive"):
        if key not in columns:
            raise ValueError(f"missing source column {key}; available {columns}")
        ordered = sorted(instructions, key=lambda r: float(r[key].replace(",", "")) if r[key] not in ("-", "") else 0, reverse=True)
        top[key] = [{"pc_offset": hex(int(r[columns[0]], 16)-base), "sass": r[columns[1]], "count": r[key]}
                    for r in ordered[:8]]
    return {"scope": "one full-set replay profile; not a performance gate", "kernel": data["Kernel Name"],
            "grid": data["Grid Size"], "block": data["Block Size"], "metrics": metrics, "top_source_counters": top}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("receipt", type=Path)
    args = ap.parse_args()
    print(json.dumps(summarize((args.receipt/"ncu-raw/RESULT.md").read_text(),
                               (args.receipt/"ncu-source/RESULT.md").read_text()), indent=2))
