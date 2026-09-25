"""R13 source-correlated PC samples: deduplicate inline views, never infer time savings."""
import argparse
import collections
import csv
import io
import json
from pathlib import Path
import re
from ncu_report import parse as parse_raw

FIELDS = ("stall_barrier", "stall_long_sb", "stall_math", "stall_not_selected", "stall_short_sb", "stall_wait")
RAW_NAMES = dict(zip(FIELDS, ("barrier", "long_scoreboard", "math_pipe_throttle", "not_selected", "short_scoreboard", "wait")))


def match_totals(pc, raw):
    for label, suffix in RAW_NAMES.items():
        metric = raw["smsp__pcsamp_warps_issue_stalled_" + suffix]
        if metric["unit"] != "warp" or int(metric["value"].replace(",", "")) != pc[label]:
            raise ValueError("source/aggregate sample-count mismatch: " + label)


def parse_source(text):
    kernels = []
    current = None
    path = function = line = source = None
    columns = None
    reader = csv.reader(io.StringIO(text))
    for row in reader:
        if not row:
            continue
        if row[0] == "File Path":
            if len(row) != 2:
                raise ValueError("bad source file marker")
            path = row[1]
            columns = None
            line = source = None
            continue
        if row[0] == "Function Name":
            if not path or len(row) != 2:
                raise ValueError("bad function marker")
            function = row[1]
            # NCU emits a complete ordered file/function bundle per launch.
            # A repeated pair starts the next launch even when its symbol matches.
            key = (path, function)
            if current is None or function != current["function"] or key in current["files"]:
                current = {"function": function, "files": set(), "pcs": {}, "source_text_repairs": 0}
                kernels.append(current)
            current["files"].add(key)
            continue
        if row[:4] == ["Line No", "Source", "Address", "Source"]:
            if not current or not all(f in row for f in FIELDS):
                raise ValueError("missing PC metric header")
            columns = {f: row.index(f) for f in FIELDS}
            continue
        # NCU 2025.3 does not escape C++ quotes in CUDA aggregate text.
        # Rejoin only that non-numeric field, with an intact typed suffix.
        # Never repair instruction rows or infer missing sample columns.
        if columns and len(row) > 10 and row[0].isdigit() and row[-8:-6] == ["-", "-"] and all(re.fullmatch(r"[0-9,]+", x) for x in row[-6:]):
            row = [row[0], ",".join(row[1:-8]), *row[-8:]]
            current["source_text_repairs"] += 1
        if columns is None or len(row) != 4 + len(FIELDS):
            raise ValueError(f"truncated or unrecognized PC source row at line {reader.line_num}: {row!r}")
        if row[0]:
            if not row[0].isdigit():
                raise ValueError("non-numeric source line")
            line, source = int(row[0]), row[1]
        if not row[2].startswith("0x"):
            # CUDA aggregate lines and ellipses are not additional samples.
            if row[2] not in ("-", "..."):
                raise ValueError("bad PC address")
            continue
        int(row[2], 16)
        if line is None:
            raise ValueError("PC lacks source context")
        counts = {f: int(row[i].replace(",", "")) for f, i in columns.items()}
        if any(v < 0 for v in counts.values()):
            raise ValueError("negative PC count")
        address, instruction = row[2], row[3].strip()
        context = (Path(path).name, line, source)
        previous = current["pcs"].get(address)
        if previous:
            if previous["instruction"] != instruction or previous["samples"] != counts:
                raise ValueError("conflicting duplicate PC view")
            previous["contexts"].add(context)
        else:
            current["pcs"][address] = {"address": address, "instruction": instruction,
                                       "samples": counts, "contexts": {context}}
    if len(kernels) != 4 or not all(k["pcs"] for k in kernels):
        raise ValueError("expected four nonempty source-correlated kernels")
    if ["activate" in k["function"] for k in kernels] != [False, False, True, False]:
        raise ValueError("unexpected gate/up/activate/down source order")
    result = []
    for role, k in zip(("gate", "up", "activation", "down"), kernels):
        totals = collections.Counter()
        lines = {}
        opcodes = collections.Counter()
        shared_loads = collections.Counter()
        pcs = []
        for pc in k["pcs"].values():
            totals.update(pc["samples"])
            contexts = sorted(pc["contexts"], key=lambda c: (c[0] != "expert_gemm.cu", c[0], c[1]))
            pc["contexts"] = contexts
            primary = contexts[0]
            key = primary[:2]
            if key not in lines:
                lines[key] = {"file": primary[0], "line": primary[1], "source": primary[2], "samples": collections.Counter()}
            lines[key]["samples"].update(pc["samples"])
            mnemonic = re.sub(r"^@!?[A-Z][A-Z0-9]*\s+", "", pc["instruction"]).split()[0]
            opcode = mnemonic.split(".")[0]
            opcodes[opcode] += 1
            if opcode == "LDS": shared_loads[mnemonic] += 1
            pcs.append(pc)
        rank = lambda item: sum(item["samples"].values())
        result.append({"role": role, "function": k["function"], "unique_pcs": len(pcs), "source_text_repairs": k["source_text_repairs"],
                       "selected_sample_totals": dict(totals), "static_opcode_counts": dict(opcodes),
                       "static_shared_load_mnemonics": dict(shared_loads),
                       "top_pcs": sorted(pcs, key=rank, reverse=True)[:8],
                       "source_lines": sorted(lines.values(), key=rank, reverse=True)})
    return result


def selftest():
    out = io.StringIO(); w = csv.writer(out)
    for name in ("gemm2", "gemm2", "activate", "gemm2"):
        for file in ("expert_gemm.cu", "inline.h"):
            w.writerow(["File Path", file]); w.writerow(["Function Name", name])
            w.writerow(["Line No", "Source", "Address", "Source", *FIELDS])
            w.writerow(["123", "fmaf(a,b,c)", "-", "-", *([3] * 6)])
            w.writerow(["", "", "0x100", "FFMA R1, R2, R3, R1", *([3] * 6)])
    text = out.getvalue(); result = parse_source(text)
    assert len(result) == 4 and all(k["unique_pcs"] == 1 for k in result)
    assert result[0]["selected_sample_totals"]["stall_wait"] == 3  # not CUDA + two inline views
    assert result[0]["static_opcode_counts"] == {"FFMA": 1}
    packed = parse_source(text.replace("FFMA R1, R2, R3, R1", "@!PT LDS.64 R2, [R4]"))
    assert packed[0]["static_shared_load_mnemonics"] == {"LDS.64": 1}
    quoted = text.replace('123,"fmaf(a,b,c)",-,-,3,3,3,3,3,3', '"123",""r"(shared), "l"(src)","-","-","3","3","3","3","3","3"', 1)
    repaired = parse_source(quoted)
    assert repaired[0]["source_text_repairs"] == 1 and repaired[0]["selected_sample_totals"]["stall_wait"] == 3
    raw = {"smsp__pcsamp_warps_issue_stalled_" + s: {"unit": "warp", "value": "3"} for s in RAW_NAMES.values()}
    match_totals(result[0]["selected_sample_totals"], raw)
    raw["smsp__pcsamp_warps_issue_stalled_wait"]["value"] = "4"
    try:
        match_totals(result[0]["selected_sample_totals"], raw)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched source/raw totals accepted")
    bads = [text.replace("stall_wait", "missing", 1), text.rsplit("0x100", 1)[0],
            text.replace("FFMA R1, R2, R3, R1", "FADD R1, R2, R3", 1)]
    for bad in bads:
        try:
            parse_source(bad)
        except ValueError:
            continue
        raise AssertionError("bad PC export accepted")
    print("HOST PASS PC parser: four launches, repeated symbol, inline/CUDA dedup, conflicting/truncated/header refusals")


def main():
    p = argparse.ArgumentParser(); p.add_argument("--selftest", action="store_true"); p.add_argument("directory", nargs="?", type=Path)
    args = p.parse_args()
    if args.selftest:
        selftest(); return
    if not args.directory:
        p.error("directory required")
    result = {}
    for m in (2, 3):
        stem = args.directory / f"profile-B2-m{m}"
        raw = parse_raw(stem.with_suffix(".csv").read_text(), "B2")
        pcs = parse_source(Path(str(stem) + "-source.csv").read_text())
        for pc, kernel in zip(pcs, raw["kernels"]):
            match_totals(pc["selected_sample_totals"], kernel["diagnostic_metrics"])
            pc["source_aggregate_totals_match"] = True
            pc["metrics"] = kernel["metrics"]
            pc["top_stalls"] = kernel["stalls"][:3]
            pc["counter_flags"] = kernel["counter_anomalies"]
        result[str(m)] = {"missing": raw["missing"], "kernels": pcs}
    (args.directory / "pc-delta.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# B2 M2/M3 source-correlated diagnostic export", "", "Selected six PC stall categories only; samples are not cycles or wall-time fractions.", "Inline/source duplicates are deduplicated by address within each launch; conflicting duplicates fail.", "Addresses are process-specific. Compare source sites/opcodes, not equal absolute addresses.", "Static opcode counts are code size, not executed instruction counts. No bandwidth gate here.", "Every selected PC-category total matches its aggregate raw counter exactly (48 comparisons).", "NCU's unescaped C++ quote/comma fields are repaired only in aggregate source text, never in addresses/instructions/counts; raw exports remain intact.", ""]
    for m, data in result.items():
        lines.append(f"## M{m}")
        for k in data["kernels"]:
            lines.extend(["", f"### {k['role']}", "", k["function"], ""])
            for label, metric in k["metrics"].items():
                lines.append(f"- {label}: {metric['value']} {metric['unit']}")
            lines.append(f"- Static opcode counts: `{json.dumps(k['static_opcode_counts'], sort_keys=True)}`")
            lines.append(f"- Selected PC sample totals: `{json.dumps(k['selected_sample_totals'], sort_keys=True)}`")
            lines.extend(["", "| Source site | Selected samples | Breakdown |", "|---|---:|---|"])
            for site in k["source_lines"][:8]:
                counts = site["samples"]
                lines.append(f"| {site['file']}:{site['line']} | {sum(counts.values())} | " + ", ".join(f"{f}={v}" for f, v in counts.items() if v) + " |")
            lines.extend(["", "| PC | Instruction | Selected samples | Primary site |", "|---|---|---:|---|"])
            for pc in k["top_pcs"]:
                ctx = pc["contexts"][0]
                lines.append(f"| {pc['address']} | `{pc['instruction']}` | {sum(pc['samples'].values())} | {ctx[0]}:{ctx[1]} |")
    (args.directory / "pc-delta.md").write_text("\n".join(lines) + "\n")
    print("PC diagnostic exported; unavailable aggregate DRAM and replay counter limitations remain explicit")


if __name__ == "__main__":
    main()
