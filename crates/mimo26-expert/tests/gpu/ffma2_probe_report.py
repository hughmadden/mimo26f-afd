#!/usr/bin/env python3
"""Local-only SASS classifier: syntax acceptance is not hardware emission."""
import collections
import csv
import json
from pathlib import Path
import re
import sys


def opcodes(text):
    ops = re.findall(r"^\s*/\*[0-9a-fA-F]+\*/\s+(?:@!?[A-Z][A-Z0-9]*\s+)?([A-Z][A-Z0-9]*)(?:\.|\s|;)", text, re.M)
    if not ops:
        raise ValueError("no SASS instructions: cannot infer absence of FFMA2")
    return dict(collections.Counter(ops))


def selftest():
    assert opcodes("Function : FFMA2\n /*0010*/ FFMA R0, R1, R2, R3;\n /*0020*/ @!P0 FFMA2.RN R4, R6, R8, R10;\n") == {"FFMA": 1, "FFMA2": 1}
    assert opcodes(" /*0010*/ NOP;\n /*0020*/ EXIT;") == {"NOP":1, "EXIT":1}
    assert "FFMA2" not in opcodes("Function : FFMA2\n /*0010*/ FADD R0, R1, R2;")
    try:
        opcodes("FFMA2 in a comment only")
    except ValueError:
        pass
    else:
        raise AssertionError("empty disassembly accepted")
    print("HOST PASS FFMA2 classifier: scalar/packed/predicated instructions, symbol false-positive and empty-SASS traps")


def main(root):
    rows = list(csv.DictReader((root / "status.tsv").open(), delimiter="\t"))
    assert {(r["arch"], r["mode"]) for r in rows} == {(a,str(m)) for a in ("100a","121a") for m in range(3)} and len(rows)==6
    for row in rows:
        for k in ("ptx_rc", "cubin_rc", "sass_rc"):
            row[k] = int(row[k])
        row["opcodes"] = opcodes((root / (row["arch"]+"-"+row["mode"]+".sass")).read_text()) if row["sass_rc"] == 0 else {}
        row["native_ffma2"] = row["opcodes"].get("FFMA2",0)>0
    controls = [r for r in rows if r["arch"]=="100a" or r["mode"]=="0"]
    assert all(r["ptx_rc"]==r["cubin_rc"]==r["sass_rc"]==0 for r in controls), "compile/disassembly control failed"
    assert all(r["native_ffma2"] for r in rows if r["arch"]=="100a" and r["mode"]!="0"), "known-target packed positive control failed"
    assert all(r["opcodes"].get("FFMA",0)+r["opcodes"].get("FFMA2",0)>0 for r in controls), "arithmetic optimized away"
    (root / "analysis.json").write_text(json.dumps({"compile_only":True,"gpu_launched":False,"rows":rows},indent=2)+"\n")
    for r in rows:
        print(f"sm_{r['arch']} mode={r['mode']} ptx={r['ptx_rc']} cubin={r['cubin_rc']} sass={r['sass_rc']} FFMA={r['opcodes'].get('FFMA',0)} FFMA2={r['opcodes'].get('FFMA2',0)}")
    print("PROBE COMPLETE: code generation only, no numerical or throughput qualification")

if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    else:
        main(Path(sys.argv[1]))
