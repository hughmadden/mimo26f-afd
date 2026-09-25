#!/usr/bin/env python3
"""Static C0 resource/accumulator inventory; never a GPU correctness gate."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def inspect(sass, build):
    resources = {}
    pattern = (r"Function properties for (\S+)\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, "
               r"(\d+) bytes spill loads\nptxas info\s*: Used (\d+) registers")
    for name, stack, stores, loads, regs in re.findall(pattern, build):
        resources[name] = dict(stack_bytes=int(stack), spill_store_bytes=int(stores),
                               spill_load_bytes=int(loads), numRegs=int(regs))
    blocks = re.split(r"\n\s*Function : ([^\n]+)\n", sass)
    result, selected = [], []
    for name, body in zip(blocks[1::2], blocks[2::2]):
        match = re.search(r"decode_pipeILi([48])ELb([01])E", name)
        if not match:
            continue
        warps, residual = map(int, match.groups())
        assert name in resources, "missing matching ptxas resource counters"
        resource = resources[name]
        assert all(resource[k] == 0 for k in ("stack_bytes", "spill_store_bytes", "spill_load_bytes")), "C0 spills/stack: stop before timing"
        instructions = []
        for line in body.splitlines():
            m = re.search(r"/\*([0-9a-f]+)\*/\s+(.*?)(?:\s+&|\s+\?|\s+/\*)", line)
            if m:
                instructions.append((m[1], m[2].strip()))
        assert not any(re.search(r"\b(?:LDL|STL)(?:\.|\s)", op) for _, op in instructions), "SASS local-memory operations"
        mma = [(pc, op) for pc, op in instructions if "HMMA." in op]
        expected = (3 if residual else 1)*12+(16//warps)*4
        assert len(mma) == expected, f"MMA count {len(mma)} != {expected}"
        destinations = sorted({re.search(r"HMMA\.\S+\s+(R\d+)", op)[1] for _, op in mma})
        minimum_sets=(3 if residual else 1)*2+(16//warps)*4
        assert len(destinations)>=minimum_sets, "too few C0 accumulator destination sets; retrace codegen"
        assert any("LDGSTS." in op for _, op in instructions), "async copy missing"
        result.append(dict(kernel=name, warps=warps, query="f32q" if residual else "bf16q",
                           **resource, static_mma_instructions=len(mma), mma_destination_sets=destinations,
                           shared_byte_load_sites=sum(bool(re.search(r"\bLDS\.U8\b", op)) for _, op in instructions),
                           mma=mma, waits=[(pc, op) for pc, op in instructions if "DEPBAR" in op]))
        selected.append(f"\nFunction : {name}\n{body}")
    assert len(result) == 4, "expected both precisions and both warp variants"
    assert len({(r['warps'], r['query']) for r in result}) == 4, "duplicate variants"
    return result, "".join(selected)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("slot", type=Path)
    ap.add_argument("--emit-sass", action="store_true")
    args = ap.parse_args()
    counters=args.slot/"ptxas.log"
    if not counters.is_file(): counters=args.slot/"receipt.log"
    report, selected = inspect((args.slot/"SASS.txt").read_text(), counters.read_text())
    if args.emit_sass:
        print(selected)
    else:
        binary = args.slot/"mimo26f-attn-bench-sm120"
        print(json.dumps(dict(scope="static accumulator-parallel decode codegen; NOT GPU qualification or timing",
                             binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), kernels=report), indent=2))
