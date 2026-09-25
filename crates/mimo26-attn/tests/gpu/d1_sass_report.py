#!/usr/bin/env python3
"""CPU-only D1 instruction inventory; never claims runtime overlap or stalls."""
import argparse
import json
from pathlib import Path
import re


def inventory(text, revision="ef95e61"):
    result = []
    blocks = re.split(r"\n\s*Function : ([^\n]+)\n", text)
    for name, body in zip(blocks[1::2], blocks[2::2]):
        if "decode_pipeILi" not in name:
            continue
        instructions = []
        for line in body.splitlines():
            m = re.search(r"/\*([0-9a-f]+)\*/\s+(.*?)(?:\s+&|\s+\?|\s+/\*)", line)
            if m:
                instructions.append((int(m[1], 16), m[2].strip()))
        async_ops = [(pc, op) for pc, op in instructions if "LDGSTS." in op]
        assert async_ops, "D1 async copy missing"
        bad = [(pc, op) for pc, op in instructions if re.search(r"\b(?:LDG(?:\.\w+)*\.U8|STS\.U8)\b", op)]
        assert not bad, f"D1 global byte load/shared byte store remains: {bad}"
        # Boundaries manually traced in this exact ef95e61 binary. Not a generic
        # optimizer proof; refuse a different layout rather than silently reuse PCs.
        warps = int(re.search(r"decode_pipeILi([48])", name)[1])
        boundaries = {
            "ef95e61": ({8: 0xfb0, 4: 0x18c0}, {8: 0x1710, 4: 0x2030}),
            # c595a10 benchmark: Q loop backedges at 0xfc0 / 0x18c0;
            # reconvergence ends at 0xfd0 / 0x18d0 before any raw priming.
            "c595a10": ({8: 0xfe0, 4: 0x18e0}, {8: 0x1800, 4: 0x2120}),
        }
        q_end = boundaries[revision][0][warps]
        expected_first = boundaries[revision][1][warps]
        shared_byte_loads = [(pc, op) for pc, op in instructions if re.search(r"\bLDS(?:\.\w+)*\.U8\b", op)]
        if revision == "c595a10":
            assert not shared_byte_loads, "F4 still has shared byte gathers"
        assert async_ops[0][0] == expected_first, "retrace Q region for this new binary"
        late_reentry = []
        for pc, op in instructions:
            branch = re.search(r"\bBRA\s+0x([0-9a-f]+)", op)
            if branch and pc >= expected_first and int(branch[1], 16) < q_end:
                late_reentry.append((hex(pc), op))
        assert not late_reentry, "post-prime branch re-enters Q prologue"
        result.append(dict(kernel=name, revision=revision, global_byte_loads=0, shared_byte_stores=0,
                           shared_byte_loads=len(shared_byte_loads),
                           manually_traced_q_region_end=hex(q_end), post_prime_branches_into_q_region=late_reentry,
                           first_async_copy_pc=hex(async_ops[0][0]),
                           async_copies=[(hex(pc), op) for pc, op in async_ops],
                           conversions=[(hex(pc), op) for pc, op in instructions if "F2FP" in op],
                           global_loads=[(hex(pc), op) for pc, op in instructions if re.search(r"\bLDG\.", op)],
                           waits=[(hex(pc), op) for pc, op in instructions if "DEPBAR" in op]))
    assert len(result) == 2, "expected original two D1 warp variants"
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sass", type=Path)
    ap.add_argument("--revision", choices=("ef95e61", "c595a10"), default="ef95e61")
    ap.add_argument("--emit-sass", action="store_true")
    args = ap.parse_args()
    text = args.sass.read_text()
    report = inventory(text, args.revision)
    if args.emit_sass:
        blocks = re.split(r"\n\s*Function : ([^\n]+)\n", text)
        print(f"// D1-only SASS extraction, manually traced revision {args.revision}; see sibling identity.")
        for name, body in zip(blocks[1::2], blocks[2::2]):
            if "decode_pipeILi" in name:
                print(f"\nFunction : {name}\n{body}")
    else:
        print(json.dumps(dict(scope="SASS inventory, not runtime efficiency", kernels=report), indent=2))
