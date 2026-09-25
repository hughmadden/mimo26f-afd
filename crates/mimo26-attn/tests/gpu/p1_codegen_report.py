#!/usr/bin/env python3
"""P1 static inventory only; not hardware correctness, residency or throughput."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def inspect(sass,build):
    pattern=(r'Function properties for (\S+)\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, '
             r'(\d+) bytes spill loads\nptxas info\s*: Used (\d+) registers, used (\d+) barriers')
    resources={n:dict(stack_bytes=int(s),spill_store_bytes=int(w),spill_load_bytes=int(l),registers=int(r),named_barriers=int(b))
               for n,s,w,l,r,b in re.findall(pattern,build)}
    blocks=re.split(r'\n\s*Function : ([^\n]+)\n',sass);result=[];selected=[]
    for name,body in zip(blocks[1::2],blocks[2::2]):
        match=re.search(r'prefill_tcILb([01])ELb([01])E',name)
        if not match:continue
        residual=int(match[1]);split=match[2]=='1';assert name in resources,'missing P1 ptxas match'
        resource=resources[name]
        assert not any(resource[k] for k in ('stack_bytes','spill_store_bytes','spill_load_bytes')),'P1 stack/spills'
        assert 0<resource['registers']<=128,'P1 register budget'
        assert resource['named_barriers']==1,'P1 requires CTA barrier only'
        if split:
            # Trial-1 occupancy variant (P1M=32): resource budget only; numeric
            # bitwise-identity is gated by the parity probe, not this inventory.
            result.append(dict(kernel=name,query='f32q-split',shared_bytes=48192,**resource))
            selected.append(f'\nFunction : {name}\n{body}')
            continue
        ops=[]
        for line in body.splitlines():
            m=re.search(r'/\*([0-9a-f]+)\*/\s+(.*?)(?:\s+&|\s+\?|\s+/\*)',line)
            if m:ops.append((int(m[1],16),m[2].strip()))
        assert not any(re.search(r'\b(?:LDL|STL)(?:\.|\s)',op) for _,op in ops),'P1 local operations'
        assert not any('LDGSTS' in op or 'SYNCS.' in op for _,op in ops),'P1 must be synchronous CTA staging'
        mma=[(pc,op) for pc,op in ops if re.search(r'\bHMMA\.',op)]
        assert len(mma)==(52 if residual else 28),f'P1 unexpected MMA count: {len(mma)}'
        barriers=[(hex(pc),op) for pc,op in ops if 'BAR.' in op]
        assert sum('BAR.SYNC' in op for _,op in ops)>=7 and any('BAR.RED' in op for _,op in ops),'P1 phase barriers missing'
        destinations={re.search(r'HMMA\.\S+\s+(R\d+)',op)[1] for _,op in mma}
        assert len(destinations)>=(22 if residual else 18),'P1 independent accumulators missing'
        result.append(dict(kernel=name,query='f32q' if residual else 'bf16q',shared_bytes=88128 if residual else 38976,
                           **resource,mma_sites=len(mma),mma_destination_registers=sorted(destinations),barriers=barriers))
        selected.append(f'\nFunction : {name}\n{body}')
    assert len(result)==3 and {r['query'] for r in result}=={'f32q','bf16q','f32q-split'},'need both P1 precisions + split'
    return result,''.join(selected)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('slot',type=Path);p.add_argument('--emit-sass',action='store_true');a=p.parse_args()
    rows,sass=inspect((a.slot/'SASS.txt').read_text(),(a.slot/'ptxas.log').read_text())
    if a.emit_sass:print(sass)
    else:print(json.dumps(dict(scope=__doc__,binary_sha256=hashlib.sha256((a.slot/'mimo26f-attn-bench-sm120').read_bytes()).hexdigest(),kernels=rows),indent=2))
