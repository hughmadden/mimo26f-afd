#!/usr/bin/env python3
"""C1 static inventory: no GPU qualification, ordering or performance claim."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def inspect(sass, build):
    pattern=(r'Function properties for (\S+)\n\s*(\d+) bytes stack frame, (\d+) bytes spill stores, '
             r'(\d+) bytes spill loads\nptxas info\s*: Used (\d+) registers, used (\d+) barriers')
    resources={name:dict(stack_bytes=int(stack),spill_store_bytes=int(stores),spill_load_bytes=int(loads),
                         registers=int(regs),named_barriers=int(bars))
               for name,stack,stores,loads,regs,bars in re.findall(pattern,build)}
    blocks=re.split(r'\n\s*Function : ([^\n]+)\n',sass)
    results=[];selected=[]
    for name,body in zip(blocks[1::2],blocks[2::2]):
        found=re.search(r'decode_c1ILb([01])E',name)
        if not found: continue
        residual=int(found[1]);assert name in resources,'missing matching ptxas'
        resource=resources[name]
        assert not any(resource[k] for k in ('stack_bytes','spill_store_bytes','spill_load_bytes')),'C1 stack/spills'
        assert resource['registers']<=128,'C1 exceeds two-CTA register byte model'
        assert resource['named_barriers']==3,'expected block/producers/QK named barriers'
        instructions=[]
        for line in body.splitlines():
            m=re.search(r'/\*([0-9a-f]+)\*/\s+(.*?)(?:\s+&|\s+\?|\s+/\*)',line)
            if m: instructions.append((int(m[1],16),m[2].strip()))
        assert not any(re.search(r'\b(?:LDL|STL)(?:\.|\s)',op) for _,op in instructions),'local memory'
        def sites(fragment,offset=None):
            return [(pc,op) for pc,op in instructions if fragment in op and (offset is None or offset in op)]
        for offset in ('0xad40','0xad50','0xad60'):
            for operation in ('SYNCS.EXCH.64','SYNCS.ARRIVE.','SYNCS.PHASECHK.','SYNCS.CCTL.IV'):
                assert sites(operation,offset),f'missing {operation} at {offset}'
        for role in (1,2):
            assert any(re.search(rf'BAR\.SYNC\S* 0x{role}, 0x40',op) for _,op in instructions),'missing 64-thread subgroup barrier'
        assert sites('LDGSTS.') and sites('DEPBAR.LE'),'missing async copy/wait'
        assert any('0x0' in op for _,op in sites('DEPBAR.LE')),'missing full async drain'
        qb=min(pc for pc,_ in sites('SYNCS.PHASECHK.','0xad40'))
        qe=min(pc for pc,_ in sites('SYNCS.ARRIVE.','0xad50'))
        vb=min(pc for pc,_ in sites('SYNCS.PHASECHK.','0xad50'))
        ve=min(pc for pc,_ in sites('SYNCS.ARRIVE.','0xad60'))
        # ptxas may place register-only tail HMMAs after the release arrival.
        # The slot lifetime ends at the LAST SHARED READ, not arithmetic issue.
        backedges=[pc for pc,op in instructions if pc>ve and
                   (m:=re.search(r'\bBRA (0x[0-9a-f]+)',op)) and int(m[1],16)<vb]
        assert backedges,'missing PV loop boundary'
        vend=min(backedges)
        assert not any('LDS' in op for pc,op in instructions if ve<pc<vend),'shared read after PV release'
        mma=sites('HMMA.')
        qm=[(pc,op) for pc,op in mma if qb<pc<qe];vm=[(pc,op) for pc,op in mma if vb<pc<vend]
        assert len(qm)==(36 if residual else 12) and len(vm)==8 and len(mma)==len(qm)+len(vm),f'unexpected role MMA sites query={residual} QK={len(qm)} PV={len(vm)} total={len(mma)} bounds={qb:x}/{qe:x}/{vb:x}/{ve:x} sites={[(hex(pc),op) for pc,op in mma]}'
        def destinations(items): return sorted({re.search(r'HMMA\.\S+\s+(R\d+)',op)[1] for _,op in items})
        qd,vd=destinations(qm),destinations(vm)
        assert len(qd)>=(6 if residual else 2) and len(vd)>=8,'lost independent role accumulators'
        results.append(dict(kernel=name,query='f32q' if residual else 'bf16q',**resource,
                            qk_mma=len(qm),pv_mma=len(vm),qk_destinations=qd,pv_destinations=vd,
                            synchronization=[(hex(pc),op) for pc,op in instructions if 'SYNCS.' in op or 'BAR.SYNC' in op or 'DEPBAR' in op]))
        selected.append(f'\nFunction : {name}\n{body}')
    assert len(results)==2 and len({r['query'] for r in results})==2,'need exactly both C1 precisions'
    return results,''.join(selected)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('slot',type=Path);p.add_argument('--emit-sass',action='store_true');a=p.parse_args()
    result,sass=inspect((a.slot/'SASS.txt').read_text(),(a.slot/'ptxas.log').read_text())
    if a.emit_sass: print(sass)
    else:
        binary=a.slot/'mimo26f-attn-bench-sm120'
        print(json.dumps(dict(scope=__doc__,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),shared_bytes=44416,kernels=result),indent=2))
