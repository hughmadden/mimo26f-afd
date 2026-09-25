#!/usr/bin/env python3
"""Strict paired P1 receipts: complete correctness first, distinct frozen gates."""
import argparse
import json
import math
from pathlib import Path
import re
import statistics

CONTEXTS={'p1-2k':(2048,4,0),'p1-swa':(32768,8,128),'p1-128k':(131072,4,0),'p1-1m':(1048576,4,0)}
MODES=('f32q','bf16q')
COUNT=2048*64*128
QCOUNT=2048*64*192

def require(ok,why):
    if not ok:raise ValueError(why)

def fields(line):
    pairs=re.findall(r'(\w+)=([^=]*?)(?=\s+\w+=|$)',line)
    require(len(pairs)==len(dict(pairs)),'duplicate field')
    return {k:v.strip() for k,v in pairs}

def num(row,key):
    try:value=float(row[key])
    except (KeyError,ValueError):raise ValueError('missing/invalid '+key)
    require(math.isfinite(value),'nonfinite '+key)
    return value

def close(row,key,value):
    require(math.isclose(num(row,key),value,rel_tol=2e-6,abs_tol=1e-6),'wrong '+key)

def exact(row,expected):
    for k,v in expected.items():require(row.get(k)==str(v),'wrong/missing '+k)

def work(s,nkv,window,native):
    visible=sum(min(s-2048+i+1,window) if window else s-2048+i+1 for i in range(2048))
    tiles=0
    # Independent visibility-interval enumeration of full M64/N16 rectangles.
    for first in range(0,2048,nkv):
        earliest=s-2048+first;latest=s-2048+min(2047,first+nkv-1)
        lower=max(0,earliest-window+1) if window else 0
        tiles+=latest//16-lower//16+1
    return 40960*visible,2*64*16*nkv*((1 if native else 3)*192+256)*tiles

def inspect(text,required):
    require(required in CONTEXTS,'unknown required context')
    require(re.search(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} (?:AEST|AEDT)$',text,re.M),'missing Sydney timestamp')
    require(len(re.findall(r'^[a-f0-9]{64}  /var/tmp/mimo26f-attn/mimo26f-[A-Za-z0-9]+/mimo26f-attn-bench-sm120$',text,re.M))==1,'missing/duplicate binary identity')
    require(not re.search(r'==PROF==|PROFILE_|Profiling',text),'profile events are not gates')
    one={};correct={};metrics={};samples={m:[] for m in MODES};progress=[];memory=[];aot=0;finished=False
    for line in text.splitlines():
        if re.search(r'RESULT: (?:FAIL|REFUSE|INCOMPLETE)',line):raise ValueError('explicit failed/incomplete receipt')
        tag=line.split(' ',1)[0]
        if tag=='MEMORY':memory.append(fields(line))
        if line=='AOT: PASS architecture, SM-count, and baked kernel launch/readback':aot+=1
        if tag in ('IDENTITY','P1_CONTEXT','P1_QUERY','P1_REFERENCE','P1_COMPLETE'):
            require(tag not in one and not finished,'duplicate/late '+tag)
            if tag=='P1_REFERENCE':require('P1_QUERY' in one,'reference before query proof')
            one[tag]=fields(line)
            if tag=='P1_COMPLETE':finished=True
        elif tag=='P1_REFERENCE_PROGRESS':
            require(not finished and 'P1_QUERY' in one and 'P1_REFERENCE' not in one,'late reference progress')
            f=fields(line);progress.append(f.get('queries'));require(0<=num(f,'elapsed_s')<=480,'reference budget')
        elif tag in ('P1_CORRECT','P1_SAMPLE','P1_METRIC'):
            require(not finished,'data after completion');f=fields(line);mode=f.get('mode');require(mode in MODES,'unknown mode')
            if tag=='P1_CORRECT':
                require('P1_REFERENCE' in one and mode not in correct and not any(samples.values()),'duplicate/late correctness')
                correct[mode]=f
            elif tag=='P1_SAMPLE':
                require(len(correct)==2 and mode not in metrics,'sample before both checks or after metric')
                require(f.get('index')==str(len(samples[mode])),'sample index gap/duplicate')
                ms=num(f,'ms');require(ms>0,'nonpositive sample');samples[mode].append(ms)
            else:
                require(mode not in metrics and len(samples[mode])==7,'duplicate/incomplete metric')
                metrics[mode]=f
        elif tag.startswith('P1_'):raise ValueError('unknown P1 record')
    require(set(one)=={'IDENTITY','P1_CONTEXT','P1_QUERY','P1_REFERENCE','P1_COMPLETE'},'missing identity/proof/completion')
    identity=one['IDENTITY'];exact(identity,dict(arch='sm_120',sms=170,baked_arch='sm_120',baked_sms=170,label='TARGET'))
    require('5090' in identity.get('gpu','') and re.fullmatch(r'[a-f0-9]{12,40}',identity.get('source','')),'GPU/source identity')
    require(aot==1,'missing/duplicate hardware AOT')
    s,nkv,window=CONTEXTS[required];ctx=one['P1_CONTEXT'];batch=2 if s==1048576 else 8
    exact(ctx,dict(cell=required,label='TARGET',T=2048,S=s,n_q=64,n_kv=nkv,QK=192,V=128,window=window,
        sink='per-Q-head' if window else 'absent',page_tokens=256,paged='reverse-256',Q_storage='f32',Q_values='BF16-exact',
        K='E4M3-unit',V_dtype='E4M3-unit',cached_V='prescaled',KV_abs_max=1.875,scope='bounded-synthetic',mma_m=64,mma_n=16,
        warmup=3,samples=7,budget_s=480,unique_KV_bytes=s*nkv*320))
    exact(one['P1_QUERY'],dict(checked=f'{QCOUNT}/{QCOUNT}',exact='PASS',finite='PASS'))
    ref=one['P1_REFERENCE'];exact(ref,dict(outputs=f'{COUNT}/{COUNT}',finite='PASS',baseline='scalar-f64-splitkv-reduce',
        slab_queries=batch,splits=256,reuse='identical-BF16-exact-inputs',coordinates='3/3'))
    require(0<num(ref,'elapsed_ms')<=480000 and 0<=num(ref,'max_coordinate_diff')<=2e-5,'reference error/budget')
    require(progress==[f'{i}/2048' for i in range(256,2049,256)],'incomplete reference slabs')
    require(memory,'missing memory reserve evidence')
    needed=s*nkv*320+QCOUNT*4+COUNT*8+(s+2048)*8+(s//256)*4+batch*64*256*130*8+(2<<30)
    require(any(num(r,'requested_bytes')>=needed for r in memory),'missing conservative allocation budget')
    for r in memory:
        free=num(r,'free_bytes');total=num(r,'total_bytes');reserve=num(r,'reserve_bytes');requested=num(r,'requested_bytes')
        require(reserve>=(4<<30) and total>=free>=reserve and 0<=requested<=free-reserve,'memory reserve violation')
    exact(one['P1_COMPLETE'],dict(modes=2,checked_per_mode=COUNT,samples_per_mode=7))
    require(text.rstrip().endswith('RESULT: PASS P1 pair harness (performance verdicts separate, no promotion)'),'missing final harness completion')
    require(set(correct)==set(metrics)==set(MODES),'both precisions required')
    result=[]
    for mode in MODES:
        native=mode=='bf16q';c=correct[mode]
        exact(c,dict(reference='bf16q-lattice-local' if native else 'f32q',checked=f'{COUNT}/{COUNT}',finite='PASS',coordinates='3/3',
                     registers=124 if native else 125,shared_bytes=38976 if native else 88128,capacity_ctas=2 if native else 1))
        for field in ('max_baseline_diff','max_coordinate_diff'):require(0<=num(c,field)<=2e-5,'candidate correctness error')
        ms=statistics.median(samples[mode]);m=metrics[mode];useful,executed=work(s,nkv,window,native)
        require(num(m,'useful_flops')==useful and num(m,'executed_mma_flops')==executed,'wrong exact FLOP accounting')
        for field,value in (('median_ms',ms),('min_ms',min(samples[mode])),('max_ms',max(samples[mode])),
                            ('mma_work_factor',executed/useful),('useful_TFLOPS',useful/(ms*1e9)),('executed_TFLOPS',executed/(ms*1e9))):close(m,field,value)
        require(executed/(ms*1e9)<=209.5,'implausible dense BF16 peak; review accounting')
        target=100 if native else 125.7;domain='useful' if native else 'executed'
        exact(m,dict(target_domain=domain));close(m,'target_TFLOPS',target)
        rate=(useful if native else executed)/(ms*1e9);verdict='PASS' if rate>=target else 'MISS'
        exact(m,dict(verdict=verdict))
        result.append(dict(mode=mode,median_ms=ms,useful_TFLOPS=useful/(ms*1e9),executed_TFLOPS=executed/(ms*1e9),target_domain=domain,target_TFLOPS=target,verdict=verdict))
    return dict(status='VALID',cell=required,source=identity['source'],full_outputs_per_mode=COUNT,reference_stage_ms=num(ref,'elapsed_ms'),modes=result)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('receipt',type=Path);p.add_argument('--required',required=True,choices=CONTEXTS);a=p.parse_args()
    try:
        result=inspect(a.receipt.read_text(),a.required);print(json.dumps(result,indent=2))
        raise SystemExit(int(any(m['verdict']=='MISS' for m in result['modes'])))
    except (ValueError,OSError) as e:print(f'INCOMPLETE/INVALID: {e}');raise SystemExit(2)
