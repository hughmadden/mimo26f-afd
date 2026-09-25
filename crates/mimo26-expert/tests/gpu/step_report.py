"""Fail-closed R18 native receipt audit. Not a promotion/gate implementation."""
import argparse
from collections import defaultdict
import json
import hashlib
import math
from pathlib import Path
import re

BYTES=3342336


def need(ok,message):
    if not ok:raise ValueError(message)


def fields(line):
    pairs=[word.split('=',1) for word in line.split() if '=' in word]
    need(len({k for k,_ in pairs})==len(pairs),'duplicate field')
    return dict(pairs)


def positive(v):
    x=float(v);need(math.isfinite(x) and x>0,'nonpositive/nonfinite timer');return x


def close(a,b):
    need(math.isclose(float(a),float(b),rel_tol=1e-6,abs_tol=1e-8),'numerical summary binding')


def r18_line(gbps):
    positive(gbps)
    return 'PASS' if gbps>=191.1 else 'BELOW_TARGET_RECORDED' if gbps>=163.8 else 'STOP'


def audit(text,case,steps,family,frozen=False,prefill=False):
    samples=defaultdict(dict);rows={};scopes={};checks=defaultdict(list);parts=defaultdict(list);final=[];allocations=[];current=None;negatives=0
    for line in text.splitlines():
        if not line.startswith('STEP '):continue
        f=fields(line)
        if line.startswith('STEP ALLOCATION '):allocations.append(f)
        elif line.startswith('STEP SCOPE '):
            current=int(f['ordinal']);need(current not in scopes,'duplicate scope');scopes[current]=f
        elif line.startswith('STEP SAMPLE '):
            i=int(f['ordinal']);j=int(f['sample']);need(j not in samples[i],'duplicate sample');samples[i][j]=f
        elif line.startswith('STEP ROW '):
            i=int(f['ordinal']);need(i not in rows,'duplicate row');rows[i]=f
        elif line.startswith('STEP CORRECT '):checks[current].append((f,len(samples[current]),len(parts[current])))
        elif line.startswith('STEP BREAKDOWN '):parts[int(f['ordinal'])].append(f)
        elif line.startswith('STEP FINAL '):final.append(f)
        elif line=='STEP NEGATIVE PASS actual backend mutation':negatives+=1
        elif line.startswith('STEP FAILURE'):raise ValueError('native failure')
    alignment='128' if family=='B2' else '16'
    need(allocations==[dict(frozen_redzone_bytes=alignment,payload_alignment=alignment)],'frozen allocator layout not reproduced')
    want=set(range(len(steps)));need(set(scopes)==set(samples)==set(rows)==set(checks)==set(parts)==want,'missing/extra step evidence')
    need(len(final)==1 and negatives==1,'missing final/negative')
    total_bytes=0;total_ms=0;breakdown=defaultdict(lambda:dict(expert_occurrences=0,native_loads=0,scheduled_expert_loads=0,replay_ms=0.0))
    all_schedules={};above_peak=0;sample_total_ms=0.0;replay_ms=[0.0]*31
    for i,step in enumerate(steps):
        hist={int(k):v for k,v in step['hist'].items()};experts=sum(hist.values());routes=sum(k*v for k,v in hist.items());distinct=experts*BYTES
        scheduled=sum(((m+7)//8)*n for m,n in hist.items())*BYTES;loads=sum((m+7)//8 for m in hist)
        if prefill:
            need(family=='B2' and len(steps)==1 and len(hist)==1 and experts==256 and next(iter(hist)) in (1,16,64) and step['layer']==1,'F5 native shape')
            scheduled=distinct;loads=1
        scope=scopes[i];need(scope['workload']==case and int(scope['weight_layer'])==step['layer'] and int(scope['route_layer'])==step.get('route_layer',step['layer']) and int(scope['frozen_order'])==int(frozen) and scope['timing']=='expert_compute_only','scope/source layer binding')
        expected=dict(family=family,workload=case,ordinal=str(i),layer=str(step['layer']),experts=str(experts),routes=str(routes),bytes=str(distinct))
        def binding(row):
            need(all(row[k]==v for k,v in expected.items()),'row/input identity mismatch')
        need(set(samples[i])==set(range(31)),'31 unique samples required')
        times=[];digests=[]
        for j,sample in samples[i].items():
            binding(sample);need(int(sample['scheduled_bytes'])==scheduled and int(sample['loads'])==loads,'repeated load accounting')
            elapsed=positive(sample['ms']);times.append(elapsed);sample_total_ms+=elapsed;replay_ms[j]+=elapsed;digest=sample['schedule'];need(re.fullmatch('[0-9a-f]{64}',digest) is not None,'schedule fingerprint')
            digests.append(digest);all_schedules[f'{i}:{j}']=digest
        need(len(set(digests))==(1 if frozen else 31),'unexpected schedule reuse')
        evidence=checks[i];need(len(evidence)==(4 if i==0 else 3),'pre/post/breakdown reference checks missing')
        need([(n,p) for _,n,p in evidence]==([(0,0),(0,0),(31,0),(31,loads)] if i==0 else [(0,0),(31,0),(31,loads)]),'reference check ordering')
        check=[v for v,_,_ in evidence]
        need([int(v['bad'])>0 for v in check]==([False,True,False,False] if i==0 else [False,False,False]),'reference/negative pattern')
        for v in check:need(v['family']==family and int(v['coordinates'])==routes*4096 and v['atol']=='1e-5' and v['rtol']=='1e-5','oracle coverage/tolerance')
        row=rows[i];binding(row);times.sort();close(row['median_ms'],times[15]);close(row['min_ms'],times[0]);close(row['max_ms'],times[-1])
        gbps=distinct/(times[15]*1e6);close(row['effective_GBps'],gbps);need(int(row['above_dram_peak'])==int(gbps>273) and row['diagnostic_only']=='yes','peak/diagnostic status');above_peak+=gbps>273
        expected_parts=([(m,m,0,n) for m,n in sorted(hist.items())] if prefill else [(m,min(8,m-first),first//8,n) for m,n in sorted(hist.items()) for first in range(0,m,8)])
        need(len(parts[i])==len(expected_parts),'breakdown extent')
        for part,(m,native,p,n) in zip(parts[i],expected_parts):
            need(part['family']==family and part['workload']==case and (int(part['original_M']),int(part['native_M']),int(part['pass']),int(part['experts']))==(m,native,p,n),'breakdown identity')
            need(part['scope']=='separate_one_shot_replay_not_additive','breakdown scope')
            b=breakdown[m];b['native_loads']+=1;b['scheduled_expert_loads']+=n;b['replay_ms']+=positive(part['ms'])
            if p==0:b['expert_occurrences']+=n
        total_bytes+=distinct;total_ms+=times[15]
    f=final[0];need(f['family']==family and f['workload']==case and int(f['steps'])==len(steps) and int(f['distinct_bytes'])==total_bytes and f['no_promotion']=='yes','final identity')
    close(f['sum_median_ms'],total_ms);close(f['aggregate_effective_GBps'],total_bytes/(total_ms*1e6))
    eligible=family=='B2' and any(s['hist']=={'1':256} for s in steps)
    need(f['m1_sanity']==('PASS' if eligible else 'NOT_APPLICABLE'),'M1 sanity status')
    if eligible:
        for i,s in enumerate(steps):
            if s['hist']=={'1':256}:need(.95<=float(rows[i]['effective_GBps'])/216.742768<=1.05,'frozen M1 reproduction')
    if prefill:
        m=next(iter(steps[0]['hist']));tile=1 if m=='1' else 8
        need(case=='prefill-M'+m,'F5 workload label')
        contracts=[fields(l) for l in text.splitlines() if l.startswith('F5 CONTRACT ')]
        need(contracts==[dict(native_max_M=m,host_passes='1',row_tile=str(tile),grid_z=str(int(m)//tile),activation_prefix_rows='8',cycle_rows='yes',report_only='yes')],'F5 native-plan/tile contract')
        reported=[fields(l) for l in text.splitlines() if l.startswith('F5 ROW ')]
        need(reported==[dict(M=m,line_GBps='163.8',at_or_above=str(int(total_bytes/(total_ms*1e6)>=163.8)),gate='no',native_wide_tile='no')],'F5 report-only line')
    return dict(family=family,workload=case,steps=len(steps),distinct_bytes=total_bytes,sum_median_ms=total_ms,effective_GBps=total_bytes/(total_ms*1e6),above_peak_steps=above_peak,
                aggregation='sum distinct bytes / sum per-step median times (preregistered)',sample_total_ms=sample_total_ms,
                pooled_samples_GBps=total_bytes*31/(sample_total_ms*1e6),median_indexed_replay_GBps=total_bytes/(sorted(replay_ms)[15]*1e6),
                per_original_M=dict(breakdown),schedule_digests=all_schedules,no_promotion=True)


def audit_mixed(text,case,steps):
    samples=defaultdict(dict);rows={};scopes={};checks=defaultdict(list);final=[];allocations=[];contracts=[];r18c=[];current=None;negatives=0
    for line in text.splitlines():
        if line.startswith('MIXED CONTRACT '):contracts.append(fields(line));continue
        if line.startswith('R18C LINE '):r18c.append(fields(line));continue
        if not line.startswith('STEP '):continue
        f=fields(line)
        if line.startswith('STEP ALLOCATION '):allocations.append(f)
        elif line.startswith('STEP SCOPE '):current=int(f['ordinal']);need(current not in scopes,'duplicate scope');scopes[current]=f
        elif line.startswith('STEP SAMPLE '):
            i=int(f['ordinal']);j=int(f['sample']);need(j not in samples[i],'duplicate sample');samples[i][j]=f
        elif line.startswith('STEP ROW '):rows[int(f['ordinal'])]=f
        elif line.startswith('STEP CORRECT '):checks[current].append(f)
        elif line.startswith('STEP FINAL '):final.append(f)
        elif line=='STEP NEGATIVE PASS actual backend mutation':negatives+=1
        elif line.startswith('STEP FAILURE'):raise ValueError('native failure')
    need(contracts==[dict(kernels='4',max_m='8',policy='phase_wise_coalesced',exact_M='1',grid_z='1',correct_entry='mixed_gemm<true>',separate_naive_entry='mixed_gemm<false>')],'mixed contract')
    need(allocations==[dict(frozen_redzone_bytes='128',payload_alignment='128')],'frozen allocator')
    need(len(r18c)==1,'R18c line')
    want=set(range(len(steps)));need(set(scopes)==set(samples)==set(rows)==set(checks)==want,'missing/extra step evidence')
    need(len(final)==1 and negatives==1,'final/negative')
    total_bytes=0;total_ms=0;sample_total_ms=0.0;replay_ms=[0.0]*31;above_peak=0;all_schedules={}
    for i,step in enumerate(steps):
        hist={int(k):v for k,v in step['hist'].items()};experts=sum(hist.values());routes=sum(k*v for k,v in hist.items());distinct=experts*BYTES
        need(all(m<=8 for m in hist),'mixed decode width>8')
        scope=scopes[i];need(scope['workload']==case and int(scope['weight_layer'])==step['layer'] and int(scope['route_layer'])==step.get('route_layer',step['layer']) and scope['frozen_order']=='0' and scope['timing']=='expert_compute_only','scope binding')
        need(set(samples[i])==set(range(31)),'31 unique samples required')
        times=[];digests=[]
        for j,s in samples[i].items():
            need(s['family']=='B2' and s['workload']==case and int(s['ordinal'])==i and int(s['layer'])==step['layer'] and int(s['experts'])==experts and int(s['routes'])==routes and int(s['bytes'])==distinct and int(s['scheduled_bytes'])==distinct and int(s['groups'])==experts,'sample identity')
            el=positive(s['ms']);times.append(el);sample_total_ms+=el;replay_ms[j]+=el;d=s['schedule'];need(re.fullmatch('[0-9a-f]{64}',d) is not None,'schedule fingerprint');digests.append(d);all_schedules[f'{i}:{j}']=d
        need(len(set(digests))==31,'schedule reuse')
        ev=checks[i];need(len(ev)==2,'pre/post correct checks')
        for v in ev:
            need(v['family']=='B2' and int(v['coordinates'])==routes*4096 and int(v['bitwise_bad'])==0 and int(v['golden_bad'])==0 and v['atol']=='1e-5' and v['rtol']=='1e-5','bitwise/reference identity')
        row=rows[i];need(row['family']=='B2' and row['workload']==case and int(row['ordinal'])==i and int(row['layer'])==step['layer'] and int(row['experts'])==experts and int(row['routes'])==routes and int(row['bytes'])==distinct and row['diagnostic_only']=='yes','row identity')
        times.sort();close(row['median_ms'],times[15]);close(row['min_ms'],times[0]);close(row['max_ms'],times[-1])
        gbps=distinct/(times[15]*1e6);close(row['effective_GBps'],gbps);need(int(row['above_dram_peak'])==int(gbps>273),'peak/diagnostic status');above_peak+=gbps>273
        total_bytes+=distinct;total_ms+=times[15]
    f=final[0];need(f['family']=='B2' and f['workload']==case and int(f['steps'])==len(steps) and int(f['distinct_bytes'])==total_bytes and f['m1_sanity']=='NOT_APPLICABLE' and f['no_promotion']=='yes','final identity')
    close(f['sum_median_ms'],total_ms);close(f['aggregate_effective_GBps'],total_bytes/(total_ms*1e6))
    gb=total_bytes/(total_ms*1e6);line=r18c[0];close(line['gbps'],gb)
    need(line['pass']=='191.1' and line['stop_below']=='163.8' and line['verdict']==r18_line(gb) and line['gate']=='conditional_on_F1' and line['no_promotion']=='yes','R18c line')
    return dict(family='B2',workload=case,steps=len(steps),distinct_bytes=total_bytes,sum_median_ms=total_ms,effective_GBps=gb,
                above_peak_steps=above_peak,aggregation='sum distinct bytes / sum per-step median times (preregistered)',sample_total_ms=sample_total_ms,
                pooled_samples_GBps=total_bytes*31/(sample_total_ms*1e6),median_indexed_replay_GBps=total_bytes/(sorted(replay_ms)[15]*1e6),
                bitwise_identity='PASS all steps vs frozen per-M launches',kernel_count=4,policy='phase_wise_coalesced',exact_M=True,
                R18c_line=r18_line(gb),R18c_thresholds_GBps={'pass':191.1,'stop_below':163.8},schedule_digests=all_schedules,no_promotion=True)


def selftest():
    case='test';step=dict(layer=1,hist={'1':1});common='family=B2 workload=test ordinal=0 layer=1 experts=1 routes=1 bytes=3342336'
    check='STEP CORRECT family=B2 coordinates=4096 bad=0 atol=1e-5 rtol=1e-5'
    lines=['STEP ALLOCATION frozen_redzone_bytes=128 payload_alignment=128','STEP SCOPE workload=test ordinal=0 weight_layer=1 route_layer=1 frozen_order=0 timing=expert_compute_only',check,check.replace('bad=0','bad=1'),'STEP NEGATIVE PASS actual backend mutation']
    for i in range(31):lines.append(f'STEP SAMPLE {common} sample={i} scheduled_bytes=3342336 loads=1 ms=1.0 schedule={i:064x}')
    lines +=[check,f'STEP ROW {common} median_ms=1 min_ms=1 max_ms=1 effective_GBps=3.342336 above_dram_peak=0 diagnostic_only=yes',
             'STEP BREAKDOWN family=B2 workload=test ordinal=0 original_M=1 native_M=1 pass=0 experts=1 ms=1 scope=separate_one_shot_replay_not_additive',check,
             'STEP FINAL family=B2 workload=test steps=1 sum_median_ms=1 distinct_bytes=3342336 aggregate_effective_GBps=3.342336 m1_sanity=NOT_APPLICABLE no_promotion=yes']
    text='\n'.join(lines);audit(text,case,[step],'B2')
    variants=[text.replace('sample=30 ','sample=29 '),text.replace('ms=1.0','ms=nan',1),text.replace('scheduled_bytes=3342336','scheduled_bytes=6684672',1),text.replace('bytes=3342336','bytes=3342337',1),text.replace('route_layer=1','route_layer=2',1),text.replace('effective_GBps=3.342336','effective_GBps=9',1),text.replace('bad=1','bad=0'),text.replace(check+'\n','',1),text.replace('pass=0','pass=1'),text.replace('m1_sanity=NOT_APPLICABLE','m1_sanity=PASS'),text.replace('no_promotion=yes','no_promotion=no'),text+'\n'+lines[-1],text.replace(f'schedule={30:064x}',f'schedule={29:064x}')]
    variants += [text.replace('payload_alignment=128','payload_alignment=16'),text.replace(lines[0]+'\n',''),text.replace('timing=expert_compute_only','timing=sum_isolated_rows')]
    for wrong in variants:
        try:audit(wrong,case,[step],'B2')
        except (ValueError,KeyError):pass
        else:raise AssertionError('unpowered receipt negative')
    wide=text
    for before,after in [('workload=test','workload=prefill-M16'),('experts=1','experts=256'),('routes=1','routes=4096'),('bytes=3342336','bytes=855638016'),('coordinates=4096','coordinates=16777216'),('original_M=1','original_M=16'),('native_M=1','native_M=16'),('ms=1.0','ms=10.0'),('median_ms=1','median_ms=10'),('min_ms=1','min_ms=10'),('max_ms=1','max_ms=10'),(' ms=1 scope',' ms=10 scope'),('effective_GBps=3.342336','effective_GBps=85.5638016')]:wide=wide.replace(before,after)
    wide='F5 CONTRACT native_max_M=16 host_passes=1 row_tile=8 grid_z=2 activation_prefix_rows=8 cycle_rows=yes report_only=yes\n'+wide+'\nF5 ROW M=16 line_GBps=163.8 at_or_above=0 gate=no native_wide_tile=no'
    wide_step=dict(layer=1,hist={'16':256});audit(wide,'prefill-M16',[wide_step],'B2',prefill=True)
    for wrong in [wide.replace('row_tile=8','row_tile=64'),wide.replace('native_M=16','native_M=8'),wide.replace('host_passes=1','host_passes=2'),wide.replace('gate=no','gate=yes'),wide.replace('activation_prefix_rows=8','activation_prefix_rows=64')]:
        try:audit(wrong,'prefill-M16',[wide_step],'B2',prefill=True)
        except (ValueError,KeyError):pass
        else:raise AssertionError('unpowered F5 receipt negative')
    for value,want in [(math.nextafter(163.8,0),'STOP'),(163.8,'BELOW_TARGET_RECORDED'),(math.nextafter(163.8,math.inf),'BELOW_TARGET_RECORDED'),(math.nextafter(191.1,0),'BELOW_TARGET_RECORDED'),(191.1,'PASS'),(math.nextafter(191.1,math.inf),'PASS')]:
        assert r18_line(value)==want
    mstep=dict(layer=1,hist={'1':1})
    mlines=['MIXED CONTRACT kernels=4 max_m=8 policy=phase_wise_coalesced exact_M=1 grid_z=1 correct_entry=mixed_gemm<true> separate_naive_entry=mixed_gemm<false>',
            'STEP ALLOCATION frozen_redzone_bytes=128 payload_alignment=128',
            'STEP SCOPE workload=test ordinal=0 weight_layer=1 route_layer=1 frozen_order=0 timing=expert_compute_only',
            'STEP CORRECT family=B2 coordinates=4096 bitwise_bad=0 golden_bad=0 atol=1e-5 rtol=1e-5',
            'STEP NEGATIVE PASS actual backend mutation']
    for i in range(31):mlines.append(f'STEP SAMPLE family=B2 workload=test ordinal=0 layer=1 sample={i} experts=1 routes=1 bytes=3342336 scheduled_bytes=3342336 groups=1 ms=1.0 schedule={i:064x}')
    mlines +=['STEP CORRECT family=B2 coordinates=4096 bitwise_bad=0 golden_bad=0 atol=1e-5 rtol=1e-5',
              'STEP ROW family=B2 workload=test ordinal=0 layer=1 experts=1 routes=1 bytes=3342336 median_ms=1 min_ms=1 max_ms=1 effective_GBps=3.342336 above_dram_peak=0 diagnostic_only=yes',
              'STEP FINAL family=B2 workload=test steps=1 sum_median_ms=1 distinct_bytes=3342336 aggregate_effective_GBps=3.342336 m1_sanity=NOT_APPLICABLE no_promotion=yes',
              'R18C LINE gbps=3.342336 pass=191.1 stop_below=163.8 verdict=STOP gate=conditional_on_F1 no_promotion=yes']
    mtext='\n'.join(mlines);audit_mixed(mtext,'test',[mstep])
    for wrong in [mtext.replace('bitwise_bad=0','bitwise_bad=1',1),mtext.replace('golden_bad=0','golden_bad=1',1),mtext.replace('STEP NEGATIVE PASS actual backend mutation','',1),mtext.replace(f'schedule={30:064x}',f'schedule={29:064x}'),mtext.replace('verdict=STOP','verdict=PASS'),mtext.replace('groups=1','groups=2'),mtext.replace('kernels=4','kernels=5')]:
        try:audit_mixed(wrong,'test',[mstep])
        except (ValueError,KeyError):pass
        else:raise AssertionError('unpowered mixed receipt negative')
    print(f'STEP REPORT HOST PASS positive native-shaped fixture, {len(variants)} powered corruptions, F5 positive/five negatives, mixed positive/seven negatives and six R18 boundary cases')


def receipt(root):
    status=fields((root/'status.txt').read_text());need(status==dict(remote_exit='0',retrieve_exit='0'),'remote/retrieval failure; retain partial evidence')
    launch=fields((root/'launch.txt').read_text());mode=launch['mode'];cases=launch['cases'].split(',');data=json.loads((root/'input.json').read_text());result={}
    manifest=(root/'source-sha256.txt').read_text().splitlines()
    for local,suffix in [('input.json','/histogram.json'),('oracle-source.json','/oracle/source.json')]:
        hits=[line.split()[0] for line in manifest if line.endswith(suffix)]
        need(hits==[hashlib.sha256((root/local).read_bytes()).hexdigest()],'receipt input/oracle manifest identity')
    def one(label,case,family,frozen=False,prefill=False):
        status=fields((root/(label+'-status.txt')).read_text());need(status==dict(native_exit='0',tee_exit='0'),'native/tee failure')
        return audit((root/(label+'.log')).read_text(),case,data[case]['steps'],family,frozen,prefill)
    if mode=='mixed':
        need(cases==['C1-w8'],'mixed cell measures C1-w8 only')
        status=fields((root/'B2-C1-w8-status.txt').read_text());need(status==dict(native_exit='0',tee_exit='0'),'mixed native/tee failure')
        row=audit_mixed((root/'B2-C1-w8.log').read_text(),'C1-w8',data['C1-w8']['steps']);row.pop('schedule_digests')
        result['B2-mixed-C1-w8']=row
        return dict(source=launch['source'],scope='phase-wise mixed-M coalesced decode (4 kernels/step), bitwise-identical to frozen per-M; explicit frozen weight-layer1 proxy',rows=result,no_promotion=True)
    if mode=='synthetic':
        need(data['synthetic-all-M1']['steps']==[dict(layer=1,hist={'1':256})],'synthetic M1 is not the frozen all256 shape')
        result['B2-frozen-M1']=one('B2-frozen-M1','synthetic-all-M1','B2',True)
    for case in cases:
        if mode=='f5':
            a=one('B2-'+case,case,'B2',prefill=True);a['F5_report_only']=True;a['F5_line_GBps']=163.8;a['F5_at_or_above_line']=a['effective_GBps']>=163.8
            a['native_plan_policy']='one native plan; unchanged tile1 for M1 or tile8/grid-z for M16/M64; cycled8-row activation prefixes'
            a['scheduled_bytes_scope']='host-plan expert weight occurrences, not tile rereads or physical DRAM traffic';result['B2-'+case]=a;continue
        a=one('B2-'+case,case,'B2');b=one('B1-'+case,case,'B1-scale-both')
        need(a['schedule_digests']==b['schedule_digests'],'B2/B1 did not execute identical schedules')
        result['B2-'+case]=a;result['B1-'+case]=b
    if mode=='synthetic':need(.95<=result['B2-synthetic-all-M1']['effective_GBps']/result['B2-frozen-M1']['effective_GBps']<=1.05,'same-run M1 order/control discrepancy')
    for value in result.values():value.pop('schedule_digests')
    if mode=='real' and 'C1-w8' in cases:
        row=result['B2-C1-w8'];need(row['steps']==512,'R18 requires all512 gate steps')
        row['R18_line']=r18_line(row['effective_GBps'])
        row['R18_thresholds_GBps']={'pass':191.1,'stop_below':163.8}
        row['scope_condition']='explicit single frozen weight-layer routing-shape replay; no full-model residency claim'
    return dict(source=launch['source'],scope='whole-step expert-compute; explicit frozen weight-layer proxy; not end-to-end/full-model residency',rows=result,no_promotion=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('receipt',type=Path,nargs='?');p.add_argument('output',type=Path,nargs='?');a=p.parse_args();selftest()
    if not a.selftest:
        if a.receipt is None or a.output is None:p.error('receipt/output required')
        result=receipt(a.receipt)
        with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
        print('STEP REPORT PASS '+json.dumps({k:v['effective_GBps'] for k,v in result['rows'].items()}))
