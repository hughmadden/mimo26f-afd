"""Lossless routing-shape normalization, not reconstruction from marginal histograms."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def unique(pairs):
    out={}
    for k,v in pairs:
        if k in out:raise ValueError('duplicate JSON key')
        out[k]=v
    return out


def integer(x,lo,hi):
    if type(x) is not int or not lo<=x<=hi:raise ValueError('invalid bounded integer')
    return x


def normalize(data,weight_layer):
    integer(weight_layer,1,47)
    out={}
    for case,tokens in (('C1-w8',8),('C4-w8',32),('C16-w8',128)):
        records=data[case]
        if not isinstance(records,list) or not records:raise ValueError('actual per-step records required')
        steps=[]
        for row in records:
            layer=integer(row['layer'],1,47);ids=row['experts'];ms=row['M']
            if not isinstance(ids,list) or not isinstance(ms,list) or not 1<=len(ids)<=256 or len(ids)!=len(ms):raise ValueError('step extent')
            for e in ids:integer(e,0,255)
            for m in ms:integer(m,1,tokens)
            if len(set(ids))!=len(ids) or sum(ms)!=tokens*8:raise ValueError('distinct experts / top8 route count')
            hist=Counter(ms)
            steps.append(dict(layer=weight_layer,route_layer=layer,hist={str(m):hist[m] for m in sorted(hist)}))
        out[case]=dict(seed=20260924,steps=steps,scope='actual routing shapes; randomized expert identities; frozen real weight-layer proxy, not full-model residency')
    return out


def selftest():
    fixture={case:[dict(layer=29,experts=list(range(8)),M=[tokens]*8)] for case,tokens in (('C1-w8',8),('C4-w8',32),('C16-w8',128))}
    result=normalize(fixture,1)
    assert result['C1-w8']['steps']==[dict(layer=1,route_layer=29,hist={'8':8})]
    import copy
    negatives=0
    for mutation in ('duplicate','count','nonint','layer','length','aggregate'):
        bad=copy.deepcopy(fixture);row=bad['C1-w8'][0]
        if mutation=='duplicate':row['experts'][1]=row['experts'][0]
        elif mutation=='count':row['M'][0]=7
        elif mutation=='nonint':row['M'][0]=True
        elif mutation=='layer':row['layer']=0
        elif mutation=='length':row['experts'].pop()
        else:bad['C1-w8']={'hist':{'1':64}}
        try:normalize(bad,1)
        except (ValueError,TypeError,KeyError):negatives+=1
        else:raise AssertionError('unpowered trace negative '+mutation)
    assert negatives==6
    print('STEP TRACE HOST PASS actual joint shape preservation, explicit weight/source layers, six powered negatives')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('--weight-layer',type=int,default=1)
    p.add_argument('input',type=Path,nargs='?');p.add_argument('output',type=Path,nargs='?');a=p.parse_args();selftest()
    if not a.selftest:
        if a.input is None or a.output is None:p.error('input/output required')
        raw=a.input.read_bytes();result=normalize(json.loads(raw,object_pairs_hook=unique),a.weight_layer)
        for v in result.values():v['source_sha256']=hashlib.sha256(raw).hexdigest()
        with a.output.open('x') as f:json.dump(result,f,indent=1);f.write('\n')
        print('STEP TRACE NORMALIZED '+json.dumps({k:len(v['steps']) for k,v in result.items()}))
