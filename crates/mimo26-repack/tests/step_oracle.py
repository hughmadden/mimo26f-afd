"""Independent real-weight, all-expert references for R18. CPU harness only."""
import argparse
import hashlib
import json
from pathlib import Path
from lattice_oracle import REPO, Q, expert, activation, tensor, unpack, np


def selftest():
    g = np.array([-.19146597385406494,.327481746673584,-.1987401247024536,.23654770851135254],np.float32)
    u = np.array([-.17092028260231018,.14912837743759155,.5958439111709595,.2174588441848755],np.float32)
    want = np.array([.014801025390625,.02838134765625,-.0533447265625,.02874755673110485],np.float32)
    np.testing.assert_array_equal(activation(g,u).view(np.uint32),want.view(np.uint32))
    wrong_e = np.exp(np.where(g>=0,-g,g))
    wrong = np.where(g>=0,g/(np.float32(1)+wrong_e),(g*wrong_e)/(np.float32(1)+wrong_e))*u
    assert np.any(wrong.view(np.uint32)!=want.view(np.uint32))
    print('STEP ORACLE HOST PASS four R17 activation witnesses; legacy exp32 mutation detected')


def generate(root, out, layers):
    out.mkdir(parents=True,exist_ok=False)
    index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    pins={b['name']:b for b in json.loads((REPO/'bench/fixtures/expert_nibble_fixture.json').read_text())['blocks']}
    x=np.random.default_rng(0x26AFD).uniform(-.5,.5,(8,4096)).astype('<f4')
    xp,xs=Q.encode_blocks(x)
    artifacts={}
    def save(name,data):
        data.tofile(out/name)
        artifacts[name]=dict(bytes=data.nbytes,sha256=hashlib.sha256((out/name).read_bytes()).hexdigest())
    save('x.f32',x);save('x-payload.u8',xp);save('x-scales.u8',xs)
    blocks=[]
    for layer in layers:
        b2=np.empty((256,8,4096),'<f4');b1=np.empty((256,8,4096),'<f8')
        for eid in range(256):
            matrices=[]
            for proj in ('gate','up','down'):
                name=f'model.layers.{layer}.mlp.experts.{eid}.{proj}_proj'
                n,k=(4096,2048) if proj=='down' else (2048,4096)
                w,wh=tensor(root,index,name+'.weight',[n,k//2]);s,sh=tensor(root,index,name+'.weight_scale',[n,k//32])
                if name in pins:
                    assert (wh,sh)==(pins[name]['weight_sha256'],pins[name]['scale_sha256'])
                blocks.append(dict(name=name,weight_sha256=wh,scale_sha256=sh))
                matrices.append(unpack(w[:,:256],s[:,:16],naive=False).astype(np.float64) if proj=='down' else unpack(w[:512],s[:512],naive=False).astype(np.float64))
            gate=x.astype(np.float64)@matrices[0].T;up=x.astype(np.float64)@matrices[1].T
            # Frozen E-FP32 reference: same FP64 formula as tp4_oracle.py.
            h=gate/(1.+np.exp(-gate))*up
            b2[eid]=h@matrices[2].T
            b1[eid]=expert(xp,xs,*matrices)['full']
            assert np.isfinite(b2[eid]).all() and np.isfinite(b1[eid]).all()
            if eid%32==0:print(f'STEP ORACLE layer={layer} expert={eid}/256',flush=True)
        save(f'b2-L{layer}.f32',b2);save(f'b1-L{layer}.f64',b1)
    manifest=dict(scope='independent rank0 all256, M8 prefixes reused for exact native passes; no GPU claim',
                  activation='E-ACT-CR32-v1',layers=layers,experts=list(range(256)),blocks=blocks,artifacts=artifacts)
    (out/'source.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'STEP ORACLE PASS layers={layers} all256 experts, B2 frozen FP64 reference and B1 R17 reference',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--selftest',action='store_true');p.add_argument('--layers',default='1')
    p.add_argument('weights',type=Path,nargs='?');p.add_argument('output',type=Path,nargs='?');a=p.parse_args();selftest()
    if not a.selftest:
        if a.weights is None or a.output is None:p.error('weights and output required')
        layers=[int(v) for v in a.layers.split(',')]
        if not layers or len(set(layers))!=len(layers) or any(not 1<=v<=47 for v in layers):p.error('unique MoE layers1..47 required')
        generate(a.weights,a.output,layers)
