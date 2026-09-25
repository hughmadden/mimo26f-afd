"""Independent E-W4A8-v1 FFN reference. Python harness only; no CUDA/Rust imports.

Dots use FP64; declared outputs and elementwise operations use FP32. This is an
ideal-accumulation lattice reference, not emulation of an MMA reduction tree.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from oracle.lattice import quant_v1 as Q
from spike.mxfp4 import unpack
from checkpoint_reader import tensor


def f32(value):
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(value, dtype=np.float32)
    if not np.isfinite(result).all():
        raise Q.NumericalFault("nonfinite FP32 boundary")
    return result


def activation(gate, up):
    """E-ACT-CR32-v1: scalar binary64 libm exp -> RN32, ordered FP32 ops.

    g >= 0: (g / (1 + exp(-g))) * u.
    g < 0: ((g * exp(g)) / (1 + exp(g))) * u.
    No branch evaluates a positive exponential; no clamp or BF16 conversion.
    """
    gate, up = f32(gate), f32(up)
    assert gate.shape == up.shape
    result = np.empty_like(gate)
    positive = gate >= 0
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        t = np.where(positive, -gate, gate)
        # R17 selects the exhaustively checked scalar libm reference, not a
        # vendor float32 exp or an unqualified vectorized float64 replacement.
        e = np.fromiter((math.exp(float(v)) for v in t.flat), dtype=np.float64,
                        count=t.size).astype(np.float32).reshape(t.shape)
        d = np.float32(1) + e
        result[positive] = gate[positive] / d[positive]
        result[~positive] = (gate[~positive] * e[~positive]) / d[~positive]
        result *= up
    return f32(result)


def expert(payload, scales, gate_w, up_w, down_w, mutation=""):
    assert mutation in ("", "drop_intermediate", "bf16_preround", "bf16_fc1", "clamp", "ftz")
    x = Q.decode_blocks(payload, scales)
    gw, uw, dw = [np.asarray(w, np.float64) for w in (gate_w, up_w, down_w)]
    assert x.ndim == 2 and gw.shape == uw.shape
    inter, hidden = gw.shape
    assert x.shape[1] == hidden and dw.shape == (hidden, inter) and inter % 128 == 0
    assert all(np.isfinite(w).all() for w in (gw, uw, dw))
    if mutation == "ftz":
        gw, uw, dw = [np.where(np.abs(w) < np.finfo(np.float32).tiny, np.copysign(0., w), w) for w in (gw, uw, dw)]
    gate = f32(x.astype(np.float64) @ gw.T)
    up = f32(x.astype(np.float64) @ uw.T)
    if mutation == "bf16_fc1":
        gate, up = Q.bf16_rne(gate), Q.bf16_rne(up)
    if mutation == "clamp":
        gate, up = np.minimum(gate, np.float32(10)), np.clip(up, -10, 10)
    h = activation(gate, up)
    quant_input = Q.bf16_rne(h) if mutation == "bf16_preround" else h
    mid_payload, mid_scales = Q.encode_blocks(quant_input)
    mid = Q.decode_blocks(mid_payload, mid_scales)
    if mutation == "drop_intermediate":
        mid = h
    local = inter // 4
    # Separately encode each contiguous rank interval. No cross-rank/K16 block.
    for r in range(4):
        sl = slice(r * local, (r + 1) * local)
        rp, rs = Q.encode_blocks(quant_input[:, sl])
        assert np.array_equal(rp, mid_payload[:, sl])
        assert np.array_equal(rs, mid_scales[:, r * local // 32:(r + 1) * local // 32])
    partial = np.stack([mid[:, r*local:(r+1)*local].astype(np.float64) @ dw[:, r*local:(r+1)*local].T for r in range(4)])
    full = mid.astype(np.float64) @ dw.T
    assert np.isfinite(partial).all() and np.isfinite(full).all()
    assert np.allclose(partial.sum(axis=0), full, atol=1e-11, rtol=1e-11)
    return dict(gate=gate, up=up, h=h, mid_payload=mid_payload, mid_scales=mid_scales,
                mid=mid, partial=partial, full=full)


def reduce_routes(partial, weights, mutation=""):
    assert mutation in ("", "weight_twice", "late_return", "per_route_return")
    raw = f32(partial)
    assert raw.ndim == 4 and raw.shape[0] == 4 and raw.shape[2] == 8
    weights = f32(weights)
    assert weights.shape == raw.shape[1:3]
    weighted = f32(raw * weights[None, :, :, None])
    if mutation == "weight_twice":
        weighted = f32(weighted * weights[None, :, :, None])
    if mutation == "per_route_return":
        weighted = Q.bf16_rne(weighted)
    rank = np.zeros((4, raw.shape[1], raw.shape[3]), np.float32)
    for slot in range(8):
        rank = f32(rank + weighted[:, :, slot])
    decoded = Q.bf16_rne(rank)
    final = np.zeros_like(rank[0])
    for r in range(4):
        final = f32(final + (rank[r] if mutation == "late_return" else decoded[r]))
    if mutation == "late_return":
        final = Q.bf16_rne(final)
    return rank, decoded, final


def selftest():
    rng = np.random.default_rng(0xB1AFD)
    x = rng.normal(size=(2, 32)).astype(np.float32)
    p, s = Q.encode_blocks(x)
    matrices = [rng.normal(size=shape).astype(np.float32) for shape in ((128, 32), (128, 32), (32, 128))]
    base = expert(p, s, *matrices)
    # Independent scalar dot (not another BLAS call) locks FP64 -> FP32 FC1.
    xd = Q.decode_blocks(p, s)
    scalar = np.array([[sum(float(xd[t,k])*float(matrices[0][n,k]) for k in range(32)) for n in range(128)] for t in range(2)], np.float32)
    assert np.array_equal(base["gate"], scalar)
    for flag in ("drop_intermediate", "bf16_preround", "bf16_fc1", "clamp"):
        wrong = expert(p, s, *matrices, mutation=flag)
        assert not np.allclose(wrong["partial"], base["partial"], atol=1e-5, rtol=1e-5), flag
        print(f"LATTICE NEGATIVE PASS {flag}")
    # FTZ matters even with v1's floor: a subnormal weight times a large finite
    # wire activation produces a normal FC1 result, then a nonzero intermediate.
    large = np.zeros((1, 32), np.float32); large[0,0] = np.float32(2.**100)
    tiny = np.zeros((128, 32), np.float32); tiny[:,0] = np.float32(2.**-127)
    uw = np.zeros_like(tiny); uw[:,0] = np.float32(2.**-73)
    dw = np.ones((32, 128), np.float32)
    ep, es = Q.encode_blocks(large)
    good = expert(ep, es, tiny, uw, dw)
    bad = expert(ep, es, tiny, uw, dw, mutation="ftz")
    assert np.all(good["full"] == 64) and np.all(bad["full"] == 0)
    print("LATTICE NEGATIVE PASS ftz (subnormal weight, normal FC1/product)")
    partial = np.stack([base["partial"] * (slot+1) / 16 for slot in range(8)], axis=2)
    weights = np.tile(np.arange(1,9,dtype=np.float32)/36, (2,1))
    rank, _, final = reduce_routes(partial, weights)
    expected = np.zeros_like(rank)
    for r in range(4):
        for t in range(2):
            for h in range(32):
                for j in range(8):
                    expected[r,t,h] = np.float32(expected[r,t,h] + np.float32(np.float32(partial[r,t,j,h])*weights[t,j]))
    assert np.array_equal(expected, rank)
    assert not np.array_equal(final, reduce_routes(partial, weights, "weight_twice")[2])
    # Non-homogeneous rank values separate rank-local rounding from late rounding.
    fixture = np.zeros((4,1,8,1), np.float64)
    fixture[:,0,0,0] = [1+2**-8, 1+2**-8, -1, 0]
    w = np.zeros((1,8), np.float32); w[0,0] = 1
    assert reduce_routes(fixture,w)[2].item() == 1
    assert reduce_routes(fixture,w,"late_return")[2].item() == 1+2**-7
    fixture.fill(0); fixture[0,0,:2,0] = [1+2**-8, 1+2**-8]; w[0,:2] = 1
    fixture[0,0,2,0] = -1; w[0,2] = 1
    assert reduce_routes(fixture,w)[2].item() != reduce_routes(fixture,w,"per_route_return")[2].item()
    fixture.fill(0); fixture[:,0,0,0] = [2**24,1,-2**24,1]
    w.fill(0); w[0,0] = 1
    assert reduce_routes(fixture,w)[2].item() == 1
    assert reduce_routes(fixture[[0,2,1,3]],w)[2].item() == 2
    print("LATTICE NEGATIVE PASS weight_twice, late_return, per_route_return, rank_order")
    g = np.array([-100., -0., 0., 100.], np.float32)
    a = activation(g, np.ones_like(g))
    assert np.isfinite(a).all() and a[0] < 0 and np.signbit(a[1]) and a[3] == 100
    for value in (np.nan, np.inf, -np.inf):
        try: activation(np.array([value]), np.ones(1))
        except Q.NumericalFault: pass
        else: raise AssertionError("nonfinite activation accepted")
    try: activation(np.array([np.finfo(np.float32).max]), np.array([2.]))
    except Q.NumericalFault: pass
    else: raise AssertionError("activation overflow accepted")
    print("LATTICE SELFTEST PASS: scalar FC1, contiguous TP4/K32, FP32 boundaries, nine FFN/wire mutations, stable exp and faults; CPU synthetic only")


def real(root, out):
    out.mkdir(parents=True, exist_ok=False)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    pinned = {b["name"]: b for b in json.loads((REPO / "bench/fixtures/expert_nibble_fixture.json").read_text())["blocks"]}
    experts = [0,7,255,1,2,3,4,5]
    x = np.random.default_rng(0x26AFD).uniform(-.5,.5,(8,4096)).astype(np.float32)
    xp, xs = Q.encode_blocks(x)
    weights = np.tile(np.arange(1,9,dtype=np.float32)/36, (8,1))
    stage = {name: np.empty((4,8,8,512), np.float32) for name in ("gate","up","h","mid")}
    stage["mid_payload"] = np.empty((4,8,8,512), np.uint8)
    stage["mid_scales"] = np.empty((4,8,8,16), np.uint8)
    partial = np.empty((4,8,8,4096), np.float64)
    full = np.empty((8,8,4096), np.float64)
    blocks = []
    for slot, eid in enumerate(experts):
        matrices = []
        for projection in ("gate_proj","up_proj","down_proj"):
            name = f"model.layers.1.mlp.experts.{eid}.{projection}"
            n,k = (4096,2048) if projection == "down_proj" else (2048,4096)
            w,wh = tensor(root,index,name+".weight",[n,k//2])
            s,sh = tensor(root,index,name+".weight_scale",[n,k//32])
            if eid in (0,7,255):
                assert wh == pinned[name]["weight_sha256"] and sh == pinned[name]["scale_sha256"]
            blocks.append(dict(name=name,weight_sha256=wh,scale_sha256=sh))
            matrices.append(unpack(w,s,naive=False).astype(np.float64))
        result = expert(xp,xs,*matrices)
        for m in (1,2,4):
            prefix = expert(xp[:m],xs[:m],*matrices)
            for name in ("gate","up","h","mid","mid_payload","mid_scales"):
                assert np.array_equal(prefix[name],result[name][:m]), (eid,m,name)
            assert np.allclose(prefix["partial"],result["partial"][:,:m],atol=1e-11,rtol=1e-11)
        partial[:,:,slot] = result["partial"]; full[:,slot] = result["full"]
        for name, dest in stage.items():
            dest[:,:,slot] = result[name].reshape(8,4,-1).transpose(1,0,2)
        print(f"LATTICE REAL expert={eid} full/TP4 identity, rank-local K32 bytes, M 1/2/4/8 prefix identity PASS", flush=True)
    rank, decoded, wire = reduce_routes(partial,weights)
    artifacts = {}
    def save(name, array, dtype):
        array = np.asarray(array,dtype=dtype)
        assert np.isfinite(array).all()
        path = out / name; array.tofile(path)
        artifacts[name] = dict(shape=list(array.shape),dtype=array.dtype.str,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    save("b1-x.f32",x,"<f4"); save("b1-x-payload.u8",xp,"u1"); save("b1-x-scales.u8",xs,"u1")
    save("b1-weights.f32",weights,"<f4")
    for name,array in stage.items(): save("b1-"+name+(".u8" if array.dtype==np.uint8 else ".f32"),array,"u1" if array.dtype==np.uint8 else "<f4")
    save("b1-partial.f64",partial,"<f8")
    save("b1-full.f64",(full*weights[:,:,None].astype(np.float64)).sum(axis=1),"<f8")
    save("b1-rank.f32",rank,"<f4"); save("b1-return-decoded.f32",decoded,"<f4"); save("b1-wire.f32",wire,"<f4")
    sources = [Path(__file__), Path(Q.__file__), REPO/"spike/mxfp4.py", Path(__file__).with_name("checkpoint_reader.py")]
    manifest = dict(lattice="E-W4A8-v1",quantizer=Q.VERSION,layer=1,experts=experts,
                    accumulation="FP64 ideal dots, FP32 declared boundaries; slot 0..7, rank 0..3",
                    wire="DS41RTE3 v3 numerical BF16 seam only, not codec/transport execution",
                    blocks=blocks,artifacts=artifacts,
                    reference_sources={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (out/"b1-source.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("LATTICE REAL ORACLE PASS: E-W4A8-v1, 8 experts, 4 ranks, 8 tokens, 16 K32 blocks/rank, 48 source hashes (18 pre-pinned); CPU reference only",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest",action="store_true")
    parser.add_argument("--real",nargs=2,metavar=("WEIGHTS","OUTPUT"))
    args = parser.parse_args()
    if args.selftest and args.real: parser.error("choose one cell")
    if args.selftest: selftest()
    elif args.real: real(*map(Path,args.real))
    else: parser.error("choose --selftest or --real")
