#!/usr/bin/env python3
"""One-shot exact-shape prebuilt vLLM FA2 reference; no JIT/install/fallback."""
import math
import statistics
import sys
import json
import time

CONTEXTS = (131072, 1048576)
RESERVE = 4 * 1024**3


def unique_bytes(s, element_bytes):
    if s <= 0 or element_bytes not in (1, 2):
        raise ValueError('invalid C5 shape/dtype')
    return s * 4 * (192 + 128) * element_bytes


def timing(s, element_bytes, samples):
    if len(samples) != 7 or any(not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError('invalid C5 samples')
    ms = statistics.median(samples)
    return dict(median_ms=ms, unique_KV_bytes=unique_bytes(s, element_bytes),
                unique_KV_GBs=unique_bytes(s, element_bytes)/(ms*1e6))


def coordinate_matches(reference, actual):
    # Vendor-lattice synthetic sanity, not a universal R7 guarantee. Reject a
    # zero-output impostor at long contexts where outputs are much below .002.
    return (math.isfinite(reference) and math.isfinite(actual)
            and abs(reference-actual) <= max(2e-5, .02*abs(reference)))


def selftest():
    assert unique_bytes(131072, 1) == 167772160
    assert unique_bytes(1048576, 1) == 1342177280
    assert unique_bytes(131072, 2) == 335544320
    assert unique_bytes(1048576, 2) == 2684354560
    assert timing(131072, 1, [2.]*7)['unique_KV_GBs'] == 83.88608
    for bad in ([0.]*7, [float('nan')]*7, [1.]*6):
        try: timing(131072, 1, bad)
        except ValueError: pass
        else: raise AssertionError('accepted invalid timing')
    no_process_build('subprocess.Popen', ('/sbin/ldconfig', ['/sbin/ldconfig', '-p']))
    no_process_build('subprocess.Popen', ('uname', ['uname', '-p']))
    denied = [('subprocess.Popen', (exe, argv)) for exe, argv in (
        ('/sbin/ldconfig', ['/sbin/ldconfig']), ('/sbin/ldconfig', ['/sbin/ldconfig','-p','-v']),
        ('/tmp/ldconfig', ['/tmp/ldconfig','-p']), ('nvcc', ['nvcc','x.cu']),
        ('ninja', ['ninja']), ('/bin/sh', ['/bin/sh','-c','ldconfig -p']),
        ('uname', ['uname', '--version']))]
    denied += [('os.system', ('ldconfig -p',)), ('os.posix_spawn', ('nvcc', ['nvcc']))]
    for event, args in denied:
        try: no_process_build(event, args)
        except RuntimeError: pass
        else: raise AssertionError('allowed a mutating/unknown subprocess')
    assert coordinate_matches(.001, .00101)
    assert not coordinate_matches(.001, 0.)
    assert not coordinate_matches(float('nan'), .001)
    print('RESULT: PASS C5 accounting/guard selftest 22/22; no vendor/GPU claim')


def no_process_build(event, args):
    # ctypes' library discovery prints the existing linker cache. This exact
    # read-only invocation does not rebuild it; all toolchains/shells stay denied.
    if (event == 'subprocess.Popen' and args[0] in ('/sbin/ldconfig', '/usr/sbin/ldconfig')
            and args[1] == [args[0], '-p']):
        return
    if (event == 'subprocess.Popen' and args[0] in ('uname', '/bin/uname', '/usr/bin/uname')
            and args[1] == [args[0], '-p']):
        return
    if event in ('subprocess.Popen', 'os.system', 'os.posix_spawn', 'os.exec'):
        raise RuntimeError(f'C5 no-build policy refuses process execution: {event} command={args[:2]!r}')


def run(import_only=False):
    # The chosen entry point dispatches a packaged shared library. Deny builds,
    # shells and unknown subprocesses; allow only exact read-only host probes.
    sys.addaudithook(no_process_build)
    import torch
    from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func
    if import_only:
        print('C5_IMPORT_PASS torch='+torch.__version__+' vllm prebuilt interface imported; no GPU availability claim', flush=True)
        return
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    p = torch.cuda.get_device_properties(0)
    if (p.major, p.minor, p.multi_processor_count) != (12, 0, 170) or '5090' not in p.name:
        raise RuntimeError(f'C5 identity mismatch: {p}')
    print('C5_IDENTITY', p.name, 'sm_120 sms=170', 'torch='+torch.__version__, flush=True)
    print('C5_SCOPE T=1 Q=BF16 P=vendor-native-BF16 QK=192 V=128 Hq=64 Hkv=4 page=256 reverse_pages=1 cached_V=prescaled reference_only=1 R7_claim=0', flush=True)

    def cell(s, dtype):
        bpe = torch.empty((), dtype=dtype).element_size()
        free, total = torch.cuda.mem_get_info()
        # Include conversion, FP64-coordinate and graph scratch headroom.
        need = unique_bytes((s+255)//256*256, bpe) + 8*1024**3
        if free - need < RESERVE:
            raise RuntimeError(f'C5 reserve refusal: free={free} planned={need}')
        print(f'C5_MEMORY free_bytes={free} total_bytes={total} planned_bytes={need} reserve_bytes={RESERVE}', flush=True)
        torch.manual_seed(501)
        pages = (s+255)//256
        q = torch.empty((1,64,192), device='cuda', dtype=torch.bfloat16).normal_(0,.25)
        k = torch.empty((pages,256,4,192), device='cuda', dtype=torch.bfloat16).uniform_(-1.875,1.875).to(dtype)
        v = torch.empty((pages,256,4,128), device='cuda', dtype=torch.bfloat16).uniform_(-1.875,1.875).mul_(.707).to(dtype)
        table = torch.arange(pages-1,-1,-1,device='cuda',dtype=torch.int32).reshape(1,-1)
        cuq = torch.tensor([0,1],device='cuda',dtype=torch.int32)
        lens = torch.tensor([s],device='cuda',dtype=torch.int32)
        def invoke():
            return flash_attn_varlen_func(q,k,v,max_seqlen_q=1,cu_seqlens_q=cuq,
                max_seqlen_k=s,seqused_k=lens,block_table=table,softmax_scale=192**-.5,
                causal=False,fa_version=2)
        out = invoke()
        torch.cuda.synchronize()
        if tuple(out.shape) != (1,64,128) or out.dtype != torch.bfloat16 or not torch.isfinite(out).all().item():
            raise RuntimeError('C5 output shape/finite check failed; no head padding allowed')
        if s == 257:
            # Independent full FP64 CPU reference also exercises a partial final
            # page under reverse mapping. Vendor BF16-P tolerance, NOT our R7.
            order = torch.tensor([p*256+j for p in range(pages-1,-1,-1) for j in range(256)][:s])
            kc = k.float().cpu().reshape(-1,4,192)[order].double()
            vc = v.float().cpu().reshape(-1,4,128)[order].double()
            qc = q.cpu().double()[0]
            expected = torch.stack([torch.softmax(kc[:,h//16] @ qc[h] * 192**-.5,0) @ vc[:,h//16] for h in range(64)])
            err = (out[0].cpu().double()-expected).abs().max().item()
            if not math.isfinite(err) or err > .002: raise RuntimeError(f'C5 full reference mismatch {err}')
            print(f'C5_NUMERICAL dtype={dtype} S=257 full_values=8192 max_abs_error={err} tolerance=.002 scope=vendor-only', flush=True)
            return
        # All tokens are visible and S is page-aligned: page permutation leaves
        # the full-attention reference invariant. Verify three FP64 coordinates.
        for h,d in ((0,0),(31,127),(63,64)):
            kk = k.reshape(s,4,192)[:,h//16].double()
            logits = kk @ q[0,h].double() * 192**-.5
            ref = torch.softmax(logits,0) @ v.reshape(s,4,128)[:,h//16,d].double()
            reference, actual = ref.item(), out[0,h,d].item()
            err = abs(reference-actual)
            if not coordinate_matches(reference, actual):
                raise RuntimeError(f'C5 FP64 coordinate mismatch {h}/{d}: {err}')
            del kk, logits, ref
        for _ in range(3): invoke()
        torch.cuda.synchronize()
        # Amortize Python/event submission using 16 native calls per graph.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(16): invoke()
        for _ in range(3): graph.replay()
        torch.cuda.synchronize()
        samples=[]
        for _ in range(7):
            a,b = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record(); graph.replay(); b.record(); b.synchronize()
            samples.append(a.elapsed_time(b)/16)
        result = timing(s,bpe,samples)
        result.update(S=s, KV=str(dtype), Q='BF16', backend='vllm-prebuilt-FA2',
                      graph_calls=16, samples_ms=samples, reference_only=True,
                      requires_roof_review=result['unique_KV_GBs']>1790)
        print('C5_METRIC '+json.dumps(result,sort_keys=True),flush=True)

    for dtype in (torch.bfloat16, torch.float8_e4m3fn):
        pending = list(CONTEXTS)
        try:
            cell(257,dtype)
            for s in CONTEXTS:
                cell(s,dtype)
                pending.remove(s)
                torch.cuda.empty_cache()
        except (RuntimeError, NotImplementedError, AssertionError) as e:
            print('C5_NOT_MEASURED '+json.dumps(dict(KV=str(dtype),contexts=pending,reason=str(e),exception=type(e).__name__)),flush=True)
            # A CUDA-context failure is not safe to probe again in this process.
            if 'CUDA error' in str(e): return
            torch.cuda.empty_cache()


if __name__ == '__main__':
    if '--selftest' in sys.argv:
        selftest()
    else:
        started=time.monotonic()
        status=0
        try: run(import_only='--import-probe' in sys.argv)
        except Exception as e:
            status=2
            print('C5_INCOMPLETE '+json.dumps(dict(reason=str(e),exception=type(e).__name__)),flush=True)
        print(f'C5_COMPLETE elapsed_seconds={time.monotonic()-started:.3f} reference_only=1 no_promotion=1 exit_status={status}',flush=True)
        sys.exit(status)
