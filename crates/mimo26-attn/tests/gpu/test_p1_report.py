#!/usr/bin/env python3
"""Synthetic receipt adversaries; never hardware performance evidence."""
import unittest
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import p1_report as p


def fixture(cell='p1-2k',passed=False,rate_override=None):
    s,nkv,window=p.CONTEXTS[cell];batch=2 if cell=='p1-1m' else 8
    needed=s*nkv*320+p.QCOUNT*4+p.COUNT*8+(s+2048)*8+(s//256)*4+batch*64*256*130*8+(2<<30)
    lines=['2026-09-24 07:00:00 AEST','a'*64+'  /var/tmp/mimo26f-attn/mimo26f-test/mimo26f-attn-bench-sm120',
           'IDENTITY gpu=NVIDIA GeForce RTX 5090 arch=sm_120 sms=170 baked_arch=sm_120 baked_sms=170 source=0123456789ab label=TARGET',
           'AOT: PASS architecture, SM-count, and baked kernel launch/readback',
           f'MEMORY free_bytes={30<<30} total_bytes={32<<30} requested_bytes={needed} reserve_bytes={4<<30}',
           f'P1_CONTEXT cell={cell} label=TARGET T=2048 S={s} n_q=64 n_kv={nkv} QK=192 V=128 window={window} sink={"per-Q-head" if window else "absent"} page_tokens=256 paged=reverse-256 Q_storage=f32 Q_values=BF16-exact K=E4M3-unit V_dtype=E4M3-unit cached_V=prescaled KV_abs_max=1.875 scope=bounded-synthetic mma_m=64 mma_n=16 warmup=3 samples=7 budget_s=480 unique_KV_bytes={s*nkv*320}',
           f'P1_QUERY checked={p.QCOUNT}/{p.QCOUNT} exact=PASS finite=PASS']
    lines += [f'P1_REFERENCE_PROGRESS queries={i}/2048 elapsed_s=1' for i in range(256,2049,256)]
    lines += [f'P1_REFERENCE outputs={p.COUNT}/{p.COUNT} finite=PASS baseline=scalar-f64-splitkv-reduce slab_queries={batch} splits=256 reuse=identical-BF16-exact-inputs elapsed_ms=1000 coordinates=3/3 max_coordinate_diff=1e-8']
    for mode in p.MODES:
        native=mode=='bf16q'
        lines += [f'P1_CORRECT mode={mode} reference={"bf16q-lattice-local" if native else "f32q"} checked={p.COUNT}/{p.COUNT} finite=PASS coordinates=3/3 max_baseline_diff=1e-8 max_coordinate_diff=1e-8 registers={124 if native else 125} shared_bytes={38976 if native else 88128} capacity_ctas={2 if native else 1}']
    for mode in p.MODES:
        native=mode=='bf16q';useful,executed=p.work(s,nkv,window,native)
        rate=rate_override or ((110 if native else 130) if passed else 10)
        ms=(useful if native else executed)/(rate*1e9)
        lines += [f'P1_SAMPLE mode={mode} index={i} ms={ms:.9f}' for i in range(7)]
        lines += [f'P1_METRIC mode={mode} median_ms={ms:.9f} min_ms={ms:.9f} max_ms={ms:.9f} useful_flops={useful} executed_mma_flops={executed} mma_work_factor={executed/useful:.9f} useful_TFLOPS={useful/(ms*1e9):.9f} executed_TFLOPS={executed/(ms*1e9):.9f} target_domain={"useful" if native else "executed"} target_TFLOPS={100 if native else 125.7} verdict={"PASS" if passed else "MISS"}']
    lines += [f'P1_COMPLETE modes=2 checked_per_mode={p.COUNT} samples_per_mode=7','RESULT: PASS P1 pair harness (performance verdicts separate, no promotion)']
    return '\n'.join(lines)

class Tests(unittest.TestCase):
    def reject(self,text):
        with self.assertRaises(ValueError):p.inspect(text,'p1-2k')
    def test_cli_status(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('MIMO26_ATTN_SELFTEST_DIR')) as tmp:
            path=Path(tmp)/'receipt.txt'
            for text,expected in ((fixture(passed=True),0),(fixture(),1),(fixture().replace('P1_QUERY','NO_QUERY'),2)):
                path.write_text(text)
                proc=subprocess.run([sys.executable,p.__file__,str(path),'--required','p1-2k'],capture_output=True,text=True)
                self.assertEqual(proc.returncode,expected,proc.stdout+proc.stderr)
    def test_all_contexts(self):
        for cell in p.CONTEXTS:self.assertEqual(p.inspect(fixture(cell),cell)['status'],'VALID')
    def test_wrong_1m_batch(self):
        with self.assertRaises(ValueError):p.inspect(fixture('p1-1m').replace('slab_queries=2','slab_queries=8'),'p1-1m')
    def test_old_1m_batch(self):
        with self.assertRaises(ValueError):p.inspect(fixture('p1-1m').replace('slab_queries=2','slab_queries=1'),'p1-1m')
    def test_wrong_short_batch(self):self.reject(fixture().replace('slab_queries=8','slab_queries=1'))
    def test_1m_reference_scratch_floor(self):
        text=fixture('p1-1m');needed=1048576*4*320+p.QCOUNT*4+p.COUNT*8+(1048576+2048)*8+(1048576//256)*4+2*64*256*130*8+(2<<30)
        with self.assertRaises(ValueError):p.inspect(text.replace(f'requested_bytes={needed}',f'requested_bytes={needed-1}'),'p1-1m')
    def test_separate_gates_no_extra_62_9(self):
        result=p.inspect(fixture(passed=True),'p1-2k');self.assertEqual([r['verdict'] for r in result['modes']],['PASS','PASS'])
        self.assertLess(result['modes'][0]['useful_TFLOPS'],62.9)
    def test_missing_mode(self):self.reject('\n'.join(l for l in fixture().splitlines() if 'mode=bf16q' not in l))
    def test_partial_reference(self):self.reject(fixture().replace(f'outputs={p.COUNT}/{p.COUNT}','outputs=8192/8192'))
    def test_partial_candidate(self):self.reject(fixture().replace(f'checked={p.COUNT}/{p.COUNT}','checked=8192/8192'))
    def test_no_query_proof(self):self.reject(fixture().replace('P1_QUERY','NO_QUERY'))
    def test_nonexact_input_scope(self):self.reject(fixture().replace('Q_values=BF16-exact','Q_values=random-f32'))
    def test_wrong_native_oracle(self):self.reject(fixture().replace('reference=bf16q-lattice-local','reference=f32q'))
    def test_duplicate(self):self.reject(fixture().replace('T=2048','T=2048 T=2048'))
    def test_duplicate_identity(self):self.reject(fixture().splitlines()[2]+'\n'+fixture())
    def test_no_sydney_timestamp(self):self.reject(fixture().replace('07:00:00 AEST','07:00:00 UTC'))
    def test_missing_binary_hash(self):self.reject(fixture().replace('a'*64,'b'*63))
    def test_instrumented(self):self.reject('==PROF==\n'+fixture())
    def test_no_aot(self):self.reject(fixture().replace('AOT: PASS','AOT: SKIP'))
    def test_proxy(self):self.reject(fixture().replace('label=TARGET','label=PROXY'))
    def test_dirty(self):self.reject(fixture().replace('source=0123456789ab','source=0123456789ab-dirty'))
    def test_reference_gap(self):self.reject(fixture().replace('queries=512/2048','queries=513/2048'))
    def test_samples(self):self.reject(fixture().replace('index=6','index=7'))
    def test_sample_early(self):self.reject(fixture().replace('P1_REFERENCE_PROGRESS queries=256','P1_SAMPLE mode=f32q index=0 ms=1\nP1_REFERENCE_PROGRESS queries=256'))
    def test_nan(self):self.reject(fixture().replace('max_baseline_diff=1e-8','max_baseline_diff=nan'))
    def test_zero_time(self):self.reject(fixture().replace('index=0 ms=','index=0 ms=0 fake='))
    def test_corrupt_error(self):self.reject(fixture().replace('max_coordinate_diff=1e-8','max_coordinate_diff=0.000021'))
    def test_n32_relabel(self):self.reject(fixture().replace('mma_n=16','mma_n=32'))
    def test_resource_scope(self):self.reject(fixture().replace('capacity_ctas=1','capacity_ctas=2'))
    def test_memory(self):self.reject(fixture().replace(f'reserve_bytes={4<<30}',f'reserve_bytes={3<<30}'))
    def test_allocation_budget(self):self.reject(fixture().replace('requested_bytes=', 'hidden_bytes='))
    def test_gate_domain(self):self.reject(fixture().replace('target_domain=executed','target_domain=useful'))
    def test_above_peak(self):self.reject(fixture(passed=True,rate_override=500))
    def test_lower_target(self):self.reject(fixture().replace('target_TFLOPS=125.7','target_TFLOPS=62.9'))
    def test_greenwashed(self):self.reject(fixture().replace('verdict=MISS','verdict=PASS'))
    def test_bad_work(self):
        u,e=p.work(2048,4,0,False);self.reject(fixture().replace(f'executed_mma_flops={e}',f'executed_mma_flops={u*2.6}'))
    def test_unknown(self):self.reject(fixture().replace('P1_COMPLETE','P1_OTHER'))
    def test_missing_completion(self):self.reject(fixture().replace('P1_COMPLETE','NOT_COMPLETE'))
    def test_after_completion(self):self.reject(fixture()+'\nP1_SAMPLE mode=f32q index=7 ms=1')
    def test_failure(self):self.reject(fixture()+'\nRESULT: INCOMPLETE stopped')
    def test_bad_median(self):self.reject(fixture().replace('median_ms=', 'lost_ms='))
    def test_nominal_factor(self):self.reject(fixture().replace('mma_work_factor=', 'missing_factor='))

if __name__=='__main__':unittest.main()
