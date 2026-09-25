"""Synthetic static-parser fixtures, not hardware receipts."""
import unittest
from p1_codegen_report import inspect

def fixture():
    sass='';build=''
    for q in (0,1):
        name=f'_test_prefill_tcILb{q}ELb0EEv'
        build+=f'Function properties for {name}\n    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\nptxas info    : Used 124 registers, used 1 barriers\n'
        ops=['BAR.SYNC 0x0;']*8+['BAR.RED.POPC R0, 0x0;']
        ops += [f'HMMA.16816.F32.BF16 R{4*(i%(22 if q else 18))}, R0, R2, R4;' for i in range(52 if q else 28)]
        sass+=f'\nFunction : {name}\n'+''.join(f' /*{i*16:04x}*/ {op} /* 0x0 */\n' for i,op in enumerate(ops))
    # Trial-1 split variant (f32q P1M=32): resource-budget-only entry, no ops.
    name='_test_prefill_tcILb1ELb1EEv'
    build+=f'Function properties for {name}\n    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\nptxas info    : Used 124 registers, used 1 barriers\n'
    sass+=f'\nFunction : {name}\n'
    return sass,build

class Tests(unittest.TestCase):
    def test_both(self):self.assertEqual(len(inspect(*fixture())[0]),3)
    def test_split_present(self):
        rows,_=inspect(*fixture())
        self.assertEqual({r['query'] for r in rows},{'f32q','bf16q','f32q-split'})
    def reject(self,s=None,b=None):
        a,c=fixture()
        with self.assertRaises(AssertionError):inspect(a if s is None else s,c if b is None else b)
    def test_spill(self):self.reject(b=fixture()[1].replace('0 bytes spill stores','4 bytes spill stores'))
    def test_registers(self):self.reject(b=fixture()[1].replace('124 registers','129 registers'))
    def test_missing_variant(self):self.reject(s=fixture()[0].replace('prefill_tcILb1','otherILb1'))
    def test_local(self):self.reject(s=fixture()[0].replace('BAR.SYNC','LDL',1))
    def test_async(self):self.reject(s=fixture()[0].replace('BAR.SYNC','LDGSTS',1))
    def test_mma(self):self.reject(s=fixture()[0].replace('HMMA.','FFMA.',1))
    def test_barrier(self):self.reject(s=fixture()[0].replace('BAR.RED','BAR.MISSING'))
    def test_missing_ptxas(self):self.reject(b='')
    def test_accumulators(self):
        import re
        self.reject(s=re.sub(r'(HMMA\.\S+ )R\d+',r'\1R0',fixture()[0]))

if __name__=='__main__':unittest.main()
