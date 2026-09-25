#!/usr/bin/env python3
import unittest
from c1_codegen_report import inspect


def fixture(late=False,late_read=False):
    sass=build=''
    for residual in (0,1):
        name=f'_test_decode_c1ILb{residual}EEv'
        ops=[]
        for off in ('0xad40','0xad50','0xad60'):ops.append(f'SYNCS.EXCH.64 URZ, [UR0+{off}], UR1;')
        ops += ['BAR.SYNC.DEFER_BLOCKING 0x1, 0x40;','BAR.SYNC.DEFER_BLOCKING 0x2, 0x40;',
                'LDGSTS.E.BYPASS.128 [R0], [R2];','DEPBAR.LE SB0, 0x0;',
                'SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [R0+0xad60], R2;',
                'SYNCS.ARRIVE.TRANS64.A1T0 RZ, [R0+0xad40], RZ;',
                'SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [R0+0xad40], R2;']
        ops += [f'HMMA.16816.F32 R{4*(i%(6 if residual else 2))}, R40, R44, R0;' for i in range(36 if residual else 12)]
        ops += ['SYNCS.ARRIVE.TRANS64.A1T0 RZ, [R0+0xad50], RZ;',
                'SYNCS.PHASECHK.TRANS64.TRYWAIT P0, [R0+0xad50], R2;']
        pv=[f'HMMA.16816.F32 R{4*i}, R40, R44, R0;' for i in range(8)]
        ops += pv[:6] if late else pv
        ops += ['SYNCS.ARRIVE.TRANS64.A1T0 RZ, [R0+0xad60], RZ;']
        if late:ops+=pv[6:]
        if late_read:ops+=['LDS R0, [R1];']
        ops+=['@P0 BRA 0x0080;']
        ops += [f'SYNCS.CCTL.IV [R0+{off}];' for off in ('0xad40','0xad50','0xad60')]
        sass += f'\nFunction : {name}\n'+''.join(f'/*{i*16:04x}*/ {op} /* 0x0 */\n' for i,op in enumerate(ops))
        build += f'Function properties for {name}\n 0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\nptxas info : Used 71 registers, used 3 barriers\n'
    return sass,build


class TestC1(unittest.TestCase):
    def test_positive(self):
        results,_=inspect(*fixture());self.assertEqual([r['qk_mma'] for r in results],[12,36])
    def fails(self,sass,build):
        with self.assertRaises(AssertionError):inspect(sass,build)
    def test_late_register_math(self):
        results,_=inspect(*fixture(late=True));self.assertEqual([r['pv_mma'] for r in results],[8,8])
    def test_late_shared_read(self):
        self.fails(*fixture(late=True,late_read=True))
    def test_spill(self):
        s,b=fixture();self.fails(s,b.replace('0 bytes spill stores','4 bytes spill stores'))
    def test_missing_variant(self):
        s,b=fixture();self.fails(s.replace('decode_c1ILb1E','otherILb1E'),b)
    def test_duplicate(self):
        s,b=fixture();self.fails(s+s,b)
    def test_local(self):
        s,b=fixture();self.fails(s.replace('LDGSTS.E.BYPASS.128','LDL'),b)
    def test_mma(self):
        s,b=fixture();self.fails(s.replace('HMMA.16816.F32','FADD',1),b)
    def test_arrival(self):
        s,b=fixture();self.fails(s.replace('SYNCS.ARRIVE.','NOP.'),b)
    def test_subgroup(self):
        s,b=fixture();self.fails(s.replace('0x2, 0x40','0x2, 0x80'),b)
    def test_register_budget(self):
        s,b=fixture();self.fails(s,b.replace('71 registers','129 registers'))


if __name__=='__main__':unittest.main()
