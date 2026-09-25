#!/usr/bin/env python3
"""P1 input/schema isolation tests; external numerical oracle remains unchanged."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import oracle_driver as oracle

class Tests(unittest.TestCase):
    def test_original_eight_unchanged(self):
        for cohort in ('tc-decode','tc-highv','tc-bf16q','tc-bf16q-highv'):
            before=oracle.gen_cases([cohort]);after=oracle.gen_cases([cohort],p1=True)
            self.assertEqual(len(before),8);self.assertEqual(len(after),10)
            for (a,x),(b,y) in zip(before,after):
                self.assertEqual(vars(a),vars(b))
                for key in x:np.testing.assert_array_equal(x[key][1],y[key][1])
    def test_multitile_and_nonmonotonic(self):
        for c,ts in oracle.tc_decode_cases(p1=True)[8:]:
            self.assertEqual(ts['q'][1].shape,(17,64,192))
            qp=ts['q_pos'][1];self.assertTrue(np.any(np.diff(qp)<0))
            self.assertEqual(ts['k_codes'][1].shape,(273,(4 if c.family=='ga' else 8)*192))
            self.assertEqual(c.tol_abs,1e-5);self.assertEqual(c.tol_rel,0)
    def test_highv_exact_bound(self):
        for c,ts in oracle.tc_highv_cases(p1=True):
            v=oracle.fb.decode_e4m3(ts['v_codes'][1]);self.assertTrue(np.all(np.isfinite(v)))
            self.assertEqual(c.tol_abs,2e-5*max(1,float(np.max(np.abs(v)))))
    def test_native_unrounded_inputs(self):
        plain=oracle.tc_decode_cases(p1=True);native=oracle.tc_bf16q_cases(p1=True)
        for (a,x),(b,y) in zip(plain,native):
            self.assertEqual(b.name,'tc_bf16q_'+a.name[3:])
            np.testing.assert_array_equal(x['q'][1],y['q'][1])
        self.assertTrue(np.any(native[-1][1]['q'][1]!=oracle.bf16_rne(native[-1][1]['q'][1])))
    def test_scope(self):
        for selection in (['tiny'],['tc-decode','tc-highv'],[]):
            with self.assertRaises(ValueError):oracle.gen_cases(selection,p1=True)

if __name__=='__main__':unittest.main()
