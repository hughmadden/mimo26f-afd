#!/usr/bin/env python3
"""CPU isolation/range checks for the separately named R7 stress cohort."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import oracle_driver as oracle


class HighValueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = oracle.tc_decode_cases()
        cls.high = oracle.tc_highv_cases()

    def test_separate_names(self):
        self.assertEqual(len(self.high), 8)
        self.assertEqual(len({c.name for c, _ in self.original+self.high}), 16)

    def test_original_checks_unchanged(self):
        for (before, tensors), (after, again) in zip(self.original, oracle.tc_decode_cases()):
            self.assertEqual((before.name, before.tol_abs, before.tol_rel), (after.name, 1e-5, 0))
            for key in tensors:
                np.testing.assert_array_equal(tensors[key][1], again[key][1])

    def test_only_cached_v_changes(self):
        for (_, before), (_, after) in zip(self.original, self.high):
            self.assertEqual(before.keys(), after.keys())
            for key in before:
                if key != "v_codes":
                    np.testing.assert_array_equal(before[key][1], after[key][1])

    def test_decoded_magnitudes_and_r7_bounds(self):
        for i, (case, tensors) in enumerate(self.high):
            v = oracle.fb.decode_e4m3(tensors["v_codes"][1]).astype(np.float64)
            self.assertTrue(np.all(np.isfinite(v)))
            magnitude = 224 if i < 4 else 448
            self.assertEqual(np.max(np.abs(v)), magnitude)
            self.assertEqual(case.tol_abs, 2e-5*magnitude)
            self.assertEqual(case.tol_rel, 0)


if __name__ == "__main__":
    unittest.main()
