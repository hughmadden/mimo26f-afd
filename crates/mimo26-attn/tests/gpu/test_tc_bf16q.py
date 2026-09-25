"""CPU tests for explicitly separate post-RoPE BF16-Q oracle inputs."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import oracle_driver as oracle


class Bf16QueryTests(unittest.TestCase):
    def test_rne_ties_even_not_truncate(self):
        q = np.array([1+1/256, 1+3/256, -1-1/256, -1-3/256], dtype=np.float32)
        np.testing.assert_array_equal(oracle.bf16_rne(q), [1, 1+1/64, -1, -1-1/64])

    def test_signed_zero_and_idempotence(self):
        q = np.array([0., -0., .125, -.5, 448], dtype=np.float32)
        np.testing.assert_array_equal(oracle.bf16_rne(q).view(np.uint32), q.view(np.uint32))

    def test_nonfinite_refuses(self):
        for x in [np.inf, -np.inf, np.nan]:
            with self.assertRaises(ValueError): oracle.bf16_rne([x])

    def test_separate_names_and_raw_inputs_unchanged(self):
        for high in [False, True]:
            original = oracle.tc_highv_cases() if high else oracle.tc_decode_cases()
            native = oracle.tc_bf16q_cases(high)
            self.assertEqual(len(native), 8)
            self.assertTrue(set(c.name for c, _ in original).isdisjoint(c.name for c, _ in native))
            for (c, tensors), (n, inputs) in zip(original, native):
                self.assertEqual((c.tol_abs, c.tol_rel), (n.tol_abs, n.tol_rel))
                for key in tensors: np.testing.assert_array_equal(tensors[key][1], inputs[key][1])

    def test_rounding_is_actually_exercised(self):
        cases = oracle.tc_bf16q_cases()
        q = next(t["q"][1] for c, t in cases if c.name.endswith("q_low"))
        self.assertTrue(np.any(q != oracle.bf16_rne(q)))

    def test_original_precision_fixtures_not_renamed(self):
        oracle.tc_bf16q_cases()
        self.assertIn("tc_q_tail", [c.name for c, _ in oracle.tc_decode_cases()])


if __name__ == "__main__": unittest.main()
