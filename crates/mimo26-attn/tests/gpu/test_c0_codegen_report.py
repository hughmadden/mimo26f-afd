"""Synthetic parser fixtures, never hardware evidence."""
import unittest
from c0_codegen_report import inspect


def fixture():
    sass, build = "", ""
    for w in (4, 8):
        for q in (0, 1):
            name = f"synthetic_decode_pipeILi{w}ELb{q}EEv"
            sass += f"\nFunction : {name}\n/*0000*/ LDGSTS.E.BYPASS.128 [R0], [R2]; /* 0 */\n"
            sets=(3 if q else 1)*2+(16//w)*4
            for i in range((3 if q else 1)*12+(16//w)*4):
                sass += f"/*{16*(i+1):04x}*/ HMMA.16816.F32.BF16 R{i%sets*4}, R4, R8, R{i%sets*4}; /* 0 */\n"
            build += f"ptxas info : Function properties for {name}\n    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\nptxas info : Used 92 registers\n"
    return sass, build


class CodegenTests(unittest.TestCase):
    def test_valid_four_variants(self):
        rows, _ = inspect(*fixture())
        self.assertEqual(len(rows), 4)
        self.assertEqual(sorted(r["static_mma_instructions"] for r in rows), [20,28,44,52])

    def test_spill_refuses(self):
        s, b = fixture()
        with self.assertRaises(AssertionError): inspect(s, b.replace("0 bytes spill stores", "4 bytes spill stores"))

    def test_local_memory_refuses(self):
        s, b = fixture()
        with self.assertRaises(AssertionError): inspect(s+"/*4000*/ LDL R0, [R2]; /* 0 */\n", b)

    def test_missing_counter_refuses(self):
        s, _ = fixture()
        with self.assertRaises(AssertionError): inspect(s, "")

    def test_wrong_mma_work_refuses(self):
        s, b = fixture()
        with self.assertRaises(AssertionError): inspect(s.replace("HMMA.", "NOOP.", 1), b)

    def test_single_chain_refuses(self):
        import re
        s, b = fixture()
        with self.assertRaises(AssertionError): inspect(re.sub(r"(HMMA\.\S+ )R\d+", r"\g<1>R0", s), b)

    def test_missing_async_refuses(self):
        s, b = fixture()
        with self.assertRaises(AssertionError): inspect(s.replace("LDGSTS.", "NOOP."), b)


if __name__ == "__main__": unittest.main()
