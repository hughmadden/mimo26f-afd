import csv
import io
import unittest
from ncu_report import REQUIRED, summarize


def fixture(kernel="void decode_pipe<8>()"):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Kernel Name", "Grid Size", "Block Size", *REQUIRED])
    writer.writerow(["", "", "", *["%" for _ in REQUIRED]])
    writer.writerow([kernel, "(85, 4, 1)", "(256, 1, 1)", *["1" for _ in REQUIRED]])
    columns = ["Address", "Source", "stall_wait", "stall_barrier", "stall_short_sb", "stall_long_sb", "L1WavefrontsSharedExcessive"]
    width = 32
    source = " ".join("-"*width for _ in columns)+"\n"
    source += " ".join(c.ljust(width) for c in columns)+"\n"
    for pc, count in (("0x100", "1"), ("0x110", "9")):
        source += " ".join(c.ljust(width) for c in [pc, "STS [R0], R1", *[count]*5])+"\n"
    return output.getvalue(), source


class NcuTests(unittest.TestCase):
    def test_valid_one_kernel_and_pc_ranking(self):
        report = summarize(*fixture())
        self.assertEqual(report["top_source_counters"]["stall_wait"][0]["pc_offset"], "0x10")
        self.assertEqual(report["metrics"][REQUIRED[0]]["value"], "1")

    def test_explicit_f32q_template(self):
        for value in ('true', '1'):
            self.assertIn(value, summarize(*fixture(f"void decode_pipe<8,{value}>()"))["kernel"])

    def test_bf16q_cannot_replace_requested_f32q_profile(self):
        for value in ('false', '0'):
            with self.assertRaisesRegex(ValueError, "requested FP32-Q"):
                summarize(*fixture(f"void decode_pipe<8,{value}>()"))

    def test_wrong_split(self):
        raw, src = fixture()
        with self.assertRaises(ValueError): summarize(raw.replace("85, 4, 1", "128, 4, 1"), src)

    def test_wrong_warps(self):
        raw, src = fixture()
        with self.assertRaises(ValueError): summarize(raw.replace("decode_pipe<8>", "decode_pipe<4>"), src)

    def test_missing_counter(self):
        raw, src = fixture()
        with self.assertRaises(ValueError): summarize(raw.replace(REQUIRED[0], "missing"), src)

    def test_nonfinite_counter(self):
        raw, src = fixture()
        with self.assertRaises(ValueError): summarize(raw.replace(",1", ",nan", 1), src)

    def test_multiple_kernels(self):
        raw, src = fixture()
        with self.assertRaises(ValueError): summarize(raw+raw.splitlines()[-1]+"\n", src)

    def test_truncated_csv(self):
        raw, src = fixture()
        with self.assertRaises(ValueError): summarize(raw.rsplit(",", 1)[0]+"\n", src)

    def test_missing_source_column(self):
        raw, src = fixture()
        with self.assertRaises(ValueError): summarize(raw, src.replace("stall_wait", "other_name"))


if __name__ == "__main__":
    unittest.main()
