import csv
import io
import tempfile
import unittest
from pathlib import Path

from p1_profile_report import REQUIRED, summarize


def raw_csv(mode):
    buf = io.StringIO()
    names = ["Kernel Name", "Grid Size", "Block Size"]
    units = ["", "", ""]
    values = [f"prefill_tc<{'0' if mode == 'bf16q' else '1'}>", "(512, 4, 1)", "(256, 1, 1)"]
    for group in REQUIRED.values():
        for i, key in enumerate(group):
            names.append(key); units.append("unit"); values.append(str(10.0 + i))
    w = csv.writer(buf); w.writerow(names); w.writerow(units); w.writerow(values)
    return buf.getvalue()


class P1ProfileReportTests(unittest.TestCase):
    def write(self, mode, text):
        self.dir = Path(tempfile.mkdtemp())
        (self.dir / mode / f"ncu-raw-{mode}").mkdir(parents=True)
        (self.dir / mode / f"ncu-raw-{mode}" / "RESULT.md").write_text(text)

    def write(self, text):
        path = Path(tempfile.mktemp(suffix=".csv"))
        path.write_text(text)
        return path

    def test_summarize_valid(self):
        for mode in ("f32q", "bf16q"):
            r = summarize(self.write(raw_csv(mode)))
            self.assertEqual(r["mode"], mode)
            self.assertEqual(r["available"], sum(len(g) for g in REQUIRED.values()))
            self.assertEqual(r["missing"], [])

    def test_wrong_kernel(self):
        with self.assertRaises(ValueError):
            summarize(self.write(raw_csv("f32q").replace("prefill_tc<1>", "decode_pipe<8,1>")))

    def test_wrong_grid(self):
        with self.assertRaises(ValueError):
            summarize(self.write(raw_csv("f32q").replace("(512, 4, 1)", "(85, 4, 1)")))

    def test_missing_counter(self):
        rows = list(csv.reader(raw_csv("f32q").splitlines()))
        i = rows[0].index("dram__bytes.sum"); rows[2][i] = "nan"
        buf = io.StringIO()
        csv.writer(buf).writerows(rows)
        r = summarize(self.write(buf.getvalue()))
        self.assertIsNone(r["metrics"]["traffic"]["dram__bytes.sum"])
        self.assertIn("dram__bytes.sum", r["missing"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
