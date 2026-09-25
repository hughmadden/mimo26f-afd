import json
import unittest

import op1_schedule as op
from op1_report import parse


def build_receipt():
    m = op.load_manifest()
    lines = [
        "2026-09-24 10:00:00 AEST",
        "a"*64 + "  /work/mimo26f-attn-bench-sm120",
        "IDENTITY gpu=NVIDIA GeForce RTX 5090 arch=sm_120 sms=170 baked_arch=sm_120 baked_sms=170 source=123456abcdef label=TARGET",
        "AOT: PASS architecture, SM-count, and baked kernel launch/readback",
        "MEMORY free_bytes=16000000000 total_bytes=33711521792 requested_bytes=2147483648 reserve_bytes=4294967296",
    ]
    for mode in ("f32q", "bf16q"):
        native = mode == "bf16q"
        for family in ("p1", "c3"):
            c3 = family == "c3"
            lines.append(f"OP1_RESOURCE mode={mode} family={family} registers={(70 if native else 92) if c3 else (124 if native else 125)} shared={49536 if c3 else (38976 if native else 88128)} capacity={2 if c3 or native else 1}")
    lines.append(f"OP1_BEGIN d7_sha={m['inputs']['d7_sha256']} cases=2651 verification=2642 short_prefill=9 layers=48 ga=9 swa=39 modes=2 paths=ga-ver-c3p8,swa-ver-c3p4,ga-pre-c3p1,swa-pre-p1 timing=direct-48-layer-span scope=attention-critical-path gate=UNSET boundary=core-post-rope-prescaled-kv Q_values=BF16-exact")
    cases = op.cell_cases(m)
    for row in cases:
        cat = op.CATEGORIES[row["category"]]
        n = row["t"] * 64 * 128
        base = f"case={row['id']} category={cat} request={row['request']} step={row['step']} phase={row['phase']}"
        lines.append(f"OP1_REFERENCE {base} layers=48 checked={n*48} coordinates=48 query_checked={row['t']*64*192*48} query_exact=PASS reference_finite=PASS max_coordinate_diff=0.000000001")
        for mode in ("f32q", "bf16q"):
            lines.append(f"OP1_PRECHECK {base} mode={mode} layers=48 checked={n*48} max_error=0.000001000 finite=PASS")
        for mode in ("f32q", "bf16q"):
            for i in range(7):
                ms = (2.0 if row["phase"] else 1.0) / (1 if mode == "f32q" else 2)
                lines.append(f"OP1_SAMPLE {base} mode={mode} timing=core index={i} ms={ms:.9f} checked_layer={(row['id']+i)%48} max_error=0.000001000")
            lines.append(f"OP1_CHECK {base} mode={mode} layers=48 checked={n*48} max_error=0.000001000 finite=PASS")
    lines.append("OP1_COMPLETE verification_cases=2642 short_prefill_cases=9 cases=2651 samples=37114 all_layer_step_measured=1")
    lines.append("RESULT: PASS OP1 cell harness (no gate, no promotion)")
    return "\n".join(lines)


class OP1ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = build_receipt()

    def rejects(self, old, new):
        self.assertIn(old, self.text)
        with self.assertRaises((ValueError, KeyError)): parse(self.text.replace(old, new, 1))

    def test_valid_ungated_stats(self):
        result = parse(self.text)
        self.assertEqual(result["status"], "VALID")
        self.assertEqual(result["gate"], "UNSET")
        self.assertEqual(result["verification_cases"], 2642)
        self.assertEqual(result["short_prefill_cases"], 9)
        self.assertEqual(result["samples"], 37114)
        for mode in ("f32q", "bf16q"):
            stats = result["modes"][mode]
            for cat in op.CATEGORIES:
                self.assertAlmostEqual(stats["categories"][cat]["mean_ms"], 1.0 / (1 if mode == "f32q" else 2))
                self.assertAlmostEqual(stats["categories"][cat]["p95_ms"], 1.0 / (1 if mode == "f32q" else 2))
                self.assertAlmostEqual(stats["short_prefill_ms"][cat], 2.0 / (1 if mode == "f32q" else 2))
            self.assertAlmostEqual(stats["category_weighted_step_ms"], 1.0 / (1 if mode == "f32q" else 2))
            self.assertAlmostEqual(stats["pooled_step_ms_diagnostic"], 1.0 / (1 if mode == "f32q" else 2))

    def test_identity_aot(self):
        self.rejects("AOT: PASS", "AOT: absent")
        self.rejects("source=123456abcdef", "source=123456abcdef-dirty")
        self.rejects("arch=sm_120", "arch=sm_89")
        self.rejects("label=TARGET", "label=PROXY")

    def test_begin_scope(self):
        self.rejects("gate=UNSET", "gate=PASS")
        self.rejects("scope=attention-critical-path", "scope=full-model-latency")
        self.rejects("cases=2651", "cases=2650")
        self.rejects("timing=direct-48-layer-span", "timing=graph-amortized")
        self.rejects("boundary=core-post-rope-prescaled-kv", "boundary=end-to-end")

    def test_missing_or_wrong_rows(self):
        self.rejects("OP1_COMPLETE", "MISSING_COMPLETE")
        self.rejects("all_layer_step_measured=1", "all_layer_step_measured=0")
        # Drop one sample -> incomplete.
        lines = self.text.splitlines()
        i = next(i for i, x in enumerate(lines) if x.startswith("OP1_SAMPLE "))
        with self.assertRaises(ValueError): parse('\n'.join(lines[:i] + lines[i+1:]))
        # Drop one precheck.
        j = next(i for i, x in enumerate(lines) if x.startswith("OP1_PRECHECK "))
        with self.assertRaises(ValueError): parse('\n'.join(lines[:j] + lines[j+1:]))
        # Drop one reference.
        k = next(i for i, x in enumerate(lines) if x.startswith("OP1_REFERENCE "))
        with self.assertRaises(ValueError): parse('\n'.join(lines[:k] + lines[k+1:]))

    def test_numeric_rejections(self):
        self.rejects("ms=1.000000000", "ms=nan")
        self.rejects("ms=0.500000000", "ms=0.000000000")
        self.rejects("ms=1.000000000", "ms=-1.000000000")
        self.rejects("max_error=0.000001000", "max_error=0.001000000")
        self.rejects("max_coordinate_diff=0.000000001", "max_coordinate_diff=0.001000000")

    def test_wrong_category_or_phase(self):
        self.rejects("category=coding", "category=other")
        self.rejects("phase=0", "phase=2")

    def test_duplicate_or_out_of_order(self):
        lines = self.text.splitlines()
        i = next(i for i, x in enumerate(lines) if x.startswith("OP1_SAMPLE "))
        line = lines.pop(i)
        j = next(i for i, x in enumerate(lines) if x.startswith("OP1_SAMPLE "))
        lines.insert(j + 1, line)
        with self.assertRaises(ValueError): parse('\n'.join(lines))

    def test_wrong_d7(self):
        self.rejects("d7_sha=", "d7_sha=deadbeef")

    def test_resource_drift(self):
        self.rejects("registers=125", "registers=126")
        self.rejects("shared=49536", "shared=49537")

    def test_failure_retained(self):
        with self.assertRaises(ValueError): parse(self.text + '\nRESULT: INCOMPLETE budget')

    def test_missing_native_mode(self):
        lines = [x for x in self.text.splitlines() if "mode=bf16q" not in x]
        with self.assertRaises(ValueError): parse('\n'.join(lines))

    def test_fields_and_bounded_helpers(self):
        from op1_report import fields, bounded, expect
        self.assertEqual(fields("a=1 b=x c=1.5"), {"a": "1", "b": "x", "c": "1.5"})
        with self.assertRaises(ValueError): fields("a=1 a=2")
        expect({"a": "1"}, a=1)
        with self.assertRaises(ValueError): expect({"a": "1"}, a=2)
        self.assertEqual(bounded("0.5", positive=True, hi=10000), 0.5)
        self.assertEqual(bounded("0.000001", hi=2e-5), 1e-6)
        with self.assertRaises(ValueError): bounded("nan")
        with self.assertRaises(ValueError): bounded("0", positive=True, hi=10000)
        with self.assertRaises(ValueError): bounded("0.001", hi=2e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
