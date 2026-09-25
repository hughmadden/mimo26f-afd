import unittest

from m0_report import parse, ARMS, CONFIGS, LENS


def build_receipt():
    lines = [
        "2026-09-24 11:00:00 AEST",
        "a"*64 + "  /work/mimo26f-attn-bench-sm120",
        "IDENTITY gpu=NVIDIA GeForce RTX 5090 arch=sm_120 sms=170 baked_arch=sm_120 baked_sms=170 source=123456abcdef label=TARGET",
        "AOT: PASS architecture, SM-count, and baked kernel launch/readback",
        "M0_BEGIN source=123456abcdef arms=3 K=1,2,4,8 threads=128,256 grids=680,1360 N=8192 scope=dense-mma-calibration gate=UNSET clock_khz=2407000 max_clock_khz=2407000",
    ]
    for arm in ARMS:
        lines.append(f"M0_ARM arm={arm} registers=64 active_CTAs_per_SM_128=2")
        for K, grid, threads, N in CONFIGS:
            exec_flops = grid * (threads / 32) * N * (2 * K) * 4096
            # monotone ms so peak = largest config
            ms = 1.0 + K * 0.1 + (grid / 1360) * 0.2
            maxcyc = int(exec_flops / (170 * 512))
            lines.append(f"M0_SAMPLE arm={arm} K={K} threads={threads} grid={grid} N={N} executed_flops={exec_flops:.0f} median_ms={ms:.6f} min_ms={ms*0.99:.6f} max_ms={ms*1.01:.6f} tflops={exec_flops/(ms*1e9):.6f} max_sm_cycles={maxcyc} flop_per_sm_cycle=512.000")
        for n in LENS:
            lines.append(f"M0_LATENCY arm={arm} N={n} cycles={n*40}")
        lines.append(f"M0_LATENCY_SLOPE arm={arm} cycles_per_mma=40.000000")
    lines.append("RESULT: PASS M0 calibration harness (calibration, not a ceiling certification)")
    return "\n".join(lines)


class M0ReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.text = build_receipt()

    def rejects(self, old, new):
        self.assertIn(old, self.text)
        with self.assertRaises((ValueError, KeyError)): parse(self.text.replace(old, new, 1))

    def test_valid(self):
        r = parse(self.text)
        self.assertEqual(r["status"], "VALID")
        self.assertEqual(r["gate"], "UNSET")
        self.assertEqual(set(r["peak_tflops"]), set(ARMS))
        self.assertEqual(set(r["latency_cycles_per_mma"]), set(ARMS))
        self.assertAlmostEqual(r["latency_cycles_per_mma"]["bf16f32"], 40.0)
        self.assertAlmostEqual(r["flop_per_sm_measured_cycle"]["bf16f32"], 512.0)
        self.assertAlmostEqual(r["flop_per_sm_measured_cycle"]["f16f16"], 512.0)

    def test_identity_aot(self):
        self.rejects("AOT: PASS", "AOT: absent")
        self.rejects("source=123456abcdef", "source=123456abcdef-dirty")
        self.rejects("arch=sm_120", "arch=sm_89")

    def test_scope(self):
        self.rejects("gate=UNSET", "gate=PASS")
        self.rejects("scope=dense-mma-calibration", "scope=ceiling-certification")
        self.rejects("arms=3", "arms=2")

    def test_wrong_flops(self):
        import re as _re
        line = next(x for x in self.text.splitlines() if x.startswith("M0_SAMPLE ") and "K=8" in x and "grid=1360" in x and "N=8192" in x)
        bad = _re.sub(r"executed_flops=\d+", "executed_flops=1", line)
        with self.assertRaises(ValueError): parse(self.text.replace(line, bad, 1))

    def test_missing(self):
        self.rejects("M0_LATENCY_SLOPE arm=f16f16", "MISSING")
        lines = self.text.splitlines()
        i = next(i for i, x in enumerate(lines) if x.startswith("M0_SAMPLE "))
        with self.assertRaises(ValueError): parse('\n'.join(lines[:i] + lines[i+1:]))

    def test_bad_timing(self):
        self.rejects("median_ms=1.200000", "median_ms=0.000000")
        self.rejects("median_ms=1.200000", "median_ms=nan")

    def test_bad_slope(self):
        self.rejects("cycles_per_mma=40.000000", "cycles_per_mma=0.000000")

    def test_failure_retained(self):
        with self.assertRaises(ValueError): parse(self.text + '\nRESULT: INCOMPLETE budget')


if __name__ == "__main__":
    unittest.main(verbosity=2)
