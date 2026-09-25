"""Synthetic N5 receipt tests, never hardware evidence."""
import unittest
from eta_report import validate, EXPECTED, InvalidReceipt


def fixture():
    lines = ["TIME 2026-09-23 00:00:00 AEST",
             "IDENTITY gpu=NVIDIA GeForce RTX 5090 arch=sm_120 sms=170 source=abcdef1 label=TARGET",
             "AOT: PASS architecture, SM-count, and baked kernel launch/readback"]
    for m, n, mode in sorted(EXPECTED):
        prefix = f"M={m} N={n} mode={mode}"
        for flag in ([1, 2] if mode == "f32q" else [2]):
            lines.append(f"ETA_NEGATIVE {prefix} flag={flag} max_abs=0.0002 detected=PASS")
        for i, ms in enumerate((11, 9, 10, 10.1, 9.9, 10.2, 9.8)):
            lines.append(f"ETA_SAMPLE {prefix} index={i} ms={ms}")
        useful = 2*m*n*320*680*32
        executed = 2*m*n*((3 if mode == "f32q" else 1)*192+2*128)*680*32
        shared = m*192*2 + n*320*2 + m*n*2 + m*16 + n*320
        rounding = "Q3" if mode == "f32q" else "BF16-RNE-post-RoPE-once"
        lines.append(f"ETA {prefix} warps={m//8} Q_round={rounding} K=E4M3-unit V=E4M3-unit "
                     f"cache=FP8-prescaled-V K_abs_max=1.875 V_abs_max=1.25 stats=f32 blocks=680 "
                     f"iterations=32 shared_bytes={shared} registers=100 active_CTAs_per_SM=1 "
                     f"median_ms=10 useful_flops={useful} executed_flops={executed} "
                     f"useful_TFLOPS={useful/1e10} executed_TFLOPS={executed/1e10} eta={executed/1e10/209.5} "
                     "load_cycles=100 qk_pack_cycles=200 softmax_hi_cycles=100 pv_lowpack_cycles=100 "
                     "softmax_hi_fraction=0.2 max_abs=0.000001 verdict=MISS scope=interior-resident-micro-not-prefill")
    return "\n".join(lines+["RESULT: PASS eta micro harness (performance verdicts per row)"])


class Tests(unittest.TestCase):
    def reject(self, text):
        with self.assertRaises(InvalidReceipt):
            validate(text)

    def test_valid(self):
        report = validate(fixture())
        self.assertEqual(report["evidence"], "VALID")
        self.assertEqual(len(report["rows"]), 6)

    def test_missing_row(self):
        self.reject("\n".join(l for l in fixture().splitlines() if not l.startswith("ETA M=64 N=64 mode=f32q")))

    def test_missing_negative(self):
        self.reject("\n".join(l for l in fixture().splitlines() if not l.startswith("ETA_NEGATIVE M=64 N=64 mode=f32q flag=1")))

    def test_pre_rope_rejected(self):
        self.reject(fixture().replace("BF16-RNE-post-RoPE-once", "BF16-pre-RoPE"))

    def test_wrong_work(self):
        self.reject(fixture().replace("iterations=32", "iterations=31"))

    def test_greenwashed_miss(self):
        self.reject(fixture().replace("verdict=MISS", "verdict=PASS"))

    def test_bad_clock_fraction(self):
        self.reject(fixture().replace("softmax_hi_fraction=0.2", "softmax_hi_fraction=0.5"))

    def test_out_of_domain(self):
        self.reject(fixture().replace("V_abs_max=1.25", "V_abs_max=448"))

    def test_failed_receipt(self):
        self.reject(fixture()+"\nRESULT: FAIL CUDA error")

    def test_missing_completion(self):
        self.reject(fixture().replace("RESULT: PASS eta micro harness (performance verdicts per row)", ""))


if __name__ == "__main__":
    unittest.main()
