import unittest

import op1_schedule as op
from op1_cold_report import parse, useful_flops, p1_mma_flops


def build_receipt():
    m = op.load_manifest()
    lines = [
        "2026-09-24 10:40:00 AEST",
        "a"*64 + "  /work/mimo26f-attn-bench-sm120",
        "IDENTITY gpu=NVIDIA GeForce RTX 5090 arch=sm_120 sms=170 baked_arch=sm_120 baked_sms=170 source=123456abcdef label=TARGET",
        "AOT: PASS architecture, SM-count, and baked kernel launch/readback",
        "MEMORY free_bytes=16000000000 total_bytes=33711521792 requested_bytes=2147483648 reserve_bytes=4294967296",
        "OP1_COLD_BEGIN d7_sha=" + m["inputs"]["d7_sha256"] + " points=4 kinds=2 modes=2 T=2048 batch=8 scope=attention-cold-prefill-multichunk-P1 gate=UNSET boundary=core-post-rope-prescaled-kv Q_values=BF16-exact",
    ]
    for S in op.COLD_PREFILL:
        for kind in ("ga", "swa"):
            nkv, window = (4, 0) if kind == "ga" else (8, 128)
            n = S // 2048
            lines.append(f"OP1_COLD_POINT S={S} kind={kind} nkv={nkv} window={window} chunks={n}")
            for i in range(n):
                full_ref = (i == 0) or (i == n // 2) or (i == n - 1)
                for mode in ("f32q", "bf16q"):
                    for j in range(7):
                        lines.append(f"OP1_COLD_SAMPLE S={S} kind={kind} chunk={i} mode={mode} index={j} ms=1.000000000 finite=PASS coord=0.000000001")
                lines.append(f"OP1_COLD_CHUNK S={S} kind={kind} chunk={i} S_c={2048*(i+1)} full_ref={1 if full_ref else 0} checked={16777216 if full_ref else 0} coordinates=3 query_exact=PASS reference_finite=PASS useful_flops={useful_flops(2048, 2048*(i+1), window):.0f} executed_f32={p1_mma_flops(2048, 2048*(i+1), nkv, window, True):.0f} executed_bf16={p1_mma_flops(2048, 2048*(i+1), nkv, window, False):.0f} max_error={'0.000001000' if full_ref else '0'} max_coordinate_diff=0.000000001")
            for mode in ("f32q", "bf16q"):
                lines.append(f"OP1_COLD_RESULT S={S} kind={kind} mode={mode} ttft_ms={n:.9f} tok_s={S/(n*1e-3):.9f} useful_flops_total={sum(useful_flops(2048, 2048*(i+1), window) for i in range(n)):.0f} executed_flops_total={sum(p1_mma_flops(2048, 2048*(i+1), nkv, window, mode=='f32q') for i in range(n)):.0f} useful_tflops=1.0 executed_tflops=1.0")
    lines.append("OP1_COLD_COMPLETE points=4 kinds=2 modes=2 chunks=106 samples=1484 full_ref_chunks=20 gate=UNSET")
    lines.append("RESULT: PASS OP1 cold prefill harness (no gate, no promotion)")
    return "\n".join(lines)


class ColdReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.text = build_receipt()

    def rejects(self, old, new):
        self.assertIn(old, self.text)
        with self.assertRaises((ValueError, KeyError)): parse(self.text.replace(old, new, 1))

    def test_valid_and_slice(self):
        result = parse(self.text)
        self.assertEqual(result["status"], "VALID")
        self.assertEqual(result["gate"], "UNSET")
        self.assertEqual(result["chunks"], 106)
        self.assertEqual(result["samples"], 1484)
        self.assertEqual(len(result["per_kind"]), 16)
        for mode in ("f32q", "bf16q"):
            self.assertEqual(set(result["attention_slice"][mode]), {"2048", "8192", "32768", "65536"})
            for S in op.COLD_PREFILL:
                row = result["attention_slice"][mode][str(S)]
                self.assertAlmostEqual(row["ttft_ms"], 48 * (S // 2048))
                self.assertEqual(row["d7_full_model_tok_s"], op.D7_BARS["cold_prefill_tok_s"][str(S)])

    def test_identity_aot(self):
        self.rejects("AOT: PASS", "AOT: absent")
        self.rejects("source=123456abcdef", "source=123456abcdef-dirty")
        self.rejects("arch=sm_120", "arch=sm_89")
        self.rejects("label=TARGET", "label=PROXY")

    def test_scope(self):
        self.rejects("gate=UNSET", "gate=PASS")
        self.rejects("scope=attention-cold-prefill-multichunk-P1", "scope=full-model")
        self.rejects("points=4", "points=3")
        self.rejects("T=2048", "T=1024")

    def test_missing_rows(self):
        self.rejects("OP1_COLD_COMPLETE", "MISSING_COMPLETE")
        lines = self.text.splitlines()
        i = next(i for i, x in enumerate(lines) if x.startswith("OP1_COLD_SAMPLE "))
        with self.assertRaises(ValueError): parse('\n'.join(lines[:i] + lines[i+1:]))
        j = next(i for i, x in enumerate(lines) if x.startswith("OP1_COLD_CHUNK "))
        with self.assertRaises(ValueError): parse('\n'.join(lines[:j] + lines[j+1:]))
        k = next(i for i, x in enumerate(lines) if x.startswith("OP1_COLD_RESULT "))
        with self.assertRaises(ValueError): parse('\n'.join(lines[:k] + lines[k+1:]))

    def test_numeric(self):
        self.rejects("ms=1.000000000", "ms=nan")
        self.rejects("ms=1.000000000", "ms=0.000000000")
        self.rejects("coord=0.000000001", "coord=0.001000000")

    def test_wrong_flops_or_chunk(self):
        self.rejects("S_c=2048", "S_c=2047")
        self.rejects("full_ref=0", "full_ref=2")
        # Corrupt the first useful_flops value.
        import re as _re
        line = next(x for x in self.text.splitlines() if x.startswith("OP1_COLD_CHUNK S=2048 kind=ga chunk=0"))
        bad = _re.sub(r"useful_flops=\d+", "useful_flops=1", line)
        with self.assertRaises(ValueError): parse(self.text.replace(line, bad, 1))

    def test_wrong_d7(self):
        self.rejects("d7_sha=", "d7_sha=deadbeef")

    def test_missing_native(self):
        lines = [x for x in self.text.splitlines() if "mode=bf16q" not in x]
        with self.assertRaises(ValueError): parse('\n'.join(lines))

    def test_failure_retained(self):
        with self.assertRaises(ValueError): parse(self.text + '\nRESULT: INCOMPLETE budget')

    def test_duplicate_index(self):
        lines = self.text.splitlines()
        i = next(i for i, x in enumerate(lines) if x.startswith("OP1_COLD_SAMPLE "))
        line = lines.pop(i)
        j = next(i for i, x in enumerate(lines) if x.startswith("OP1_COLD_SAMPLE "))
        lines.insert(j + 1, line)
        with self.assertRaises(ValueError): parse('\n'.join(lines))


if __name__ == "__main__":
    unittest.main(verbosity=2)
