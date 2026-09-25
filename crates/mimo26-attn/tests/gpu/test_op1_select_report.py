import unittest

import op1_schedule as op
from op1_select_report import parse


def receipt():
    m = op.load_manifest()
    lines = ["2026-09-24 09:10:00 AEST", "a"*64 + "  /work/mimo26f-attn-bench-sm120",
             "IDENTITY gpu=NVIDIA GeForce RTX 5090 arch=sm_120 sms=170 baked_arch=sm_120 baked_sms=170 source=123456abcdef label=TARGET",
             "AOT: PASS architecture, SM-count, and baked kernel launch/readback",
             "MEMORY free_bytes=16000000000 total_bytes=33711521792 requested_bytes=2147483648 reserve_bytes=4294967296"]
    for mode in ("f32q", "bf16q"):
        native = mode == "bf16q"
        for family in ("p1", "c3"):
            c3 = family == "c3"
            lines.append(f"OP1_RESOURCE mode={mode} family={family} registers={(70 if native else 92) if c3 else (124 if native else 125)} shared={49536 if c3 else (38976 if native else 88128)} capacity={2 if c3 or native else 1}")
    lines.append(f"OP1_SELECT_BEGIN d7_sha={m['inputs']['d7_sha256']} cases=36 kinds=2 modes=2 paths=5 repeats=16 warmup=3 samples=7 scope=selection-only-context-proxy gate=UNSET Q=post-RoPE-f32 Q_values=BF16-exact KV=E4M3-unit V=prescaled page=reverse-256 boundary=attention-core")
    for c in op.cases(m):
        if not c["select"]:
            continue
        for kind in ("ga", "swa"):
            base = f"id={c['id']} kind={kind}"
            n = c["t"] * 64 * 128
            lines.append(f"OP1_SELECT_CASE {base} category={op.CATEGORIES[c['category']]} step={c['step']} T={c['t']} S={c['swa_s'] if kind == 'swa' else c['s']} prefix={c['prefix']} first={c['swa_start'] if kind == 'swa' else 0} query_checked={c['t']*64*192} reference_checked={n} query_exact=PASS reference_finite=PASS coordinates=3 max_coordinate_diff=1e-8")
            for mode in ("f32q", "bf16q"):
                for path in range(5):
                    lines.append(f"OP1_SELECT_CORRECT {base} mode={mode} path={path} checked={n} guard=128 max_error=1e-6")
            for path in range(5):
                for mode in ("f32q", "bf16q"):
                    lines.append(f"OP1_SELECT_GRAPH {base} mode={mode} path={path} nodes={16 if path == 0 else 32}")
                for i in range(7):
                    for mode in ("f32q", "bf16q"):
                        ms = .01 * (path + 1) * (2 if mode == "f32q" else 1)
                        lines.append(f"OP1_SELECT_SAMPLE {base} mode={mode} path={path} index={i} graph_ms={16*ms:.9f} per_call_ms={ms:.9f} checked={n} guard=128 max_error=1e-6")
    lines.append("OP1_SELECT_COMPLETE shape_kinds=72 candidates=720 samples=5040 all_layer_step_measured=0")
    return "\n".join(lines)


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.text = receipt()

    def rejects(self, old, new):
        self.assertIn(old, self.text)
        with self.assertRaises((ValueError, KeyError)): parse(self.text.replace(old, new, 1))

    def test_valid_ungated_and_no_step_claim(self):
        result = parse(self.text)
        self.assertEqual(result["status"], "VALID")
        self.assertEqual(result["gate"], "UNSET")
        self.assertEqual(result["samples"], 5040)
        self.assertEqual(len(result["recommendations"]), 4)
        self.assertTrue(all(x["selected_path"] == 0 for x in result["recommendations"]))

    def test_aot(self): self.rejects("AOT: PASS", "AOT: absent")
    def test_dirty(self): self.rejects("source=123456abcdef", "source=123456abcdef-dirty")
    def test_wrong_arch(self): self.rejects("arch=sm_120", "arch=sm_89")
    def test_wrong_input(self): self.rejects("d7_sha=", "other_sha=")
    def test_gate_invention(self): self.rejects("gate=UNSET", "gate=PASS")
    def test_missing_completion(self): self.rejects("OP1_SELECT_COMPLETE", "MISSING_COMPLETE")
    def test_wrong_layer_claim(self): self.rejects("all_layer_step_measured=0", "all_layer_step_measured=1")
    def test_wrong_repeat(self): self.rejects("repeats=16", "repeats=8")
    def test_missing_reduction(self): self.rejects("nodes=32", "nodes=16")
    def test_resource_drift(self): self.rejects("registers=125", "registers=126")
    def test_missing_query_proof(self): self.rejects("query_exact=PASS", "query_exact=absent")
    def test_wrong_count(self): self.rejects("reference_checked=401408", "reference_checked=8192")
    def test_nan_coordinate(self): self.rejects("max_coordinate_diff=1e-8", "max_coordinate_diff=nan")
    def test_nan_output(self): self.rejects("max_error=1e-6", "max_error=nan")
    def test_error_bound(self): self.rejects("max_error=1e-6", "max_error=0.001")
    def test_guard_missing(self): self.rejects("guard=128", "guard=0")
    def test_wrong_sample_divisor(self): self.rejects("per_call_ms=0.020000000", "per_call_ms=0.320000000")
    def test_duplicate_index(self): self.rejects("index=1", "index=0")
    def test_no_bonus_context(self): self.rejects("prefix=49", "prefix=48")
    def test_drop_last_token(self): self.rejects("S=49", "S=48")
    def test_missing_native(self):
        with self.assertRaises(ValueError): parse('\n'.join(x for x in self.text.splitlines() if "mode=bf16q" not in x))
    def test_late_check(self):
        lines = self.text.splitlines()
        i = next(i for i, x in enumerate(lines) if x.startswith("OP1_SELECT_CORRECT"))
        line = lines.pop(i)
        j = next(i for i, x in enumerate(lines) if x.startswith("OP1_SELECT_SAMPLE"))
        lines.insert(j+1, line)
        with self.assertRaises(ValueError): parse('\n'.join(lines))
    def test_low_reserve(self): self.rejects("free_bytes=16000000000", "free_bytes=4294967295")
    def test_failure_retained(self):
        with self.assertRaises(ValueError): parse(self.text + '\nRESULT: INCOMPLETE budget')


if __name__ == "__main__":
    unittest.main(verbosity=2)
