"""Synthetic report fixtures ONLY: these tests are not hardware receipts."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from bench_report import InvalidReceipt, SPECS, parse_receipt, visible_pairs

DONE = "RESULT: PASS timing harness (performance verdicts are per-cell PASS/MISS)"
AOT = "AOT: PASS architecture, SM-count, and baked kernel launch/readback"


def fixture(names=("decode-128k",), *, proxy=False, ms_override=None):
    label, arch, sms, name = ("PROXY", 89, 128, 4090) if proxy else ("TARGET", 120, 170, 5090)
    lines = ["TIME 2026-09-23 00:00:00 AEST",  # synthetic test timestamp, not a run
             f"IDENTITY gpu=NVIDIA GeForce RTX {name} arch=sm_{arch} sms={sms} "
             f"baked_arch=sm_{arch} baked_sms={sms} source=27c28a0f3e9e label={label}",
             "MEMORY free_bytes=17179869184 total_bytes=25769803776 requested_bytes=0 reserve_bytes=4294967296", AOT]
    for cell in names:
        t, s, nkv, window = SPECS[cell]
        ops = 2 * 64 * 320 * visible_pairs(t, s, window)
        ms = ms_override if ms_override is not None else (ops / (120e9) if t > 1 else 1.0)
        byte_count = s * nkv * 320
        gb = byte_count / (ms * 1e6)
        tf = ops / (ms * 1e9)
        passed = tf >= 100 if t > 1 else gb >= 1253
        passed = passed and (cell != "decode-1m" or 9*ms <= 10)
        lines.extend([
            f"CELL {cell} label={label} Q=f32 KV=E4M3-unit cached_V=prescaled n_q=64 n_kv={nkv} "
            f"QK=192 V=128 T={t} S={s} window={window} sink={'per-Q-head' if window else 'absent'} "
            f"splits={0 if t>1 else 256} warmup=2 samples=5 paged=reverse-256 kernel=synthetic-selftest-only",
            "CORRECT sampled_oracle=3/3 full_finite_scan=PASS max_abs=0.0000001",
        ])
        for i, v in enumerate((ms*1.05, ms*0.95, ms, ms*1.02, ms*0.98)):
            lines.append(f"SAMPLE cell={cell} index={i} ms={v:.6f}")
        lines.append(
            f"METRIC cell={cell} label={label} median_ms={ms:.6f} min_ms={ms*0.95:.6f} max_ms={ms*1.05:.6f} "
            f"unique_KV_bytes={byte_count} useful_flops={ops} effective_KV_GBs={gb:.3f} "
            f"pct_5090_peak={gb/17.9:.3f} BF16_equiv_TFLOPS={tf:.6f} "
            f"GA9_extrapolated_ms={9*ms if t==1 else 0:.6f} verdict={'PASS' if passed else 'MISS'} "
            "target_GBs=1253 target_TFLOPS=100")
    return "\n".join(lines + [DONE]) + "\n"


def tc_fixture(splits=256, cell="decode-128k"):
    s=SPECS[cell][1]
    text = fixture((cell,)).replace("kernel=synthetic-selftest-only",
        "kernel=tc-q3-p2-d01 Q_values=BF16-exact KV_abs_max=1.875 precision_scope=bounded-synthetic")
    text = text.replace("max_abs=0.0000001", "max_abs=0.0000001 full_baseline=8192/8192 baseline_max_abs=0.000001")
    text = text.replace("splits=256", f"splits={splits}")
    padded = sum(((s*(i+1)//splits - s*i//splits+31)//32)*32 for i in range(splits))
    executed = 106496*padded
    return text.replace("target_TFLOPS=100", f"target_TFLOPS=100 executed_mma_flops={executed} "
                        f"executed_mma_TFLOPS={executed/1e9:.6f} mma_work_factor={executed/(40960*s):.8f}")


def pipe_fixture(warps=4):
    return tc_fixture(85).replace("kernel=tc-q3-p2-d01",
        f"kernel=tc-q3-p2-d1-w{warps} pipe_warps={warps} pipe_registers=61 pipe_shared_bytes=64192 pipe_active_ctas=1")


def bf16q_fixture(warps=8):
    text = pipe_fixture(warps).replace("tc-q3-p2", "tc-bf16q-p2").replace("Q=f32", "Q=bf16-RNE-post-RoPE Q_storage=f32 reference=bf16q-lattice-local K_dtype=E4M3-unit V_dtype=E4M3-unit")
    padded = 85*49*32
    old, new = 106496*padded, 57344*padded
    return text.replace(f"executed_mma_flops={old}", f"executed_mma_flops={new}").replace(
        f"executed_mma_TFLOPS={old/1e9:.6f}", f"executed_mma_TFLOPS={new/1e9:.6f}").replace(
        f"mma_work_factor={old/5368709120:.8f}", f"mma_work_factor={new/5368709120:.8f}")


def c1_fixture(native=False,cell="decode-128k",splits=85):
    import re
    text=tc_fixture(splits,cell).replace("tc-q3-p2-d01", "tc-bf16q-p2-c1-w8" if native else "tc-q3-p2-c1-w8")
    text=text.replace("Q_values=", "pipe_warps=8 pipe_registers=71 pipe_shared_bytes=44416 pipe_active_ctas=2 mma_tile_n=16 Q_values=")
    if native:
        text=text.replace("Q=f32", "Q=bf16-RNE-post-RoPE Q_storage=f32 reference=bf16q-lattice-local K_dtype=E4M3-unit V_dtype=E4M3-unit")
    s=SPECS[cell][1]
    padded=sum(((s*(i+1)//splits-s*i//splits+15)//16)*16 for i in range(splits))
    executed=(57344 if native else 106496)*padded
    for key,value in (("executed_mma_flops",str(executed)),("executed_mma_TFLOPS",f"{executed/1e9:.6f}"),("mma_work_factor",f"{executed/(40960*s):.8f}")):
        text=re.sub(rf"{key}=\S+",f"{key}={value}",text)
    return text


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.receipt = fixture()
        self.required = ["decode-128k"]

    def reject(self, text, required=None):
        with self.assertRaises(InvalidReceipt):
            parse_receipt(text, required if required is not None else self.required)

    def test_c1_six_exact_padded_cells(self):
        for native in (False,True):
            for cell,splits,padded in (("decode-128k",85,131920),("decode-1m",255,1048816),("decode-1m",510,1052640)):
                r=parse_receipt(c1_fixture(native,cell,splits),[cell])["cells"][0]
                self.assertEqual(r["executed_mma_flops"],padded*(57344 if native else 106496))
                self.assertEqual(r["mma_tile_n"],16)
    def test_c1_wrong_tile(self):
        self.reject(c1_fixture().replace("mma_tile_n=16","mma_tile_n=32"))
    def test_c1_missing_tile(self):
        self.reject(c1_fixture().replace("mma_tile_n=16 ",""))
    def test_c1_cannot_relabel_n32_work(self):
        self.reject(pipe_fixture(8).replace("-d1-","-c1-").replace("pipe_shared_bytes=64192","pipe_shared_bytes=44416").replace("pipe_active_ctas=1","pipe_active_ctas=2 mma_tile_n=16"))
    def test_c1_wrong_shared(self):
        self.reject(c1_fixture().replace("pipe_shared_bytes=44416","pipe_shared_bytes=49536"))
    def test_c1_wrong_capacity(self):
        self.reject(c1_fixture().replace("pipe_active_ctas=2","pipe_active_ctas=1"))
    def test_c1_no_four_warp_variant(self):
        self.reject(c1_fixture().replace("-c1-w8","-c1-w4").replace("pipe_warps=8","pipe_warps=4"))
    def test_c1_register_budget(self):
        self.reject(c1_fixture().replace("pipe_registers=71","pipe_registers=129"))
    def test_c1_full_baseline_required(self):
        self.reject(c1_fixture(True).replace("full_baseline=8192/8192","full_baseline=missing"))

    def test_c3_both_precision_rows_and_resource_scope(self):
        for text in (pipe_fixture(), bf16q_fixture()):
            compact=text.replace("-p2-d1-w", "-p2-c3-w").replace("pipe_shared_bytes=64192", "pipe_shared_bytes=49536").replace("pipe_active_ctas=1", "pipe_active_ctas=2")
            r=parse_receipt(compact, self.required)["cells"][0]
            self.assertEqual(r["pipe_active_ctas"],2)
            self.reject(compact.replace("pipe_active_ctas=2", "pipe_active_ctas=1"))

    def test_c3_cannot_relabel_old_shared_footprint(self):
        self.reject(pipe_fixture().replace("-p2-d1-w", "-p2-c3-w").replace("pipe_active_ctas=1", "pipe_active_ctas=2"))

    def test_c0_both_precision_rows(self):
        for text in (pipe_fixture(), bf16q_fixture()):
            report = parse_receipt(text.replace("-p2-d1-w", "-p2-c0-w"), self.required)
            self.assertIn("-c0-", report["cells"][0]["kernel"])

    def test_bf16q_explicit_scope_and_padded_work(self):
        for w in (4, 8):
            r = parse_receipt(bf16q_fixture(w), self.required)["cells"][0]
            self.assertEqual(r["executed_mma_flops"], 57344*85*49*32)
            self.assertEqual(r["reference_scope"], "bf16q-lattice-local")

    def test_bf16q_cannot_claim_f32_reference(self):
        self.reject(bf16q_fixture().replace("reference=bf16q-lattice-local", "reference=f32q"))

    def test_bf16q_cannot_hide_cast(self):
        self.reject(bf16q_fixture().replace("Q=bf16-RNE-post-RoPE", "Q=f32"))

    def test_bf16q_nominal_factor_not_padded_work(self):
        import re
        self.reject(re.sub(r"mma_work_factor=\S+", "mma_work_factor=1.40000000", bf16q_fixture()))

    def test_bf16q_needs_full_baseline(self):
        self.reject(bf16q_fixture().replace("full_baseline=8192/8192", "full_baseline=missing"))

    def test_bf16q_needs_kv_identity(self):
        self.reject(bf16q_fixture().replace("V_dtype=E4M3-unit", "V_dtype=unknown"))

    def test_profile_events_are_not_gate_evidence(self):
        self.reject("PROFILE_TIMING_NOT_A_GATE: Nsight replay\n"+self.receipt)

    def test_pipe_warp_variants(self):
        for w in (4, 8):
            row = parse_receipt(pipe_fixture(w), self.required)["cells"][0]
            self.assertEqual(row["pipe_warps"], w)
            self.assertEqual(row["executed_mma_flops"], 14193786880)

    def test_pipe_wrong_dispatch(self):
        self.reject(pipe_fixture().replace("pipe_warps=4", "pipe_warps=8"))

    def test_pipe_wrong_resources(self):
        self.reject(pipe_fixture().replace("pipe_shared_bytes=64192", "pipe_shared_bytes=43456"))
        self.reject(pipe_fixture().replace("pipe_active_ctas=1", "pipe_active_ctas=2"))
        self.reject(pipe_fixture().replace("pipe_registers=61", "pipe_registers=0"))

    def test_pipe_missing_full_baseline(self):
        self.reject(pipe_fixture().replace("full_baseline=8192/8192", "full_baseline=not-applicable"))

    def test_unknown_tc_kernel(self):
        self.reject(pipe_fixture().replace("tc-q3-p2-d1-w4", "tc-unqualified"))

    def test_tc_explicit_work(self):
        report = parse_receipt(tc_fixture(), self.required)
        self.assertEqual(report["cells"][0]["mma_work_factor"], 2.6)

    def test_tc_split_padding(self):
        report = parse_receipt(tc_fixture(85), self.required)
        self.assertEqual(report["cells"][0]["executed_mma_flops"], 14193786880)

    def test_tc_nominal_factor_refused(self):
        self.reject(tc_fixture().replace("mma_work_factor=2.60000000", "mma_work_factor=2.00000000"))

    def test_tc_wrong_executed_rate(self):
        self.reject(tc_fixture().replace("executed_mma_TFLOPS=13.958644", "executed_mma_TFLOPS=20"))

    def test_tc_missing_baseline(self):
        self.reject(tc_fixture().replace("full_baseline=8192/8192", "full_baseline=not-applicable"))

    def test_tc_baseline_error(self):
        self.reject(tc_fixture().replace("baseline_max_abs=0.000001", "baseline_max_abs=0.001"))

    def test_tc_scope_missing(self):
        self.reject(tc_fixture().replace("precision_scope=bounded-synthetic", "precision_scope=general"))

    def test_valid_miss_retained(self):
        report = parse_receipt(self.receipt, self.required)
        self.assertEqual(report["target_verdict"], "MISS")
        self.assertAlmostEqual(report["cells"][0]["effective_KV_GBs"], 167.77216)

    def test_valid_target_pass(self):
        report = parse_receipt(fixture(ms_override=0.12), self.required)
        self.assertEqual(report["target_verdict"], "PASS")

    def test_proxy_cannot_promote(self):
        report = parse_receipt(fixture(proxy=True), self.required)
        self.assertEqual(report["label"], "PROXY")
        self.assertTrue(report["promotion"].startswith("NONE"))

    def test_all_shapes_causal_accounting(self):
        report = parse_receipt(fixture(SPECS), list(SPECS))
        self.assertEqual(len(report["cells"]), 8)
        self.assertEqual(report["cells"][4]["useful_flops"], 85941288960)
        self.assertEqual(report["cells"][7]["useful_flops"], 10737418240)

    def test_short_decode_is_diagnostic(self):
        report = parse_receipt(fixture(["decode-4k"]), ["decode-4k"])
        self.assertEqual(report["target_verdict"], "NOT_APPLICABLE")

    def test_correct_ga9_extrapolation(self):
        report = parse_receipt(fixture(["decode-1m"]), ["decode-1m"])
        self.assertEqual(report["cells"][0]["GA9_extrapolated_ms"], 9)
        self.assertEqual(report["cells"][0]["unique_KV_bytes"], 1342177280)

    def test_missing_memory_receipt(self):
        self.reject("\n".join(l for l in self.receipt.splitlines() if not l.startswith("MEMORY ")))

    def test_insufficient_memory_reserve(self):
        self.reject(self.receipt.replace("free_bytes=17179869184", "free_bytes=4294967295"))

    def test_memory_allocation_breaks_reserve(self):
        self.reject(self.receipt.replace("requested_bytes=0", "requested_bytes=17179869184"))

    def test_wrong_memory_floor(self):
        self.reject(self.receipt.replace("reserve_bytes=4294967296", "reserve_bytes=2147483648"))

    def test_missing_cell(self):
        self.reject(self.receipt, list(SPECS))

    def test_no_completion(self):
        self.reject(self.receipt.replace(DONE, ""))

    def test_no_launch_gate(self):
        self.reject(self.receipt.replace(AOT, "AOT: PASS architecture and SM-count (hardware readback)"))

    def test_wrong_arch(self):
        self.reject(self.receipt.replace("baked_arch=sm_120", "baked_arch=sm_89"))

    def test_wrong_sms(self):
        self.reject(self.receipt.replace("baked_sms=170", "baked_sms=188"))

    def test_wrong_proxy_label(self):
        self.reject(fixture(proxy=True).replace("label=PROXY", "label=TARGET"))

    def test_wrong_gpu(self):
        self.reject(self.receipt.replace("RTX 5090", "RTX 4090"))

    def test_no_sydney_timestamp(self):
        self.reject(self.receipt.replace("TIME 2026-09-23 00:00:00 AEST", ""))

    def test_no_source(self):
        self.reject(self.receipt.replace("source=27c28a0f3e9e", "source=unknown"))

    def test_duplicate_identity(self):
        self.reject(self.receipt + self.receipt)

    def test_duplicate_field(self):
        self.reject(self.receipt.replace("sms=170 baked_arch", "sms=170 sms=170 baked_arch"))

    def test_duplicate_metric(self):
        metric = next(l for l in self.receipt.splitlines() if l.startswith("METRIC "))
        self.reject(self.receipt.replace(DONE, metric+"\n"+DONE))

    def test_reordered_samples(self):
        self.reject(self.receipt.replace("index=1", "index=4"))

    def test_incomplete_samples(self):
        self.reject(self.receipt.replace("samples=5", "samples=7"))

    def test_nonpositive_time(self):
        self.reject(self.receipt.replace("index=0 ms=1.050000", "index=0 ms=0"))

    def test_nonfinite_metric(self):
        self.reject(self.receipt.replace("median_ms=1.000000", "median_ms=nan"))

    def test_nonmedian_average_claim(self):
        self.reject(self.receipt.replace("median_ms=1.000000", "median_ms=1.100000"))

    def test_gqa_byte_inflation(self):
        self.reject(self.receipt.replace("unique_KV_bytes=167772160", "unique_KV_bytes=2684354560"))

    def test_full_square_prefill_inflation(self):
        text = fixture(["prefill-2k"])
        self.reject(text.replace("useful_flops=85941288960", "useful_flops=171798691840"), ["prefill-2k"])

    def test_wrong_rate(self):
        self.reject(self.receipt.replace("effective_KV_GBs=167.772", "effective_KV_GBs=1677.720"))

    def test_greenwashed_miss(self):
        self.reject(self.receipt.replace("verdict=MISS", "verdict=PASS"))

    def test_wrong_target(self):
        self.reject(self.receipt.replace("target_GBs=1253", "target_GBs=700"))

    def test_ga_sink_forbidden(self):
        self.reject(self.receipt.replace("sink=absent", "sink=per-Q-head"))

    def test_no_numerical_gate(self):
        self.reject(self.receipt.replace("sampled_oracle=3/3", "sampled_oracle=2/3"))

    def test_numerical_error(self):
        self.reject(self.receipt.replace("max_abs=0.0000001", "max_abs=0.001"))

    def test_explicit_failure_retained(self):
        for failure in ("FAIL", "REFUSE", "INCOMPLETE"):
            with self.subTest(failure=failure):
                self.reject(self.receipt + f"RESULT: {failure} retained\n")

    def test_measurement_after_completion(self):
        sample = next(l for l in self.receipt.splitlines() if l.startswith("SAMPLE "))
        self.reject(self.receipt + sample + "\n")

    def test_empty_required_list(self):
        self.reject(self.receipt, [])

    def test_duplicate_required_cells(self):
        self.reject(self.receipt, self.required * 2)

    def test_implausible_peak_requires_review(self):
        self.reject(fixture(ms_override=0.01))

    def test_rounded_threshold_refuses_certification(self):
        self.reject(fixture(ms_override=167772160 / (1253e6)))

    def cli(self, text):
        # All temporary artifacts remain in the runner's workspace build slot.
        with tempfile.TemporaryDirectory(dir=os.environ["MIMO26_ATTN_SELFTEST_DIR"]) as directory:
            receipt = Path(directory) / "synthetic-selftest-only.log"
            receipt.write_text(text)
            process = subprocess.run([sys.executable, str(Path(__file__).with_name("bench_report.py")),
                                      str(receipt), "--required", "decode-128k"],
                                     capture_output=True, text=True, check=False)
            return process.returncode, json.loads(process.stdout)

    def test_cli_valid_miss_exit_one(self):
        rc, report = self.cli(self.receipt)
        self.assertEqual(rc, 1)
        self.assertEqual(report["evidence"], "VALID")
        self.assertEqual(report["target_verdict"], "MISS")

    def test_cli_valid_pass_exit_zero(self):
        rc, report = self.cli(fixture(ms_override=0.12))
        self.assertEqual(rc, 0)
        self.assertEqual(report["target_verdict"], "PASS")

    def test_cli_incomplete_exit_two(self):
        rc, report = self.cli(self.receipt.replace(DONE, ""))
        self.assertEqual(rc, 2)
        self.assertEqual(report["evidence"], "INVALID_OR_INCOMPLETE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
