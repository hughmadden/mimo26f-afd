import copy
import json
import unittest

import op1_schedule as op

DISTINCT_LENGTHS = {
    "coding": {(49, 173), (49, 194), (49, 200)},
    "format": {(92, 117), (92, 118)},
    "json": {(70, 97), (70, 99)},
    "math": {(82, 169), (82, 171), (82, 172), (82, 176), (82, 177)},
    "narrative": {(47, 132), (47, 140), (47, 143), (47, 144), (47, 146), (47, 156), (47, 161)},
    "prose": {(44, 110), (44, 128), (44, 133), (44, 136), (44, 139), (44, 141), (44, 149)},
    "reasoning": {(69, 200)},
    "structured": {(120, 101), (120, 102), (120, 103)},
    "summary": {(309, 100), (309, 113), (309, 119), (309, 122), (309, 123), (309, 125)},
}
PER_CATEGORY_STEPS = {"coding": 222, "format": 126, "json": 150, "math": 210,
                      "narrative": 567, "prose": 453, "reasoning": 439,
                      "structured": 100, "summary": 375}


class OP1ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads(op.D7.read_text())
        self.pattern = op.layer_pattern(op.CONFIG.read_text())

    def manifest(self):
        return op.build_manifest(self.data, self.pattern)

    def test_exact_nine_cohort_and_c1_values(self):
        rows = self.manifest()["categories"]
        self.assertEqual([x["category"] for x in rows], list(op.CATEGORIES))
        self.assertEqual([x["prompt_tokens"] for x in rows], [49, 70, 47, 44, 82, 69, 309, 120, 92])
        self.assertEqual([x["completion_tokens"] for x in rows], [173, 97, 156, 133, 169, 200, 113, 101, 118])
        self.assertEqual([x["accept_hundredths"] for x in rows], [588, 380, 80, 106, 483, 228, 129, 567, 556])

    def test_exact_layer_order(self):
        manifest = self.manifest()
        self.assertEqual([x["id"] for x in manifest["layers"] if x["kind"] == "ga"],
                         [0, 5, 11, 17, 23, 29, 35, 41, 47])
        self.assertEqual(len(manifest["layers"]), 48)
        for layer in manifest["layers"]:
            self.assertEqual((layer["qk"], layer["v"], layer["n_q"]), (192, 128, 64))
            self.assertEqual(layer["n_kv"], 8 if layer["kind"] == "swa" else 4)
            self.assertEqual(layer["window"], 128 if layer["kind"] == "swa" else 0)
            self.assertEqual(layer["sink"], "per-Q-head" if layer["kind"] == "swa" else "absent")

    def test_full_distribution_counts(self):
        manifest = self.manifest()
        for row in manifest["categories"]:
            self.assertEqual(len(row["requests"]), 7)
            self.assertEqual([r["c"] for r in row["requests"]].count(1), 1)
            self.assertEqual([r["c"] for r in row["requests"]].count(6), 6)
            self.assertEqual(sum(len(r["steps"]) for r in row["requests"]),
                             PER_CATEGORY_STEPS[row["category"]])
        self.assertEqual(sum(sum(len(r["steps"]) for r in row["requests"])
                             for row in manifest["categories"]), 2642)

    def test_max_p_plus_c_is_434(self):
        manifest = self.manifest()
        maxima = {row["category"]: max(p + c for r in row["requests"]
                                       for p, c in [(r["prompt_tokens"], r["completion_tokens"])])
                  for row in manifest["categories"]}
        self.assertEqual(maxima["summary"], 434)
        self.assertTrue(all(v <= 434 for v in maxima.values()))

    def test_length_identity_note(self):
        for row in self.manifest()["categories"]:
            actual = {(r["prompt_tokens"], r["completion_tokens"]) for r in row["requests"]}
            self.assertEqual(actual, DISTINCT_LENGTHS[row["category"]])

    def test_every_step_context_coverage_and_tail(self):
        for row in self.manifest()["categories"]:
            for req in row["requests"]:
                committed = 0
                for i, step in enumerate(req["steps"]):
                    self.assertEqual(step["index"], i)
                    self.assertEqual(step["cache_prefix"], req["prompt_tokens"] + committed)
                    self.assertEqual(step["T"], 8)
                    self.assertEqual(step["S"], step["cache_prefix"] + 8)
                    self.assertEqual(step["query_end"], step["S"] - 1)
                    self.assertEqual(step["query_start"], step["cache_prefix"])
                    self.assertEqual(step["advance_model"], step["draft_accepted_model"] + 1)
                    self.assertLessEqual(step["advance_committed"], step["advance_model"])
                    self.assertGreaterEqual(step["advance_committed"], 1)
                    self.assertLessEqual(step["advance_model"], 8)
                    self.assertEqual(step["terminal_cap"], step["advance_committed"] != step["advance_model"])
                    if step["terminal_cap"]:
                        self.assertEqual(i, len(req["steps"]) - 1)
                    committed += step["advance_committed"]
                self.assertEqual(committed, req["completion_tokens"] - 1)
                self.assertLessEqual(req["steps"][-1]["S"],
                                     req["prompt_tokens"] + req["completion_tokens"] + 6)
                self.assertEqual(req["prefill"],
                                 {"T": req["prompt_tokens"], "S": req["prompt_tokens"],
                                  "query_start": 0, "query_end": req["prompt_tokens"] - 1})

    def test_bonus_not_omitted_at_subunit_acceptance(self):
        narrative = self.manifest()["categories"][2]
        c1 = next(r for r in narrative["requests"] if r["c"] == 1)
        self.assertEqual([x["advance_model"] for x in c1["steps"][:5]], [1, 2, 2, 2, 2])
        self.assertEqual(len(c1["steps"]), 87)
        self.assertNotEqual(len(c1["steps"]), 194)  # ceil(155/0.8) is the wrong denominator.

    def test_prefix_acceptance_discrepancy_below_one(self):
        for row in self.manifest()["categories"]:
            for req in row["requests"]:
                accepted = 0
                for n, step in enumerate(req["steps"], 1):
                    accepted += step["draft_accepted_model"]
                    self.assertLessEqual(accepted * 100, n * req["accept_hundredths"])
                    self.assertLess(n * req["accept_hundredths"] - accepted * 100, 100)

    def test_lookahead_not_clamped_to_committed_length(self):
        summary = self.manifest()["categories"][6]
        longest = max(summary["requests"], key=lambda r: r["completion_tokens"])
        self.assertGreater(longest["steps"][-1]["S"],
                           longest["prompt_tokens"] + longest["completion_tokens"])
        self.assertEqual(longest["steps"][-1]["T"], 8)

    def test_swa_union_preserves_each_query_window(self):
        for case in op.cell_cases(self.manifest()):
            if case["phase"] != 0:
                continue
            union = set()
            for q in range(case["prefix"], case["prefix"] + 8):
                union.update(range(max(0, q - 127), q + 1))
            actual = set(range(case["swa_start"], case["swa_start"] + case["swa_s"]))
            self.assertEqual(actual, union)
            self.assertLessEqual(case["swa_s"], 135)

    def test_cold_prefill_points_and_d7_bars(self):
        manifest = self.manifest()
        self.assertEqual(manifest["cold_prefill"], [2048, 8192, 32768, 65536])
        self.assertEqual(manifest["d7_bars"]["c1_mean_ttft_s"], 0.251)
        self.assertEqual(manifest["d7_bars"]["cold_prefill_tok_s"],
                         {"2048": 2999, "8192": 2975, "32768": 2671, "65536": 2114})
        self.assertEqual([s["prompt_tokens"] for s in manifest["short_prefill"]],
                         [49, 70, 47, 44, 82, 69, 309, 120, 92])

    def test_selection_cases_unchanged(self):
        rows = op.cases(self.manifest())
        self.assertEqual(len(rows), 378)  # 369 C1 steps + 9 prefill
        self.assertEqual(sum(r["select"] for r in rows), 36)
        self.assertEqual([r["id"] for r in rows], list(range(378)))

    def test_cell_cases_count_and_phases(self):
        rows = op.cell_cases(self.manifest())
        self.assertEqual(len(rows), 2651)  # 2642 verification + 9 short prefill
        self.assertEqual(sum(r["phase"] == 0 for r in rows), 2642)
        self.assertEqual(sum(r["phase"] == 1 for r in rows), 9)
        for row in rows:
            if row["phase"] == 0:
                self.assertGreaterEqual(row["request"], 0)
                self.assertGreaterEqual(row["step"], 0)
                self.assertEqual(row["t"], 8)
            else:
                self.assertEqual(row["request"], -1)
                self.assertEqual(row["step"], -1)
                self.assertGreaterEqual(row["t"], 44)

    def test_cell_header_shape(self):
        header = op.cell_header(op.load_manifest())
        self.assertEqual(header.count('},'), 2651)
        self.assertIn("struct CellCase {", header)
        self.assertIn("namespace op1cell {", header)
        self.assertIn("cold_prefill[] = {2048,8192,32768,65536}", header)
        self.assertIn(op.load_manifest()["inputs"]["d7_sha256"], header)

    def test_selection_header_shape(self):
        header = op.cpp_header(op.load_manifest())
        self.assertEqual(header.count('},'), 378)
        self.assertIn("struct Case {", header)
        self.assertIn("select;", header)

    def test_ceiling_count_excluded(self):
        self.assertTrue(any(b["category"] == "ceiling_count" for b in self.data["batches"]))
        manifest = self.manifest()
        self.assertTrue(all(r["category"] != "ceiling_count" for r in manifest["categories"]))

    def test_duplicate_category_rejected(self):
        self.data["batches"].append(copy.deepcopy(next(b for b in self.data["batches"] if b["category"] == "coding")))
        with self.assertRaises(ValueError): self.manifest()

    def test_missing_category_rejected(self):
        self.data["batches"] = [b for b in self.data["batches"] if b["category"] != "coding"]
        with self.assertRaises(ValueError): self.manifest()

    def test_unknown_category_rejected(self):
        next(b for b in self.data["batches"] if b["category"] == "coding")["category"] = "other"
        with self.assertRaises(ValueError): self.manifest()

    def test_wrong_identity_rejected(self):
        for key in ("label", "prompt_set"):
            old = self.data[key]
            self.data[key] = "wrong"
            with self.assertRaises(ValueError): self.manifest()
            self.data[key] = old

    def test_wrong_request_count_rejected(self):
        batch = next(b for b in self.data["batches"] if b["category"] == "coding" and b["c"] == 6)
        batch["requests"] = batch["requests"][:5]
        with self.assertRaises(ValueError): self.manifest()

    def test_wrong_concurrency_rejected(self):
        batch = next(b for b in self.data["batches"] if b["category"] == "coding" and b["c"] == 6)
        batch["c"] = 3
        with self.assertRaises(ValueError): self.manifest()

    def test_acceptance_invalid(self):
        for value in (None, True, "5.88", float("nan"), float("inf"), -1, 7.01, 0.123):
            with self.subTest(value=value), self.assertRaises(ValueError):
                op.accepted_hundredths(value)
        self.assertEqual(op.accepted_hundredths(0), 0)
        self.assertEqual(op.accepted_hundredths(7), 700)

    def test_zero_and_full_acceptance_progress(self):
        coding = next(b for b in self.data["batches"] if b["category"] == "coding" and b["c"] == 1)
        for rate, first_advance in ((0, 1), (7, 8)):
            coding["accept_per_draft"] = rate
            req = next(r for r in self.manifest()["categories"][0]["requests"] if r["c"] == 1)
            self.assertEqual(req["steps"][0]["advance_model"], first_advance)
            self.assertEqual(sum(x["advance_committed"] for x in req["steps"]), 172)

    def test_token_counts_invalid(self):
        request = next(b for b in self.data["batches"] if b["category"] == "coding" and b["c"] == 1)["requests"][0]
        for field in ("prompt_tokens", "completion_tokens"):
            old = request[field]
            for bad in (0, -1, True, 1.2, "49", 1000):
                request[field] = bad
                with self.subTest(field=field, bad=bad), self.assertRaises(ValueError): self.manifest()
            request[field] = old

    def test_batch_total_mismatch(self):
        batch = next(b for b in self.data["batches"] if b["category"] == "coding" and b["c"] == 1)
        batch["tokens"] += 1
        with self.assertRaises(ValueError): self.manifest()

    def test_no_decode_after_one_completion_token(self):
        batch = next(b for b in self.data["batches"] if b["category"] == "coding" and b["c"] == 1)
        batch["tokens"] = batch["requests"][0]["completion_tokens"] = 1
        req = next(r for r in self.manifest()["categories"][0]["requests"] if r["c"] == 1)
        self.assertEqual(req["steps"], [])

    def test_layer_shape_rejections(self):
        for pattern in ([1]*48, self.pattern[:-1], [True] + self.pattern[1:], [2] + self.pattern[1:]):
            with self.assertRaises(ValueError): op.build_manifest(self.data, pattern)
        for source in ("class MiMoConfig: pass", "class MiMoConfig:\n hybrid_layer_pattern: tuple = (1,)*48"):
            with self.assertRaises(ValueError): op.layer_pattern(source)

    def test_digest_identity_and_gate_scope(self):
        manifest = op.load_manifest()
        self.assertEqual(set(manifest["inputs"]),
                         {"d7_sha256", "d7_result_sha256", "config_sha256", "counter_source_sha256"})
        self.assertTrue(all(len(x) == 64 for x in manifest["inputs"].values()))
        self.assertEqual(manifest["gate"], "UNSET-pending-R19a")
        self.assertEqual(manifest["schema"], "mimo26-op1-schedule-v2")
        self.assertEqual(manifest["amendment"], "519a721-full-distributions-plus-cold-prefill")

    def test_nearest_rank_p95_and_mean(self):
        stats = op.step_statistics(list(range(1, 21)))
        self.assertEqual(stats, {"mean_ms": 10.5, "p95_ms": 19, "sample_count": 20})
        self.assertEqual(op.step_statistics([7, 1, 4])["p95_ms"], 7)

    def test_nonfinite_empty_or_nonpositive_times_rejected(self):
        for samples in ([], [0], [-1], [float("nan")], [float("inf")], [True], ["1"]):
            with self.assertRaises(ValueError): op.step_statistics(samples)

    def test_equal_category_not_pooled_step_weighting(self):
        samples = {name: [i+1] for i, name in enumerate(op.CATEGORIES)}
        samples["coding"] *= 100
        result = op.category_statistics(samples)
        self.assertEqual(result["category_weighted_step_ms"], 5)
        self.assertAlmostEqual(result["pooled_step_ms_diagnostic"], 144/108)
        for name in ("ceiling_count", "missing"):
            bad = copy.deepcopy(samples)
            if name == "missing": bad.pop("coding")
            else: bad[name] = [1]
            with self.assertRaises(ValueError): op.category_statistics(bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
