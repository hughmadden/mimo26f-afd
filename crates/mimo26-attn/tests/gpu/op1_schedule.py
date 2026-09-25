"""R19 OP1 aggregate-derived context schedule (amended 519a721); not a trace.

The D7 acceptance counter counts accepted draft tokens, not the guaranteed
bonus/replacement token, so context advance is 1 + accept_per_draft. Prefill
emits the first completion token; verification covers completion_tokens - 1
more. Every verification step evaluates the current uncached anchor plus seven
candidates (always T = 8).

R19 amendment scope:
- Verification replays the FULL per-category length distributions (both C1 and
  C6 batches, 7 requests per category, maximum prompt+completion = 434). The
  D7 lengths are non-identical under deterministic decoding (identity note).
- Prefill has two arms: short single-chunk TTFT at the C1 prompt lengths, and
  cold full-request multi-chunk P1 at 2K / 8K / 32K / 64K.
- 128K/1M decode and 1M/SWA-32K chunk rows remain recorded MISSes (I5 debt).
"""
from __future__ import annotations

import argparse
import ast
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import statistics

CATEGORIES = ("coding", "json", "narrative", "prose", "math", "reasoning",
              "summary", "structured", "format")
COLD_PREFILL = (2048, 8192, 32768, 65536)
D7_BARS = {"c1_mean_ttft_s": 0.251,
           "cold_prefill_tok_s": {"2048": 2999, "8192": 2975, "32768": 2671, "65536": 2114}}

ROOT = Path(__file__).resolve().parents[4]
D7 = ROOT / "runs/20260923-d7-tp4-baseline/bench/bench-d7-ourfleet-tp4-gmu84.json"
D7_RESULT = ROOT / "runs/20260923-d7-tp4-baseline/RESULT.md"
CONFIG = ROOT / "oracle/mimo26/config.py"
COUNTERS = ROOT / "harness/fleet/tonyd2wild/mimobench.py"


def integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def accepted_hundredths(value):
    if type(value) not in (int, float):
        raise ValueError("accept_per_draft must be numeric")
    try:
        d = Decimal(str(value))
        if not d.is_finite() or d < 0 or d > 7 or d * 100 != (d * 100).to_integral_value():
            raise ValueError("accept_per_draft outside block-8 / two-decimal contract")
        return int(d * 100)
    except InvalidOperation as exc:
        raise ValueError("invalid acceptance counter") from exc


def layer_pattern(source):
    """Read the frozen oracle's literal config without executing oracle code."""
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "MiMoConfig":
            for field in node.body:
                if (isinstance(field, ast.AnnAssign) and isinstance(field.target, ast.Name)
                        and field.target.id == "hybrid_layer_pattern"):
                    values = ast.literal_eval(field.value)
                    if (not isinstance(values, tuple) or len(values) != 48
                            or any(type(x) is not int or x not in (0, 1) for x in values)
                            or values.count(0) != 9 or values.count(1) != 39):
                        raise ValueError("expected exact 9 GA + 39 SWA layer geometry")
                    return list(values)
    raise ValueError("missing literal hybrid_layer_pattern")


def request_steps(prompt, completion, accept_hundredths):
    """Integer floor-difference acceptance; advance = 1 + accepted drafts.

    Terminal committed progress is capped to the completion budget; the model
    still evaluates T = 8 speculative queries at every step.
    """
    a = accept_hundredths
    generated = 0
    steps = []
    while generated < completion - 1:
        i = len(steps)
        # Integer discrepancy < 1 draft token at every prefix; no float drift.
        accepted = ((i + 1) * a) // 100 - (i * a) // 100
        advance = 1 + accepted
        committed = min(advance, completion - 1 - generated)
        prefix = prompt + generated
        swa_start = max(0, prefix - 127)
        steps.append({"index": i, "cache_prefix": prefix, "T": 8,
                      "S": prefix + 8, "swa_key_start": swa_start,
                      "swa_S": prefix + 8 - swa_start, "query_start": prefix,
                      "query_end": prefix + 7, "draft_accepted_model": accepted,
                      "advance_model": advance, "advance_committed": committed,
                      "terminal_cap": committed != advance})
        generated += committed
    return steps


def build_request(batch, request):
    p = integer(request.get("prompt_tokens"), "prompt_tokens")
    c = integer(request.get("completion_tokens"), "completion_tokens")
    if not (44 <= p <= 309 and p + c <= 434):
        raise ValueError("outside frozen D7 prompt/completion range")
    a = accepted_hundredths(batch.get("accept_per_draft"))
    concurrency = integer(batch.get("c"), "concurrency")
    if concurrency not in (1, 6):
        raise ValueError("only C1/C6 concurrency cohorts are replayed")
    return {"c": concurrency, "prompt_tokens": p, "completion_tokens": c,
            "accept_hundredths": a,
            "steps": request_steps(p, c, a),
            "prefill": {"T": p, "S": p, "query_start": 0, "query_end": p - 1}}


def build_manifest(data, pattern):
    if data.get("label") != "d7-ourfleet-tp4-gmu84" or data.get("prompt_set") != "v1":
        raise ValueError("wrong D7 receipt identity")
    by_category = {}
    for batch in data["batches"]:
        if type(batch.get("c")) is not int or type(batch.get("tokens")) is not int:
            raise ValueError("concurrency/tokens must be integers")
        name = batch.get("category")
        if name == "ceiling_count":
            continue
        if name not in CATEGORIES:
            raise ValueError("unknown headline category")
        requests = batch.get("requests")
        if not isinstance(requests, list) or not requests:
            raise ValueError("batch must carry its requests")
        if integer(batch["tokens"], "batch tokens") != sum(
                integer(r["completion_tokens"], "completion") for r in requests):
            raise ValueError("completion count does not match batch total")
        for request in requests:
            by_category.setdefault(name, []).append(build_request(batch, request))
    if set(by_category) != set(CATEGORIES):
        raise ValueError("missing headline category")
    for name in CATEGORIES:
        rows = by_category[name]
        if len(rows) != 7 or [r["c"] for r in rows].count(1) != 1 or [r["c"] for r in rows].count(6) != 6:
            raise ValueError("each category must replay exactly one C1 and six C6 requests")
    if (len(pattern) != 48 or any(type(x) is not int or x not in (0, 1) for x in pattern)
            or pattern.count(0) != 9 or pattern.count(1) != 39):
        raise ValueError("wrong all-layer pattern")
    categories = []
    for name in CATEGORIES:
        rows = by_category[name]
        c1 = next(r for r in rows if r["c"] == 1)
        categories.append({
            "category": name,
            # C1 convenience view, kept for the selection controls.
            "prompt_tokens": c1["prompt_tokens"],
            "completion_tokens": c1["completion_tokens"],
            "accept_hundredths": c1["accept_hundredths"],
            "steps": c1["steps"],
            "prefill": c1["prefill"],
            "requests": rows,
        })
    short_prefill = [{"category": name, "prompt_tokens": row["prompt_tokens"]}
                     for name, row in zip(CATEGORIES, categories)]
    return {"schema": "mimo26-op1-schedule-v2",
            "scope": "aggregate-derived-synthetic-context-proxy",
            "amendment": "519a721-full-distributions-plus-cold-prefill",
            "acceptance_semantics": "accepted-drafts-plus-one-bonus",
            "first_completion_token": "emitted-by-prefill-not-a-verification-step",
            "progression": "floor-difference-two-decimal-acceptance; terminal advance capped",
            "speculative_queries": "one-uncached-anchor-plus-seven-candidates; always-T8",
            "cache_semantics": "commit-only-accepted-prefix; overwrite-rejected-lookahead",
            "swa_view": "union-of-all-eight-query-windows; absolute-positions; at-most-135-keys",
            "prefill_arm": "short-single-chunk-TTFT-at-C1-prompts; cold-multichunk-P1-2k-8k-32k-64k",
            "verification_replay": "full-per-category-distributions-C1-and-C6; max-p-plus-c-434",
            "length_identity_note": "deterministic-decode-yields-non-identical-completions-per-prompt",
            "category_aggregation": "equal-one-ninth-of-category-means; pooled-step-mean-separate",
            "step_tail": "nearest-rank-p95; retain-every-sample",
            "gate": "UNSET-pending-R19a", "layer_pattern": pattern,
            "d7_bars": D7_BARS,
            "layers": [{"id": i, "kind": "swa" if v else "ga", "n_q": 64,
                        "n_kv": 8 if v else 4, "qk": 192, "v": 128,
                        "window": 128 if v else 0, "sink": "per-Q-head" if v else "absent"}
                       for i, v in enumerate(pattern)],
            "categories": categories,
            "short_prefill": short_prefill,
            "cold_prefill": list(COLD_PREFILL)}


def step_statistics(samples):
    if not samples or any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in samples):
        raise ValueError("step times must be nonempty, finite and positive")
    ordered = sorted(samples)
    return {"mean_ms": statistics.mean(samples),
            "p95_ms": ordered[(95 * len(ordered) + 99) // 100 - 1],
            "sample_count": len(samples)}


def category_statistics(samples):
    if set(samples) != set(CATEGORIES):
        raise ValueError("exact nine headline categories required")
    rows = {name: step_statistics(samples[name]) for name in CATEGORIES}
    return {"categories": rows,
            "category_weighted_step_ms": statistics.mean(row["mean_ms"] for row in rows.values()),
            "pooled_step_ms_diagnostic": statistics.mean(x for name in CATEGORIES for x in samples[name])}


def load_manifest(d7=D7, config=CONFIG):
    payload, cfg = d7.read_bytes(), config.read_bytes()
    result = build_manifest(json.loads(payload), layer_pattern(cfg.decode()))
    result["inputs"] = {"d7_sha256": hashlib.sha256(payload).hexdigest(),
                        "d7_result_sha256": hashlib.sha256(D7_RESULT.read_bytes()).hexdigest(),
                        "config_sha256": hashlib.sha256(cfg).hexdigest(),
                        "counter_source_sha256": hashlib.sha256(COUNTERS.read_bytes()).hexdigest()}
    return result


# ---- selection controls (frozen; consumed by the retired-but-retained cell) ----
def cases(manifest):
    """Per-category C1 first/middle/last + short prefill (selection controls)."""
    result = []
    for category, row in enumerate(manifest["categories"]):
        result.append({"id": len(result), "category": category, "step": -1,
                       "t": row["prompt_tokens"], "prefix": 0, "s": row["prompt_tokens"],
                       "swa_start": 0, "swa_s": row["prompt_tokens"], "select": True})
        selected = {0, len(row["steps"]) // 2, len(row["steps"]) - 1}
        for step in row["steps"]:
            result.append({"id": len(result), "category": category, "step": step["index"],
                           "t": 8, "prefix": step["cache_prefix"], "s": step["S"],
                           "swa_start": step["swa_key_start"], "swa_s": step["swa_S"],
                           "select": step["index"] in selected})
    return result


def cpp_header(manifest):
    lines = ["// Generated by op1_schedule.py; selection controls (C1 first/middle/last).",
             "#pragma once", "namespace op1 {",
             "struct Case { int id, category, step, t, prefix, s, swa_start, swa_s, select; };",
             'constexpr const char* d7_sha = "' + manifest["inputs"]["d7_sha256"] + '";',
             'constexpr const char* categories[] = {' + ','.join(json.dumps(x) for x in CATEGORIES) + '};',
             "constexpr int layers[] = {" + ','.join(map(str, manifest["layer_pattern"])) + "};",
             "constexpr Case cases[] = {"]
    for row in cases(manifest):
        lines.append("{" + ','.join(str(int(x)) for x in row.values()) + "},")
    lines.extend(["};", "} // namespace op1"])
    return '\n'.join(lines) + '\n'


# ---- full-driver cases (verification replay + short prefill + cold sizes) ----
def cell_cases(manifest):
    """Full distribution: every request's verification steps, then short prefill.

    phase 0 = verification (request index and step index valid); phase 1 =
    short single-chunk prefill (request/step = -1).
    """
    result = []
    for category, row in enumerate(manifest["categories"]):
        for request, req in enumerate(row["requests"]):
            for step in req["steps"]:
                result.append({"id": len(result), "category": category,
                               "request": request, "step": step["index"], "phase": 0,
                               "t": 8, "prefix": step["cache_prefix"], "s": step["S"],
                               "swa_start": step["swa_key_start"], "swa_s": step["swa_S"]})
        result.append({"id": len(result), "category": category, "request": -1,
                       "step": -1, "phase": 1, "t": row["prompt_tokens"],
                       "prefix": 0, "s": row["prompt_tokens"],
                       "swa_start": 0, "swa_s": row["prompt_tokens"]})
    return result


def cell_header(manifest):
    lines = ["// Generated by op1_schedule.py; full-distribution OP1 cell cases.",
             "#pragma once", "namespace op1cell {",
             "struct CellCase { int id, category, request, step, phase, t, prefix, s, swa_start, swa_s; };",
             'constexpr const char* d7_sha = "' + manifest["inputs"]["d7_sha256"] + '";',
             'constexpr const char* categories[] = {' + ','.join(json.dumps(x) for x in CATEGORIES) + '};',
             "constexpr int layers[] = {" + ','.join(map(str, manifest["layer_pattern"])) + "};",
             "constexpr int cold_prefill[] = {" + ','.join(map(str, COLD_PREFILL)) + "};",
             "constexpr CellCase cases[] = {"]
    for row in cell_cases(manifest):
        lines.append("{" + ','.join(str(int(x)) for x in row.values()) + "},")
    lines.extend(["};", "} // namespace op1cell"])
    return '\n'.join(lines) + '\n'


def _summary(manifest):
    print("OP1_SCHEDULE scope=aggregate-derived-synthetic-context-proxy gate=UNSET amendment=519a721")
    print("INPUTS " + json.dumps(manifest["inputs"], sort_keys=True))
    print("D7_BARS " + json.dumps(manifest["d7_bars"], sort_keys=True))
    total = 0
    for row in manifest["categories"]:
        lengths = sorted({(r["prompt_tokens"], r["completion_tokens"]) for r in row["requests"]})
        n = sum(len(r["steps"]) for r in row["requests"])
        total += n
        max_pc = max(p + c for p, c in lengths)
        print(f"CATEGORY {row['category']} requests=7 c1_prompt={row['prompt_tokens']} "
              f"c1_accept={row['accept_hundredths']/100:.2f} steps={n} max_p_plus_c={max_pc} "
              f"distinct_lengths={len(lengths)}")
    print("TOTAL_VERIFICATION_STEPS " + str(total))
    print("SHORT_PREFILL " + ",".join(str(s["prompt_tokens"]) for s in manifest["short_prefill"]))
    print("COLD_PREFILL " + ",".join(map(str, manifest["cold_prefill"])))
    print("LAYER_PATTERN " + ",".join(str(x) for x in manifest["layer_pattern"]))
    print("RESULT: PASS CPU schedule only; no GPU timing or captured trace")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--summary", action="store_true")
    output.add_argument("--header", action="store_true", help="selection-control header")
    output.add_argument("--header-cell", action="store_true", help="full-driver cell header")
    args = parser.parse_args()
    result = load_manifest()
    if args.header:
        print(cpp_header(result), end="")
    elif args.header_cell:
        print(cell_header(result), end="")
    elif args.summary:
        _summary(result)
    else:
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
