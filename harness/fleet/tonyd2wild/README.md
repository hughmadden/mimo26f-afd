# Vendored fleet probes: tonyd2wild MiMo recipe @ `13621bb` (MIT)

These are byte-verbatim copies. REUSE rows are in `docs/REUSE.md`, and the pins
live in `harness/selftests/test_fleet_tools.py`. They are L5/L6 fleet drivers:
never run them in CI, and never point one at a model until its selftest row is
complete (TEST-PLAN R6).

| File | Gate / trap | What it proves |
|---|---|---|
| `stress-corrupt.py` | **G8** (T26), tool probe (T24) | Runs 24 concurrent strict 300-line hex outputs; the pass bar is 0 hard-bad lines. The tool probe counts calls per response. |
| `replay_exact.py` | T24 regression | Replays a captured agent request body N times and counts tool calls. With the cap: ≤ 6 calls and `finish_reason: tool_calls`. |
| `mimobench.py` | Bench of record | Prompt set v1, C1–C6 across 9 categories, and cold prefill. Tokens come from the usage block and concurrency is real. Results compare 1:1 with the measured TP4 bar (ADVISOR-I3 §10.1). |
| `mimo_needle.py` | **G4** needle ladder | Filler text with a code at depths 0.1 / 0.5 / 0.9. Temperature 0, thinking off. |
| `toolcap-proxy.cjs` | Reference for T24 | Defines the semantics the engine-side `tool_call_cap` must reproduce: cut when call 7 opens, drop the partial call, end with `finish_reason: tool_calls`, abort upstream. |

The endpoint paths and model name inside the scripts are upstream's: pass our
URL on the command line, and our served model name where a flag exists. Do not
edit the files. Wrap them instead, so the pins keep meaning something.
