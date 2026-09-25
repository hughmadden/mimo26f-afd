"""spike/twin_loop.py — P-103: bare prefill + greedy decode loop (topology C).

Pipeline: tokenize short prompt -> prefill -> >=8 greedy decode steps -> print
tokens.  Honors T3-T8 on the twin: T9 start_pos continuation (model.py:110),
SWA eviction ``min(batch_pos) - window + 1`` (kv.py:82), T3 GA window drop,
T5 query==context v_scale, partial rotary 0.334.  Sink (T6/T7) is N/A at
sink_dim=0 in the tiny config (called out in the probe output).

Salad policy (AGENTS.md §3): if a transcript is salad, LOCALIZE — the probe
block below reruns each trap invariant and names the failing stage
(loader / attn / rope / start_pos / eviction).  No sampling hacks: greedy =
argmax, always.

Topology C = CPU/tiny synthetic weights (this file, ``--tiny``).  Synthetic
weights are random -> their transcripts are mechanically coherent (incremental
== full recompute gate) but semantically meaningless; the three REAL
non-salad transcripts come from topology A (real weights, the coordinator RTX 5090).

Run ONLY via ``scripts/dev.sh spike cpu``.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from spike import attn as attn_mod
from spike import kv as kvmod
from spike.model import SpikeModel, init_weights

V_SCALE = 0.707
CFG = {
    "n_layers": 2,
    "d_model": 8,
    "vocab": 258,
    "sliding_window": 4,
    "is_ga": [True, False],
    "sink_dim": 0,
    "v_scale": V_SCALE,
}

PROMPTS = {
    "factual": "Paris is the capital of",
    "arithmetic": "2+3=",
    "continuation": "The quick brown",
}


def tokenize(s: str) -> list[int]:
    return [b + 1 for b in s.encode("utf-8")]


def detokenize(ids) -> str:
    return bytes(max(0, i - 1) for i in ids).decode("utf-8", "replace")


def greedy_loop(model: SpikeModel, prompt_ids: list[int], steps: int = 8) -> list[int]:
    out = model.forward(np.asarray(prompt_ids, np.int64))          # prefill
    tok = int(np.argmax(out[-1]))
    gen: list[int] = []
    for _ in range(steps):
        gen.append(tok)
        lg = model.forward(np.asarray([tok], np.int64))            # start_pos -> cache.tokens (T9)
        tok = int(np.argmax(lg[-1]))
    return gen


def probes(model_factory) -> list[tuple[str, bool, str]]:
    """Salad localization: one row per trap stage; a False row NAMES the stage."""
    rows: list[tuple[str, bool, str]] = []

    ga = attn_mod.ga_cache_config(CFG)
    rows.append(("T3 ga window None", ga["sliding_window"] is None, "attn/window"))

    blk = attn_mod.SelfAttention(8, np.zeros((24, 8), np.float32), np.eye(8, dtype=np.float32),
                                 sliding_window=None, v_scale=V_SCALE)
    rng = np.random.default_rng(3)
    v_pre = rng.standard_normal((2, 8)).astype(np.float32)
    _, _, v_q = blk.project_qkv(rng.standard_normal((2, 8)).astype(np.float32))
    _, v_c = blk.inject_context_kv(v_pre, v_pre)
    ok = np.allclose(v_c, v_pre * V_SCALE, rtol=1e-6)
    rows.append(("T5 context v scaled", bool(ok), "attn/v_scale"))

    # F7 (P-105): env-default impl — the probe must be able to FAIL on the
    # naive run; the old hardcoded naive=False made the T8 row unfalsifiable.
    kv = kvmod.KVCache(1)
    k = np.zeros((1, 8), np.float32)
    for p in range(6, 11):
        kv.append(0, np.asarray([p]), k, k, window=4)
    # AGENTS.md §3: evict only entries older than min(batch_pos) - window + 1.
    # Final batch_pos=10, window=4 -> keep >= 7 -> [7, 8, 9, 10].
    ok = (list(kv.pos_of(0)) == [7, 8, 9, 10]
          and int(kv.pos_of(0)[0]) == 10 - 4 + 1)
    # batch case: keep only entries not older than min(batch_pos) - window + 1
    kv.append(0, np.asarray([12, 13, 14]), k, k, window=4)
    actual = list(int(p) for p in kv.pos_of(0))
    expect = [p for p in [7, 8, 9, 10, 12, 13, 14] if p >= 12 - 4 + 1]
    ok = ok and actual == expect
    rows.append(("T8 eviction min(batch_pos)-w+1", bool(ok),
                 f"kv/eviction (kept {actual}, rule says {expect})"))

    m_full = model_factory()
    full = m_full.logits([1, 2, 3, 4, 5])[-1]
    m_inc = model_factory()
    m_inc.logits([1, 2, 3, 4])
    inc = m_inc.forward(np.asarray([5]))[-1]
    ok = np.allclose(inc, full, rtol=1e-5, atol=1e-6)
    rows.append(("T9 start_pos continuation", bool(ok), "start_pos/rope"))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true", help="topology C: synthetic tiny weights")
    ap.add_argument("--steps", type=int, default=8)
    args = ap.parse_args()

    def factory():
        return SpikeModel(init_weights(CFG, 0), CFG)

    print("== spike twin loop (topology C, tiny synthetic weights)")
    rows = probes(factory)
    bad = [(n, s) for n, ok, s in rows if not ok]
    for n, ok, s in rows:
        print(f"probe {n}: {'PASS' if ok else 'FAIL -> ' + s}")
    print("probe T6 sink: N/A at sink_dim=0 in the tiny config — covered by "
          "test_p101_addendum at sink_dim=1; T7 dual-θ is LIVE here "
          "(rope_theta_for: GA 1e7 / SWA 1e4), not waived")
    if bad:
        print(f"RESULT: FAIL salad localized at: {[s for _, s in bad]}")
        return 1

    for name, text in PROMPTS.items():
        model = factory()
        ids = tokenize(text)
        gen = greedy_loop(model, ids, args.steps)
        print(f"[{name}] prompt={text!r} -> {detokenize(gen)!r} ids={gen}")
    print("RESULT: PASS (coherence gate green; semantic transcripts require topology A real weights)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
