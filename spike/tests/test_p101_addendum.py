"""P-106 addendum — T2-real-pad-64 / T4 / T6 / T7 / T10 / T14 trap negatives
+ P-105 F6/F7/F8 review fixes + F1-re-review close-out (c1/c2 + hardening).

Closes the P-105 review holes (runs/20260923-i1-spike/P-105-review.md):
  F1/T4  attention-path QK192/V128 negative (scaling + verified-rows gate +
         THROUGH-__call__ call-site pin, c2 hardening)
  F2/T6  sink LIVE on SWA (extra softmax-logit column), family-gated (c1)
  F2/T7  rotary 64-dim / dual-θ negative
  F5/T2  real GA pad=64: 3392 rows/shard → 27-row local grid, grid row 26 is
         the HALF-PAD row — poisoned here at block=128 real segs, with a
         ROW-IDENTITY pin (not just the count — F1-re-review hardening)
  F6/T3  window asserts retargeted to MODEL-BUILT layers (decision site
         model.py:63), not the dead ga_cache_config path
  F7/T8  eviction negative on CACHE CONTENTS with the env-default impl in
         BOTH runs
  F8/T11 missing-bias / missing-name negative (model.py:50-56 fail-loud guards)

F1-re-review REFUSE close-out (P-106 task-6, 23 Sep 2026 AEST): T10 E8M0-255
clamp and T14 nibble order have REAL naive-negative flips HERE (the golden
fixture's scales_u8 contain only {127,128} — byte 255 was never exercised):
  test_t10_e8m0_clamp_255_negative / test_t10_unpack_255_scale_byte_negative
  test_t14_nibble_order_negative
  (c1) test_c1_sink_family_gated_negative — GA-with-sink == GA-without-sink
       BITWISE + SWA contrast observable (flips: naive drops the sink)
  (c2) test_c2_attn_scale_through_call_negative — d_qk=5, d_v=2 through
       __call__ vs an independent causal reference (flips: naive scales by
       the V width); a swapped call-site is observable in the same test.
       test_c2_attn_scale_arg_order_pin (both runs, explicit naive=False)
       pins the attn_scale (d_qk, d_v) argument order.

External oracle (AGENTS §4.4; spec use only, code bodies lifted: NONE):
  the coordinator:/srv/models/XiaomiMiMo/MiMo-V2.6-Flash-RL/modeling_mimo_v2.py
  sha256 a8c3cb3aae473bcc15f023010547c919f15eba6546e6ed7efb61a8937b12f3ad
  sink extra-logit column :89-96 · rotary inv_freq/rotate-half :28-42 ·
  score scaling head_dim**-0.5 :61-63 · verified config: rope_theta 1e7 (GA) /
  swa_rope_theta 1e4 (SWA) / partial_rotary_factor 0.334 → 64 of 192 /
  add_swa_attention_sink_bias=true / add_full_attention_sink_bias=false.
  MXFP4 spec: OCP MXFP4 E8M0-32, byte 255 reserved (clamp 2^127); T14
  low-nibble = even k (mxfp4_golden.json e2m1_codebook_by_nibble).

Two-run classification (scripts/dev.sh spike tests vs spike tests --naive):
  TRAP NEGATIVES — FAIL on the naive run (env-default impl flips wrong):
    test_t2_real_pad64_half_pad_scale_row_negative  (row identity pinned too)
    test_t4_score_scale_uses_qk_dim_negative
    test_c2_attn_scale_through_call_negative   (NEW c2: swapped call-site)
    test_t6_sink_live_on_swa_negative
    test_c1_sink_family_gated_negative         (NEW c1: family gate + SWA live)
    test_t7_dual_theta_negative
    test_t10_e8m0_clamp_255_negative           (NEW T10, env-default)
    test_t10_unpack_255_scale_byte_negative    (NEW T10, synthetic 255 byte)
    test_t14_nibble_order_negative             (NEW T14)
    test_f6_window_on_model_built_layers_negative
    test_f7_t8_eviction_keeps_rule_set_negative
  BOTH RUNS PASS (fail-loud guards + detection-power pairs + spec pins;
  explicit naive=False where the semantics are env-invariant):
    test_t2_naive_global_mapping_mispoison_is_visible
    test_t4_v_as_192_rows_gate_rejects
    test_c2_attn_scale_arg_order_pin
    test_t6_sink_drop_bug_is_visible
    test_t7_rope_dim_64_pin / test_t7_apply_rotary_matches_oracle_formula
    test_t10_clamp_255_detectable
    test_f7_t8_naive_eviction_difference_is_visible
    test_f8_t11_missing_router_bias_negative
    test_f8_t11_missing_required_name_negative
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np
import pytest

from spike import attn, kv, mxfp4, model
from spike import real_loader as RL


def _tiny_cfg(n_layers: int = 2, d: int = 8, vocab: int = 16) -> dict:
    return {
        "n_layers": n_layers,
        "d_model": d,
        "vocab": vocab,
        "sliding_window": 4,
        # spike's is_ga is BOOLEAN GA-ness (True = GA) — cf. twin_loop CFG
        # [True, False] and model.py:63 ``None if cfg["is_ga"][i] else window``.
        # NB: the REAL config's hybrid_layer_pattern has the OPPOSITE polarity
        # (0 = GA, 1 = SWA); real_loop maps it — fixtures must not copy it raw
        # (that inversion mis-targeted F6 on the first two-run record).
        "is_ga": [True, False][:n_layers],  # layer 0 GA, layer 1 SWA
        "sink_dim": 0,
        "v_scale": 1.0,
    }


# ---------------------------------------------------------------------------
# T2 real-pad-64 (F5): real GA per-shard geometry, half-pad grid row poisoned
# ---------------------------------------------------------------------------

_CODE = 0x38  # any nonzero E4M3 code (exact decode value is immaterial here)


def _pad_case() -> tuple[list[dict], list[dict]]:
    """Real-geometry GA pad fixture at block=128.

    Verified layout (real_loader.py:3-18, P-102 evidence): GA per shard 3392
    rows = 26.5 block-rows at br=128 → LOCAL grid 27 rows; grid row 26 is the
    HALF-PAD row covering shard rows 3328..3391 = 64 real rows (the F5
    "pad=64 / half-pad last grid row" that synthetic fixtures never exercise).
    The trap: scale rows are indexed by the LOCAL row ``r//128`` per shard;
    a naive GLOBAL ``row//128`` mapping drifts onto other shards' grid rows —
    including this half-pad row.
    """
    per = RL.PER_SHARD_GA  # (3072, 192, 128) real segs
    rows = sum(per)
    assert rows == 3392 and rows % 128 != 0  # the half-pad condition, live
    cols = 128  # exactly one FP8 block wide (RL.FP8_BLOCK = (128, 128))

    def grid(poison: bool) -> np.ndarray:
        g = np.ones((27, 1), np.float32)  # 27 local rows, 32-col grid → 1 here
        if poison:
            g[26, 0] = 8.0  # distinct scale on the HALF-PAD row
        return g

    clean = [
        {"w8": np.full((rows, cols), _CODE, np.uint8),
         "s": grid(False), "block": RL.FP8_BLOCK}
        for _ in range(4)
    ]
    pois = [
        {"w8": np.full((rows, cols), _CODE, np.uint8),
         "s": grid(True), "block": RL.FP8_BLOCK}
        for _ in range(4)
    ]
    return pois, clean


def _affected_rows(shards: list[dict], clean: list[dict], naive: bool | None) -> np.ndarray:
    """Rows whose dequantized value changed by the row-26 scale poison.

    Layout- and codebook-agnostic: poison scales those rows by exactly 8x
    (power of two ⇒ exact in f32), so the affected set is the set of rows the
    mapping assigned to local grid row 26.
    """
    a = RL.reconstruct_layer_qkv_now(shards, RL.PER_SHARD_GA, naive=naive)
    b = RL.reconstruct_layer_qkv_now(clean, RL.PER_SHARD_GA, naive=naive)
    return np.any(a != b, axis=1)


def test_t2_real_pad64_half_pad_scale_row_negative():
    """TRAP NEGATIVE — FAILS on the naive run (F5 + row-identity hardening)."""
    pois, clean = _pad_case()
    got_env = _affected_rows(pois, clean, naive=None)  # env-default impl
    got_ok = _affected_rows(pois, clean, naive=False)
    # Correct local mapping: ONLY the 64 real rows under the half-pad block,
    # per shard: 4 x 64 = 256 rows (externally anchored: 3392 = 26x128 + 64,
    # the verified per-shard row count — not a re-derived formula constant).
    assert int(got_ok.sum()) == 256
    # ROW-IDENTITY pin (hardening): exact affected row set, from VERIFIED
    # geometry literals only (P-102 evidence — not formula-derived):
    #   output layout [Q 12288; K 768; V 512] (loader.py:85-86 regroups
    #   [q|k|v] projection-major; quant.split_shard_major_fused concatenates
    #   each projection's per-rank chunks in rank order);
    #   V base = 12288 + 768 = 13056; shard c's V_c sits at 13056 + c*128;
    #   the half-pad scale row 26 covers shard rows 3328..3391 = V_c TAIL
    #   rows 64..127 (V_c starts at shard row 3264 = 3072 + 192).
    expect = np.concatenate([
        np.arange(13056 + c * 128 + 64, 13056 + c * 128 + 128) for c in range(4)
    ])
    np.testing.assert_array_equal(np.flatnonzero(got_ok), expect)
    # The env-default implementation must track the correct mapping: under
    # MIMO26_SPIKE_NAIVE=1 this is exactly where the global row//128 bug shows.
    np.testing.assert_array_equal(got_env, got_ok)


def test_t2_naive_global_mapping_mispoison_is_visible():
    """Detection power (both runs): the wrong mapping poisons a DIFFERENT row
    set than the half-pad row — the bug cannot hide behind equal outputs."""
    pois, clean = _pad_case()
    got_ok = _affected_rows(pois, clean, naive=False)
    got_bad = _affected_rows(pois, clean, naive=True)
    assert not np.array_equal(got_bad, got_ok), (
        "T2: naive global row//128 must pick wrong scale rows (visible)"
    )


# ---------------------------------------------------------------------------
# T4 QK192/V128 attention path (F1) + c2 call-site hardening
# ---------------------------------------------------------------------------

def test_t4_score_scale_uses_qk_dim_negative():
    """TRAP NEGATIVE — FAILS on the naive run (F1/T4).

    Oracle :61-63: scaling = head_dim**-0.5 with head_dim = the QK dim (192);
    the V dim (128) is the VALUE width and never enters scaling.
    """
    s = attn.attn_scale(192, 128)  # env-default impl
    assert s == pytest.approx(192.0 ** -0.5)
    assert s != pytest.approx(128.0 ** -0.5), "QK/V scaling mixup undetected"


def test_t4_v_as_192_rows_gate_rejects():
    """Both runs (fail-loud gate): a V-as-192 variant cannot even fit the
    VERIFIED stored rows — QK192/V128 is enforced before any math runs."""
    good = RL.qkv_segments(64, 192, 4, 128)
    assert sum(good) == 13568  # verified header rows (external fact, P-102)
    bad = RL.qkv_segments(64, 192, 4, 192)  # V wrongly QK-sized
    with pytest.raises(ValueError, match="fused rows"):
        RL.verify_stored("ga", (sum(bad), RL.HIDDEN), (108, 32))


def test_c2_attn_scale_arg_order_pin():
    """Both runs (c2, explicit naive=False): the (d_qk, d_v) argument order is
    pinned — the FIRST argument governs, and swapping them is observable. Any
    call-site written as attn_scale(d_v, d_qk) changes the value and gets
    caught by test_c2_attn_scale_through_call_negative."""
    assert attn.attn_scale(192, 128, naive=False) == pytest.approx(192.0 ** -0.5)  # d_qk governs
    assert attn.attn_scale(128, 192, naive=False) == pytest.approx(128.0 ** -0.5)  # swapped -> different
    assert attn.attn_scale(192, 128, naive=False) != pytest.approx(attn.attn_scale(128, 192, naive=False)), \
        "attn_scale arg order unpinned — a swapped call-site would pass"


def test_c2_attn_scale_through_call_negative():
    """TRAP NEGATIVE — FAILS on the naive run (c2 + T4 through __call__).

    At asymmetric widths d_qk=5, d_v=2, ``SelfAttention.__call__`` must scale
    scores by ``d_qk**-0.5`` taken from the LIVE tensor widths and match the
    INDEPENDENT causal reference below (built from the test's own weights —
    zero code under test reused). A swapped call-site (V-dim scale) must be
    observable — asserted against the same reference at ``d_v**-0.5``.
    Naive flip: MIMO26_SPIKE_NAIVE=1 scales by the V width (the QK/V mixup).
    (d_qk=5 ⇒ rot = int(5*0.334) = 1 → evened to 0: no rotary confound.)
    """
    rng = np.random.default_rng(62)
    d_model, d_qk, d_v = 4, 5, 2
    wq = (rng.standard_normal((d_qk, d_model)) * 0.5).astype(np.float32)
    wk = (rng.standard_normal((d_qk, d_model)) * 0.5).astype(np.float32)
    wv = (rng.standard_normal((d_v, d_model)) * 0.5).astype(np.float32)
    ow = (rng.standard_normal((d_model, d_v)) * 0.5).astype(np.float32)
    qkv = np.concatenate([wq, wk, wv], axis=0)  # rows [q d_qk | k d_qk | v d_v]
    x = rng.standard_normal((3, d_model)).astype(np.float32)
    pos = np.arange(3, dtype=np.int64)
    blk = attn.SelfAttention(d_qk, qkv, ow, sliding_window=None,
                             d_v=d_v)  # env-default impl
    out = blk(x, pos, kv.KVCache(1), 0)

    def ref(scale: float) -> np.ndarray:  # independent causal reference
        q, k, v = x @ wq.T, x @ wk.T, x @ wv.T
        att = np.where(np.arange(3)[None, :] <= pos[:, None], q @ k.T * scale, -1e30)
        p = np.exp(att - att.max(axis=-1, keepdims=True))
        p = p / p.sum(axis=-1, keepdims=True)
        return (p @ v) @ ow.T

    np.testing.assert_allclose(out, ref(d_qk ** -0.5), rtol=1e-6, atol=1e-7)
    assert not np.allclose(out, ref(d_v ** -0.5), atol=1e-6), \
        "c2: a swapped call-site (V-dim scale) must be observable at d_qk=5, d_v=2"
    # Detection (both runs): the naive call-site IS exactly the V-width mixup
    # (out_naive == ref(d_v**-0.5) and observably != the d_qk reference).
    out_naive = attn.SelfAttention(d_qk, qkv, ow, sliding_window=None,
                                   d_v=d_v, naive=True)(
        x, pos, kv.KVCache(1, naive=True), 0)
    np.testing.assert_allclose(out_naive, ref(d_v ** -0.5), rtol=1e-6, atol=1e-7)
    assert not np.allclose(out_naive, ref(d_qk ** -0.5), atol=1e-6), \
        "c2: naive V-width scaling must differ from the d_qk reference"


# ---------------------------------------------------------------------------
# T6 sink LIVE on SWA (F2) + c1 family gate
# ---------------------------------------------------------------------------

def _sink_block(naive: bool | None, is_ga: bool = False, sink_dim: int = 1) -> np.ndarray:
    d = 4
    rng = np.random.default_rng(60)
    qkv = rng.standard_normal((3 * d, d)).astype(np.float32)
    ow = rng.standard_normal((d, d)).astype(np.float32)
    x = rng.standard_normal((3, d)).astype(np.float32)
    blk = attn.SelfAttention(
        d, qkv, ow,
        sliding_window=128 if not is_ga else None,
        sink_dim=sink_dim,
        sink_bias=+40.0,
        is_ga=is_ga,
        naive=naive,
    )
    return blk(x, np.arange(3), kv.KVCache(1, naive=naive), 0)


def test_t6_sink_live_on_swa_negative():
    """TRAP NEGATIVE — FAILS on the naive run (F2/T6).

    Oracle :89-96: the per-Q-head sink bias is an EXTRA softmax-logit column,
    dropped AFTER softmax — its mass is absorbed, never renormalized onto the
    keys. With a +40 logit the sink swallows ~all mass ⇒ output ~ 0. The naive
    bug (sink dropped entirely) leaves a normal O(1) weighted mix.
    """
    out = _sink_block(naive=None)  # env-default impl, SWA (is_ga=False)
    assert float(np.max(np.abs(out))) < 1e-3, (
        "T6: sink must be live on SWA (extra softmax column absorbs the mass)"
    )


def test_t6_sink_drop_bug_is_visible():
    """Detection power (both runs): sink-live vs sink-dropped outputs differ."""
    out_ok = _sink_block(naive=False)
    out_bad = _sink_block(naive=True)
    assert float(np.max(np.abs(out_ok))) < 1e-3
    assert float(np.max(np.abs(out_bad))) > 1e-1
    assert not np.allclose(out_ok, out_bad, atol=1e-6)


def test_c1_sink_family_gated_negative():
    """TRAP NEGATIVE — FAILS on the naive run (c1 + T6 gate).

    The sink is FAMILY-GATED to SWA only (verified config
    add_swa_attention_sink_bias=true / add_full_attention_sink_bias=false;
    real_loop.py:301 ``sink = (self.sink_swa if is_swa else self.sink_ga)``).
    (1) GA-with-sink == GA-without-sink BITWISE: a family-agnostic gate (the
        c1 bug — sink live on GA too) breaks this pin and is caught on the
        correct impl. (2) The SWA twin is sink-live: the +40 sink column
        absorbs ~all mass so the contrast is observable (not vacuous).
    Naive flip: MIMO26_SPIKE_NAIVE=1 drops the sink column entirely — the
    SWA contrast below vanishes and this test FAILS.
    """
    ga_sink = _sink_block(naive=None, is_ga=True, sink_dim=1)  # env-default impl
    ga_none = _sink_block(naive=None, is_ga=True, sink_dim=0)
    assert (ga_sink.view(np.uint32) == ga_none.view(np.uint32)).all(), \
        "c1: GA must have NO sink (add_full_attention_sink_bias=false) — bitwise"
    swa_sink = _sink_block(naive=None, is_ga=False, sink_dim=1)
    swa_none = _sink_block(naive=None, is_ga=False, sink_dim=0)
    assert not np.allclose(swa_sink, swa_none, atol=1e-6), \
        "c1: SWA must be sink-live (contrast pair; naive drops the sink -> flip)"
    assert float(np.max(np.abs(swa_sink))) < 1e-3  # +40 sink absorbed the mass
    assert float(np.max(np.abs(swa_none))) > 1e-1


# ---------------------------------------------------------------------------
# T7 rotary 64-dim + dual-θ (F2)
# ---------------------------------------------------------------------------

def test_t7_rope_dim_64_pin():
    """Both runs (spec pin): partial_rotary_factor 0.334 on head_dim 192 ⇒
    rope_dim 64 (even), split [rope 64 | nope 128] (oracle :28-42)."""
    rot = int(192 * 0.334)
    assert rot == 64 and rot % 2 == 0
    assert 192 - rot == 128


def test_t7_dual_theta_negative():
    """TRAP NEGATIVE — FAILS on the naive run (F2/T7).

    Verified config: GA rope_theta = 1e7 vs SWA swa_rope_theta = 1e4 — two
    DIFFERENT thetas across layer families. Naive: one theta for both.
    """
    assert attn.rope_theta_for(True) == pytest.approx(1.0e7)  # GA
    assert attn.rope_theta_for(False) == pytest.approx(1.0e4)  # SWA
    rng = np.random.default_rng(7)
    x = rng.standard_normal((2, 32)).astype(np.float32)  # rot=10, half=5
    pos = np.array([0, 7], np.int64)
    ga = attn.apply_rotary(x, pos, theta=attn.rope_theta_for(True))
    swa = attn.apply_rotary(x, pos, theta=attn.rope_theta_for(False))
    assert not np.allclose(ga, swa, atol=1e-6), (
        "T7: dual-θ must materially rotate differently (GA 1e7 vs SWA 1e4)"
    )


def test_t7_apply_rotary_matches_oracle_formula():
    """Both runs (external-oracle pin at real dims 192/64): inv_freq =
    1/(θ**(arange(0,rot,2)/rot)); rotate-half over the first rot dims,
    out = x*cos + rotate_half(x)*sin (oracle :28-42)."""
    theta, rot = 1.0e7, 64
    rng = np.random.default_rng(8)
    x = rng.standard_normal((3, 192)).astype(np.float32)
    pos = np.array([0, 3, 11], np.int64)
    freqs = 1.0 / (theta ** (np.arange(0, rot, 2) / rot))
    ang = pos[:, None] * freqs[None, :]
    cos, sin = np.cos(ang), np.sin(ang)
    a, b = x[:, : rot // 2], x[:, rot // 2 : rot]
    exp = x.copy()
    exp[:, : rot // 2] = a * cos - b * sin
    exp[:, rot // 2 : rot] = a * sin + b * cos
    np.testing.assert_allclose(
        attn.apply_rotary(x, pos, theta=theta), exp, rtol=1e-6, atol=1e-6
    )


# ---------------------------------------------------------------------------
# T10 E8M0-255 clamp + T14 nibble order (F1-re-review REFUSE close-out)
#
# The golden fixture's scales_u8_hex holds only {127, 128} — byte 255 (OCP
# RESERVED) is exercised here synthetically; goldens stay READ-ONLY (I-Gold).
# ---------------------------------------------------------------------------

def test_t10_e8m0_clamp_255_negative():
    """TRAP NEGATIVE — FAILS on the naive run (T10).

    OCP MXFP4: E8M0 byte 255 is RESERVED; the scale clamps to 2^127. The
    naive decode 2^128 is a poison scale (overflow-class garbage downstream).
    """
    got = float(mxfp4.scale_byte_to_float(np.uint8(255)))  # env-default impl
    assert got == 2.0 ** 127, (
        f"T10: E8M0 byte 255 must clamp to 2^127, got {got} "
        "(naive unclamped 2^128 is a poison scale)"
    )
    # Flip value pinned: under MIMO26_SPIKE_NAIVE=1 the same call returns
    # exactly 2^128 (asserted here via the env-equivalent naive flag).
    assert float(mxfp4.scale_byte_to_float(np.uint8(255), naive=True)) == 2.0 ** 128


def test_t10_unpack_255_scale_byte_negative():
    """TRAP NEGATIVE — FAILS on the naive run (T10, through unpack).

    Full codec path with a 255 scale byte (the case the golden fixture lacks):
    the decode must use the clamped 2^127 scale, byte-exact in f32.
    """
    packed = np.full((1, 16), 0x12, np.uint8)          # 32 E2M1 nibbles
    scales = np.full((1, 1), 255, np.uint8)            # reserved E8M0 byte
    vals = np.tile(mxfp4.E2M1_CODEBOOK[[2, 1]].astype(np.float64), 16)  # low=even k
    want = (vals * np.exp2(127.0)).astype(np.float32).reshape(1, 32)  # f64 math -> f32
    got = mxfp4.unpack(packed, scales)                 # env-default impl
    assert got.view(np.uint32).tolist() == want.view(np.uint32).tolist(), \
        "T10: 255 scale byte must decode at the clamped 2^127 scale, byte-exact"
    # Detection power (both runs): the unclamped variant is visibly different.
    bad = mxfp4.unpack(packed, scales, naive=True)
    assert bad.view(np.uint32).tolist() != want.view(np.uint32).tolist()
    assert float(mxfp4.scale_byte_to_float(np.uint8(255), naive=True)) == 2.0 ** 128


def test_t10_clamp_255_detectable():
    """Both runs (detection power): clamped 2^127 vs unclamped 2^128 differ."""
    ok = float(mxfp4.scale_byte_to_float(np.uint8(255), naive=False))
    bad = float(mxfp4.scale_byte_to_float(np.uint8(255), naive=True))
    assert ok == 2.0 ** 127 and bad == 2.0 ** 128 and ok != bad


def test_t14_nibble_order_negative():
    """TRAP NEGATIVE — FAILS on the naive run (T14).

    Low nibble = even k (golden e2m1_codebook_by_nibble layout). Byte 0x12
    decodes to (k=0 -> nibble 2 -> 1.0, k=1 -> nibble 1 -> 0.5); the naive
    high-nibble-first order swaps the pair.
    """
    packed = np.zeros((1, 16), np.uint8)
    packed[0, 0] = 0x12
    scales = np.full((1, 1), 127, np.uint8)            # 2^0 = 1.0
    want = np.zeros((1, 32), np.float32)
    want[0, 0], want[0, 1] = 1.0, 0.5
    got = mxfp4.unpack(packed, scales)                 # env-default impl
    assert got.view(np.uint32).tolist() == want.view(np.uint32).tolist(), \
        "T14: low nibble = even k (golden order)"
    bad = mxfp4.unpack(packed, scales, naive=True)     # detection (both runs)
    assert bad[0, 0] == 0.5 and bad[0, 1] == 1.0


# ---------------------------------------------------------------------------
# F6/T3: window asserts on MODEL-BUILT layers (decision site model.py:63)
# ---------------------------------------------------------------------------

def test_f6_window_on_model_built_layers_negative():
    """TRAP NEGATIVE — FAILS on the naive run (F6).

    P-105 F6: test_t3_swa_window asserted ga_cache_config, which NO run path
    calls. The real decision site is model.py:63 (``sliding_window=None if
    cfg["is_ga"][i] else self.window``) — assert on the layers SpikeModel
    actually builds.
    """
    cfg = _tiny_cfg()
    w = model.init_weights(cfg)
    m = model.SpikeModel(w, cfg)  # env-default impl
    assert m.layers[0].is_ga and m.layers[0].window is None, (
        "F6/T3: GA model-built layer must have NO window (model.py:63)"
    )
    assert (not m.layers[1].is_ga) and m.layers[1].window == 4
    # Detection power (both runs): the naive model builds GA with a window.
    mn = model.SpikeModel(w, cfg, naive=True)
    assert mn.layers[0].window is not None


# ---------------------------------------------------------------------------
# F7/T8: eviction negative on CACHE CONTENTS, env-default in BOTH runs
# ---------------------------------------------------------------------------

def _evict(positions_batches: list[list[int]], window: int,
           naive: bool | None = None) -> list[int]:
    c = kv.KVCache(1, naive=naive)  # env-default when None — exactly F7's ask
    for batch in positions_batches:
        p = np.asarray(batch, np.int64)
        kvv = np.zeros((len(p), 1, 2), np.float32)
        c.append(0, p, kvv, kvv, window=window)
    return [int(x) for x in c.pos_of(0)]


def test_f7_t8_eviction_keeps_rule_set_negative():
    """TRAP NEGATIVE — FAILS on the naive run (F7/T8).

    kv.py:56-62 (mirrors mimo26/kv.py:77-92): evict only entries older than
    ``min(batch_pos) - window + 1``; the old trim rule is an UPPER BOUND.
    Assert the KEPT POSITIONS against the rule, computed here independently.
    """
    # Batch case: gap at 11, batch mins 8 then 12 -> keep_from = 12-4+1 = 9.
    batches = [[8, 9, 10], [12, 13, 14]]
    rule = [p for b in batches for p in b if p >= min(batches[-1]) - 4 + 1]
    assert _evict(batches, 4) == rule == [9, 10, 12, 13, 14]
    # Sequential catch-up: 0..9 in two appends, window 4 -> keep_from = 5-4+1.
    batches2 = [list(range(0, 5)), list(range(5, 10))]
    rule2 = [p for b in batches2 for p in b if p >= min(batches2[-1]) - 4 + 1]
    assert _evict(batches2, 4) == rule2 == [2, 3, 4, 5, 6, 7, 8, 9]


def test_f7_t8_naive_eviction_difference_is_visible():
    """Detection power (both runs): correct vs naive kept sets differ."""
    batches = [[8, 9, 10], [12, 13, 14]]
    ok = _evict(batches, 4, naive=False)
    bad = _evict(batches, 4, naive=True)
    assert ok != bad, "T8: trim-only eviction must visibly differ from the rule"


# ---------------------------------------------------------------------------
# F8/T11: missing-bias / missing-name negatives (model.py:50-56 fail-loud)
# ---------------------------------------------------------------------------

def test_f8_t11_missing_router_bias_negative():
    """Both runs (fail-loud guard): missing
    ``layers.{i}.mlp.gate.e_score_correction_bias`` must raise KeyError —
    the T11 name-audit class (mirrors model.py:85-88)."""
    cfg = _tiny_cfg()
    w = model.init_weights(cfg)
    del w["layers.1.mlp.gate.e_score_correction_bias"]
    with pytest.raises(KeyError, match="e_score_correction_bias"):
        model.SpikeModel(w, cfg)


def test_f8_t11_missing_required_name_negative():
    """Both runs (fail-loud guard): a missing REQUIRED name raises ValueError
    listing it (mirrors model.py:34-42) — no silent default, no skip."""
    cfg = _tiny_cfg()
    w = model.init_weights(cfg)
    del w["lm_head.weight"]
    with pytest.raises(ValueError, match="missing required weights"):
        model.SpikeModel(w, cfg)
