"""I1b needle builder pins (spike/needle_prompt.py) — env-invariant (BOTH RUNS
green; no naive class: the builder is deterministic text, not numerics).

Trap class: needle CONTAMINATION and PLACEMENT DRIFT — if the filler leaks the
code/question words, or the fact/question wander out of their windows, the
spike silently stops testing 4K GA reach and the I3 exit criterion lies.
"""
from __future__ import annotations

import pytest

from spike.needle_prompt import FACT, FACT_CODE, QUESTION, build_text, filler_doc, place


def test_build_deterministic_byte_equal():
    """Two builds byte-identical (no RNG, no clocks); pool cycles but the
    Document-N index keeps every document distinct."""
    a, b = build_text(3, 120), build_text(3, 120)
    assert a == b and a.encode("utf-8") == b.encode("utf-8")
    assert filler_doc(1) != filler_doc(11)
    assert filler_doc(1).split(": ", 1)[1] == filler_doc(11).split(": ", 1)[1]


def test_fact_once_question_last_no_contamination():
    """The needle appears EXACTLY once, the question ends the prompt, and the
    filler contains no needle words (contamination would fake retrieval)."""
    t = build_text(5, 200)
    assert t.count(FACT) == 1 and t.count(FACT_CODE) == 1
    assert t.endswith(QUESTION)
    filler_only = t.replace(FACT, " ").replace(QUESTION, " ")
    for needle in (FACT_CODE, "north gate", "access code", "kestrel"):
        assert needle.lower() not in filler_only.lower(), needle


def test_placement_near_targets_stub_encoder():
    """Fact at/after token 100, question at/after token 4,000, both within the
    +-48 tolerance, re-measured on the composed text (whitespace stub is
    compositional, so prefix measurement is exact here)."""
    enc = lambda s: s.split()  # noqa: E731 — deterministic stub tokenizer
    text, meta = place(enc, fact_at=100, question_at=4000, tol=48)
    assert 100 <= meta["fact_start"] <= 148, meta
    assert 4000 <= meta["question_start"] <= 4048, meta
    assert len(enc(text[: text.index(FACT)].rstrip())) == meta["fact_start"]
    assert text.count(FACT) == 1 and text.endswith(QUESTION)
    text2, meta2 = place(enc, fact_at=100, question_at=4000, tol=48)
    assert (text2, meta2["fact_start"], meta2["question_start"]) == \
           (text, meta["fact_start"], meta["question_start"])


def test_query_chunked_prefill_equals_full():
    """I1b fire-time delta (tenant-adjusted VRAM): the MIMO26_SPIKE_CHUNK
    query-chunked prefill must equal the unchunked path on a small case —
    env-invariant (BOTH RUNS), torch-gated (importorskip: the dev host test venv has
    no torch; the fire venv mimo26f-torch does).  Pinned atol=1e-6: each query
    row's softmax spans its full key row either way (identical semantics);
    bitwise is not promised because GEMM-shape changes can reorder the d_qk
    reduction.  Covers sink / no-sink (T6) and a SWA-style window mask (T3/T8)."""
    torch = pytest.importorskip("torch")
    from spike.real_loop import attention_core
    torch.manual_seed(11)
    T, Tk, n_q, d_qk, d_v = 37, 41, 4, 8, 6
    q = torch.randn(T, n_q, d_qk)
    kk = torch.randn(Tk, n_q, d_qk)
    vv = torch.randn(Tk, n_q, d_v)
    qpos = torch.arange(T)
    kpos = torch.arange(Tk)
    keep = (kpos[None, :] <= qpos[:, None]) & ((qpos[:, None] - kpos[None, :]) < 12)  # causal+window
    sink = torch.randn(n_q, 1, 1)
    for bias in (sink, None):
        full = attention_core(q, kk, vv, keep, bias, chunk=0)
        assert full.shape == (T, n_q, d_v)
        for chunk in (1, 7, 16, 512):
            got = attention_core(q, kk, vv, keep, bias, chunk=chunk)
            torch.testing.assert_close(got, full, rtol=0, atol=1e-6)


def test_segmentwise_prefill_equals_monolithic():
    """I1b fix (2): KV-accumulating segmentwise prefill == one monolithic
    computation on the tiny seeded case (atol=1e-6; GEMM-shape changes can
    reorder the d_qk reduction).  Pins exactly the seams segmentwise exercises:
    KV growth across calls (cat rebinding — prior tensors released, no O(T^2)
    history), pos0 = kv.tokens() (T9), and causal masks over the GROWN cache.
    Env-invariant (BOTH RUNS); torch-gated (importorskip)."""
    torch = pytest.importorskip("torch")
    from spike.real_loop import KV, attention_core
    torch.manual_seed(13)
    T, n_q, d_qk, d_v, seg = 24, 4, 8, 6, 8
    q = torch.randn(T, n_q, d_qk)
    k = torch.randn(T, n_q, d_qk)
    v = torch.randn(T, n_q, d_v)

    def run_segments(segmented: bool):
        kv = KV(1, "cpu")
        outs = []
        spans = ([(i, min(i + seg, T)) for i in range(0, T, seg)] if segmented else [(0, T)])
        for lo, hi in spans:
            kv.append(0, list(range(lo, hi)), k[lo:hi], v[lo:hi], None)
            keep = kv.pos[0][None, :] <= torch.arange(lo, hi)[:, None]   # causal over grown cache
            outs.append(attention_core(q[lo:hi], kv.k[0], kv.v[0], keep, None, chunk=0))
        return torch.cat(outs, 0), kv

    mono, kv_m = run_segments(False)
    seg_out, kv_s = run_segments(True)
    torch.testing.assert_close(seg_out, mono, rtol=0, atol=1e-6)
    assert torch.equal(kv_s.k[0], kv_m.k[0]) and kv_s.k[0].shape[0] == T  # ONE accumulated tensor
    assert torch.equal(kv_s.pos[0], torch.arange(T))


def test_moe_expert_major_equals_cached():
    """I1b fix (2b): expert-major residency (moe_expert_major) == the pre-fix
    all-experts-cached loop (fire rows #1/#2: the cache held idx.unique() across
    ALL T tokens — 256 x 96 MiB = 24 GiB class at T=4018) up to f32
    accumulation order — pinned atol=1e-6.  Env-invariant (BOTH RUNS);
    torch-gated (importorskip)."""
    torch = pytest.importorskip("torch")
    from spike.real_loop import moe_expert_major
    torch.manual_seed(17)
    T, H, E, n_exp, top_k = 9, 6, 5, 7, 3
    x = torch.randn(T, H)
    idx = torch.randint(0, n_exp, (T, top_k))
    w = torch.rand(T, top_k)
    weights = {e: (torch.randn(E, H), torch.randn(E, H), torch.randn(H, E)) for e in range(n_exp)}
    got = moe_expert_major(x, idx, w, weights.__getitem__)
    cache = {int(e): weights[int(e)] for e in idx.unique().tolist()}  # pre-fix structure
    out = torch.zeros_like(x)
    for tok in range(T):
        h = x[tok : tok + 1]
        for j in range(top_k):
            gg, uu, dd = cache[int(idx[tok, j])]
            out[tok] += ((torch.nn.functional.silu(h @ gg.T) * (h @ uu.T)) @ dd.T).squeeze(0) * w[tok, j]
    torch.testing.assert_close(got, out, rtol=0, atol=1e-6)


def test_unpack_torch_bitwise_matches_reference():
    """Device MXFP4 dequant is BITWISE-identical to the numpy reference on both
    naive paths (ADVISOR-I3 §11.3 pin).  int32-view equality — catches sign-of-
    zero and 1-ulp slips that `torch.equal` float semantics would miss.  The
    fixtures hit every trap seam: all 16 nibbles (incl. -0.0 at 8), T10
    reserved-255 clamp vs naive 2^128 poison, saturation to +-f32max, exact-fit
    1.5*2^127 (must NOT clamp), denormal 0.5*2^-127, and per-block scale
    changes across the 32-element block boundary (T10/T14 layout)."""
    torch = pytest.importorskip("torch")
    import numpy as np

    from spike import mxfp4

    out, inn = 7, 64
    half = inn // 2
    rng = np.random.default_rng(0)
    packed = rng.integers(0, 256, size=(out, half), dtype=np.uint8)
    # every nibble value present in row 0 (bytes 0x00..0x77 -> 0..7, 0x88..0xFF -> 8..15)
    packed[0, 0:8] = np.arange(0, 8, dtype=np.uint8) * 0x11
    packed[0, 8:16] = np.arange(0, 8, dtype=np.uint8) * 0x11 + 0x88
    scales = np.empty((out, inn // 32), dtype=np.uint8)
    scales[0] = [127, 255]     # boundary exponent + reserved clamp (T10)
    scales[1] = [128, 254]
    scales[2] = [0, 253]       # 2^-127 denormal-adjacent products
    scales[3] = [255, 255]     # saturation to +-f32max
    scales[4] = [1, 126]
    scales[5] = [127, 127]     # 1.5*2^127 exact-fit must not clamp
    scales[6] = rng.choice([0, 1, 64, 127, 128, 254, 255], size=inn // 32).astype(np.uint8)

    for naive in (False, True):
        ref = mxfp4.unpack(packed, scales, naive=naive)
        got = mxfp4.unpack_torch(packed, scales, naive=naive, device="cpu")
        assert got.dtype == torch.float32 and tuple(got.shape) == (out, inn)
        assert torch.equal(got.view(torch.int32),
                           torch.from_numpy(ref).view(torch.int32)), f"bitwise slip (naive={naive})"
        if torch.cuda.is_available():
            got_dev = mxfp4.unpack_torch(packed, scales, naive=naive, device="cuda")
            assert torch.equal(got_dev.view(torch.int32).cpu(),
                               torch.from_numpy(ref).view(torch.int32)), \
                f"cuda bitwise slip (naive={naive})"


def test_chat_render_deterministic_and_template_pinned():
    """X1a render (ADVISOR-I4 §3.0): the chat_template.jinja is the served one
    (sha 853650be…), the render is deterministic, and it wraps the needle text
    as ONE user turn with a generation prompt (thinking off)."""
    import hashlib
    import os

    from spike.needle_prompt import FACT, QUESTION, build_text, render_chat

    tpl = os.path.join(
        os.path.expanduser("~/models/XiaomiMiMo/MiMo-V2.6-Flash-RL"), "chat_template.jinja")
    if not os.path.exists(tpl):
        pytest.skip("chat_template.jinja not on this host (local weights copy)")
    sha = hashlib.sha256(open(tpl, "rb").read()).hexdigest()
    assert sha.startswith("853650bee57b"), f"template drifted: {sha[:12]}"
    t = build_text(2, 3)
    a, b = render_chat(t), render_chat(t)
    assert a == b and a.encode("utf-8") == b.encode("utf-8")
    assert a.count(FACT) == 1 and a.count(QUESTION) == 1


def test_t28_chat_output_checks():
    """T28 (ADVISOR-I4 §3.5) negative pins: the checker must flag the raw-mode
    artifacts (REWARD:, role-less assistant marker) and pass a clean answer."""
    from spike.needle_prompt import IM_START, check_chat_output

    assert check_chat_output("KESTREL-41") == []
    bad = check_chat_output("Answer:KESTREL-41REWARD:True")
    assert any("REWARD" in v for v in bad), bad
    bad = check_chat_output("hi " + IM_START + "user")
    assert any("role-less assistant marker" in v for v in bad), bad
