"""Quantizer e4m3fn-k32-v1 reference codec (docs/design/lattice-v1.md §4): exhaustive and edge tests.

Exact rationals (fractions.Fraction) serve as the independent referee for scale selection and RNE.
The torch fake-quant twin must equal the numpy codec bit for bit; that test is skipped where torch is absent.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from lattice import quant_v1 as Q

F32MAX = np.finfo(np.float32).max
CODES = np.arange(256, dtype=np.uint8)
FINITE = CODES[(CODES & 0x7F) != 0x7F]


def grid():
    """Sorted non-negative E4M3FN magnitudes as exact Fractions."""
    vals = set()
    for c in range(0x7F):
        e, m = c >> 3, c & 7
        vals.add(Fraction(m, 512) if e == 0 else Fraction(8 + m, 8) * Fraction(2) ** (e - 7))
    return sorted(vals)


def rne_ref(y: Fraction) -> Fraction:
    """Independent RNE on the E4M3FN magnitude grid, ties to the value with an even mantissa code."""
    g = grid()
    a = abs(y)
    for lo, hi in zip(g, g[1:]):
        if lo <= a <= hi:
            if a - lo < hi - a:
                return lo
            if hi - a < a - lo:
                return hi
            lo_code = int(Q.e4m3fn_encode(np.array([float(lo)]))[0])
            return lo if lo_code % 2 == 0 else hi
    return g[-1]


def test_decode_table_and_nan_codes():
    v = Q.e4m3fn_decode(FINITE)
    assert v.max() == 448.0 and v.min() == -448.0
    assert sorted(set(np.abs(v[v != 0]))) [0] == 2.0 ** -9
    assert np.signbit(Q.e4m3fn_decode(np.array([0x80], np.uint8)))[0]
    for bad in (0x7F, 0xFF):
        with pytest.raises(Q.NumericalFault):
            Q.e4m3fn_decode(np.array([bad], np.uint8))


def test_every_finite_code_round_trips_including_signed_zero():
    assert (Q.e4m3fn_encode(Q.e4m3fn_decode(FINITE)) == FINITE).all()


def test_rne_at_every_midpoint_and_its_neighbours():
    g = grid()
    ys = []
    for lo, hi in zip(g, g[1:]):
        mid = (lo + hi) / 2
        for y in (mid, mid - Fraction(1, 2 ** 40), mid + Fraction(1, 2 ** 40)):
            ys.append(y)
    got = Q.e4m3fn_decode(Q.e4m3fn_encode(np.array([float(y) for y in ys])))
    want = [float(rne_ref(y)) for y in ys]
    assert np.array_equal(got, want)
    # the 0 <-> 2**-9 tie goes to zero, and below it rounds to signed zero
    assert Q.e4m3fn_encode(np.array([2.0 ** -10, -(2.0 ** -10), -(2.0 ** -12)])).tolist() == [0x00, 0x80, 0x80]


@pytest.mark.parametrize("a", [1e-4, 1.0, 448.0, 448.0000305, 449.0, 896.0, 0.875, 0.8750001, 3.0e-3,
                               1.17549435e-38, float(F32MAX), 2.0 ** 100 * 448, 2.0 ** 100 * 448 * (1 + 2 ** -23)])
def test_scale_exponent_is_the_smallest_power_of_two_that_fits(a):
    a32 = np.float32(max(a, 1e-4))
    k = Q.scale_exponent(a32)
    fa = Fraction(float(a32))
    assert fa <= 448 * Fraction(2) ** k and not fa <= 448 * Fraction(2) ** (k - 1)


def test_scale_byte_range_floor_and_zero_block():
    zeros = np.zeros(32, np.float32)
    zeros[3] = -0.0
    codes, scales = Q.encode_blocks(zeros)
    assert scales.tolist() == [105] and codes[3] == 0x80 and (np.delete(codes, 3) == 0).all()
    tiny = np.full(32, 3e-9, np.float32)  # amax below the floor: the floor scale applies
    assert Q.encode_blocks(tiny)[1].tolist() == [105]
    big = np.zeros(32, np.float32)
    big[0] = F32MAX
    assert Q.encode_blocks(big)[1].tolist() == [247]


def test_block_encode_matches_exact_referee_on_random_blocks():
    rng = np.random.default_rng(20260923)
    for trial in range(200):
        mag = 10.0 ** rng.uniform(-12, 12)
        h = (rng.standard_normal(32) * mag).astype(np.float32)
        h[rng.integers(0, 32, 3)] *= np.float32(10.0 ** rng.uniform(-8, 0))  # near-zero channels in the block
        codes, scales = Q.encode_blocks(h)
        k = int(scales[0]) - 127
        want = [rne_ref(Fraction(float(x)) / Fraction(2) ** k) for x in h]
        got = Q.e4m3fn_decode(codes)
        assert [Fraction(float(v)) for v in got] == [abs(w) if w == 0 else (w if x >= 0 else -w)
                                                      for w, x in zip(want, h)]
        assert (np.signbit(got) == np.signbit(h)).all()


def test_no_payload_saturation_and_amax_maps_within_range():
    rng = np.random.default_rng(7)
    h = (rng.standard_normal((1000, 64)) * 10.0 ** rng.uniform(-6, 6, (1000, 1))).astype(np.float32)
    codes, _ = Q.encode_blocks(h)
    assert not np.any((codes & 0x7F) == 0x7F)


def test_nonfinite_and_out_of_range_reconstruction_are_faults():
    for bad in (np.nan, np.inf, -np.inf):
        h = np.zeros(32, np.float32)
        h[5] = bad
        with pytest.raises(Q.NumericalFault):
            Q.encode_blocks(h)
    h = np.zeros(32, np.float32)
    h[0] = F32MAX  # k = 120; payload rounds to 256 -> 2**128 on reconstruction
    codes, scales = Q.encode_blocks(h)
    with pytest.raises(Q.NumericalFault):
        Q.decode_blocks(codes, scales)
    with pytest.raises(Q.NumericalFault):
        Q.decode_blocks(codes, np.array([104], np.uint8))


def test_named_naive_variants_are_detected():
    rng = np.random.default_rng(11)
    h = (rng.standard_normal((64, 32)) * 3).astype(np.float32)
    ref = Q.decode_blocks(*Q.encode_blocks(h))
    k = (Q.encode_blocks(h)[1].astype(np.int64) - 127)[:, None]
    y = h.astype(np.float64) * np.ldexp(1.0, -k)
    trunc = np.sign(y) * np.floor(np.abs(y) * 8) / 8  # truncation instead of RNE (in the [1,2) binade)
    assert not np.array_equal(ref, np.ldexp(Q.e4m3fn_decode(Q.e4m3fn_encode(trunc)), k).astype(np.float32))
    # round-nearest scale (instead of round-up) picks a smaller scale for amax just above 448 * 2**k
    a = np.float32(449.0)
    assert Q.scale_exponent(a) == 1 and int(np.rint(np.log2(a / 448.0))) == 0
    # K16 blocks differ from K32 on a block with a large outlier in its first half
    h16 = np.ones(32, np.float32) * np.float32(1.1e-3)  # not a power of two: K16 and K32 grids differ
    h16[0] = 100.0
    k32 = Q.decode_blocks(*Q.encode_blocks(h16))
    k16 = np.concatenate([Q.decode_blocks(*Q.encode_blocks(np.concatenate([h16[:16], h16[:16]])))[:16],
                          Q.decode_blocks(*Q.encode_blocks(np.concatenate([h16[16:], h16[16:]])))[:16]])
    assert not np.array_equal(k32, k16)
    # removing the floor changes blocks whose amax is below 1e-4
    tiny = np.full(32, 3e-9, np.float32)
    tiny[0] = 7e-9
    floored = Q.decode_blocks(*Q.encode_blocks(tiny))
    k_nofloor = Q.scale_exponent(np.float32(7e-9))  # the same round-up rule without the 1e-4 floor
    nofloor = (Q.e4m3fn_decode(Q.e4m3fn_encode(tiny.astype(np.float64) * 2.0 ** -k_nofloor))
               * 2.0 ** k_nofloor).astype(np.float32)
    assert not np.array_equal(floored, nofloor)


def test_bf16_rne_ties_signed_zero_and_faults():
    x = np.array([1.0 + 2 ** -8, 1.0 + 3 * 2 ** -8, -0.0, 1.0 + 2 ** -8 + 2 ** -20], np.float32)
    r = Q.bf16_rne(x)
    assert r.tolist()[:2] == [1.0, 1.0 + 2 ** -6] and np.signbit(r[2]) and r[3] == np.float32(1.0 + 2 ** -7)
    with pytest.raises(Q.NumericalFault):
        Q.bf16_rne(np.array([np.nan], np.float32))
    with pytest.raises(Q.NumericalFault):
        Q.bf16_rne(np.array([F32MAX], np.float32))


def test_torch_fake_quant_is_bit_equal_to_the_numpy_codec():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(20260923)
    parts = [
        (rng.standard_normal((4096, 32)) * 10.0 ** rng.uniform(-10, 10, (4096, 1))).astype(np.float32),
        np.zeros((2, 32), np.float32),
        np.full((1, 32), 3e-9, np.float32),
    ]
    g = [float(v) for v in grid()]
    mids = np.array([(a + b) / 2 for a, b in zip(g, g[1:])] * 1, np.float64)
    tie_block = np.zeros((len(mids) // 32 + 1) * 32, np.float32)
    tie_block[: len(mids)] = mids.astype(np.float32)
    tie_block[31::32] = 448.0  # pin k = 0 so the midpoints stay midpoints
    parts.append(tie_block.reshape(-1, 32))
    h = np.concatenate(parts)
    h[1, 7] = -0.0
    ref = Q.decode_blocks(*Q.encode_blocks(h))
    got = Q.fake_quant_torch(torch.from_numpy(h)).numpy()
    assert np.array_equal(ref.view(np.uint32), got.view(np.uint32))
