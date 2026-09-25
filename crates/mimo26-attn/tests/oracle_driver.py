#!/usr/bin/env python3
"""Parity driver for `crates/mimo26-attn` — the numpy oracle side of the
manifest protocol (see `crates/mimo26-attn/tests/common/mod.rs`).

Consumed read-only (I-Gold): `oracle/mimo26` is the byte-verified CPU twin /
reference; nothing here imports code under test. The same manifest format
drives `kernels/parity/attn_parity.cu` on the GPU cell.

Modes
-----
  eval --in DIR --out DIR
      Read `DIR/manifest.txt` (case + tensor records + raw LE binaries — the
      inputs a Rust test just wrote), compute each case's expected outputs with
      `mimo26.nn.layers`, write `OUT/manifest.txt` (case + expect records) and
      the expect binaries.

  gen --out DIR --shapes csv
      Self-contained golden generation (inputs + expects) for the GPU cell and
      the harness selftest. Shape sets: tiny, decode-4k, decode-32k, prefill-4k,
      prefill-32k, decode-128k, decode-1m, rope-1m, kv-store. Deterministic
      (fixed seeds) — regenerating is byte-identical.

Case types
----------
  attn | decode | prefill   f32 in-memory KV. Tensor `v` is CACHED V (post
                            `v_scale`, T18): the oracle is called with
                            `value_scale=1.0` on it.
  decode_fp8_unit | decode_fp8_pth | prefill_fp8_unit | prefill_fp8_pth
                            FP8-cached KV (`k_codes`/`v_codes` u8 +
                            `k_scales`/`v_codes` f32 planes); the oracle decodes
                            with `mimo26.quant.fp8_block` then attends on the
                            decoded f32 values (so the comparison bounds the
                            attention math, not E4M3 rounding).
  rope                      `x` [T,H,D] + `pos` -> `y` (FP64-angle reference).
  kv_store_unit | kv_store_pth
                            `k`/`v_raw` (RAW V — the store path applies
                            `v_scale` before quantizing, T18) -> `k_dec`,
                            `v_dec`, `clip` (i64 [1], the amax clip gate).

Layout (T20): FP8 scale planes are PER TOKEN × HEAD with K and V SEPARATE —
mirrors `mimo26_attn::fp8kv` (`k_scale_off`/`v_scale_off`) exactly: plane scale
`= (amax / 448) as f32` (f32 rounding), quantization divides in f64.

Declared tolerances (carried in the manifest case line; `tol_abs`, `tol_rel`):
the oracle's `np.einsum` accumulates f32 per dot; the twin folds f64. Small
cases bound that at 1e-5; the GPU cell bounds f32 accumulation over S keys at
1e-4 (tiny) / 2e-4 (4K) / 5e-4 (32K) / 2e-3 (128K–1M). Rope tolerances are the
derived FP32-vs-FP64 angle bound `3·eps_f32·|pos| + 12·eps_f32` (COHERENCE-TRAPS
T19: this is HF's f32 `pos × inv_freq` behaviour — the tolerance is declared,
not tuned). kv_store compares exact decode round-trips at 1e-6.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

EPS_F32 = float(np.finfo(np.float32).eps)
E4M3_MAX = 448.0


def _repo_root() -> pathlib.Path:
    for cand in pathlib.Path(__file__).resolve().parents:
        if (cand / "oracle" / "mimo26").is_dir():
            return cand
    raise SystemExit("oracle_driver: cannot locate repo root (oracle/mimo26)")


ROOT = _repo_root()
sys.path.insert(0, str(ROOT / "oracle"))  # `import mimo26` -> oracle/mimo26
from mimo26.nn import layers as L  # noqa: E402
from mimo26.quant import fp8_block as fb  # noqa: E402

# ---------------------------------------------------------------------------
# manifest I/O (mirror of tests/common/mod.rs)
# ---------------------------------------------------------------------------

HEADER = "mimo26-attn-parity-manifest 1"
DT = {"f32": np.dtype("<f4"), "i64": np.dtype("<i8"), "u8": np.dtype(np.uint8)}
WIDTH = {"f32": 4, "i64": 8, "u8": 1}


class Tensor:
    def __init__(self, name, dtype, shape, file, is_expect):
        self.name = name
        self.dtype = dtype
        self.shape = list(shape)
        self.file = file
        self.is_expect = is_expect


class Case:
    def __init__(self, name, ty, family, window, sink, theta, partial, vscale, tol_abs, tol_rel):
        self.name = name
        self.ty = ty
        self.family = family
        self.window = int(window)
        self.sink = int(sink)
        self.theta = float(theta)
        self.partial = float(partial)
        self.vscale = float(vscale)
        self.tol_abs = float(tol_abs)
        self.tol_rel = float(tol_rel)
        self.tensors = []

    def line(self):
        return "case {} {} {} {} {} {!r} {!r} {!r} {!r} {!r}".format(
            self.name, self.ty, self.family, self.window, self.sink,
            self.theta, self.partial, self.vscale, self.tol_abs, self.tol_rel,
        )

    def tensor(self, name):
        for t in self.tensors:
            if t.name == name:
                return t
        raise KeyError(f"case {self.name}: tensor {name!r} missing")


def tensor_line(t: Tensor) -> str:
    kind = "expect" if t.is_expect else "tensor"
    shape = ",".join(str(d) for d in t.shape)
    return f"{kind} {t.name} {t.dtype} {shape} {t.file}"


def parse_manifest(path: pathlib.Path):
    cases = []
    for ln, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        f = line.split()
        if f[0] == "mimo26-attn-parity-manifest":
            assert f[1] == "1", f"manifest version (line {ln})"
        elif f[0] == "case":
            assert len(f) == 11, f"case line {ln} needs 10 fields"
            cases.append(Case(*f[1:11]))
        elif f[0] in ("tensor", "expect"):
            assert len(f) == 5, f"tensor line {ln} needs 4 fields"
            shape = [int(d) for d in f[3].split(",")] if f[3] != "0" else []
            cases[-1].tensors.append(Tensor(f[1], f[2], shape, f[4], f[0] == "expect"))
        else:
            raise SystemExit(f"manifest line {ln}: unknown record {f[0]!r}")
    return cases


def write_manifest(path: pathlib.Path, cases):
    out = [HEADER]
    for c in cases:
        out.append(c.line())
        for t in c.tensors:
            out.append(tensor_line(t))
    path.write_text("\n".join(out) + "\n")


def read_bin(directory: pathlib.Path, t: Tensor) -> np.ndarray:
    n = 1
    for d in t.shape:
        n *= d
    raw = np.fromfile(directory / t.file, dtype=DT[t.dtype])
    assert raw.size == n, f"{t.file}: expected {n} elements, got {raw.size}"
    return raw.reshape(t.shape) if t.shape else raw


def put(case: Case, directory: pathlib.Path, name: str, dtype: str, arr: np.ndarray, is_expect: bool):
    arr = np.asarray(arr)
    fname = f"{case.name}.{name}.bin"
    arr.astype(DT[dtype]).tofile(directory / fname)
    shape = list(arr.shape) if arr.ndim > 0 else [1]
    case.tensors.append(Tensor(name, dtype, shape, fname, is_expect))
    return case


# ---------------------------------------------------------------------------
# expectations
# ---------------------------------------------------------------------------

def _kv_plane_decoded(codes_2d: np.ndarray, scales_2d_or_none: np.ndarray | None, width: int):
    """codes [S,n_kv*width] (flat rows) + optional per-token×head scales
    [S, n_kv] -> decoded f32 [S, n_kv, width] (Rust `decode_kv` semantics)."""
    s = codes_2d.shape[0]
    n_kv = codes_2d.shape[1] // width
    dec = fb.decode_e4m3(codes_2d.astype(np.uint8)).astype(np.float32)
    if scales_2d_or_none is not None:
        sc = np.asarray(scales_2d_or_none, dtype=np.float32).reshape(s, n_kv)
        dec = dec.reshape(s, n_kv, width) * sc[:, :, None]
        return dec
    return dec.reshape(s, n_kv, width)


def _plane_scale(row: np.ndarray) -> np.float32:
    amax = float(np.max(np.abs(row.astype(np.float64)))) if row.size else 0.0
    return np.float32(amax / E4M3_MAX) if amax > 0.0 else np.float32(1.0)


def expected_kv_store(mode: str, k, v_raw, vscale):
    """Mirror of `mimo26_attn::fp8kv` encode/decode for the two pinned layouts.
    T18: V is scaled BEFORE quantization (the plane amax sees scaled V too);
    K is never scaled. Unit-scale: scale 1.0 and the amax clip gate fires on
    anything past ±448."""
    pth = mode == "kv_store_pth"
    n_tok, n_kv, d_qk = k.shape
    d_v = v_raw.shape[2]
    v_scaled = (v_raw.astype(np.float32) * np.float32(vscale)).astype(np.float32)
    vals_k = k.astype(np.float64)
    vals_v = v_scaled.astype(np.float64)
    k_dec = np.empty((n_tok, n_kv, d_qk), dtype=np.float32)
    v_dec = np.empty((n_tok, n_kv, d_v), dtype=np.float32)
    clips = 0
    for t in range(n_tok):
        for h in range(n_kv):
            s_k = _plane_scale(k[t, h]) if pth else np.float32(1.0)
            s_v = _plane_scale(v_scaled[t, h]) if pth else np.float32(1.0)
            # T18/T20 quantizer input is f32: the kernel computes `val/s` in f32
            # and clamps on the f32 magnitude (`m26::e4m3_encode`, kv_cache_fp8.cu
            # `store_kernel`). Dividing in f64 nudged values sitting exactly at
            # the ±448 amax boundary (the 700 probe → amax → 448 after the plane
            # scale) just over it, over-counting clips. Mirror the f32 quantizer
            # exactly (`k[t,h]/s_k`, `v_scaled[t,h]/s_v`) so the clip count and
            # codes match `e4m3_encode` bit for bit.
            qk = np.float32(vals_k[t, h]) / s_k
            qv = np.float32(vals_v[t, h]) / s_v
            clips += int(np.sum(np.abs(qk) > E4M3_MAX)) + int(np.sum(np.abs(qv) > E4M3_MAX))
            k_dec[t, h] = fb.decode_e4m3(fb.encode_e4m3(qk)).astype(np.float32) * s_k
            v_dec[t, h] = fb.decode_e4m3(fb.encode_e4m3(qv)).astype(np.float32) * s_v
    return k_dec, v_dec, clips


def compute_expects(c: Case, directory: pathlib.Path):
    """Return {name: (dtype, array)} of expected outputs for one case."""
    g = lambda n: read_bin(directory, c.tensor(n))
    if c.ty in ("attn", "decode", "prefill"):
        q, k, v = g("q"), g("k"), g("v")
        q_pos, k_pos = g("q_pos"), g("k_pos")
        sink = g("sink") if c.sink else None
        window = c.window if c.window > 0 else None
        out = L.attention(
            q, k, v, q_pos=q_pos, k_pos=k_pos, window=window,
            sink_bias=sink, value_scale=1.0,  # `v` is CACHED V (T18)
        )
        return {"o": ("f32", out)}
    if c.ty in ("decode_fp8_unit", "decode_fp8_pth", "prefill_fp8_unit", "prefill_fp8_pth"):
        pth = c.ty.endswith("_pth")
        q = g("q")
        if c.name.startswith("tc_bf16q_"):
            q = bf16_rne(q)  # post-RoPE Q; independent lattice-local oracle
        q_pos, k_pos = g("q_pos"), g("k_pos")
        # A GA caller may supply a sink tensor, but the layer-family contract
        # says it is ignored. The tc corpus deliberately supplies one as a trap.
        sink = g("sink") if c.sink and c.family == "swa" else None
        k_codes, v_codes = g("k_codes"), g("v_codes")
        s = k_codes.shape[0]
        # flat row-major codes [S, n_kv*width]; scales [S, n_kv] when pth
        k_scales = g("k_scales") if pth else None
        v_scales = g("v_scales") if pth else None
        k_dec = _kv_plane_decoded(k_codes.reshape(s, -1), k_scales, D_QK)
        v_dec = _kv_plane_decoded(v_codes.reshape(s, -1), v_scales, D_V)
        window = c.window if c.window > 0 else None
        out = L.attention(
            q, k_dec, v_dec, q_pos=q_pos, k_pos=k_pos, window=window,
            sink_bias=sink, value_scale=1.0,
        )
        if c.name in ("tc_q_low", "tc_p_low", "tc_q_tail"):
            argument = {"tc_q_low": 448 * 2**-8 / np.sqrt(192),
                        "tc_p_low": 1 / (2 * np.sqrt(192)),
                        "tc_q_tail": 448 * 2**-17 / np.sqrt(192)}[c.name]
            if not np.all(np.abs(out - np.tanh(argument)) < 1e-7):
                raise AssertionError("external oracle disagrees with independent tanh identity")
        return {"o": ("f32", out)}
    if c.ty == "rope":
        x, pos = g("x"), g("pos")
        y = L.apply_rotary(x, pos, theta=c.theta, partial_rotary_factor=c.partial)
        return {"y": ("f32", y)}
    if c.ty in ("kv_store_unit", "kv_store_pth"):
        k, v_raw = g("k"), g("v_raw")
        k_dec, v_dec, clips = expected_kv_store(c.ty, k, v_raw, c.vscale)
        return {
            "k_dec": ("f32", k_dec),
            "v_dec": ("f32", v_dec),
            "clip": ("i64", np.asarray([clips], dtype=np.int64)),
        }
    raise SystemExit(f"oracle_driver: unknown case type {c.ty!r}")


# ---------------------------------------------------------------------------
# gen: shape sets (real dims — oracle/mimo26/config.py)
# ---------------------------------------------------------------------------

N_Q_GA, N_KV_GA, N_Q_SWA, N_KV_SWA = 64, 4, 64, 8
D_QK, D_V, WINDOW = 192, 128, 128
THETA_GA, THETA_SWA, PARTIAL, VSCALE = 1e7, 1e4, 0.334, 0.707


def rope_tol(max_pos: int) -> float:
    """Declared FP32-vs-FP64 angle bound at `max_pos` (T19): |Δangle| ≤
    3·eps·|pos| + 8·eps, output bound ≈ that (|x| ≤ 0.5) + 4·eps."""
    return 3.0 * EPS_F32 * max_pos + 12.0 * EPS_F32


def _attn_case(rng, name, ty, family, t_len, s_len, q_pos, k_pos, window, sink, tol):
    n_q, n_kv, theta = (N_Q_GA, N_KV_GA, THETA_GA) if family == "ga" else (N_Q_SWA, N_KV_SWA, THETA_SWA)
    c = Case(name, ty, family, window, 1 if sink else 0, theta, PARTIAL, 1.0, tol, 0.0)
    return c, {
        "q": ("f32", rng.uniform(-0.5, 0.5, (t_len, n_q, D_QK)).astype(np.float32)),
        "k": ("f32", rng.uniform(-0.5, 0.5, (s_len, n_kv, D_QK)).astype(np.float32)),
        "v": ("f32", rng.uniform(-0.5, 0.5, (s_len, n_kv, D_V)).astype(np.float32)),  # CACHED V (T18)
        "q_pos": ("i64", np.asarray(q_pos, dtype=np.int64)),
        "k_pos": ("i64", np.asarray(k_pos, dtype=np.int64)),
        **({"sink": ("f32", rng.uniform(-1.0, 1.0, n_q).astype(np.float32))} if sink else {}),
    }


def _fp8_case(rng, name, family, s_len, pth, tol):
    """decode_fp8_* case: builds the FP8 KV exactly like the engine stores it
    (V scaled before quantization — T18), expected from the DECODED values."""
    n_q, n_kv, theta = (N_Q_GA, N_KV_GA, THETA_GA) if family == "ga" else (N_Q_SWA, N_KV_SWA, THETA_SWA)
    ty = f"decode_fp8_{'pth' if pth else 'unit'}"
    c = Case(name, ty, family, WINDOW if family == "swa" else 0, 1 if family == "swa" else 0,
             theta, PARTIAL, 1.0, tol, 0.0)
    k = rng.uniform(-0.5, 0.5, (s_len, n_kv, D_QK)).astype(np.float32)
    v_cached = (rng.uniform(-0.5, 0.5, (s_len, n_kv, D_V)).astype(np.float32) * np.float32(VSCALE))
    tensors = {
        "q": ("f32", rng.uniform(-0.5, 0.5, (1, n_q, D_QK)).astype(np.float32)),
        "q_pos": ("i64", np.asarray([s_len - 1], dtype=np.int64)),
        "k_pos": ("i64", np.arange(s_len, dtype=np.int64)),
    }
    if family == "swa":
        tensors["sink"] = ("f32", rng.uniform(-1.0, 1.0, n_q).astype(np.float32))
    for tag, rows, width in (("k", k, D_QK), ("v", v_cached, D_V)):
        flat = rows.reshape(s_len, n_kv * width)
        if pth:
            codes = np.empty_like(flat, dtype=np.uint8)
            scales = np.empty((s_len, n_kv), dtype=np.float32)
            for t in range(s_len):
                for h in range(n_kv):
                    s = _plane_scale(rows[t, h])
                    scales[t, h] = s
                    sl = flat[t, h * width:(h + 1) * width]
                    codes[t, h * width:(h + 1) * width] = fb.encode_e4m3(sl.astype(np.float64) / np.float64(s))
            tensors[f"{tag}_codes"] = ("u8", codes)
            tensors[f"{tag}_scales"] = ("f32", scales)
        else:
            tensors[f"{tag}_codes"] = ("u8", fb.encode_e4m3(flat.astype(np.float64)).astype(np.uint8))
    return c, tensors


def tc_decode_cases(p1=False):
    """New inputs for the EXISTING external oracle; no implementation import.

    Cached V is authored directly. Eight cases, each checked flat and paged.
    The absolute bound is 1e-5, no looser than existing tiny attention tests.
    """
    rng = np.random.default_rng(202609231)
    out = []
    specs = [
        ("tc_ga_pages", "ga", 257, [1000, 1130, 1256]),
        ("tc_swa_sink", "swa", 289, [1000, 1127, 1288]),
        ("tc_empty", "swa", 3, [998, 999]),
        ("tc_short", "ga", 3, [1000, 1002]),
        ("tc_q_low", "ga", 16, [1015]),
        ("tc_p_low", "ga", 16, [1015]),
        ("tc_rescale", "ga", 512, [1511]),
        ("tc_q_tail", "ga", 16, [1015]),
    ]
    if p1:
        # Append, never mutate the original eight fixtures. Nonmonotonic query
        # positions cross M64 query-tile boundaries and mix window visibility.
        specs += [("tc_p1_ga_tiles", "ga", 273, [999+(i*37+13)%301 for i in range(17)]),
                  ("tc_p1_swa_tiles", "swa", 273, [999+(i*41+7)%301 for i in range(17)])]
    for name, family, size, positions in specs:
        c, tensors = _attn_case(rng, name, "decode_fp8_unit", family,
                               len(positions), size, positions, range(1000, 1000+size),
                               WINDOW if family == "swa" else 0, family == "swa", 1e-5)
        if name == "tc_ga_pages":
            c.sink = 1  # Supplied but MUST be ignored on GA.
            tensors["sink"] = ("f32", np.linspace(-4, 4, 64, dtype=np.float32))
        if family == "swa":
            tensors["sink"] = ("f32", np.linspace(-4, 4, 64, dtype=np.float32))
        if name in ("tc_q_low", "tc_p_low", "tc_rescale", "tc_q_tail"):
            q, k, v = (tensors[tag][1] for tag in ("q", "k", "v"))
            q.fill(0); k.fill(0)
            if name in ("tc_q_low", "tc_q_tail"):
                if name == "tc_q_low":
                    q[:, :, 0] = 1 + 2**-8; q[:, :, 1] = 1
                else:
                    q[:, :, 0] = 1 + 2**-9 + 2**-17; q[:, :, 1] = 1 + 2**-9
                k[0::2, :, 0] = 448; k[0::2, :, 1] = -448
                k[1::2, :, 0] = -448; k[1::2, :, 1] = 448
                v[0::2] = 1; v[1::2] = -1
            elif name == "tc_p_low":
                q[:, :, 0] = 1; k[0::2, :, 0] = 1
                v[0::2] = 1; v[1::2] = -1
            else:
                q[:, :, 0] = 2
                sign = np.where(np.arange(size) % 64 < 32, -1, 1).astype(np.float32)
                k[:, :, 0] = sign[:, None] * 8
                v[:] = sign[:, None, None]
        for tag in ("k", "v"):
            values = tensors.pop(tag)[1]
            tensors[f"{tag}_codes"] = ("u8", fb.encode_e4m3(values.reshape(size, -1)).astype(np.uint8))
        out.append((c, tensors))
    return out


def tc_highv_cases(p1=False):
    """Separate R7 stress cohort; never alter the original eight fixtures."""
    cases = tc_decode_cases(p1=p1)
    for c, tensors in cases:
        c.name = "tc_highv_" + c.name[3:]
        cached = fb.decode_e4m3(tensors["v_codes"][1]).astype(np.float32)
        codes = fb.encode_e4m3(cached * np.float32(448)).astype(np.uint8)
        decoded = fb.decode_e4m3(codes).astype(np.float64)
        if not np.all(np.isfinite(decoded)) or np.max(np.abs(decoded)) > 448:
            raise ValueError("high-value fixture must stay within finite E4M3")
        tensors["v_codes"] = ("u8", codes)
        c.tol_abs = 2e-5 * max(1.0, float(np.max(np.abs(decoded))))
        c.tol_rel = 0.0
    return cases


def bf16_rne(q):
    """Independent finite FP32 -> BF16 RNE -> FP32 reference, not kernel code."""
    q = np.asarray(q, dtype=np.float32)
    if not np.all(np.isfinite(q)):
        raise ValueError("BF16-Q fixtures require finite post-RoPE inputs")
    u = q.view(np.uint32)
    return ((u + np.uint32(0x7fff) + ((u >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)


def tc_bf16q_cases(highv=False, p1=False):
    cases = tc_highv_cases(p1=p1) if highv else tc_decode_cases(p1=p1)
    for c, _ in cases:
        c.name = "tc_bf16q_" + c.name[3:]
    # Original FP32 input tensors remain unrounded: the GPU must do the cast.
    return cases


SHAPE_SETS = ["tiny", "decode-4k", "decode-32k", "prefill-4k", "prefill-32k",
              "decode-128k", "decode-1m", "rope-1m", "kv-store", "tc-decode", "tc-highv", "tc-bf16q", "tc-bf16q-highv"]


def gen_cases(sets, p1=False):
    if p1 and (len(sets)!=1 or sets[0] not in ("tc-decode","tc-highv","tc-bf16q","tc-bf16q-highv")):
        raise ValueError("P1 extension requires one explicit TC cohort")
    out = tc_decode_cases(p1=p1) if "tc-decode" in sets else []
    if "tc-highv" in sets:
        out.extend(tc_highv_cases(p1=p1))
    if "tc-bf16q" in sets:
        out.extend(tc_bf16q_cases(p1=p1))
    if "tc-bf16q-highv" in sets:
        out.extend(tc_bf16q_cases(highv=True,p1=p1))
    if "tiny" in sets:
        rng = np.random.default_rng(20260923)
        out.append(_attn_case(rng, "tiny_ga_basic", "attn", "ga", 3, 11, [3, 7, 10], list(range(11)), 0, False, 1e-5))
        out.append(_attn_case(rng, "tiny_swa_sink", "attn", "swa", 3, 13, [5, 11, 12], list(range(13)), 4, True, 1e-5))
        out.append(_attn_case(rng, "tiny_decode_start_pos", "decode", "ga", 1, 13, [1000], list(range(987, 1000)), 0, False, 1e-5))
        out.append(_attn_case(rng, "tiny_prefill_swa", "prefill", "swa", 3, 13, [10, 11, 12], list(range(13)), 4, True, 1e-5))
        for name, heads, theta in (("tiny_rope_ga", 4, THETA_GA), ("tiny_rope_swa", 8, THETA_SWA)):
            c = Case(name, "rope", "na", 0, 0, theta, PARTIAL, 1.0, 3.0 * EPS_F32 * 300 + 12 * EPS_F32, 0.0)
            out.append((c, {
                "x": ("f32", rng.uniform(-0.5, 0.5, (2, heads, D_QK)).astype(np.float32)),
                "pos": ("i64", np.asarray([1, 300], dtype=np.int64)),
            }))
        for name, mode in (("tiny_kv_pth", "kv_store_pth"), ("tiny_kv_unit", "kv_store_unit")):
            c = Case(name, mode, "na", 0, 0, 0.0, 0.0, VSCALE, 1e-6, 0.0)
            v_raw = rng.uniform(-0.5, 0.5, (3, 4, D_V)).astype(np.float32)
            # amax clip probe: survives the 0.707 pre-scale (T18) and fires the
            # unit-scale gate (clip == 1) while pth's plane scale absorbs it
            v_raw[0, 0, 0] = 700.0
            out.append((c, {
                "k": ("f32", rng.uniform(-0.5, 0.5, (3, 4, D_QK)).astype(np.float32)),
                "v_raw": ("f32", v_raw),
            }))
        out.append(_fp8_case(rng, "tiny_decode_fp8_pth", "ga", 33, True, 1e-5))
    for size, s_len, tol in (("4k", 4096, 2e-4), ("32k", 32768, 5e-4)):
        if f"decode-{size}" in sets:
            rng = np.random.default_rng(4240 + s_len)
            out.append(_attn_case(rng, f"ga_decode_{size}", "decode", "ga", 1, s_len, [s_len - 1], list(range(s_len)), 0, False, tol))
            out.append(_attn_case(rng, f"swa_decode_{size}", "decode", "swa", 1, s_len, [s_len - 1], list(range(s_len)), WINDOW, True, tol))
            out.append(_fp8_case(rng, f"ga_decode_fp8_{size}", "ga", s_len, True, tol))
            if size == "32k":
                out.append(_fp8_case(rng, f"ga_decode_fp8_unit_{size}", "ga", s_len, False, tol))
        if f"prefill-{size}" in sets:
            rng = np.random.default_rng(8480 + s_len)
            t_len = 256 if size == "4k" else 512
            q_pos = list(range(s_len - t_len, s_len))
            out.append(_attn_case(rng, f"ga_prefill_{size}", "prefill", "ga", t_len, s_len, q_pos, list(range(s_len)), 0, False, tol))
            out.append(_attn_case(rng, f"swa_prefill_{size}", "prefill", "swa", t_len, s_len, q_pos, list(range(s_len)), WINDOW, True, tol))
    if "decode-128k" in sets:
        rng = np.random.default_rng(131072)
        out.append(_fp8_case(rng, "ga_decode_fp8_128k", "ga", 131072, True, 2e-3))
    if "decode-1m" in sets:
        rng = np.random.default_rng(1048576)
        out.append(_fp8_case(rng, "ga_decode_fp8_1m", "ga", 1048576, True, 2e-3))
    if "rope-1m" in sets:
        rng = np.random.default_rng(7)
        c = Case("rope_ga_1m", "rope", "na", 0, 0, THETA_GA, PARTIAL, 1.0, rope_tol(1048575), 0.0)
        out.append((c, {
            "x": ("f32", rng.uniform(-0.5, 0.5, (3, N_Q_GA, D_QK)).astype(np.float32)),
            "pos": ("i64", np.asarray([1, 131072, 1048575], dtype=np.int64)),
        }))
    if "kv-store" in sets:
        rng = np.random.default_rng(1024)
        for name, mode in (("kv_pth_1k", "kv_store_pth"), ("kv_unit_1k", "kv_store_unit")):
            c = Case(name, mode, "na", 0, 0, 0.0, 0.0, VSCALE, 1e-6, 0.0)
            out.append((c, {
                "k": ("f32", rng.uniform(-0.5, 0.5, (1024, N_KV_GA, D_QK)).astype(np.float32)),
                "v_raw": ("f32", rng.uniform(-0.5, 0.5, (1024, N_KV_GA, D_V)).astype(np.float32)),
            }))
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def cmd_eval(args):
    in_dir, out_dir = pathlib.Path(args.i), pathlib.Path(args.o)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = parse_manifest(in_dir / "manifest.txt")
    out_cases = []
    for c in cases:
        exp = compute_expects(c, in_dir)
        oc = Case(c.name, c.ty, c.family, c.window, c.sink, c.theta, c.partial, c.vscale, c.tol_abs, c.tol_rel)
        for name, (dtype, arr) in exp.items():
            put(oc, out_dir, name, dtype, arr, is_expect=True)
        out_cases.append(oc)
    write_manifest(out_dir / "manifest.txt", out_cases)
    print(f"oracle_driver eval: {len(out_cases)} case(s) -> {out_dir}")


def audit_tc_values(cases):
    """R7: audit decoded, already-prescaled cached V; keep fixture tolerances."""
    count = 0
    for c, tensors in cases:
        if not c.name.startswith("tc_"):
            continue
        values = fb.decode_e4m3(tensors["v_codes"][1]).astype(np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"nonfinite cached V in {c.name}")
        magnitude = float(np.max(np.abs(values), initial=0))
        bound = 2e-5 * max(1.0, magnitude)
        if c.tol_abs > bound or c.tol_rel != 0:
            raise ValueError(f"fixture tolerance exceeds R7 in {c.name}")
        print(f"V_AUDIT case={c.name} family={c.family} cache=E4M3-unit-prescaled-V "
              f"max_abs_v={magnitude:.9g} R7_bound={bound:.9g} "
              f"fixture_abs={c.tol_abs:.9g} fixture_rel={c.tol_rel:.9g}")
        count += 1
    return count


def cmd_audit_tc(args):
    count = audit_tc_values(gen_cases([args.shapes],p1=args.p1))
    expected=10 if args.p1 else 8
    if count != expected:
        raise ValueError(f"incomplete {expected}-case audit: {count}")
    print(f"RESULT: PASS V magnitude audit {count}/{expected} (CPU metadata, not a new GPU requalification)")


def cmd_gen(args):
    out_dir = pathlib.Path(args.o)
    out_dir.mkdir(parents=True, exist_ok=True)
    sets = SHAPE_SETS.copy() if args.shapes in ("all", "--all") else [s for s in args.shapes.split(",") if s]
    for s in sets:
        if s not in SHAPE_SETS:
            raise SystemExit(f"unknown shape set {s!r}; known: {SHAPE_SETS}")
    if args.list_only:
        print("SHAPES " + ",".join(sets))
        return
    cases = []
    generated = gen_cases(sets,p1=args.p1)
    if "tc-decode" in sets or "tc-highv" in sets:
        audit_tc_values(generated)
    for c, tensors in generated:
        for name, (dtype, arr) in tensors.items():
            put(c, out_dir, name, dtype, arr, is_expect=False)
        exp = compute_expects(c, out_dir)
        for name, (dtype, arr) in exp.items():
            put(c, out_dir, name, dtype, arr, is_expect=True)
        cases.append(c)
    write_manifest(out_dir / "manifest.txt", cases)
    print(f"oracle_driver gen [{','.join(sets)}]: {len(cases)} case(s) -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    audit = sub.add_parser("audit-tc", help="audit decoded V magnitudes without GPU or oracle evaluation")
    audit.add_argument("--shapes", choices=("tc-decode", "tc-highv", "tc-bf16q", "tc-bf16q-highv"), default="tc-decode")
    audit.add_argument("--p1", action="store_true", help="append independent multi-query P1 inputs")
    audit.set_defaults(fn=cmd_audit_tc)
    ev = sub.add_parser("eval", help="compute oracle expects for a Rust-written input manifest")
    ev.add_argument("--in", dest="i", required=True)
    ev.add_argument("--out", dest="o", required=True)
    ev.set_defaults(fn=cmd_eval)
    gn = sub.add_parser("gen", help="generate inputs + oracle goldens for the GPU cell")
    gn.add_argument("--out", dest="o", required=True)
    gn.add_argument("--shapes", default="tiny", help="comma list of: " + ",".join(SHAPE_SETS))
    gn.add_argument("--list-only", action="store_true", help="validate/print selection without generating tensors")
    gn.add_argument("--p1", action="store_true", help="append independent multi-query P1 inputs")
    gn.set_defaults(fn=cmd_gen)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
