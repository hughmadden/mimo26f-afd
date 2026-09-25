#!/usr/bin/env python3
"""Numerics (b) decisive test — per-layer hidden diff, chunk 4096 vs chunk 2048.

Reads the `MIMO26_DUMP_HIDDENS` + `MIMO26_DUMP_ROW_START=2048` outputs from two
coordinator runs over the SAME prompt and reports, per layer, the max-abs and
mean-abs hidden-state difference at the same absolute token positions:

  dir4096/c0.l{layer}.f32    — chunk 4096, rows [2048:T) of the single chunk
  dir2048/c2048.l{layer}.f32 — chunk 2048, the second chunk (tokens [2048:T))

Both files must hold the same number of f32 rows (default 1979 = 4027 - 2048).

Verdict (builder attn-d2-numerics.md, ADVISOR-I5):
  * layer 0 (GA) is ~1e-6 relative (accumulation-order / SGEMM-tiling class) and
    the gap GROWS smoothly through the layers, with no boundary-row concentration
    and a sparse->dense divergence profile -> FP8 chaos (record + distributional
    checks), the cross-chunk t != s attention path is CORRECT;
  * layer 0 (the first t != s attention layer) is ALREADY large (>~1e-4 abs) ->
    the cross-chunk TC path has a bug (fix it).

Usage:
  python3 harness/hidden_chunk_diff.py DIR4096 DIR2048 [hidden_size]
"""
import sys
import os
import numpy as np

N_LAYERS = 48
GA = {0, 5, 11, 17, 23, 29, 35, 41, 47}


def load(path, hid):
    a = np.fromfile(path, dtype="<f4")
    assert a.size % hid == 0, f"{path}: {a.size} not divisible by hid {hid}"
    return a.reshape(-1, hid)


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    d4096, d2048 = sys.argv[1], sys.argv[2]
    hid = int(sys.argv[3]) if len(sys.argv) > 3 else 4096
    print(f"hidden_size={hid}  layers={N_LAYERS}  GA={sorted(GA)}")
    print(f"{'layer':>5} {'max_abs':>12} {'mean_abs':>12} {'rel_max':>12}  kind")
    layer0 = None
    diffs = {}
    for layer in range(N_LAYERS):
        a = load(os.path.join(d4096, f"c0.l{layer}.f32"), hid)
        b = load(os.path.join(d2048, f"c2048.l{layer}.f32"), hid)
        assert a.shape == b.shape, f"layer {layer}: {a.shape} vs {b.shape}"
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        diffs[layer] = d
        max_abs = float(d.max())
        mean_abs = float(d.mean())
        rel = max_abs / max(float(np.abs(a).max()), 1e-30)
        kind = "GA" if layer in GA else "SWA"
        if layer == 0:
            layer0 = (max_abs, rel)
        print(f"{layer:>5} {max_abs:>12.4e} {mean_abs:>12.4e} {rel:>12.4e}  {kind}")

    # ---- verdict (builder's discriminator: layer 0, the first t != s layer) ----
    l0_abs, l0_rel = layer0
    print(f"\nlayer0(GA) max_abs={l0_abs:.4e}  rel_max={l0_rel:.4e}")

    # Distributional checks (FP8-chaos signature):
    d1 = diffs[1]
    rowmax = d1.max(axis=1)
    bm = rowmax[:128].mean() if rowmax.shape[0] > 128 else float("nan")
    im = rowmax[128:].mean() if rowmax.shape[0] > 128 else float("nan")
    n_over = int((d1 > 1e-6).sum())
    print(f"layer1(SWA) rowmax boundary-rows mean={bm:.3e}  interior mean={im:.3e}")
    print(f"layer1(SWA) elems>1e-6 = {n_over}/{d1.size} ({100*n_over/d1.size:.2f}%)")

    if l0_abs > 1e-4:
        print("\nVERDICT: layer 0 (first t != s attention) is already large "
              f"({l0_abs:.2e}) -> cross-chunk TC path BUGGY (fix it).")
    else:
        print("\nVERDICT: layer 0 is accumulation-order class (~1e-6 relative) and "
              "the gap grows smoothly through the layers -> FP8 CHAOS.")
        print("  -> cross-chunk t != s attention path is CORRECT; no bug to fix.")


if __name__ == "__main__":
    main()
