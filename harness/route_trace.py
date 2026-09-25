#!/usr/bin/env python3
"""Routing trace for the R16 expert-target re-plan (ADVISOR-I4 §9 R15/R16).

Runs the S arm (FP32 oracle, no lattice change) teacher-forced over the frozen natural-4096 prompts and records the
top-8 expert ids chosen at every MoE layer for every token. The router math is RealModel.mlp's, recomputed here only
to read the choice; the forward itself is the unmodified S path.

Output: an .npz with one int16 array per prompt, shape [n_moe_layers, T, 8], plus the MoE layer list.
Analysis (tokens per expert per step, M histograms for C1 block-8 and synthetic concurrency) is route_hist.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x1c_arms as XA  # noqa: E402

RL = XA.RL


class TraceModel(XA.X1cModel):
    trace: dict

    def mlp(self, layer, x):
        if self.moe_freq[layer]:
            dt, sh, gw = self.r.get(f"model.layers.{layer}.mlp.gate.weight")
            g = RL.to_f32(dt, gw.reshape(sh)).to(self.device)
            dt, sh, eb = self.r.get(f"model.layers.{layer}.mlp.gate.e_score_correction_bias")
            e_bias = RL.to_f32(dt, eb.reshape(sh)).to(self.device)
            choice = torch.sigmoid((x.float() @ g.T).float()) + e_bias
            gs = choice.view(-1, self.n_group, self.n_exp // self.n_group).topk(2, -1)[0].sum(-1)
            gi = gs.topk(self.topk_group, -1)[1]
            mask = torch.zeros_like(gs, dtype=torch.bool).scatter_(1, gi, True)
            sm = mask[:, :, None].expand(-1, -1, self.n_exp // self.n_group).reshape(-1, self.n_exp)
            idx = choice.masked_fill(~sm, float("-inf")).topk(self.top_k, -1)[1]
            self.trace.setdefault(layer, []).append(idx.to(torch.int16).cpu().numpy())
        return super().mlp(layer, x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-from", default="runs/20260923-i4/x1c/corpus/natural-4096.json")
    ap.add_argument("--limit", type=int, default=0, help="first n prompts only (0 = all)")
    ap.add_argument("--n", type=int, default=0, help="first n ids per prompt (0 = all; smoke tests)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    prompts = json.load(open(a.ids_from))["prompts"]
    prompts = prompts[: a.limit] if a.limit else prompts
    model = TraceModel(RL.Reader())
    moe = [i for i, f in enumerate(model.moe_freq) if f]
    out, t0 = {"moe_layers": np.array(moe, np.int16)}, time.time()
    for pr in prompts:
        ids = pr["prompt_ids"][: a.n] if a.n else pr["prompt_ids"]
        model.trace = {}
        XA.run(model, ids)
        arr = np.stack([np.concatenate(model.trace[l], 0) for l in moe], 0)
        assert arr.shape == (len(moe), len(ids), model.top_k), arr.shape
        out[pr["name"]] = arr
        print(f"[route] {pr['name']} T={len(ids)} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    np.savez_compressed(a.out, **out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
