#!/usr/bin/env python3
"""R18a falsifier F1: routing on D7 prompt-set v1 and on the model's own greedy continuations (ADVISOR-I4 §9 R18a).

For each D7 category (harness/fleet/tonyd2wild/mimobench.py, prompt set v1, the nine categories), the prompt is
rendered with the project chat template (spike/needle_prompt.render_chat), prefilled teacher-forced, then continued
greedily for up to --gen tokens (capped at the category's max_tokens; EOS stops early). The S-arm router is recorded
for every routed token: the prompt tokens (prefill) and each generated token when it is fed back as input, which is
how a block-8 verification step routes accepted and drafted tokens. Output: .npz with, per category, "<cat>/prompt"
[47, T_prompt, 8] and "<cat>/gen" [47, T_gen, 8], plus the generated token ids. Token ids only (T29).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import route_trace as RT  # noqa: E402  (TraceModel: the S path with a read-only router record)
import x1c_arms as XA  # noqa: E402
from spike import needle_prompt as NP  # noqa: E402


def d7_categories():
    spec = importlib.util.spec_from_file_location("mimobench", os.path.join(HERE, "fleet/tonyd2wild/mimobench.py"))
    mb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mb)
    assert mb.PROMPT_SET_VERSION == "v1"
    return mb.CATEGORIES


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, default=96, help="greedy tokens per category (capped at its max_tokens)")
    ap.add_argument("--only", default="", help="comma-separated category subset (smoke tests)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(XA.RL.BASE, "tokenizer.json"))  # the spike's effective weights dir
    cats = [c for c in d7_categories() if not a.only or c[0] in a.only.split(",")]
    model = RT.TraceModel(XA.RL.Reader())
    moe = [i for i, f in enumerate(model.moe_freq) if f]
    out, t0 = {"moe_layers": np.array(moe, np.int16)}, time.time()
    for name, prompt, max_tokens in cats:
        ids = tok.encode(NP.render_chat(prompt)).ids
        model.trace = {}
        _, _, _, gen = XA.run(model, ids, greedy_steps=min(a.gen, max_tokens), eos=set(NP.EOS_IDS))
        arr = np.stack([np.concatenate(model.trace[l], 0) for l in moe], 0)  # [47, T_prompt + fed, 8]
        fed = arr.shape[1] - len(ids)  # generated tokens that were fed back (EOS is not fed)
        assert 0 <= fed <= len(gen), (name, arr.shape, len(ids), len(gen))
        out[f"{name}/prompt"] = arr[:, : len(ids)]
        out[f"{name}/gen"] = arr[:, len(ids):]
        out[f"{name}/gen_ids"] = np.array(gen, np.int32)
        print(f"[route-d7] {name} prompt={len(ids)} gen={len(gen)} fed={fed} {time.time() - t0:.0f}s",
              file=sys.stderr, flush=True)
    np.savez_compressed(a.out, **out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
