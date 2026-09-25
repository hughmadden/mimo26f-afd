#!/usr/bin/env python3
"""Vision reference (perf reset V2): the checkpoint's own MiMo vision tower, for goldens.

Two steps, two interpreters (neither needs transformers):

  prep  (system python3: Pillow + numpy) — HF Qwen2VLImageProcessor semantics for this checkpoint
        (patch 16, merge 2, temporal 2, min_pixels 3136, max_pixels 12845056, CLIP mean/std):
        exif_transpose, convert_to_rgb (alpha over white), smart_resize (factor 32), Pillow BICUBIC,
        rescale + normalize, patchify.
            python3 harness/vision_ref.py prep IMAGE OUT.npz
  vit   (a torch venv) — runs `MiMoVisionTransformer` exec'd from the checkpoint's
        `modeling_mimo_v2.py` (the "Vision encoder" section, verbatim, never copied into this repo)
        on the `visual.*` tensors; missing merger biases are zero, as transformers initialises them.
            python harness/vision_ref.py vit MODELING.py CONFIG.json VISUAL.safetensors IN.npz OUT.npz [--bf16]

OUT.npz holds `embeds` [tokens, 4096] float32 and, with --dump, `blocks` [28, patches, 1280] (the
residual stream after each block, in row order) and `patch` (after the patch embedding).
"""
from __future__ import annotations

import json
import math
import sys


MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
PATCH, MERGE, TEMPORAL, FACTOR = 16, 2, 2, 32
MIN_PIXELS, MAX_PIXELS = 3136, 12845056


def save(out, **arrays):
    """OUT.npz plus one OUT.<name>.npy per array (the Rust check reads .npy)."""
    import numpy as np
    np.savez(out, **arrays)
    stem = out[:-4] if out.endswith(".npz") else out
    for k, a in arrays.items():
        np.save(f"{stem}.{k}.npy", a)


def smart_resize(height, width, factor=FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    if max(height, width) / min(height, width) > 200:
        raise ValueError("absolute aspect ratio must be smaller than 200")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def prep(path, out):
    import numpy as np
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(Image.open(path))
    if img.mode != "RGB":  # transformers.image_transforms.convert_to_rgb
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255))
        img = Image.alpha_composite(bg, rgba).convert("RGB")
    h, w = img.height, img.width
    rh, rw = smart_resize(h, w)
    if (rh, rw) != (h, w):
        img = img.resize((rw, rh), Image.BICUBIC)
    a = np.asarray(img, dtype=np.uint8).astype(np.float32) * np.float32(1 / 255)
    a = (a - np.array(MEAN, np.float32)) / np.array(STD, np.float32)  # [H, W, C]
    gh, gw = rh // PATCH, rw // PATCH
    x = a.transpose(2, 0, 1).reshape(3, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
    x = x.transpose(1, 4, 2, 5, 0, 3, 6)  # [gh/2, gw/2, 2, 2, C, 16, 16]
    x = np.repeat(x[:, :, :, :, :, None], TEMPORAL, axis=5)  # [.., C, T, 16, 16]
    pv = np.ascontiguousarray(x.reshape(gh * gw, 3 * TEMPORAL * PATCH * PATCH), dtype=np.float32)
    save(out, pixel_values=pv, grid=np.array([1, gh, gw], np.int64), size=np.array([h, w, rh, rw]))
    print(f"prep {path}: {w}x{h} -> {rw}x{rh}, grid {gh}x{gw}, {gh * gw} patches, {gh * gw // 4} tokens -> {out}")


def load_safetensors(path):
    import numpy as np
    import torch
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        hdr = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            s, e = v["data_offsets"]
            f.seek(base + s)
            raw = np.frombuffer(f.read(e - s), dtype=np.uint16 if v["dtype"] == "BF16" else np.float32)
            if v["dtype"] == "BF16":
                raw = (raw.astype(np.uint32) << 16).view(np.float32)
            out[k] = torch.from_numpy(raw.copy()).reshape(v["shape"])
        return out


def vision_classes(modeling_path):
    """Exec the checkpoint's "Vision encoder" section (verbatim) with torch and a SiLU ACT2FN."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    src = open(modeling_path).read()
    start = src.index("def _rotate_half_vision")
    end = src.index("# Audio encoder")
    ns = {"torch": torch, "nn": nn, "F": F, "math": math, "ACT2FN": {"silu": nn.SiLU()}}
    exec(compile(src[start:end], modeling_path + " [vision section]", "exec"), ns)
    return ns


def vit(modeling, config, weights, inp, out, bf16=False, dump=False):
    import types
    import numpy as np
    import torch
    ns = vision_classes(modeling)
    vc = json.load(open(config))["vision_config"]
    model = ns["MiMoVisionTransformer"](types.SimpleNamespace(**vc))
    sd = {k[len("visual."):]: v for k, v in load_safetensors(weights).items() if k.startswith("visual.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert set(missing) == {"merger.ln_q.bias", "merger.mlp.0.bias", "merger.mlp.2.bias"}, missing
    with torch.no_grad():
        for name in missing:  # transformers' _init_weights zeroes a missing Linear/LayerNorm bias
            model.get_parameter(name).zero_()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev, dtype=torch.bfloat16 if bf16 else torch.float32).eval()
    z = np.load(inp)
    pv = torch.from_numpy(z["pixel_values"]).to(dev)
    grid = torch.from_numpy(z["grid"]).reshape(1, 3).to(dev)
    blocks = []
    if dump:
        def hook(i):
            def f(_m, _a, o):
                blocks.append((i, o.detach().float().cpu().numpy()))
            return f
        for i, b in enumerate(model.blocks):
            b.register_forward_hook(hook(i))
    with torch.no_grad():
        patch = model.patch_embed(pv.to(model.dtype)).float().cpu().numpy() if dump else None
        emb = model(pv, grid).float().cpu().numpy()
    extra = {}
    if dump:
        # Block outputs are in the block's own token order; put column-ordered ones back in row order.
        col = model.get_window_index_1d(grid.cpu(), col=True).numpy()
        rows = []
        for i, o in blocks:
            if model.vit_window_attn_types[i] == 1:
                o = o.reshape(-1, 4, o.shape[-1])[np.argsort(col)].reshape(-1, o.shape[-1])
            rows.append(o)
        extra = {"blocks": np.stack(rows), "patch": patch}
    save(out, embeds=emb, **extra)
    print(f"vit {inp}: {'bf16' if bf16 else 'fp32'} on {dev}: embeds {emb.shape}, |x| mean {np.abs(emb).mean():.4f} -> {out}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "prep" and len(a) == 3:
        prep(a[1], a[2])
    elif a and a[0] == "vit" and len(a) >= 6:
        vit(*a[1:6], bf16="--bf16" in a, dump="--dump" in a)
    else:
        raise SystemExit(__doc__)
