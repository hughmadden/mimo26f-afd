#!/usr/bin/env python3
"""Generate the goldens for crates/mimo26-image (system python3 + Pillow + numpy only).

Oracle
------
Every expected value comes from Pillow's *decoder* / resampler and numpy, never
from the encoders in this file:

* decode goldens: the HF load path
      Image.open(BytesIO(data)) -> ImageOps.exif_transpose -> convert_to_rgb
  where convert_to_rgb mirrors transformers.image_transforms.convert_to_rgb
  (non-RGB modes -> RGBA -> alpha_composite over opaque white -> RGB).
* resize goldens: Image.fromarray(rgb).resize((w, h), Image.BICUBIC).
* smart_resize goldens: the Qwen2-VL smart_resize below (factor 32,
  min_pixels 3136, max_pixels 12845056), copied from transformers.
* pixel_values goldens: hf_pixel_values below mirrors transformers
  Qwen2VLImageProcessor._preprocess (slow / PIL path) with this checkpoint's
  preprocessor_config: patch_size 16, merge_size 2, temporal_patch_size 2,
  BICUBIC resample, rescale_factor 1/255, CLIP image_mean / image_std.

Inputs are written by Pillow's encoders or by the small PNG / baseline-JPEG
encoders below, which exist only to produce layouts Pillow cannot write
(Adam7, per-row filter choice, sub-byte depths with tRNS, 4:1:1 / 4:4:0 and
odd sampling factors, Adobe-RGB, missing DHT, 16-bit DQT, fill bytes, ...).

Usage:  python3 crates/mimo26-image/tests/gen_goldens.py   (rewrites tests/goldens/)
Output is deterministic for a given Pillow / numpy / libjpeg-turbo / zlib build.
"""

import hashlib
import io
import math
import os
import shutil
import struct
import sys
import zlib

import numpy as np
import PIL
from PIL import Image, ImageOps, features

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "goldens")

MEAN = [0.48145466, 0.4578275, 0.40821073]
STD = [0.26862954, 0.26130258, 0.27577711]
RESCALE = 0.00392156862745098  # preprocessor_config rescale_factor
assert RESCALE == 1 / 255

# ------------------------------------------------------------ HF oracles ---


def convert_to_rgb(image):
    """transformers.image_transforms.convert_to_rgb (PIL branch)."""
    if image.mode == "RGB":
        return image
    image_rgba = image.convert("RGBA")
    background = Image.new("RGBA", image_rgba.size, (255, 255, 255))
    alpha_composite = Image.alpha_composite(background, image_rgba)
    return alpha_composite.convert("RGB")


def hf_decode(data):
    im = Image.open(io.BytesIO(data))
    im = ImageOps.exif_transpose(im)
    im = convert_to_rgb(im)
    return np.asarray(im, dtype=np.uint8), Image.open(io.BytesIO(data)).mode


def smart_resize(height, width, factor=32, min_pixels=3136, max_pixels=12845056):
    """transformers Qwen2-VL smart_resize (as specified for mimo26-image)."""
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


def hf_pixel_values(rgb):
    """Qwen2VLImageProcessor._preprocess for one RGB uint8 (H, W, 3) image."""
    h, w = rgb.shape[:2]
    rh, rw = smart_resize(h, w)
    # image_transforms.resize -> to_pil_image (uint8, no rescale) -> PIL resize
    # (reducing_gap=None) -> np.array. PIL returns a copy when the size matches.
    resized = np.asarray(Image.fromarray(rgb).resize((rw, rh), resample=Image.BICUBIC))
    # rescale: image.astype(float64) * scale, then astype(float32)
    x = (resized.astype(np.float64) * RESCALE).astype(np.float32)
    # normalize (channels last): (image - mean) / std with float32 mean/std
    x = (x - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    x = x.transpose(2, 0, 1)[None]  # (1, C, H, W), data_format=FIRST
    x = np.concatenate([x, x], axis=0)  # pad to temporal_patch_size by repeating
    gt, gh, gw = 1, rh // 16, rw // 16
    x = x.reshape(gt, 2, 3, gh // 2, 2, 16, gw // 2, 2, 16)
    x = x.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flat = np.ascontiguousarray(x.reshape(gt * gh * gw, 3 * 2 * 16 * 16), dtype="<f4")
    return flat, (gt, gh, gw), np.ascontiguousarray(resized)


# ------------------------------------------------------- synthetic images ---


def synth(w, h, seed, noise=10.0):
    """Photo-like RGB: smooth color fields, hard edges and mild noise."""
    rng = np.random.RandomState(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    r = 128 + 100 * np.sin(x / 7.0 + seed) * np.cos(y / 11.0)
    g = 128 + 90 * np.cos((x + y) / 13.0 + 0.5 * seed)
    b = (x * 255.0 / max(w - 1, 1) + y * 255.0 / max(h - 1, 1)) / 2
    img = np.stack([r, g, b], -1)
    img[(x > w * 0.3) & (x < w * 0.6) & (y > h * 0.2) & (y < h * 0.5)] = [230, 30, 40]
    img[np.abs(x - y * w / max(h, 1)) < 1.5] = [20, 200, 60]
    img += rng.normal(0, noise, img.shape)
    return np.clip(np.round(img), 0, 255).astype(np.uint8)


def procedural(w, h):
    """Deterministic pattern also generated by the Rust test (tests/common)."""
    y, x = np.mgrid[0:h, 0:w].astype(np.uint64)
    n = ((x * np.uint64(2654435761) + y * np.uint64(40503)) & np.uint64(0xFFFFFFFF)) >> np.uint64(16)
    r = (x + np.uint64(2) * y + ((x * y) >> np.uint64(9))) & np.uint64(255)
    g = (np.uint64(128) + ((x >> np.uint64(3)) ^ (y >> np.uint64(3))) * np.uint64(3)) & np.uint64(255)
    b = ((x >> np.uint64(2)) + (y >> np.uint64(2)) + (n & np.uint64(31))) & np.uint64(255)
    return np.stack([r, g, b], -1).astype(np.uint8)


# ------------------------------------------------------------ PNG writer ---

ADAM7 = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]


def png_chunk(ctype, data):
    return struct.pack(">I", len(data)) + ctype + data + struct.pack(">I", zlib.crc32(ctype + data) & 0xFFFFFFFF)


def pack_row(samples, depth):
    samples = np.asarray(samples).reshape(-1)
    if depth == 8:
        return samples.astype(np.uint8).tobytes()
    if depth == 16:
        return samples.astype(">u2").tobytes()
    per = 8 // depth
    out = bytearray((len(samples) * depth + 7) // 8)
    for i, v in enumerate(samples):
        out[i // per] |= int(v) << (8 - depth * (i % per + 1))
    return bytes(out)


def filter_row(ft, row, prev, bpp):
    cur = np.frombuffer(row, np.uint8).astype(np.int32)
    up = np.frombuffer(prev, np.uint8).astype(np.int32) if prev is not None else np.zeros_like(cur)
    out = bytearray(len(cur))
    for i in range(len(cur)):
        a = int(cur[i - bpp]) if i >= bpp else 0
        b = int(up[i])
        c = int(up[i - bpp]) if i >= bpp else 0
        if ft == 0:
            p = 0
        elif ft == 1:
            p = a
        elif ft == 2:
            p = b
        elif ft == 3:
            p = (a + b) // 2
        else:
            pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
            p = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
        out[i] = (int(cur[i]) - p) & 255
    return bytes([ft]) + bytes(out)


def make_png(samples, depth, ctype, *, filters=(0, 1, 2, 3, 4), interlace=False, plte=None, trns=None,
             exif=None, exif_after=None, level=9, strategy=zlib.Z_DEFAULT_STRATEGY, idat_sizes=None,
             pre_chunks=(), post_chunks=(), zdata=None):
    """samples: (h, w, ch) raw sample values (palette indices for ctype 3)."""
    h, w, ch = samples.shape
    bpp = max(1, depth * ch // 8)
    raw = bytearray()
    fi = 0
    for (x0, y0, dx, dy) in (ADAM7 if interlace else [(0, 0, 1, 1)]):
        sub = samples[y0::dy, x0::dx]
        if sub.shape[0] == 0 or sub.shape[1] == 0:
            continue
        prev = None
        for r in range(sub.shape[0]):
            row = pack_row(sub[r], depth)
            raw += filter_row(filters[fi % len(filters)], row, prev, bpp)
            fi += 1
            prev = row
    if zdata is None:
        c = zlib.compressobj(level, zlib.DEFLATED, 15, 9, strategy)
        zdata = c.compress(bytes(raw)) + c.flush()
    out = b"\x89PNG\r\n\x1a\n"
    out += png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, ctype, 0, 0, 1 if interlace else 0))
    for t, d in pre_chunks:
        out += png_chunk(t, d)
    if plte is not None:
        out += png_chunk(b"PLTE", plte)
    if trns is not None:
        out += png_chunk(b"tRNS", trns)
    if exif is not None:
        out += png_chunk(b"eXIf", exif)
    parts = []
    rest = zdata
    for n in (idat_sizes or []):
        parts.append(rest[:n])
        rest = rest[n:]
    parts.append(rest)
    for p in parts:
        out += png_chunk(b"IDAT", p)
    for t, d in post_chunks:
        out += png_chunk(t, d)
    if exif_after is not None:
        out += png_chunk(b"eXIf", exif_after)
    out += png_chunk(b"IEND", b"")
    return out


def pil_bytes(im, fmt, **kw):
    b = io.BytesIO()
    im.save(b, fmt, **kw)
    return b.getvalue()


# ----------------------------------------------------------- JPEG writer ---

NATURAL = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
           13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52,
           45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63]
STD_LUM_Q = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
             14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
             92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]
STD_CHR_Q = [17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
             47, 66, 99, 99, 99, 99, 99, 99] + [99] * 32


def std_huff_tables():
    """Huffman tables libjpeg-turbo writes by default (parsed from a Pillow JPEG)."""
    d = pil_bytes(Image.new("RGB", (8, 8), (10, 20, 30)), "JPEG", quality=75)
    tables = {}
    i = 2
    while i < len(d):
        assert d[i] == 0xFF
        m = d[i + 1]
        ln = struct.unpack(">H", d[i + 2:i + 4])[0]
        seg = d[i + 4:i + 2 + ln]
        if m == 0xC4:
            j = 0
            while j < len(seg):
                idx = seg[j]
                bits = list(seg[j + 1:j + 17])
                n = sum(bits)
                tables[(idx >> 4, idx & 15)] = (bits, list(seg[j + 17:j + 17 + n]))
                j += 17 + n
        if m == 0xDA:
            break
        i += 2 + ln
    return tables


STD_HUFF = std_huff_tables()


def huff_codes(bits, vals):
    codes = {}
    code = 0
    k = 0
    for length in range(1, 17):
        for _ in range(bits[length - 1]):
            codes[vals[k]] = (code, length)
            code += 1
            k += 1
        code <<= 1
    return codes


class BitWriter:
    def __init__(self):
        self.out = bytearray()
        self.acc = 0
        self.n = 0

    def put(self, value, nbits):
        for i in range(nbits - 1, -1, -1):
            self.acc = (self.acc << 1) | ((value >> i) & 1)
            self.n += 1
            if self.n == 8:
                self.out.append(self.acc)
                if self.acc == 0xFF:
                    self.out.append(0)
                self.acc = 0
                self.n = 0

    def flush(self):
        if self.n:
            self.put((1 << (8 - self.n)) - 1, 8 - self.n)


def qtable(base, quality):
    scale = 5000 // quality if quality < 50 else 200 - quality * 2
    return [min(max((q * scale + 50) // 100, 1), 255) for q in base]


def make_jpeg(rgb, *, sampling=((2, 2), (1, 1), (1, 1)), quality=85, color="ycc", jfif=True, adobe=None,
              comp_ids=None, dht=True, separate_scans=False, restart=0, q16=False, fill_ff=False,
              extra_q=None, dqt_override=None, app=()):
    """Minimal baseline (SOF0) Huffman encoder. rgb: (h, w, 3) uint8 or (h, w) gray."""
    rgb = np.asarray(rgb)
    if color == "gray":
        planes = [rgb.astype(np.float64)]
    else:
        r, g, b = [rgb[..., i].astype(np.float64) for i in range(3)]
        if color == "ycc":
            planes = [0.299 * r + 0.587 * g + 0.114 * b,
                      -0.168736 * r - 0.331264 * g + 0.5 * b + 128,
                      0.5 * r - 0.418688 * g - 0.081312 * b + 128]
        else:
            planes = [r, g, b]
    nc = len(planes)
    sampling = list(sampling[:nc])
    h, w = planes[0].shape
    maxh = max(s[0] for s in sampling)
    maxv = max(s[1] for s in sampling)
    mcux = -(-w // (8 * maxh))
    mcuy = -(-h // (8 * maxv))
    ids = comp_ids or ([1] if nc == 1 else [1, 2, 3])
    qts = [qtable(STD_LUM_Q, quality), qtable(STD_CHR_Q, quality)]
    if extra_q is not None:
        qts[0] = extra_q
    tq = [0] + [1] * (nc - 1)
    C = np.array([[math.sqrt((1 if k == 0 else 2) / 8) * math.cos((2 * n + 1) * k * math.pi / 16)
                   for n in range(8)] for k in range(8)])
    comps = []
    for ci, (p, (hs, vs)) in enumerate(zip(planes, sampling)):
        he, ve = maxh // hs, maxv // vs
        dw, dh = -(-w * hs // maxh), -(-h * vs // maxv)
        pp = np.pad(p, ((0, dh * ve - h), (0, dw * he - w)), mode="edge")
        ds = pp.reshape(dh, ve, dw, he).mean(axis=(1, 3))
        pbw, pbh = mcux * hs, mcuy * vs
        full = np.pad(ds, ((0, pbh * 8 - dh), (0, pbw * 8 - dw)), mode="edge") - 128.0
        q = np.array(qts[tq[ci]], dtype=np.float64).reshape(8, 8)
        blocks = np.zeros((pbh, pbw, 64), dtype=np.int64)
        for by in range(pbh):
            for bx in range(pbw):
                f = C @ full[by * 8:by * 8 + 8, bx * 8:bx * 8 + 8] @ C.T
                blocks[by, bx] = np.round(f / q).astype(np.int64).reshape(64)
        comps.append(dict(blocks=blocks, h=hs, v=vs, bw=-(-dw // 8), bh=-(-dh // 8)))
    dc_codes = [huff_codes(*STD_HUFF[(0, 0)]), huff_codes(*STD_HUFF[(0, 1)])]
    ac_codes = [huff_codes(*STD_HUFF[(1, 0)]), huff_codes(*STD_HUFF[(1, 1)])]

    def enc_block(bw, blk, pred, t):
        zz = [int(blk[NATURAL[k]]) for k in range(64)]
        diff = zz[0] - pred
        s = abs(diff).bit_length()
        code, ln = dc_codes[t][s]
        bw.put(code, ln)
        if s:
            bw.put(diff if diff >= 0 else diff + (1 << s) - 1, s)
        run = 0
        for k in range(1, 64):
            v = zz[k]
            if v == 0:
                run += 1
                continue
            while run > 15:
                bw.put(*ac_codes[t][0xF0])
                run -= 16
            s = abs(v).bit_length()
            bw.put(*ac_codes[t][(run << 4) | s])
            bw.put(v if v >= 0 else v + (1 << s) - 1, s)
            run = 0
        if run:
            bw.put(*ac_codes[t][0x00])
        return zz[0]

    def encode_scan(scan_comps):
        bw = BitWriter()
        preds = {ci: 0 for ci in scan_comps}
        if len(scan_comps) == 1:
            c = comps[scan_comps[0]]
            units = [(my, mx) for my in range(c["bh"]) for mx in range(c["bw"])]
        else:
            units = [(my, mx) for my in range(mcuy) for mx in range(mcux)]
        for n, (my, mx) in enumerate(units):
            if restart and n and n % restart == 0:
                bw.flush()
                if fill_ff:
                    bw.out += b"\xff\xff"
                bw.out += bytes([0xFF, 0xD0 + ((n // restart - 1) & 7)])
                preds = {ci: 0 for ci in scan_comps}
            for ci in scan_comps:
                c = comps[ci]
                t = 0 if ci == 0 else 1
                if len(scan_comps) == 1:
                    preds[ci] = enc_block(bw, c["blocks"][my, mx], preds[ci], t)
                else:
                    for by in range(c["v"]):
                        for bx in range(c["h"]):
                            preds[ci] = enc_block(bw, c["blocks"][my * c["v"] + by, mx * c["h"] + bx],
                                                  preds[ci], t)
        bw.flush()
        return bytes(bw.out)

    def seg(marker, payload):
        return bytes([0xFF, marker]) + struct.pack(">H", len(payload) + 2) + payload

    out = b"\xff\xd8"
    if jfif:
        out += seg(0xE0, b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00")
    if adobe is not None:
        out += seg(0xEE, b"Adobe" + struct.pack(">HHHB", 100, 0, 0, adobe))
    for m, payload in app:
        out += seg(m, payload)
    for t in range(1 if nc == 1 else 2):
        written = dqt_override if dqt_override is not None else qts[t]
        if q16:
            out += seg(0xDB, bytes([0x10 | t]) + b"".join(struct.pack(">H", written[NATURAL[k]])
                                                         for k in range(64)))
        else:
            out += seg(0xDB, bytes([t]) + bytes(qts[t][NATURAL[k]] for k in range(64)))
    sof = struct.pack(">BHHB", 8, h, w, nc)
    for ci in range(nc):
        sof += bytes([ids[ci], (sampling[ci][0] << 4) | sampling[ci][1], tq[ci]])
    out += seg(0xC0, sof)
    if dht:
        for (cls, slot) in [(0, 0), (1, 0), (0, 1), (1, 1)]:
            bits, vals = STD_HUFF[(cls, slot)]
            out += seg(0xC4, bytes([(cls << 4) | slot]) + bytes(bits) + bytes(vals))
    if restart:
        out += seg(0xDD, struct.pack(">H", restart))
    scans = [[ci] for ci in range(nc)] if separate_scans else [list(range(nc))]
    for sc in scans:
        payload = bytes([len(sc)])
        for ci in sc:
            t = 0 if ci == 0 else 1
            payload += bytes([ids[ci], (t << 4) | t])
        payload += bytes([0, 63, 0])
        out += seg(0xDA, payload)
        out += encode_scan(sc)
        if fill_ff:
            out += b"\xff\xff\xff"
    out += b"\xff\xd9"
    return out


def insert_after_soi(jpeg, segments):
    return jpeg[:2] + b"".join(segments) + jpeg[2:]


def app1(payload):
    return b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload


def tiff_orientation(value, typ=3, le=True, pad=0):
    """Minimal TIFF blob with IFD0 = one Orientation entry."""
    e = "<" if le else ">"
    ifd = 8 + pad
    head = (b"II" if le else b"MM") + struct.pack(e + "HI", 42, ifd) + b"\0" * pad
    if typ == 3:
        val = struct.pack(e + "H", value) + b"\0\0"
    else:
        val = bytes([value, 0, 0, 0])
    return head + struct.pack(e + "H", 1) + struct.pack(e + "HHI", 0x0112, typ, 1) + val + struct.pack(e + "I", 0)


# ------------------------------------------------------------- manifests ---


class Goldens:
    def __init__(self):
        if os.path.isdir(OUT):
            shutil.rmtree(OUT)
        for d in ("inputs", "expected", "resize", "pixel_values"):
            os.makedirs(os.path.join(OUT, d))
        self.decode = []
        self.errors = []
        self.resize = []

    def write(self, rel, data):
        with open(os.path.join(OUT, rel), "wb") as f:
            f.write(data)

    def add_decode(self, name, data, note=""):
        ext = "png" if data[:4] == b"\x89PNG" else "jpg"
        rel = f"inputs/{name}.{ext}"
        self.write(rel, data)
        rgb, mode = hf_decode(data)
        exp = f"expected/{name}.rgb"
        self.write(exp, rgb.tobytes())
        h, w = rgb.shape[:2]
        self.decode.append(f"{rel} {w} {h} {exp} {mode}{(' # ' + note) if note else ''}")

    def add_error(self, name, data, needle, ext=None):
        ext = ext or ("png" if data[:4] == b"\x89PNG" else "jpg" if data[:2] == b"\xff\xd8" else "dat")
        rel = f"inputs/{name}.{ext}"
        self.write(rel, data)
        try:
            hf_decode(data)
            pillow = "pillow-ok"
        except Exception as e:  # noqa: BLE001
            pillow = "pillow-error:" + type(e).__name__
        self.errors.append(f"{rel} {pillow} | {needle}")


# --------------------------------------------------------------- the cases ---


def png_cases(g):
    rng = np.random.RandomState(7)
    rgb = synth(37, 53, 1)
    g.add_decode("png_rgb8_filters", make_png(rgb, 8, 2))
    g.add_decode("png_rgb8_257x129_pillow", pil_bytes(Image.fromarray(synth(257, 129, 2)), "PNG"))
    g.add_decode("png_rgb8_1x1", make_png(np.array([[[12, 200, 77]]]), 8, 2))
    g.add_decode("png_rgb8_3x7_paeth", make_png(synth(3, 7, 3), 8, 2, filters=(4,)))
    rgba = np.concatenate([synth(37, 53, 4), rng.randint(0, 256, (53, 37, 1))], -1)
    rgba[0, :, 3] = 0
    rgba[1, :, 3] = 255
    g.add_decode("png_rgba8", make_png(rgba, 8, 6))
    # Exhaustive composite table: column = value, row = alpha.
    v = np.arange(256)
    tab = np.zeros((256, 256, 4), np.int64)
    tab[..., 0] = v[None, :]
    tab[..., 1] = 255 - v[None, :]
    tab[..., 2] = (v[None, :] * 7) & 255
    tab[..., 3] = v[:, None]
    g.add_decode("png_rgba8_alpha_table", make_png(tab, 8, 6, filters=(1,)))
    g.add_decode("png_rgba8_pillow", pil_bytes(Image.fromarray(rgba.astype(np.uint8)), "PNG"))

    pal = rng.randint(0, 256, (200, 3)).astype(np.uint8).tobytes()
    idx = rng.randint(0, 200, (29, 37, 1))
    g.add_decode("png_pal8", make_png(idx, 8, 3, plte=pal))
    g.add_decode("png_pal8_trns", make_png(idx, 8, 3, plte=pal, trns=rng.randint(0, 256, 150).astype(np.uint8).tobytes()))
    trns_simple = bytearray(b"\xff" * 20)
    trns_simple[5] = 0
    g.add_decode("png_pal8_trns_simple", make_png(idx % 20, 8, 3, plte=pal, trns=bytes(trns_simple)))
    g.add_decode("png_pal8_trns_long", make_png(idx, 8, 3, plte=pal, trns=b"\x00" + b"\xff" * 299))
    g.add_decode("png_pal8_out_of_range", make_png(rng.randint(0, 256, (11, 13, 1)), 8, 3, plte=pal[:30]))
    g.add_decode("png_pal8_no_plte", make_png(rng.randint(0, 4, (3, 5, 1)), 8, 3))
    for depth in (1, 2, 4):
        n = 1 << depth
        pl = rng.randint(0, 256, (n, 3)).astype(np.uint8).tobytes()
        ix = rng.randint(0, n, (7, 13, 1))
        g.add_decode(f"png_pal{depth}", make_png(ix, depth, 3, plte=pl))
        g.add_decode(f"png_pal{depth}_trns", make_png(ix, depth, 3, plte=pl, trns=bytes([0, 128, 255][:n])))
        g.add_decode(f"png_pal{depth}_adam7", make_png(rng.randint(0, n, (21, 19, 1)), depth, 3, plte=pl,
                                                      interlace=True))

    for depth in (1, 2, 4, 8, 16):
        mx = (1 << depth) - 1
        gray = rng.randint(0, mx + 1, (53, 37, 1))
        if depth == 16:
            gray[0, :8, 0] = [0, 1, 255, 256, 257, 1000, 232, 65535]
        g.add_decode(f"png_gray{depth}", make_png(gray, depth, 0))
        g.add_decode(f"png_gray{depth}_3x7", make_png(rng.randint(0, mx + 1, (7, 3, 1)), depth, 0))
    g.add_decode("png_gray1_trns1", make_png(rng.randint(0, 2, (9, 17, 1)), 1, 0, trns=struct.pack(">H", 1)))
    g.add_decode("png_gray1_trns0", make_png(rng.randint(0, 2, (9, 17, 1)), 1, 0, trns=struct.pack(">H", 0)))
    g.add_decode("png_gray2_trns2", make_png(rng.randint(0, 4, (9, 17, 1)), 2, 0, trns=struct.pack(">H", 2)),
                 "Pillow compares the scaled L value with the raw tRNS value: nothing matches")
    g.add_decode("png_gray4_trns170", make_png(rng.randint(0, 16, (9, 17, 1)), 4, 0,
                                               trns=struct.pack(">H", 170)))
    g8 = rng.randint(90, 110, (21, 23, 1))
    g.add_decode("png_gray8_trns", make_png(g8, 8, 0, trns=struct.pack(">H", 100)))
    g16 = rng.randint(0, 1200, (21, 23, 1))
    g16[0, :4, 0] = [232, 1000, 255, 256]
    g.add_decode("png_gray16_trns1000", make_png(g16, 16, 0, trns=struct.pack(">H", 1000)),
                 "I;16 clips to 255 and compares with tRNS & 0xff (232)")
    g.add_decode("png_gray16_trns255", make_png(g16, 16, 0, trns=struct.pack(">H", 255)))
    la = np.concatenate([rng.randint(0, 256, (29, 31, 1)), rng.randint(0, 256, (29, 31, 1))], -1)
    g.add_decode("png_la8", make_png(la, 8, 4))
    g.add_decode("png_la16", make_png(rng.randint(0, 65536, (29, 31, 2)), 16, 4))
    rgb16 = rng.randint(0, 65536, (23, 29, 3))
    g.add_decode("png_rgb16", make_png(rgb16, 16, 2))
    g.add_decode("png_rgb16_trns", make_png(rgb16, 16, 2, trns=struct.pack(">HHH", 1, 2, 3)))
    g.add_decode("png_rgba16", make_png(rng.randint(0, 65536, (23, 29, 4)), 16, 6))
    rgb8t = synth(21, 17, 5)
    t = rgb8t[3, 4]
    g.add_decode("png_rgb8_trns_ignored", make_png(rgb8t, 8, 2, trns=struct.pack(">HHH", *[int(c) for c in t])),
                 "convert_to_rgb returns mode RGB unchanged, tRNS ignored")

    g.add_decode("png_adam7_rgb8", make_png(synth(37, 53, 6), 8, 2, interlace=True))
    g.add_decode("png_adam7_rgba16", make_png(rng.randint(0, 65536, (19, 23, 4)), 16, 6, interlace=True))
    g.add_decode("png_adam7_gray16", make_png(rng.randint(0, 400, (19, 23, 1)), 16, 0, interlace=True))
    g.add_decode("png_adam7_gray2", make_png(rng.randint(0, 4, (19, 23, 1)), 2, 0, interlace=True))
    g.add_decode("png_adam7_la8", make_png(rng.randint(0, 256, (13, 11, 2)), 8, 4, interlace=True))
    for (w, h) in [(1, 1), (2, 2), (3, 3), (5, 1), (1, 9), (9, 2)]:
        g.add_decode(f"png_adam7_{w}x{h}", make_png(synth(w, h, w * 10 + h), 8, 2, interlace=True))

    data = synth(64, 40, 8)
    z = make_png(data, 8, 2)
    comp = zlib.compress(b"", 9)
    g.add_decode("png_multi_idat", make_png(data, 8, 2, idat_sizes=[1, 1, 0, 7, 100, 3]))
    g.add_decode("png_stored_blocks", make_png(synth(200, 120, 9), 8, 2, level=0, filters=(0,)),
                 "zlib level 0: stored blocks, two of them (72 KB raw)")
    g.add_decode("png_fixed_huffman", make_png(synth(64, 40, 10), 8, 2, strategy=zlib.Z_FIXED))
    g.add_decode("png_huffman_only", make_png(synth(64, 40, 11), 8, 2, strategy=zlib.Z_HUFFMAN_ONLY))
    g.add_decode("png_rle", make_png(synth(64, 40, 12), 8, 2, strategy=zlib.Z_RLE))
    bomb = zlib.compress(b"\x00" * (10 << 20), 9)
    g.add_decode("png_zlib_bomb_1x1", make_png(np.zeros((1, 1, 1)), 8, 0, zdata=bomb),
                 "10 MiB zlib stream for a 1x1 image: only 2 bytes are inflated")
    raw_extra = zlib.compress(b"\x00\x05" + b"\x07" * 5000, 9)
    g.add_decode("png_trailing_zdata", make_png(np.zeros((1, 1, 1)), 8, 0, zdata=raw_extra))
    g.add_decode("png_private_chunks", make_png(synth(17, 11, 13), 8, 2,
                                                pre_chunks=[(b"prVt", b"hello"), (b"tEXt", b"k\0v")],
                                                post_chunks=[(b"zzZz", b"x" * 10)]))
    ex = Image.Exif()
    ex[0x0112] = 6
    g.add_decode("png_exif_o6_pillow", pil_bytes(Image.fromarray(synth(23, 13, 14)), "PNG", exif=ex.tobytes()))
    g.add_decode("png_exif_o3_after_idat", make_png(synth(23, 13, 15), 8, 2,
                                                    exif_after=tiff_orientation(3, le=False)))
    g.add_decode("png_exif_o8_before_idat", make_png(synth(23, 13, 16), 8, 2, exif=tiff_orientation(8)))
    del z, comp


def jpeg_cases(g):
    photo = synth(96, 72, 20)
    g.add_decode("jpg_444_q95", pil_bytes(Image.fromarray(photo), "JPEG", quality=95, subsampling=0))
    g.add_decode("jpg_422_q90", pil_bytes(Image.fromarray(photo), "JPEG", quality=90, subsampling=1))
    g.add_decode("jpg_420_q90", pil_bytes(Image.fromarray(photo), "JPEG", quality=90, subsampling=2))
    g.add_decode("jpg_420_q50_250x131", pil_bytes(Image.fromarray(synth(250, 131, 21)), "JPEG", quality=50,
                                                  subsampling=2))
    g.add_decode("jpg_444_q50_33x17", pil_bytes(Image.fromarray(synth(33, 17, 22)), "JPEG", quality=50,
                                                subsampling=0))
    g.add_decode("jpg_420_33x17", pil_bytes(Image.fromarray(synth(33, 17, 23)), "JPEG", quality=85,
                                            subsampling=2))
    g.add_decode("jpg_422_37x53", pil_bytes(Image.fromarray(synth(37, 53, 24)), "JPEG", quality=85,
                                            subsampling=1))
    g.add_decode("jpg_420_250x131_q95", pil_bytes(Image.fromarray(synth(250, 131, 25, noise=25)), "JPEG",
                                                  quality=95, subsampling=2))
    gray = synth(64, 48, 26)[..., 1]
    g.add_decode("jpg_gray_q90", pil_bytes(Image.fromarray(gray), "JPEG", quality=90))
    g.add_decode("jpg_gray_37x23", pil_bytes(Image.fromarray(synth(37, 23, 27)[..., 0]), "JPEG", quality=70))
    g.add_decode("jpg_prog_420_250x131", pil_bytes(Image.fromarray(synth(250, 131, 28)), "JPEG", quality=85,
                                                   subsampling=2, progressive=True))
    g.add_decode("jpg_prog_444", pil_bytes(Image.fromarray(synth(64, 48, 29)), "JPEG", quality=90,
                                           subsampling=0, progressive=True))
    g.add_decode("jpg_prog_422_37x53", pil_bytes(Image.fromarray(synth(37, 53, 30)), "JPEG", quality=75,
                                                 subsampling=1, progressive=True))
    g.add_decode("jpg_prog_gray", pil_bytes(Image.fromarray(synth(48, 40, 31)[..., 2]), "JPEG",
                                            quality=80, progressive=True))
    g.add_decode("jpg_optimize_420", pil_bytes(Image.fromarray(photo), "JPEG", quality=80, optimize=True))
    g.add_decode("jpg_restart_blocks3", pil_bytes(Image.fromarray(photo), "JPEG", quality=85,
                                                  restart_marker_blocks=3))
    g.add_decode("jpg_restart_rows1_422", pil_bytes(Image.fromarray(synth(250, 131, 32)), "JPEG", quality=90,
                                                    subsampling=1, restart_marker_rows=1))
    g.add_decode("jpg_prog_restart", pil_bytes(Image.fromarray(photo), "JPEG", quality=85, progressive=True,
                                               restart_marker_blocks=2))
    for (w, h) in [(1, 1), (2, 3), (3, 7), (4, 4), (5, 5), (17, 2)]:
        g.add_decode(f"jpg_420_{w}x{h}", pil_bytes(Image.fromarray(synth(w, h, 40 + w + h)), "JPEG",
                                                   quality=90, subsampling=2))
    g.add_decode("jpg_422_4x4", pil_bytes(Image.fromarray(synth(4, 4, 50)), "JPEG", quality=90, subsampling=1))
    g.add_decode("jpg_422_5x3", pil_bytes(Image.fromarray(synth(5, 3, 51)), "JPEG", quality=90, subsampling=1))

    small = Image.fromarray(synth(33, 17, 60))
    for o in range(1, 9):
        ex = Image.Exif()
        ex[0x0112] = o
        g.add_decode(f"jpg_exif_o{o}", pil_bytes(small, "JPEG", quality=90, exif=ex.tobytes()))
    base = pil_bytes(small, "JPEG", quality=90)
    xmp = (b"http://ns.adobe.com/xap/1.0/\x00<x:xmpmeta><rdf:Description "
           b"tiff:Orientation=\"6\"/></x:xmpmeta>")
    g.add_decode("jpg_xmp_o6", insert_after_soi(base, [app1(xmp)]))
    xmp_el = b"http://ns.adobe.com/xap/1.0/\x00<tiff:Orientation>7</tiff:Orientation>"
    g.add_decode("jpg_xmp_element_o7", insert_after_soi(base, [app1(xmp_el)]))
    tiff = tiff_orientation(3, pad=24)
    g.add_decode("jpg_exif_two_segments_o3",
                 insert_after_soi(base, [app1(b"Exif\0\0" + tiff[:20]), app1(b"Exif\0\0" + tiff[20:])]),
                 "Pillow concatenates Exif APP1 payloads")
    g.add_decode("jpg_exif_byte_type_blocks_xmp",
                 insert_after_soi(base, [app1(b"Exif\0\0" + tiff_orientation(6, typ=1)), app1(xmp)]),
                 "BYTE-typed Orientation is present but never equals an int: no transpose, no XMP fallback")
    g.add_decode("jpg_exif_bigendian_o5", insert_after_soi(base, [app1(b"Exif\0\0" + tiff_orientation(5, le=False))]))

    # Truncated scan + EOI: libjpeg zero-fills ("premature end of data segment").
    b = pil_bytes(Image.fromarray(photo), "JPEG", quality=90)
    sos = b.index(b"\xff\xda")
    cut = sos + (len(b) - sos) * 3 // 5
    g.add_decode("jpg_truncated_plus_eoi", b[:cut] + b"\xff\xd9")
    b = pil_bytes(Image.fromarray(photo), "JPEG", quality=90, restart_marker_blocks=4)
    sos = b.index(b"\xff\xda")
    cut = sos + (len(b) - sos) // 2
    g.add_decode("jpg_restart_truncated_plus_eoi", b[:cut] + b"\xff\xd9")

    # Hand-encoded layouts Pillow cannot write.
    g.add_decode("jpg_enc_411_67x29", make_jpeg(synth(67, 29, 70), sampling=((4, 1), (1, 1), (1, 1))))
    g.add_decode("jpg_enc_440_40x33", make_jpeg(synth(40, 33, 71), sampling=((1, 2), (1, 1), (1, 1))))
    g.add_decode("jpg_enc_mixed_45x37", make_jpeg(synth(45, 37, 72), sampling=((2, 2), (2, 1), (1, 2))),
                 "Cb h1v2 fancy, Cr h2v1 fancy")
    g.add_decode("jpg_enc_luma_small_30x22", make_jpeg(synth(30, 22, 73), sampling=((1, 1), (2, 2), (2, 2))),
                 "luma upsampled h2v2 from chroma-sized grid")
    g.add_decode("jpg_enc_31_50x20", make_jpeg(synth(50, 20, 74), sampling=((3, 1), (1, 1), (1, 1))))
    g.add_decode("jpg_enc_42_41x19", make_jpeg(synth(41, 19, 75), sampling=((4, 2), (1, 1), (1, 1))))
    g.add_decode("jpg_enc_420_tiny_4x9", make_jpeg(synth(4, 9, 76), sampling=((2, 2), (1, 1), (1, 1))),
                 "chroma width 2: box h2v2")
    g.add_decode("jpg_enc_adobe_rgb", make_jpeg(synth(40, 24, 77), sampling=((1, 1),) * 3, color="rgb",
                                                jfif=False, adobe=0, comp_ids=[1, 2, 3]))
    g.add_decode("jpg_enc_ids_rgb", make_jpeg(synth(40, 24, 78), sampling=((1, 1),) * 3, color="rgb",
                                              jfif=False, comp_ids=[82, 71, 66]))
    g.add_decode("jpg_enc_ids_other", make_jpeg(synth(40, 24, 79), jfif=False, comp_ids=[5, 6, 7]))
    g.add_decode("jpg_enc_adobe_ycc", make_jpeg(synth(40, 24, 80), jfif=False, adobe=1))
    g.add_decode("jpg_enc_no_dht", make_jpeg(synth(48, 32, 81), dht=False), "libjpeg-turbo default tables")
    g.add_decode("jpg_enc_separate_scans", make_jpeg(synth(45, 37, 82), separate_scans=True, restart=5),
                 "non-interleaved sequential: one scan per component")
    g.add_decode("jpg_enc_restart2_fill", make_jpeg(synth(48, 40, 83), restart=2, fill_ff=True),
                 "0xFF fill bytes before markers")
    g.add_decode("jpg_enc_gray_h2v2", make_jpeg(synth(37, 21, 84)[..., 0], sampling=((2, 2),), color="gray"))
    q16 = [min(8 + 7 * k, 400) for k in range(64)]
    g.add_decode("jpg_enc_q16", make_jpeg(synth(40, 24, 85), q16=True, extra_q=q16), "16-bit DQT entries up to 400")
    g.add_decode("jpg_enc_extreme_dqt", make_jpeg(synth(40, 24, 86, noise=40), quality=100, q16=True,
                                                  dqt_override=[3000 + 97 * k for k in range(64)]),
                 "coefficients quantized with q=1 but DQT says ~3000: dequantized values overflow 16 bits "
                 "(libjpeg-turbo AVX2 IDCT wraps/saturates)")

    # Progressive files cut at / inside a scan and closed with EOI: libjpeg-turbo
    # applies inter-block smoothing (jdcoefct.c decompress_smooth_data).
    for (q, sub) in [(75, 0), (90, 2)]:
        b = pil_bytes(Image.fromarray(synth(64, 48, 87 + sub)), "JPEG", quality=q, subsampling=sub,
                      progressive=True)
        sos = [i for i in range(len(b) - 1) if b[i] == 0xFF and b[i + 1] == 0xDA]
        for k in (1, 2, 4, 6, 9):
            g.add_decode(f"jpg_prog{q}_{sub}_cut_at_scan{k + 1}", b[:sos[k]] + b"\xff\xd9")
        for k in (1, 4, 8):
            g.add_decode(f"jpg_prog{q}_{sub}_cut_in_scan{k}", b[:(sos[k - 1] + sos[k]) // 2] + b"\xff\xd9")


def error_cases(g):
    base = pil_bytes(Image.fromarray(synth(40, 24, 90)), "JPEG", quality=85)
    sof = base.index(b"\xff\xc0")
    twelve = bytearray(base)
    twelve[sof + 4] = 12
    g.add_error("err_jpeg_12bit", bytes(twelve), "12-bit JPEG is not supported")
    for marker, needle, name in [(0xC3, "lossless", "err_jpeg_lossless_sof3"),
                                 (0xC9, "arithmetic", "err_jpeg_arith_sof9"),
                                 (0xCA, "arithmetic", "err_jpeg_arith_sof10"),
                                 (0xC5, "hierarchical", "err_jpeg_hier_sof5")]:
        b = bytearray(base)
        b[sof + 1] = marker
        g.add_error(name, bytes(b), needle)
    g.add_error("err_jpeg_cmyk", pil_bytes(Image.new("CMYK", (16, 16), (10, 20, 30, 40)), "JPEG"), "CMYK")
    g.add_error("err_jpeg_truncated", base[: len(base) * 2 // 3], "truncated")
    g.add_error("err_jpeg_no_eoi", base[:-2], "truncated")
    huge = bytearray(base)
    huge[sof + 5:sof + 9] = struct.pack(">HH", 65535, 65535)
    g.add_error("err_jpeg_too_large", bytes(huge), "too large")
    g.add_error("err_jpeg_no_frame", b"\xff\xd8\xff\xd9", "no image data")
    dup = base[:-2] + base[base.index(b"\xff\xda"):]
    g.add_error("err_jpeg_repeated_scan", dup, "extra scan")
    wide = bytearray(base)
    wide[sof + 5:sof + 9] = struct.pack(">HH", 8, 65501)
    g.add_error("err_jpeg_side_over_65500", bytes(wide), "65500")
    prog = pil_bytes(Image.fromarray(synth(40, 24, 93)), "JPEG", quality=85, progressive=True)
    out, i = bytearray(prog[:2]), 2
    while prog[i + 1] != 0xDA:  # copy header segments except DHT
        ln = struct.unpack(">H", prog[i + 2:i + 4])[0]
        if prog[i + 1] != 0xC4:
            out += prog[i:i + 2 + ln]
        i += 2 + ln
    rest = prog[i:]
    while True:  # drop DHT segments that sit between scans
        j = rest.find(b"\xff\xc4")
        if j < 0:
            break
        ln = struct.unpack(">H", rest[j + 2:j + 4])[0]
        rest = rest[:j] + rest[j + 2 + ln:]
    g.add_error("err_jpeg_progressive_no_dht", bytes(out + rest), "Huffman table")

    png = make_png(synth(16, 8, 91), 8, 2)
    g.add_error("err_png_truncated", png[: len(png) // 2], "truncated")
    ihdr = bytearray(png)
    ihdr[16:24] = struct.pack(">II", 100000, 100000)
    g.add_error("err_png_too_large", bytes(ihdr), "too large")
    g.add_error("err_png_bad_filter", make_png(synth(4, 4, 92), 8, 2, filters=(5,)), "filter")
    g.add_error("err_png_bad_depth", make_png(np.zeros((2, 2, 3)), 4, 2), "unsupported PNG bit depth")
    bad_z = bytearray(png)
    idat = bad_z.index(b"IDAT")
    bad_z[idat + 6] ^= 0xFF
    g.add_error("err_png_corrupt_zlib", bytes(bad_z), "PNG image data")
    g.add_error("err_png_trns_too_long", make_png(np.zeros((2, 2, 1)), 8, 3, plte=b"\x00" * 30,
                                                  trns=b"\x07" * 300), "tRNS")
    g.add_error("err_not_an_image", b"hello, world", "unsupported image format", ext="dat")
    g.add_error("err_gif", b"GIF89a\x01\x00\x01\x00\x00\x00\x00;", "GIF", ext="gif")


def resize_cases(g):
    srcs = {
        "src_64x48": synth(64, 48, 100),
        "src_1x1": np.array([[[200, 10, 90]]], np.uint8),
        "src_1x17": synth(1, 17, 101),
        "src_23x1": synth(23, 1, 102),
        "src_5x3": synth(5, 3, 103, noise=60),
        "src_97x61": synth(97, 61, 104, noise=40),
    }
    for k, v in srcs.items():
        g.write(f"resize/{k}.rgb", v.tobytes())
    targets = {
        "src_64x48": [(97, 61), (31, 23), (7, 5), (1, 1), (128, 96), (64, 96), (200, 48), (63, 47), (33, 100),
                      (256, 1), (13, 11), (64, 48)],
        "src_1x1": [(3, 2), (32, 32)],
        "src_1x17": [(5, 40), (1, 3)],
        "src_23x1": [(7, 1), (50, 4)],
        "src_5x3": [(64, 32), (2, 2)],
        "src_97x61": [(64, 32), (96, 64), (192, 128), (40, 61)],
    }
    for k, lst in targets.items():
        src = srcs[k]
        sh, sw = src.shape[:2]
        for (dw, dh) in lst:
            out = np.asarray(Image.fromarray(src).resize((dw, dh), resample=Image.BICUBIC))
            rel = f"resize/{k}_to_{dw}x{dh}.rgb"
            g.write(rel, out.tobytes())
            g.resize.append(f"resize/{k}.rgb {sw} {sh} {dw} {dh} {rel}")


def smart_resize_cases():
    rng = np.random.RandomState(5)
    pairs = set()
    for h in range(1, 80, 3):
        for w in (1, 2, 15, 16, 17, 31, 32, 33, 48, 80, 112, 200):
            pairs.add((h, w))
    for k in range(0, 60):  # exact .5 ties: 32k + 16
        pairs.add((32 * k + 16, 32 * (k % 7) + 16))
        pairs.add((32 * k + 16, 640))
    pairs |= {(1, 1), (1, 200), (200, 1), (1, 201), (2, 401), (2, 400), (17, 3400), (17, 3401), (3400, 17),
              (480, 640), (1080, 1920), (2160, 3840), (4320, 7680), (3000, 4000), (4000, 3000), (3584, 3584),
              (3585, 3585), (100000, 100000), (65535, 65535), (4294967295, 4294967295),
              (4294967295, 21474837), (21474836, 4294967295), (12845056, 1), (3136, 1), (56, 56), (55, 57),
              (0, 5), (5, 0), (0, 0), (3968, 3232), (3900, 3400), (375, 500), (30, 20), (28, 28), (27, 116)}
    for _ in range(1500):
        h = int(math.exp(rng.uniform(0, math.log(60000))))
        w = int(math.exp(rng.uniform(0, math.log(60000))))
        pairs.add((max(h, 1), max(w, 1)))
    lines = []
    for (h, w) in sorted(pairs):
        try:
            if min(h, w) == 0:
                raise ZeroDivisionError
            rh, rw = smart_resize(h, w)
            lines.append(f"{h} {w} {rh} {rw} {(rh // 32) * (rw // 32)}")
        except (ValueError, ZeroDivisionError):
            lines.append(f"{h} {w} ERR")
    return lines


def pv_case(g, name, rgb, lines, full=False, input_note=None):
    flat, (gt, gh, gw), resized = hf_pixel_values(rgb)
    n = flat.shape[0]
    rows = list(range(n)) if full else sorted({0, 1, 2, 3, n // 3, n // 2, n - 4, n - 3, n - 2, n - 1})
    g.write(f"pixel_values/{name}.rows.f32", flat[rows].tobytes())
    sha_pv = hashlib.sha256(flat.tobytes()).hexdigest()
    sha_rs = hashlib.sha256(resized.tobytes()).hexdigest()
    h, w = rgb.shape[:2]
    lines.append(f"{name} {input_note} {w} {h} {gt} {gh} {gw} {resized.shape[1]} {resized.shape[0]} "
                 f"{sha_rs} {sha_pv} {','.join(map(str, rows))}")


def pixel_value_cases(g):
    lines = []
    small = synth(20, 30, 200)
    g.write("pixel_values/min_20x30.rgb", small.tobytes())
    pv_case(g, "min_20x30", small, lines, full=True, input_note="rgb:pixel_values/min_20x30.rgb")
    tiny = synth(3, 5, 201)  # hits min_pixels from a very small source
    g.write("pixel_values/tiny_5x3.rgb", tiny.tobytes())
    pv_case(g, "tiny_5x3", tiny, lines, full=True, input_note="rgb:pixel_values/tiny_5x3.rgb")
    for (w, h, q, seed) in [(640, 480, 90, 202), (500, 375, 85, 203)]:
        jpg = pil_bytes(Image.fromarray(synth(w, h, seed)), "JPEG", quality=q)
        rel = f"pixel_values/photo_{w}x{h}.jpg"
        g.write(rel, jpg)
        rgb, _ = hf_decode(jpg)
        pv_case(g, f"photo_{w}x{h}", rgb, lines, input_note=f"jpeg:{rel}")
    rgba = np.concatenate([synth(97, 61, 204), np.random.RandomState(3).randint(0, 256, (61, 97, 1))],
                          -1).astype(np.uint8)
    png = pil_bytes(Image.fromarray(rgba), "PNG")
    g.write("pixel_values/rgba_97x61.png", png)
    rgb, _ = hf_decode(png)
    pv_case(g, "rgba_97x61", rgb, lines, input_note="png:pixel_values/rgba_97x61.png")
    big = procedural(3900, 3400)
    pv_case(g, "procedural_3900x3400", big, lines, input_note="procedural:3900x3400")
    return lines


def main():
    g = Goldens()
    png_cases(g)
    jpeg_cases(g)
    error_cases(g)
    resize_cases(g)
    sr = smart_resize_cases()
    pv = pixel_value_cases(g)
    hdr = (f"# generated by tests/gen_goldens.py with Python {sys.version.split()[0]}, Pillow {PIL.__version__} "
           f"(libjpeg-turbo {features.version_feature('libjpeg_turbo')}, zlib {features.version('zlib')}), "
           f"numpy {np.__version__}\n")
    g.write("decode.txt", (hdr + "# input width height expected.rgb pillow-mode [# note]\n"
                           + "\n".join(g.decode) + "\n").encode())
    g.write("errors.txt", (hdr + "# input pillow-behaviour(HF load path) | substring our error must contain\n"
                           + "\n".join(g.errors) + "\n").encode())
    g.write("resize.txt", (hdr + "# src.rgb src_w src_h dst_w dst_h expected.rgb (Pillow BICUBIC)\n"
                           + "\n".join(g.resize) + "\n").encode())
    g.write("smart_resize.txt", (hdr + "# height width -> resized_h resized_w merged_tokens | ERR\n"
                                 + "\n".join(sr) + "\n").encode())
    g.write("pixel_values.txt", (hdr + "# name input width height grid_t grid_h grid_w resized_w resized_h "
                                 "sha256(resized u8) sha256(pixel_values f32 LE) stored_rows\n"
                                 + "\n".join(pv) + "\n").encode())
    total = 0
    for root, _, files in os.walk(OUT):
        total += sum(os.path.getsize(os.path.join(root, f)) for f in files)
    print(f"wrote {len(g.decode)} decode, {len(g.errors)} error, {len(g.resize)} resize, {len(sr)} smart_resize, "
          f"{len(pv)} pixel_values goldens; {total / 1e6:.2f} MB in {OUT}")


if __name__ == "__main__":
    main()
