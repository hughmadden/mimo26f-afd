#!/usr/bin/env python3
"""Image input check (perf reset V2) against a live endpoint.

Images are generated with Pillow (system python3) and sent as data URLs:
1. colours: a 224x224 image in four coloured quadrants; the answer must name red, green, blue and
   yellow;
2. text: a 640x480 image reading "MIMO VISION 42"; the answer must contain 42 and VISION;
3. repeat: the same text request again; it must return the identical answer (prefix cache; the
   server log shows `device hit`);
4. swap: the same size with different text ("MIMO VISION 77"): the answer must contain 77 and
   not 42 (a different image never reuses the first one's cached tokens);
5. stream: the colours request streamed; content only, no markup;
6. refusals: a remote image URL gets a 400.

usage: l5_vision.py [--base http://coordinator:8100/v1] [--model mimo-v2.6-flash] [--key-file F] [--out DIR]
(`--key-file`: a file holding an `Authorization: Bearer ...` header line, for LiteLLM.)
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import pathlib
import time
import urllib.error
import urllib.request

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def data_url(img):
    b = io.BytesIO()
    img.save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def quad():
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (224, 224))
    d = ImageDraw.Draw(im)
    for box, c in (([0, 0, 111, 111], (220, 30, 30)), ([112, 0, 223, 111], (30, 200, 40)),
                   ([0, 112, 111, 223], (30, 60, 220)), ([112, 112, 223, 223], (240, 220, 30))):
        d.rectangle(box, fill=c)
    return im


def text_image(s):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGB", (640, 480), (250, 250, 250))
    d = ImageDraw.Draw(im)
    d.text((40, 180), s, fill=(10, 10, 10), font=ImageFont.truetype(FONT, 64))
    return im


def post(base, body, headers, stream=False, timeout=600):
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **headers})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if not stream:
                d = json.loads(r.read())
                return 200, d["choices"][0]["message"].get("content") or "", d.get("usage"), time.time() - t0
            text, ttft = "", None
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                for ch in json.loads(line[5:]).get("choices", []):
                    c = (ch.get("delta") or {}).get("content")
                    if c:
                        ttft = ttft or time.time() - t0
                        text += c
            return 200, text, None, ttft
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace"), None, time.time() - t0


def ask(base, model, headers, url, question, stream=False):
    body = {"model": model, "temperature": 0, "max_tokens": 64, "stream": stream,
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                                      {"type": "text", "text": question}]}]}
    return post(base, body, headers, stream)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://coordinator:8100/v1")
    ap.add_argument("--model", default="mimo-v2.6-flash")
    ap.add_argument("--key-file", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    headers = {}
    if a.key_file:
        k, v = pathlib.Path(a.key_file).read_text().strip().split(":", 1)
        headers[k.strip()] = v.strip()
    rows = {}

    def check(name, r, ok, note=""):
        status, text, usage, secs = r
        rows[name] = {"status": status, "text": text, "usage": usage, "seconds": secs, "pass": ok}
        print(f"  {name:8} {status} {secs:6.2f} s  {'PASS' if ok else 'FAIL'}  {text[:90]!r} {note}", flush=True)

    colours = "What are the colours of the four quadrants? Answer with four colour names."
    r = ask(a.base, a.model, headers, data_url(quad()), colours)
    t = r[1].lower()
    check("colours", r, r[0] == 200 and all(c in t for c in ("red", "green", "blue", "yellow")))
    read = "What text is written in the image? Reply with the text only."
    r1 = ask(a.base, a.model, headers, data_url(text_image("MIMO VISION 42")), read)
    check("text", r1, r1[0] == 200 and "42" in r1[1] and "VISION" in r1[1].upper(),
          f"prompt tokens {r1[2] and r1[2].get('prompt_tokens')}")
    r2 = ask(a.base, a.model, headers, data_url(text_image("MIMO VISION 42")), read)
    check("repeat", r2, r2[0] == 200 and r2[1] == r1[1])
    r3 = ask(a.base, a.model, headers, data_url(text_image("MIMO VISION 77")), read)
    check("swap", r3, r3[0] == 200 and "77" in r3[1] and "42" not in r3[1])
    r4 = ask(a.base, a.model, headers, data_url(quad()), colours, stream=True)
    t = r4[1].lower()
    check("stream", r4, r4[0] == 200 and all(c in t for c in ("red", "green", "blue", "yellow")) and "<" not in r4[1])
    r5 = ask(a.base, a.model, headers, "https://example.com/a.png", read)
    check("remote", r5, r5[0] == 400)
    ok = all(v["pass"] for v in rows.values())
    print(f"RESULT: {'PASS' if ok else 'FAIL'} L5 vision ({sum(v['pass'] for v in rows.values())}/{len(rows)})", flush=True)
    if a.out:
        pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
        (pathlib.Path(a.out) / "l5-vision.json").write_text(json.dumps(rows, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
