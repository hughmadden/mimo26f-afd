"""Selftests for the L5 ladder runner (harness/l5_ladder.py), before it is pointed at the engine.

A scripted fake model on a loopback server answers every ladder prompt correctly, or wrongly on one case per row. The
runner must pass the good server, fail exactly the rows the bad server breaks, retain failed rows without retrying, and
score the three-needle prompt by label. Offline; the needle sizes are shrunk with the runner's hidden flags.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import subprocess
import sys

import pytest

HARNESS = pathlib.Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FT = _load("fleet_bench_fake", HARNESS / "selftests" / "test_fleet_bench_tools.py")  # Fake, send_json, send_sse
L5 = _load("l5_ladder_under_test", HARNESS / "l5_ladder.py")


def answer(prompt, bad):
    """What a correct model says; `bad` breaks one case per row."""
    if prompt.startswith("Reply exactly APPLE"):
        return "APPLES" if bad else "APPLE"
    if prompt.startswith("Calculate 17 times 23"):
        return "391"
    if prompt.startswith("Reply with exactly this JSON"):
        return '{"ok":true,"n":3}'
    m = re.match(r"Count (down )?from (\d+) to (\d+)", prompt)
    if m:
        a, b = int(m.group(2)), int(m.group(3))
        seq = list(range(a, b - 1, -1)) if m.group(1) else list(range(a, b + 1))
        if bad and a == 1 and b == 50:
            seq.remove(37)  # a dropped number is a COUNT failure
        return ", ".join(map(str, seq))
    m = re.match(r"List the even numbers from (\d+) to (\d+)", prompt)
    if m:
        return ", ".join(map(str, range(int(m.group(1)), int(m.group(2)) + 1, 2)))
    m = re.search(r"The secret vault code is (\d{6})", prompt)
    if m:
        shallow = prompt.index(m.group(0)) < len(prompt) * 0.3  # the depth-0.1 needle
        return "000000" if bad and shallow else m.group(1)
    found = dict(re.findall(r"The vault code for (\w+) is (\d{6})", prompt))
    if found:
        if bad:
            found["Gamma"] = "111111"
        return "\n".join(f"{k}: {v}" for k, v in found.items())
    return "A refrigerator moves heat out of the box using a refrigerant that evaporates and condenses."


def reply_factory(bad):
    def reply(fake, h, body):
        text = answer(body["messages"][0]["content"], bad)
        usage = {"prompt_tokens": len(body["messages"][0]["content"]) // 4, "completion_tokens": 12, "total_tokens": 0}
        if body.get("stream"):
            return FT.send_sse(h, [{"choices": [{"index": 0, "delta": {"content": text}}]},
                                   {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                                   {"choices": [], "usage": usage}])
        FT.send_json(h, {"choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                      "finish_reason": "stop"}], "usage": usage})
    return reply


class ModelsFake(FT.Fake):
    """The fake plus GET /v1/models (A8 readiness)."""

    def __init__(self, reply):
        super().__init__(reply)
        get_metrics = self.srv.RequestHandlerClass.do_GET

        def do_GET(h):
            if h.path == "/v1/models":
                return FT.send_json(h, {"object": "list", "data": [{"id": L5.MODEL, "object": "model"}]})
            return get_metrics(h)
        self.srv.RequestHandlerClass.do_GET = do_GET


def run_ladder(tmp_path, bad, cell="ladder"):
    with ModelsFake(reply_factory(bad)) as fake:
        p = subprocess.run([sys.executable, str(HARNESS / "l5_ladder.py"), "--base", fake.url + "/v1", "--out", str(tmp_path),
                            "--cell", cell, "--needle-targets", "400,800", "--needle128k-target", "1200"],
                           capture_output=True, text=True, timeout=120)
        n_requests = len(fake.bodies)
    rec = json.loads((tmp_path / f"l5-{cell}.json").read_text())
    return p, rec, n_requests


def test_good_model_passes_every_row(tmp_path):
    p, rec, n = run_ladder(tmp_path, bad=False)
    assert p.returncode == 0, p.stdout + p.stderr
    assert [r["row"] for r in rec["rows"]] == ["G1", "G2", "G3", "G4", "G4c", "G5"]
    assert all(r["pass"] for r in rec["rows"]) and rec["pass"] and rec["within_budget"]
    assert n == 3 + 5 + 6 + 1 + 4  # G2 + G3 + mimo_needle (2 sizes x 3 depths) + G4c + G5 (warm-free C1 + C3)
    assert "RESULT: PASS L5 ladder" in p.stdout
    md = (tmp_path / "l5-ladder.md").read_text()
    assert md.startswith("# L5 ladder: PASS") and "| G4c | PASS |" in md


def test_bad_model_fails_exactly_the_broken_rows_and_retains_them(tmp_path):
    p, rec, n = run_ladder(tmp_path, bad=True)
    assert p.returncode == 1
    verdict = {r["row"]: r["pass"] for r in rec["rows"]}
    assert verdict == {"G1": True, "G2": False, "G3": False, "G4": False, "G4c": False, "G5": True}
    g2 = next(r for r in rec["rows"] if r["row"] == "G2")
    assert g2["note"] == "2/3" and [c["pass_"] for c in g2["cases"]] == [False, True, True]
    g3 = next(r for r in rec["rows"] if r["row"] == "G3")
    assert g3["note"] == "4/5"
    g4 = next(r for r in rec["rows"] if r["row"] == "G4")
    assert g4["note"].startswith("4/6 ") and sum(" FAIL " in line for line in g4["lines"]) == 2
    assert n == 3 + 5 + 6 + 1 + 4  # failed rows were not retried
    assert "RESULT: FAIL L5 ladder" in p.stdout


def test_needle128k_cell_is_one_request_with_three_labelled_needles(tmp_path):
    p, rec, n = run_ladder(tmp_path, bad=False, cell="needle128k")
    assert p.returncode == 0 and n == 1
    (r,) = rec["rows"]
    assert r["row"] == "G4-128K" and r["pass"] and r["got"] == r["codes"]
    p, rec, n = run_ladder(tmp_path, bad=True, cell="needle128k")
    assert p.returncode == 1 and rec["rows"][0]["got"]["Gamma"] == "111111" and n == 1


def test_needle3_prompt_is_deterministic_and_places_needles_by_depth():
    a, codes = L5.needle3_prompt(8000)
    b, codes_b = L5.needle3_prompt(8000)
    assert a == b and codes == codes_b and len(set(codes.values())) == 3
    for lab, depth in zip(L5.NEEDLE_LABELS, L5.NEEDLE_DEPTHS):
        assert a.count(f"The vault code for {lab} is") == 1
        assert a.index(f"The vault code for {lab} is") / len(a) == pytest.approx(depth, abs=0.03)
    got, ok = L5.score_needle3("Alpha: {Alpha}\nBeta: {Beta}\nGamma: {Gamma}".format(**codes), codes)
    assert ok and got == codes
    assert not L5.score_needle3("Alpha: {Alpha}\nBeta: {Beta}".format(**codes), codes)[1]  # a missing line fails


def test_partial_rows_run_only_those_and_never_pass(tmp_path):
    with ModelsFake(reply_factory(False)) as fake:
        p = subprocess.run([sys.executable, str(HARNESS / "l5_ladder.py"), "--base", fake.url + "/v1", "--out", str(tmp_path),
                            "--rows", "G1,G2", "--needle-targets", "400,800"], capture_output=True, text=True, timeout=120)
        n = len(fake.bodies)
    rec = json.loads((tmp_path / "l5-ladder.json").read_text())
    assert [r["row"] for r in rec["rows"]] == ["G1", "G2"] and all(r["pass"] for r in rec["rows"])
    assert rec["partial"] and not rec["pass"] and p.returncode == 1 and n == 3  # G2's 3 requests only
    assert (tmp_path / "l5-ladder.md").read_text().startswith("# L5 ladder: PARTIAL (rows G1,G2)")


def test_needle3_cell_labels_its_target(tmp_path):
    with ModelsFake(reply_factory(False)) as fake:
        p = subprocess.run([sys.executable, str(HARNESS / "l5_ladder.py"), "--base", fake.url + "/v1", "--out", str(tmp_path),
                            "--cell", "needle3", "--needle3-target", "1500"], capture_output=True, text=True, timeout=120)
    rec = json.loads((tmp_path / "l5-needle3.json").read_text())
    assert p.returncode == 0 and [r["row"] for r in rec["rows"]] == ["G4-2K"] and rec["rows"][0]["pass"]
