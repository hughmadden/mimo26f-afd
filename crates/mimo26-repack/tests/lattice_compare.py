"""Fail-closed stage-artifact comparator; execution provenance is a separate gate."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from lattice_oracle import Q, activation, reduce_routes

EXPERTS = [0, 7, 255, 1, 2, 3, 4, 5]
ATOL = RTOL = 1e-5
COMPUTE_TERM = 2.4e-6


def schema(m, reference):
    result = {
        "x_payload": ("b1-x-payload.u8", "|u1", (m, 4096)),
        "x_scales": ("b1-x-scales.u8", "|u1", (m, 128)),
        "weights": ("b1-weights.f32", "<f4", (m, 8)),
        "mid_payload": ("b1-mid_payload.u8", "|u1", (4, m, 8, 512)),
        "mid_scales": ("b1-mid_scales.u8", "|u1", (4, m, 8, 16)),
    }
    for name in ("gate", "up", "h"):
        result[name] = (f"b1-{name}.f32", "<f4", (4, m, 8, 512))
    for name, filename in (("rank", "rank"), ("decoded", "return-decoded"), ("wire", "wire")):
        result[name] = (f"b1-{filename}.f32", "<f4", (m, 4096) if name == "wire" else (4, m, 4096))
    suffix = "f64" if reference else "f32"
    result["partial"] = (f"b1-partial.{suffix}", "<f8" if reference else "<f4", (4, m, 8, 4096))
    if reference:
        result["full"] = ("b1-full.f64", "<f8", (m, 4096))
    return result


def load(root, reference):
    root = Path(root).resolve()
    manifest = json.loads((root / "b1-source.json").read_text())
    if (manifest.get("lattice"), manifest.get("quantizer"), manifest.get("layer"), manifest.get("experts")) != (
            "E-W4A8-v1", Q.VERSION, 1, EXPERTS):
        raise ValueError("wrong lattice/codec/layer/route identity")
    artifacts = manifest["artifacts"]
    m = artifacts["b1-x-payload.u8"]["shape"][0]
    if type(m) is not int or m not in range(1, 9):
        raise ValueError("unsupported token count")
    arrays = {}
    for key, (name, dtype, shape) in schema(m, reference).items():
        meta = artifacts[name]
        if meta["dtype"] != dtype or meta["shape"] != list(shape):
            raise ValueError(f"{name}: wrong dtype/shape")
        path = root / name
        if path.is_symlink() or path.resolve().parent != root:
            raise ValueError(f"{name}: non-local artifact")
        expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
        if path.stat().st_size != expected:
            raise ValueError(f"{name}: wrong byte length")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != meta["sha256"]:
            raise ValueError(f"{name}: digest mismatch")
        arrays[key] = np.frombuffer(data, dtype=dtype).reshape(shape)
        if not np.isfinite(arrays[key]).all():
            raise Q.NumericalFault(f"{name}: nonfinite values")
    # Exact source identities are compared between bundles, not inferred from a
    # filename or accepted as evidence of an actual GPU execution.
    blocks = manifest["blocks"]
    if not isinstance(blocks, list) or len(blocks) != 24:
        raise ValueError("expected 24 projection source identities")
    names = [f"model.layers.1.mlp.experts.{e}.{p}_proj" for e in EXPERTS for p in ("gate", "up", "down")]
    for block, name in zip(blocks, names):
        if block["name"] != name:
            raise ValueError("source projection order/identity")
        for key in ("weight_sha256", "scale_sha256"):
            digest = block[key]
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("invalid source digest")
    return arrays, blocks


def prefix(reference, m):
    if reference["x_payload"].shape[0] < m:
        raise ValueError("candidate exceeds reference token count")
    return {key: value[:m] if key in ("x_payload", "x_scales", "weights", "wire", "full") else value[:, :m]
            for key, value in reference.items()}


def compare(ref, got):
    rows = {}
    def exact(name, a, b):
        # Bit comparison retains signed zero at the ordered reduction boundaries.
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"{name}: array contract mismatch")
        count = int(np.count_nonzero(a.view(np.uint8).reshape(-1) != b.view(np.uint8).reshape(-1)))
        rows[name] = {"bad_bytes": count, "pass": count == 0}
    def numeric(name, a, b, bound=None):
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise Q.NumericalFault(f"{name}: shape/nonfinite numerical boundary")
        delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
        limit = ATOL + RTOL * np.abs(b.astype(np.float64)) if bound is None else bound
        bad = int(np.count_nonzero(delta > limit))
        rows[name] = {"bad": bad, "max_abs": float(delta.max()), "pass": bad == 0}
    for key in ("x_payload", "x_scales", "weights"):
        exact("input_" + key, got[key], ref[key])
    Q.decode_blocks(got["x_payload"], got["x_scales"])
    for key in ("gate", "up", "h", "partial", "rank"):
        numeric(key + "_vs_ideal", got[key], ref[key])
    numeric("activation_from_reported_fc1", got["h"], activation(got["gate"], got["up"]))
    payload, scales = Q.encode_blocks(got["h"])
    exact("codec_payload_same_fp32", got["mid_payload"], payload)
    exact("codec_scales_same_fp32", got["mid_scales"], scales)
    Q.decode_blocks(got["mid_payload"], got["mid_scales"])
    rank, _, _ = reduce_routes(got["partial"], got["weights"])
    exact("ordered_weight_once_rank", got["rank"], rank)
    # Diagnose each boundary using its reported input, not a combined error.
    exact("bf16_rank_return", got["decoded"], Q.bf16_rne(got["rank"]))
    ordered_wire = np.zeros_like(got["wire"])
    for r in range(4):
        ordered_wire = (ordered_wire + got["decoded"][r]).astype(np.float32)
    exact("ordered_rank_sum", got["wire"], ordered_wire)
    numeric("compute_presum_vs_full", got["rank"].astype(np.float64).sum(axis=0), ref["full"], COMPUTE_TERM)
    bound = COMPUTE_TERM + np.abs(got["rank"].astype(np.float64)).sum(axis=0) * 2**-8
    bound += 3 * 2**-24 * np.abs(got["decoded"].astype(np.float64)).sum(axis=0)
    numeric("wire_r8_vs_full", got["wire"], ref["full"], bound)
    crossings = {key: int(np.count_nonzero(got[key] != ref[key])) for key in ("mid_payload", "mid_scales")}
    return {"pass": all(row["pass"] for row in rows.values()), "rows": rows,
            "ideal_intermediate_byte_differences": crossings,
            "atol": ATOL, "rtol": RTOL, "compute_term": COMPUTE_TERM,
            "scope": "artifact comparison only; execution provenance separate; no bound widening for bin crossings"}


def run(reference, candidate):
    ref, rb = load(reference, True)
    got, gb = load(candidate, False)
    if rb != gb:
        raise ValueError("checkpoint source digest mismatch")
    return compare(prefix(ref, got["x_payload"].shape[0]), got)


def seal_dump(reference, candidate, m, source_revision):
    m = int(m)
    ref, blocks = load(reference, True)
    if m not in range(1, 9) or m > ref["x_payload"].shape[0]:
        raise ValueError("dump token count")
    root = Path(candidate)
    artifacts = {}
    for _, (name, dtype, shape) in schema(m, False).items():
        path = root / name
        if path.is_symlink() or path.stat().st_size != int(np.prod(shape)) * np.dtype(dtype).itemsize:
            raise ValueError("dump extent/path")
        artifacts[name] = dict(shape=list(shape), dtype=dtype, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest = dict(lattice="E-W4A8-v1", quantizer=Q.VERSION, layer=1, experts=EXPERTS,
                    blocks=blocks, artifacts=artifacts, source_revision=source_revision,
                    provenance="raw-artifact manifest; execution evidence must be supplied separately")
    with (root / "b1-source.json").open("x") as out:
        json.dump(manifest, out, indent=2); out.write("\n")
    print("DUMP MANIFEST recorded; numerical comparison and execution provenance are separate gates")


def reference_control(root):
    """Consume a retained real reference with CPU-derived candidate boundaries.

    This exercises artifacts/prefixes, not an independently executed candidate.
    """
    ref, _ = load(root, True)
    reports = {}
    for m in (1, 2, 4, 8):
        if m > ref["x_payload"].shape[0]: continue
        part = prefix(ref, m)
        candidate = {key: np.asarray(part[key], dtype=dtype)
                     for key, (_, dtype, _) in schema(m, False).items()}
        reports[str(m)] = compare(part, candidate)
    return {"pass": all(report["pass"] for report in reports.values()), "prefixes": reports,
            "scope": "CPU-derived reference controls only; no GPU candidate or new checkpoint read"}


def fixture():
    rng = np.random.default_rng(20260924)
    got = {key: np.zeros(shape, dtype) for key, (_, dtype, shape) in schema(1, False).items()}
    got["x_payload"], got["x_scales"] = Q.encode_blocks(np.zeros((1, 4096), np.float32))
    got["weights"][:] = np.arange(1, 9, dtype=np.float32) / 36
    got["gate"][:] = rng.uniform(-1, 1, got["gate"].shape)
    got["up"][:] = rng.uniform(-.1, .1, got["up"].shape)
    got["h"] = activation(got["gate"], got["up"])
    got["mid_payload"], got["mid_scales"] = Q.encode_blocks(got["h"])
    got["partial"][:] = rng.uniform(-.01, .01, got["partial"].shape)
    got["rank"], got["decoded"], got["wire"] = reduce_routes(got["partial"], got["weights"])
    ref = copy.deepcopy(got)
    ref["partial"] = ref["partial"].astype(np.float64)
    ref["full"] = (ref["partial"] * ref["weights"][None, :, :, None].astype(np.float64)).sum(axis=(0, 2))
    return ref, got


def selftest():
    ref, good = fixture()
    assert compare(ref, good)["pass"]
    mutations = {
        "wrong_input": ("x_payload", 1), "fc1": ("gate", 1), "activation": ("h", 1),
        "codec_payload": ("mid_payload", 1), "codec_scale": ("mid_scales", 1),
        "fc2": ("partial", 1), "weighted_presum": ("rank", 1),
        "bf16_return": ("decoded", 1), "rank_sum": ("wire", 1),
    }
    for label, (key, delta) in mutations.items():
        bad = copy.deepcopy(good)
        if key == "mid_payload": bad[key].flat[0] ^= 128
        else: bad[key].flat[0] += delta
        assert not compare(ref, bad)["pass"], label
    bad = copy.deepcopy(good); bad["h"].flat[0] = np.nan
    try: compare(ref, bad)
    except Q.NumericalFault: pass
    else: raise AssertionError("nonfinite accepted")
    for mutation in ("weight_twice", "late_return", "per_route_return"):
        bad = copy.deepcopy(good)
        bad["rank"], bad["decoded"], bad["wire"] = reduce_routes(bad["partial"], bad["weights"], mutation)
        assert not compare(ref, bad)["pass"], mutation
    ordered = copy.deepcopy(good)
    ordered["weights"][:] = 0; ordered["weights"][:, 0] = 1
    ordered["partial"][:, 0, 0, 0] = [2**24, 1, -2**24, 1]
    ordered["rank"], ordered["decoded"], ordered["wire"] = reduce_routes(ordered["partial"], ordered["weights"])
    order_ref = copy.deepcopy(ordered); order_ref["partial"] = order_ref["partial"].astype(np.float64)
    order_ref["full"] = (order_ref["partial"] * order_ref["weights"][None, :, :, None]).sum(axis=(0, 2))
    assert compare(order_ref, ordered)["pass"]
    bad = copy.deepcopy(ordered); bad["wire"] = np.zeros_like(bad["wire"])
    for r in (0, 2, 1, 3): bad["wire"] = (bad["wire"] + bad["decoded"][r]).astype(np.float32)
    assert ordered["wire"][0, 0] == 1 and bad["wire"][0, 0] == 2
    assert not compare(order_ref, bad)["pass"], "arrival-order sum"
    doubled = {key: np.concatenate([value, value], axis=0 if key in ("x_payload", "x_scales", "weights", "wire", "full") else 1)
               for key, value in ref.items()}
    assert compare(prefix(doubled, 1), good)["pass"]
    # A one-ULP h change crosses the 288/320 tie at 304 * 2^-12.
    # Codec correctness is conditioned on the actual h, not the ideal bytes.
    boundary = copy.deepcopy(good); boundary_ref = copy.deepcopy(ref)
    target = np.float32(304 * 2**-12)
    for data in (boundary, boundary_ref):
        data["gate"].reshape(-1)[:32] = 1
        data["up"].reshape(-1)[:32] = target / activation(np.ones(1, np.float32), np.ones(1, np.float32))[0]
        data["h"] = activation(data["gate"], data["up"])
        data["h"].reshape(-1)[:32] = target
        data["mid_payload"], data["mid_scales"] = Q.encode_blocks(data["h"])
    boundary["h"].flat[0] = np.nextafter(target, np.float32(0))
    boundary["mid_payload"], boundary["mid_scales"] = Q.encode_blocks(boundary["h"])
    report = compare(boundary_ref, boundary)
    assert report["pass"] and report["rows"]["codec_payload_same_fp32"]["pass"]
    assert report["ideal_intermediate_byte_differences"] == {"mid_payload": 1, "mid_scales": 0}
    boundary["partial"].flat[0] += np.float32(.1)
    boundary["rank"], boundary["decoded"], boundary["wire"] = reduce_routes(boundary["partial"], boundary["weights"])
    report = compare(boundary_ref, boundary)
    assert not report["pass"] and not report["rows"]["compute_presum_vs_full"]["pass"]
    with tempfile.TemporaryDirectory() as tmp:
        roots = [Path(tmp) / "reference", Path(tmp) / "candidate"]
        blocks = [{"name": f"model.layers.1.mlp.experts.{e}.{p}_proj", "weight_sha256": "0"*64, "scale_sha256": "1"*64}
                  for e in EXPERTS for p in ("gate", "up", "down")]
        for root, data, reference in zip(roots, (ref, good), (True, False)):
            root.mkdir(); artifacts = {}
            for key, (name, dtype, shape) in schema(1, reference).items():
                raw = np.asarray(data[key], dtype=dtype).tobytes(); (root/name).write_bytes(raw)
                artifacts[name] = dict(shape=list(shape), dtype=dtype, sha256=hashlib.sha256(raw).hexdigest())
            manifest = dict(lattice="E-W4A8-v1", quantizer=Q.VERSION, layer=1, experts=EXPERTS, blocks=blocks, artifacts=artifacts)
            (root/"b1-source.json").write_text(json.dumps(manifest))
        (roots[1] / "b1-source.json").unlink()
        seal_dump(roots[0], roots[1], 1, "CPU-selftest-no-GPU")
        assert run(*roots)["pass"]
        path = roots[1] / "b1-source.json"; original = json.loads(path.read_text())
        for mutation in ("hash", "shape", "dtype", "lattice", "source", "missing"):
            manifest = copy.deepcopy(original)
            if mutation == "hash": manifest["artifacts"]["b1-h.f32"]["sha256"] = "0"*64
            elif mutation == "shape": manifest["artifacts"]["b1-h.f32"]["shape"][-1] = 256
            elif mutation == "dtype": manifest["artifacts"]["b1-h.f32"]["dtype"] = "<f8"
            elif mutation == "lattice": manifest["lattice"] = "E-W4A8-p"
            elif mutation == "source": manifest["blocks"][0]["weight_sha256"] = "2"*64
            else: del manifest["artifacts"]["b1-h.f32"]
            path.write_text(json.dumps(manifest))
            try: run(*roots)
            except (ValueError, KeyError): pass
            else: raise AssertionError(mutation)
        path.write_text(json.dumps(original))
        binary = roots[1] / "b1-h.f32"; raw = binary.read_bytes(); binary.write_bytes(raw[:-1])
        try: run(*roots)
        except ValueError: pass
        else: raise AssertionError("truncated artifact")
        outside = Path(tmp) / "outside.bin"; outside.write_bytes(raw)
        binary.unlink(); binary.symlink_to(outside)
        try: run(*roots)
        except ValueError: pass
        else: raise AssertionError("symlink artifact")
    print("HOST PASS stage comparator: synthetic positive, 9 data mutations, 4 ordered-reduction mutations, nonfinite, token prefix, one-ULP crossing without waiver, 8 artifact refusals; no GPU evidence")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--selftest", action="store_true")
    group.add_argument("--compare", nargs=2, metavar=("REFERENCE", "CANDIDATE"))
    group.add_argument("--reference-control", metavar="REFERENCE")
    group.add_argument("--seal-dump", nargs=4, metavar=("REFERENCE", "CANDIDATE", "M", "SOURCE"))
    args = parser.parse_args()
    try:
        if args.selftest: selftest()
        elif args.seal_dump: seal_dump(*args.seal_dump)
        else:
            report = reference_control(args.reference_control) if args.reference_control else run(*args.compare)
            print(json.dumps(report, indent=2, allow_nan=False))
            raise SystemExit(0 if report["pass"] else 3)
    except Q.NumericalFault as error:
        print(f"NUMERICAL_FAULT {error}"); raise SystemExit(3)
    except (ValueError, KeyError, OSError, TypeError, IndexError) as error:
        print(f"INPUT_FAILURE {error}"); raise SystemExit(2)
