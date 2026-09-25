# Spark-side round-trip overhead attribution and I6 design proposals

**Author:** kernel-lead. **Date:** 2026-09-25 AEST. **Status:** desk-work design
proposal (no fleet time, no code change). Feeds the maintainer's I6 scope decision per the
builder's steer 2026-09-25 05:21 AEST ("a Spark-side round-trip overhead
attribution plus a design proposal for I6"). Every number is a receipt (with its
SHA) or labelled **model**; every proposal is marked **PROPOSAL** and is a
proposal, not a commitment.

## 1. The measured picture (sourced)

The 2K-prefill per-layer wall is ~250 ms and is **round-trip-bound, not
FFN-bound** (integration-i5 §4.1, `c6c93a0`; the §4 falsifier fired at 174 tok/s
vs ~730 modelled):

| Block | ms/layer | Share | Source |
|---|---:|---:|---|
| Coordinator compute (attn 3.2, qkv 2.5, o_proj 9.5, router 12.6, KV 1.5, RoPE 0.1) | ~29 | ~12% | `c6c93a0` |
| MoE round trip (`wire_moe_chunk`) | ~220 | ~88% | `c6c93a0` |
| ─ Spark FFN (B2 E-FP32 SIMT, 51.8 = the modelled 840 tok/s) | 51.8 | ~21% | `a6f3d50` |
| ─ Spark overheads (receive+CRC+gather+reduce+seal+send) | ~49 | ~20% | `a6f3d50` |
| ─ coordinator wire handling (quantize 15, send, receive, deserialize) + queueing | ~120 (remainder) | ~48% | `c6c93a0` |

The Spark-side (this doc's scope) is the **~49 ms of non-FFN overhead** around the
51.8 ms FFN, measured per rank by the offset-free four-timestamp table
(`a6f3d50`, `runs/20260923-i4/packets/kernel-lead.md:2447-2463`), spark1's own
clock, one 2K prefill layer:

| Stage | ms | What it is (code) |
|---|---:|---|
| receive | 8.85 | `TcpTransport::recv` blocking read of the ~8.9 MB request frame |
| CRC | 1.2 | `StreamReceiver::accept` → `decode_frame` → `crc32c` over the frame |
| gather | 16.0 | `serve_return` decode_hidden (CPU E4M3→f32) + `RoutePlan` + `upload_and_gather` (H2D + zero-fill + gather kernel) |
| FFN | 51.8 | the B2 SIMT FFN (the TP-2 kernel target, separate track) |
| reduce | 3.4 | `route_reduce` device reduce + D2H download of 16.8 MB BF16 |
| seal | 6.9 | `tx.encode_return` (`StreamSender::encode_return` clone + `frame::encode_return` + CRC) |
| send | 12.4 | `transport.send` blocking `write_all` of the ~16.8 MB return frame |

Byte anchors (all sourced from wire layout, `crates/mimo26-wire/src/layout.rs`):
- request frame ≈ 2048 × (4096 E4M3 + 128 scale = 4224 B) = ~8.65 MB hidden +
  routes/descriptors ⇒ ~8.9 MB on the wire (the packet's "8.9 MB receive");
- decoded hidden f32 = 2048 × 4096 × 4 B = **33.5 MB** (uploaded in gather);
- return = 2048 × 8192 B (`RETURN_ROW_BYTES = HIDDEN*2`) = **16.8 MB**.

## 2. Attribution: per stage, copying / allocation / serialization / waiting

| Stage | ms | Dominant cost | Detail |
|---|---:|---|---|
| receive | 8.85 | **waiting** + allocation | blocking `read_exact` into a freshly allocated `vec![0u8; wb]` per frame (`transport.rs:200`); the wait is the coordinator's serialize/send upstream, the allocation is per-request |
| CRC | 1.2 | **serialization** | one `crc32c` pass over the full ~8.9 MB frame (`l4.rs:88` → `frame::decode_frame`) |
| gather | 16.0 | **copying + serialization + allocation** | CPU `decode_hidden` E4M3→f32 of 8.4 M elements (serialization, `serve.rs:33`); H2D upload of the 33.5 MB f32 hidden + src/dst index arrays + zero-fill of the padded x (copying, `decode.rs:613-691`); `Vec<i32>` src/dst and `Vec<f32>` hidden allocated per request (allocation) |
| reduce | 3.4 | **copying** | D2H download of the 16.8 MB BF16 return (`decode.rs:600-602`) + the reduce kernel |
| seal | 6.9 | **copying + serialization** | `StreamSender::encode_return` clones the whole `ReturnFrame` (16.8 MB of codes, `l4.rs:57`) then `frame::encode_return` serializes + `crc32c` over the return |
| send | 12.4 | **waiting** | blocking `write_all` of 16.8 MB (`transport.rs:163`); the socket buffer/peer drain is the wait |

**Reading:** the ~49 ms is ~2/3 **copying+waiting** (receive 8.85, gather's H2D,
reduce's D2H, seal's clone, send 12.4) and ~1/3 **serialization** (decode_hidden,
CRC×2, frame encode). The FFN (51.8) is the only compute term and it is the
separate TP-2 kernel track. The transport itself is *not* the wall: network+queue
is 1.6–5.6 ms per rank (`a6f3d50`), which is what the offset-free split proves.

## 3. I6 design proposals

Each is a **PROPOSAL**. Predictions are **model** (per-layer ms at the 2K shape),
not measured, and each carries the discount rule (I-Hon: quote with a 3–5×
discount until a live gate measures it).

### P1 — Persistent and pinned buffers (PROPOSAL)

Reuse the receive frame buffer, the hidden f32 staging, the src/dst index arrays,
and the return-frame buffer across requests instead of allocating per request.
The `Scratch` pool already pins the H2D staging (`decode.rs:605-610`); extend the
same idea to the transport's per-frame `vec![0u8; wb]` and to `StreamSender`'s
clone target.

- **Attribution it attacks:** allocation in receive (8.85), gather (16.0), seal (6.9).
- **Prediction (model):** removes ~2–4 ms/layer of allocation+clone. The dominant
  term it leaves untouched is the actual socket wait and the H2D/D2H bytes.

### P2 — Zero-copy send (PROPOSAL)

Encode the return directly into a persistent, page-aligned buffer and `write_all`
that buffer, eliminating `StreamSender::encode_return`'s `f.clone()` (a 16.8 MB
copy, `l4.rs:57`) and the intermediate encode buffer.

- **Attribution it attacks:** seal's clone (6.9) + part of send (12.4).
- **Prediction (model):** removes ~3–5 ms/layer (the clone is the biggest pure copy
  left after P1); the socket drain itself is untouched.

### P3 — Overlap gather with the FFN (PROPOSAL)

Run `upload_and_gather` (H2D upload + zero-fill + gather) on a second stream so
layer *L+1*'s gather overlaps layer *L*'s FFN, instead of serialising on the null
stream (`decode.rs:685-686`). Requires the hidden decode to move off the critical
path (see P4) since `decode_hidden` is CPU-bound on the main thread.

- **Attribution it attacks:** gather's 16.0 (copying) — hidden behind the 51.8 FFN.
- **Prediction (model):** removes up to ~16 ms/layer once the CPU decode is not
  serialising the stream; realistically ~8–12 ms (the upload still shares DRAM
  bandwidth with the FFN).

### P4 — Fuse seal and CRC into the send path (PROPOSAL)

Compute the CRC incrementally as the return rows are appended, so the seal's
separate `crc32c` pass over 16.8 MB and the receive-side CRC over 8.9 MB are
folded into the encode/decode passes rather than a second full-frame walk.
(Hardware CRC32C already runs at ~100 GB/s on the Sparks — `crc32c.rs` — so the
saving is the *extra pass*, not the CRC cost itself.)

- **Attribution it attacks:** CRC (1.2) + part of seal (6.9).
- **Prediction (model):** removes ~2–4 ms/layer.

### P5 — Per-rank pipelining (PROPOSAL)

Pipeline receive → decode → gather → FFN → reduce → seal → send across
consecutive layers within a rank (double-buffer the scratch, one in flight, one
serving), so the 8.85 receive + 12.4 send + 16.0 gather of layer *L+1* overlap
the 51.8 FFN of layer *L*. This is the Spark-side analogue of the coordinator's
ping-pong (§3.3 of `integration-i5.md`).

- **Attribution it attacks:** waiting terms — receive 8.85 + send 12.4 — plus the
  serialization between stages.
- **Prediction (model):** hides most of the ~49 ms non-FFN overhead behind the FFN,
  i.e. per-layer Spark time → toward the max(FFN, non-FFN) ≈ 51.8 ms. The residual
  is the serial (non-overlapped) tail, modelled at ~10–20 ms until measured.

### What this deliberately does NOT touch

- **The FFN 51.8 ms** — that is the TP-2 kernel track (TP-1 stopped at F2/F3,
  `65a8fba`; a TP-2 needs a fresh consult pre-registration and is I6 scope).
- **The coordinator ~120 ms** (quantize/send/receive/deserialize + queueing) — the
  largest single block but coordinator-side; owned by attn-lead/builder, not this
  doc's filespace.

## 4. Prediction table (all model, per-layer ms removed at the 2K shape)

| Proposal | Removes | Model Δ (discount 3–5×) |
|---|---:|---:|
| P1 persistent+pinned buffers | allocation/clone | 2–4 → 0.5–1.5 |
| P2 zero-copy send | 16.8 MB clone | 3–5 → 0.7–1.7 |
| P3 overlap gather with FFN | H2D gather | 8–12 → 2–4 |
| P4 fuse seal+CRC | extra CRC passes | 2–4 → 0.5–1.5 |
| P5 per-rank pipelining | receive+send+gather wait | ~49 → FFN-bound (~51.8) |

**Bottom line (model):** P1–P4 alone shave ~15–25 ms/layer of the ~49 ms overhead;
P5 is the one that structurally removes the wait and is the natural I6
centrepiece. None of them is a kernel change — they are transport/scheduling
changes in `crates/mimo26-spark` and the wire layer.

## Provenance

`c6c93a0` (integration-i5 §4.1 re-derivation: 174/151/139 tok/s vs 730/669/516,
29/220/51.8/49/120 split), `a6f3d50` (offset-free four-timestamp, 8.85/1.2/16.0/
51.8/3.4/6.9/12.4 split + 1.6–5.6 ms net+queue), `80f7a4e` (cell B 32K),
`6194b02`/`aa51f5b` (timeline instrumentation). Wire bytes:
`crates/mimo26-wire/src/layout.rs` (`HIDDEN=4096`, `HIDDEN_ROW_BYTES=4224`,
`RETURN_ROW_BYTES=8192`). No GPU work, no timing claims beyond the sourced
receipts and the labelled model predictions.
