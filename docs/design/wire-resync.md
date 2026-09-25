# Wire resync after a panic mid-forward (D5 residual)

**Author:** kernel-lead. **Date:** 2026-09-25 AEST. **Status:** design note +
loopback-transport test (no fleet time, no serving-path change). Complements the
attn-lead's D5 (the engine survives a request panic).

## 1. The failure mode

A panic caught mid-forward (D5, `emit_delta`'s `catch_unwind`) abandons an
in-flight MoE round trip on the coordinator↔Spark wire. What is left behind is
unrecoverable in place:

1. **Diverged L4 sequence counters.** The coordinator's `StreamSender` stamped
   the request (its `next_seq` advanced) and its `StreamReceiver` still expects
   that request's return. The Spark either accepted the request (its receiver
   advanced) and maybe never sent a return, or never saw the frame. The two
   ends' counters have drifted by an unknown amount; the next frame in either
   direction is classified `OutOfOrder` (a `Retry`-class error that a bounded
   retry cannot fix, because it re-sends the *same* stale counter).
2. **An indeterminate byte stream.** On a real TCP transport a partial write or
   a partial read leaves the stream at an unknown offset. The only framing the
   wire has is the 128-byte header's `wire_bytes`; there is no in-band
   "skip to next frame" marker, so a receiver cannot re-frame a stream that was
   cut mid-frame.

`WireError::disposition()` already classifies `OutOfOrder`/`Truncated`/`Corrupt`
as `Retry`, but a retry cannot recover a *diverged counter* — it re-transmits the
same sequence the peer has already moved past. The only safe recovery is a full
reconnect.

## 2. The resync primitive: reconnect, not in-band resync

The L4 sequence state (`StreamSender.next_seq` / `StreamReceiver.expected`) is
**per-connection** and starts at 0 on a fresh connection on both ends:

- coordinator `SparkConn::connect` (`crates/mimo26-coordinator/src/wire.rs`)
  builds a fresh `StreamSender`/`StreamReceiver`;
- the Spark serve loop (`crates/mimo26-spark/src/main.rs`) builds a fresh
  `StreamReceiver`/`StreamSender` per accepted connection.

So **reconnecting is the resync**: drop the connection (discarding any partial
frame bytes and the drifted counters together) and re-establish with both ends
back at sequence 0. There is deliberately no in-band resync message: it would
have to be trusted over a stream whose framing is already indeterminate, which
is exactly the state a panic leaves.

## 3. What must change (PROPOSAL, for the coordinator, not done here)

`WireClient` (`crates/mimo26-coordinator/src/wire.rs`) currently keeps its four
`SparkConn`s for the process lifetime and has no reconnect path: a failed
`moe_layer` returns `Err` and leaves `self.conns` empty or stale (the parallel
send/collect paths `std::mem::take` the conns and never restore them on error).

**PROPOSAL:** after any `moe_layer` error, or any caught panic mid-forward, the
request loop must re-establish the wire before the next request — a
`WireClient::reconnect()` (drop all conns, `WireClient::connect` again). This is
a coordinator-side change owned by attn-lead's D5 work; this note records the
contract so the resync is a reconnect, never an in-band retry.

## 4. The loopback-transport test

`crates/mimo26-spark/src/transport.rs` gains
`resync_after_mid_forward_panic_is_a_reconnect`, which drives the loopback
through the three states:

1. **Abandoned mid-forward.** Coordinator sends request seq 0; the Spark accepts
   it; the Spark's serve "panics" before producing a return. The counters have
   diverged (coordinator sender at seq 1, a freshly connected Spark receiver at
   expected 0).
2. **The desync (negative).** Reusing the stale coordinator sender against a
   freshly connected Spark receiver is `OutOfOrder` (`Retry` disposition) —
   proving the stale connection cannot be reused.
3. **The resync (positive).** Both ends reconnect (fresh `StreamSender` /
   `StreamReceiver`, seq 0) and a full request→return round trip succeeds with
   the return's `request_id` matching.

The test asserts the contract in §2: after a panic mid-forward, a reconnect
resyncs; reuse does not.

## Provenance

`emit_delta` catch_unwind: `crates/mimo26-coordinator/src/api.rs:75`. L4
sequence policy: `crates/mimo26-wire/src/l4.rs` (`StreamSender`/`StreamReceiver`,
`OutOfOrder` = `Retry`); disposition table `crates/mimo26-wire/src/error.rs`.
Coordinator per-connection state `wire.rs:141-159`, Spark per-accept state
`main.rs:238-239`. No GPU work, no timing claim.
