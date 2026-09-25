//! fabric_rate — go-window step 4 (A2 R1): sustained coordinator<->Sparks throughput
//! over the serving transport, plus the per-layer decode round-trip.
//!
//! Runs on the coordinator against the four Spark daemons (they must be idle — this is a
//! post-X1a / post-ladder measurement). Inference traffic runs only on the RDMA
//! fabric: both ends refuse a connection whose local address is not on a RoCE v2
//! port of at least `MIMO26_WIRE_MIN_GBPS` (100) Gb/s, so the LAN/10G path (the
//! API path) cannot carry the wire. `MIMO26_SPARK_ADDRS` has no default.
//!
//! ```text
//! MIMO26_SPARK_ADDRS="192.0.2.1:8600,192.0.2.2:8600,192.0.2.4:8600,192.0.2.5:8600" \
//!   cargo run -p mimo26-coordinator --example fabric_rate -- --tokens 4096 --iters 8
//! ```
//!
//! (A LAN comparison needs `MIMO26_WIRE_ALLOW_LAN=1` on both ends; a test-only override.)
//!
//! Reports, per iteration: wall time, and the sustained send (coordinator->Spark) and
//! return (Spark->coordinator) byte rates at production frame sizes (a `tokens`-token
//! prefill chunk: ~4.3 KB/token out, 8,192 B/token back, 4 ranks in parallel).
//! Then a 1-token decode round-trip loop (mean/min/max RTT).

use std::time::{Duration, Instant};

use mimo26_coordinator::wire::{quantize_hidden, WireClient};
use mimo26_wire::frame::{RequestFrame, ReturnFrame, ReturnRow, RouteEntry, RowDescriptor};
use mimo26_wire::l4::StreamSender;
use mimo26_wire::{SourceKind, Status, WireNaive, HIDDEN};

fn main() {
    let addrs: Vec<String> = std::env::var("MIMO26_SPARK_ADDRS")
        .map(|s| s.split(',').map(str::to_string).collect())
        .expect("MIMO26_SPARK_ADDRS: the four Sparks' RDMA-fabric addresses (host:port, rank order); no LAN default");
    let mut tokens = 4096usize;
    let mut iters = 8usize;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--tokens" => tokens = args.next().and_then(|s| s.parse().ok()).unwrap_or(tokens),
            "--iters" => iters = args.next().and_then(|s| s.parse().ok()).unwrap_or(iters),
            _ => eprintln!("ignoring unknown arg {a}"),
        }
    }
    let topk: usize = 8;

    let mut wire = WireClient::connect(&addrs).expect("WireClient::connect");
    eprintln!(
        "fabric_rate: addrs={addrs:?} tokens={tokens} iters={iters} topk={topk}"
    );

    // Synthetic activations (all 0.5) and routes (8 experts per token, weight 1.0).
    let hidden = vec![0.5f32; tokens * HIDDEN];
    let routes: Vec<(u32, f32)> = (0..tokens * topk).map(|i| ((i % 256) as u32, 1.0)).collect();

    // Exact on-wire sizes, from the same frames `moe_layer` encodes.
    let (req_bytes, ret_bytes) = frame_sizes(tokens, topk, &hidden);
    eprintln!(
        "frame sizes: request={req_bytes} B ({:.2} MB) return={ret_bytes} B ({:.2} MB), 4 ranks",
        req_bytes as f64 / 1e6,
        ret_bytes as f64 / 1e6
    );

    println!("iter,wall_ms,send_GBs,return_GBs,aggregate_GBs");
    let mut total_send = 0.0f64;
    let mut total_ret = 0.0f64;
    for i in 0..iters {
        let t0 = Instant::now();
        let _out = wire
            .moe_layer(1, &hidden, &routes, topk)
            .expect("moe_layer");
        let dt = t0.elapsed().as_secs_f64();
        // 4 ranks in parallel: the per-direction bytes are the 4-rank totals.
        let send_gbs = (4.0 * req_bytes as f64) / dt / 1e9;
        let ret_gbs = (4.0 * ret_bytes as f64) / dt / 1e9;
        println!(
            "{i},{:.4},{send_gbs:.3},{ret_gbs:.3},{:.3}",
            dt * 1e3,
            send_gbs + ret_gbs
        );
        total_send += send_gbs;
        total_ret += ret_gbs;
    }
    eprintln!(
        "sustained (mean over {iters} iters, 4 ranks): send {:.3} GB/s ({:.1} Gbps), return {:.3} GB/s ({:.1} Gbps), aggregate {:.3} GB/s ({:.1} Gbps)",
        total_send / iters as f64,
        total_send / iters as f64 * 8.0,
        total_ret / iters as f64,
        total_ret / iters as f64 * 8.0,
        (total_send + total_ret) / iters as f64,
        (total_send + total_ret) / iters as f64 * 8.0
    );

    // Per-layer decode round-trip: 1 token, 4 ranks in parallel.
    let one = vec![0.5f32; HIDDEN];
    let one_routes: Vec<(u32, f32)> = (0..topk).map(|k| (k as u32, 1.0)).collect();
    let decode_iters = 64;
    let mut rtts = Vec::with_capacity(decode_iters);
    for _ in 0..decode_iters {
        let t0 = Instant::now();
        wire.moe_layer(1, &one, &one_routes, topk).expect("decode moe_layer");
        rtts.push(t0.elapsed());
    }
    let min = rtts.iter().min().unwrap();
    let max = rtts.iter().max().unwrap();
    let mean = rtts.iter().sum::<Duration>() / decode_iters as u32;
    eprintln!(
        "decode 1-token round-trip (4 ranks, {} iters): mean {:.3} ms, min {:.3} ms, max {:.3} ms",
        decode_iters,
        mean.as_secs_f64() * 1e3,
        min.as_secs_f64() * 1e3,
        max.as_secs_f64() * 1e3
    );
}

/// Encode the exact request/return frames `moe_layer` produces and report their
/// on-wire byte lengths (the header carries `wire_bytes` at byte 76).
fn frame_sizes(tokens: usize, topk: usize, hidden: &[f32]) -> (usize, usize) {
    let mut rows = Vec::with_capacity(tokens);
    let mut routes = Vec::with_capacity(tokens * topk);
    let mut hidden_rows = Vec::with_capacity(tokens);
    for t in 0..tokens {
        rows.push(RowDescriptor {
            row_id: t as u64,
            source_kind: SourceKind::Decode,
            source_request_id: 1,
            token_position: t as u64,
            route_offset: (t * topk) as u32,
            route_count: topk as u32,
        });
        for k in 0..topk {
            routes.push(RouteEntry {
                row_index: t as u32,
                expert_id: (k % 256) as u32,
                gate_weight: 1.0,
            });
        }
        hidden_rows.push(quantize_hidden(&hidden[t * HIDDEN..(t + 1) * HIDDEN]).expect("quantize"));
    }
    let req = RequestFrame {
        request_id: 1,
        placement_version: 1,
        layer_id: 1,
        executor_id: 0,
        source_kind: SourceKind::Decode,
        token_position: 0,
        flags: 0,
        seq: 0,
        rows,
        routes,
        hidden_rows,
    };
    let mut sender = StreamSender::new(WireNaive::NONE);
    let req_bytes = sender.encode_request(&req).expect("encode request").len();

    let ret = ReturnFrame {
        request_id: 1,
        placement_version: 1,
        layer_id: 1,
        executor_id: 0,
        token_position: 0,
        status: Status::Ok,
        flags: 0,
        route_count: topk,
        seq: 0,
        rows: (0..tokens).map(|_| ReturnRow { codes: vec![0u16; HIDDEN] }).collect(),
    };
    let ret_bytes = sender.encode_return(&ret).expect("encode return").len();
    (req_bytes, ret_bytes)
}
