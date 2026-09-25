//! The Spark serve loop: request decode → FFN → route reduce → return encode.
//!
//! `serve` glues the pieces together for one request frame: decode each hidden
//! row (E4M3 + UE8M0 K32 → f32), build the route plan, replicate the hidden
//! rows, run the FFN (injected — `decode_ffn`/`prefill_ffn` on the device, or
//! the CPU oracle in tests), reduce to the per-token pre-sum, and encode the
//! compact BF16 return. The FFN is injected so the whole loop is CPU-testable
//! and the device kernel is a drop-in.

use std::time::Instant;

use mimo26_expert::slice::HIDDEN;
use mimo26_expert::NaiveBits;
use mimo26_wire::RequestFrame;

use crate::route::RoutePlan;
use crate::wire;

/// Per-request serve timings (ms), filled by `serve_return`. The server composes
/// these with its recv/lazy-load/send timings into one per-request line
/// (go-window step 4: the Spark half of the prefill budget).
#[derive(Default, Clone, Copy)]
pub struct Timings {
    pub plan_ms: f64,
    pub ffn_ms: f64,
    pub reduce_ms: f64,
}

/// Decode one hidden row (4,096 E4M3 codes + 128 UE8M0 K32 scales) to f32.
///
/// `value[k] = decode_e4m3(payload[k]) * 2^(scales[k/32] - 127)`, saturated to
/// `±f32::MAX` (byte 255 clamps to `2^127`, the T10 rule).
pub fn decode_hidden(payload: &[u8], scales: &[u8]) -> Result<Vec<f32>, String> {
    if payload.len() != HIDDEN {
        return Err(format!(
            "hidden: payload has {} bytes, expected {HIDDEN}",
            payload.len()
        ));
    }
    let scale_len = HIDDEN / 32;
    if scales.len() != scale_len {
        return Err(format!(
            "hidden: scales has {} bytes, expected {scale_len}",
            scales.len()
        ));
    }
    let mut out = Vec::with_capacity(HIDDEN);
    let f32_max = f32::MAX as f64;
    let tab = mimo26_load::e4m3::decode_table();
    for b in 0..(HIDDEN / 32) {
        // The E8M0 scale is constant over a 32-element block; hoist it out of the
        // inner loop (it was recomputed per element — an exp2 per element).
        let s = mimo26_expert::mxfp4::e8m0_scale(scales[b], NaiveBits::NONE);
        for k in 0..32 {
            let v = tab[payload[b * 32 + k] as usize];
            let prod = (v * s).clamp(-f32_max, f32_max);
            out.push(prod as f32);
        }
    }
    Ok(out)
}

/// Serve one request frame end to end, returning the un-stamped compact return
/// frame (the server stamps the L4 sequence before encoding).
///
/// `grouped` is the layer's resident grouped image pointer (256 quarter slices
/// ordered by expert id, so `expert_id == slot index`) — device-resident in the
/// serving daemon, host-resident in the CPU tests; `ffn` runs the expert FFN +
/// route reduce from the token-major `hidden` and the `RoutePlan` (device
/// gather/decode/prefill, or the CPU oracle in tests).
pub fn serve_return<F>(
    request: &RequestFrame,
    grouped: *const u8,
    ffn: F,
    _naive: mimo26_wire::WireNaive,
    timings: &mut Timings,
) -> Result<mimo26_wire::ReturnFrame, String>
where
    F: Fn(*const u8, &[f32], &RoutePlan) -> Result<Vec<u16>, String>,
{
    let tokens = request.rows.len();
    if request.hidden_rows.len() != tokens {
        return Err(format!(
            "serve: {} hidden rows for {tokens} rows",
            request.hidden_rows.len()
        ));
    }

    // 1. Decode hidden rows (token-major).
    let mut hidden = Vec::with_capacity(tokens * HIDDEN);
    for h in &request.hidden_rows {
        hidden.extend_from_slice(&decode_hidden(&h.payload, &h.scales)?);
    }

    // 2. Route plan (the ffn gathers/replicates the padded x internally).
    let t = Instant::now();
    let rp = RoutePlan::from_routes(&request.routes, tokens)?;
    timings.plan_ms = t.elapsed().as_secs_f64() * 1e3;
    // M-padding waste telemetry (I5-R11): the ladder-risk factor per layer.
    if crate::timeline::trace() {
        eprintln!(
            "waste layer={} factor={:.3}x padded={} real={}",
            request.layer_id,
            rp.waste_factor(),
            rp.padded_rows(),
            rp.routed_rows()
        );
    }

    // 3. FFN + route reduce (grouped output collapsed to per-token BF16 rows).
    let t = Instant::now();
    let bf16 = ffn(grouped, &hidden, &rp)?;
    timings.ffn_ms = t.elapsed().as_secs_f64() * 1e3;
    timings.reduce_ms = 0.0;

    // 4. Compact BF16 return frame (8,192 B per token once encoded).
    wire::rank_bf16_to_return_frame(
        &bf16,
        tokens,
        request.request_id,
        request.placement_version,
        request.layer_id,
        request.executor_id,
        request.token_position,
    )
    .map_err(|e| e.to_string())
}

/// [`serve_return`] on the B1 path (perf reset R3): the wire payload goes to the
/// B1 tensor-core FFN as received (E4M3 rows + K32 scales, no host decode), with
/// the token-major top-8 routes; returns the same compact BF16 return frame.
/// The B1 inputs of one request frame: E4M3 payload + K32 scales (token-major,
/// contiguous) and the token-major top-8 expert ids + route weights.
#[cfg(feature = "cuda")]
pub fn b1_inputs(request: &RequestFrame) -> Result<(Vec<u8>, Vec<u8>, Vec<i32>, Vec<f32>), String> {
    let tokens = request.rows.len();
    if request.hidden_rows.len() != tokens {
        return Err(format!("serve b1: {} hidden rows for {tokens} rows", request.hidden_rows.len()));
    }
    let mut payload = Vec::with_capacity(tokens * HIDDEN);
    let mut scales = Vec::with_capacity(tokens * HIDDEN / 32);
    for h in &request.hidden_rows {
        if h.payload.len() != HIDDEN || h.scales.len() != HIDDEN / 32 {
            return Err("serve b1: hidden row is not 4,096 E4M3 + 128 scales".into());
        }
        payload.extend_from_slice(&h.payload);
        scales.extend_from_slice(&h.scales);
    }
    let mut ids = Vec::with_capacity(tokens * 8);
    let mut weights = Vec::with_capacity(tokens * 8);
    for (i, row) in request.rows.iter().enumerate() {
        if row.route_count != 8 {
            return Err(format!("serve b1: row {i} has {} routes, want 8", row.route_count));
        }
        let off = row.route_offset as usize;
        let rs = request.routes.get(off..off + 8).ok_or("serve b1: route range out of bounds")?;
        for r in rs {
            if r.row_index as usize != i {
                return Err(format!("serve b1: route row {} under row {i}", r.row_index));
            }
            ids.push(r.expert_id as i32);
            weights.push(r.gate_weight);
        }
    }
    Ok((payload, scales, ids, weights))
}

/// The return frame's metadata for `request` (no rows): what
/// `wire::rank_bf16_to_return_frame` stamps, for a frame assembled in place.
pub fn return_meta(request: &RequestFrame) -> mimo26_wire::ReturnFrame {
    mimo26_wire::ReturnFrame {
        request_id: request.request_id,
        placement_version: request.placement_version,
        layer_id: request.layer_id,
        executor_id: request.executor_id,
        token_position: request.token_position,
        status: mimo26_wire::layout::Status::Ok,
        flags: mimo26_wire::frame::FLAG_RETURN_REQUIRED,
        route_count: 8,
        seq: 0,
        rows: Vec::new(),
    }
}

/// [`return_meta`] for a request validated in place.
pub fn return_meta_view(v: &mimo26_wire::frame::RequestView) -> mimo26_wire::ReturnFrame {
    mimo26_wire::ReturnFrame {
        request_id: v.request_id,
        placement_version: v.placement_version,
        layer_id: v.layer_id,
        executor_id: v.executor_id,
        token_position: v.token_position,
        status: mimo26_wire::layout::Status::Ok,
        flags: mimo26_wire::frame::FLAG_RETURN_REQUIRED,
        route_count: 8,
        seq: 0,
        rows: Vec::new(),
    }
}

/// [`serve_b1_into`] for a request validated in place (perf reset: Spark
/// zero-copy receive): the top-8 routes are gathered straight from the frame's
/// entries with `b1_inputs`'s shape checks, and the hidden rows are copied to
/// the device from where the NIC landed them (the registered receive slot).
#[cfg(feature = "cuda")]
pub fn serve_b1_view(
    view: &mimo26_wire::frame::RequestView,
    layer: &crate::b1::Layer,
    scratch: &mut crate::b1::Scratch,
    timings: &mut Timings,
    out: &mut [u16],
) -> Result<(), String> {
    let tokens = view.rows;
    if view.row_stride != HIDDEN + HIDDEN / 32 {
        return Err(format!("serve b1: hidden row stride {} is not 4,096 E4M3 + 128 scales", view.row_stride));
    }
    let t = Instant::now();
    let mut ids = Vec::with_capacity(tokens * 8);
    let mut weights = Vec::with_capacity(tokens * 8);
    for i in 0..tokens {
        let row = view.row(i).map_err(|e| e.to_string())?;
        if row.route_count != 8 {
            return Err(format!("serve b1: row {i} has {} routes, want 8", row.route_count));
        }
        let off = row.route_offset as usize; // parse checked off + 8 <= routes
        for j in off..off + 8 {
            let r = view.route(j).map_err(|e| e.to_string())?;
            if r.row_index as usize != i {
                return Err(format!("serve b1: route row {} under row {i}", r.row_index));
            }
            ids.push(r.expert_id as i32);
            weights.push(r.gate_weight);
        }
    }
    timings.plan_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let st = crate::b1::ffn_strided(layer, scratch, view.hidden(), view.row_stride, &ids, &weights, tokens, out)?;
    timings.ffn_ms = t.elapsed().as_secs_f64() * 1e3;
    timings.reduce_ms = 0.0;
    if crate::timeline::trace() {
        eprintln!(
            "b1 layer={} rows={} groups={} stage={:.3} upload_plan={:.3} gpu={:.3} [fc1={:.3} q={:.3} fc2={:.3} red={:.3}] tail={:.3} ms (zero-copy)",
            view.layer_id, tokens, st.groups, timings.plan_ms, st.upload_plan_ms, st.gpu_ms,
            st.phase_ms[0], st.phase_ms[1], st.phase_ms[2], st.phase_ms[3], st.tail_ms
        );
    }
    Ok(())
}

/// [`serve_return_b1`] writing the rank's BF16 partial straight into `out`
/// (`[tokens * 4096]`, the transport's registered send buffer after the header):
/// no return-row copies (perf reset R2 zero-copy return).
#[cfg(feature = "cuda")]
pub fn serve_b1_into(
    request: &RequestFrame,
    layer: &crate::b1::Layer,
    scratch: &mut crate::b1::Scratch,
    timings: &mut Timings,
    out: &mut [u16],
) -> Result<(), String> {
    let tokens = request.rows.len();
    let t = Instant::now();
    let (payload, scales, ids, weights) = b1_inputs(request)?;
    timings.plan_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let st = crate::b1::ffn(layer, scratch, &payload, &scales, &ids, &weights, tokens, out)?;
    timings.ffn_ms = t.elapsed().as_secs_f64() * 1e3;
    timings.reduce_ms = 0.0;
    if crate::timeline::trace() {
        eprintln!(
            "b1 layer={} rows={} groups={} stage={:.3} upload_plan={:.3} gpu={:.3} [fc1={:.3} q={:.3} fc2={:.3} red={:.3}] tail={:.3} ms (in place)",
            request.layer_id, tokens, st.groups, timings.plan_ms, st.upload_plan_ms, st.gpu_ms,
            st.phase_ms[0], st.phase_ms[1], st.phase_ms[2], st.phase_ms[3], st.tail_ms
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub fn serve_return_b1(
    request: &RequestFrame,
    layer: &crate::b1::Layer,
    scratch: &mut crate::b1::Scratch,
    timings: &mut Timings,
) -> Result<mimo26_wire::ReturnFrame, String> {
    let tokens = request.rows.len();
    if request.hidden_rows.len() != tokens {
        return Err(format!("serve b1: {} hidden rows for {tokens} rows", request.hidden_rows.len()));
    }
    let t = Instant::now();
    let mut payload = Vec::with_capacity(tokens * HIDDEN);
    let mut scales = Vec::with_capacity(tokens * HIDDEN / 32);
    for h in &request.hidden_rows {
        if h.payload.len() != HIDDEN || h.scales.len() != HIDDEN / 32 {
            return Err("serve b1: hidden row is not 4,096 E4M3 + 128 scales".into());
        }
        payload.extend_from_slice(&h.payload);
        scales.extend_from_slice(&h.scales);
    }
    let mut ids = Vec::with_capacity(tokens * 8);
    let mut weights = Vec::with_capacity(tokens * 8);
    for (i, row) in request.rows.iter().enumerate() {
        if row.route_count != 8 {
            return Err(format!("serve b1: row {i} has {} routes, want 8", row.route_count));
        }
        let off = row.route_offset as usize;
        let rs = request.routes.get(off..off + 8).ok_or("serve b1: route range out of bounds")?;
        for r in rs {
            if r.row_index as usize != i {
                return Err(format!("serve b1: route row {} under row {i}", r.row_index));
            }
            ids.push(r.expert_id as i32);
            weights.push(r.gate_weight);
        }
    }
    timings.plan_ms = t.elapsed().as_secs_f64() * 1e3;
    let mut bf16 = vec![0u16; tokens * HIDDEN];
    let t = Instant::now();
    let st = crate::b1::ffn(layer, scratch, &payload, &scales, &ids, &weights, tokens, &mut bf16)?;
    timings.ffn_ms = t.elapsed().as_secs_f64() * 1e3;
    timings.reduce_ms = 0.0;
    if crate::timeline::trace() {
        eprintln!(
            "b1 layer={} rows={} groups={} stage={:.3} upload_plan={:.3} gpu={:.3} [fc1={:.3} q={:.3} fc2={:.3} red={:.3}] tail={:.3} ms",
            request.layer_id, tokens, st.groups, timings.plan_ms, st.upload_plan_ms, st.gpu_ms,
            st.phase_ms[0], st.phase_ms[1], st.phase_ms[2], st.phase_ms[3], st.tail_ms
        );
    }
    wire::rank_bf16_to_return_frame(
        &bf16,
        tokens,
        request.request_id,
        request.placement_version,
        request.layer_id,
        request.executor_id,
        request.token_position,
    )
    .map_err(|e| e.to_string())
}

/// Serve one request frame end to end, encoded (no L4 sequence stamp).
pub fn serve<F>(
    request: &RequestFrame,
    grouped: *const u8,
    ffn: F,
    naive: mimo26_wire::WireNaive,
) -> Result<Vec<u8>, String>
where
    F: Fn(*const u8, &[f32], &RoutePlan) -> Result<Vec<u16>, String>,
{
    let frame = serve_return(request, grouped, ffn, naive, &mut Timings::default())?;
    mimo26_wire::frame::encode_return(&frame, naive).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mimo26_repack::geom::QUARTER_SLICE_BYTES;
    use mimo26_wire::frame::{decode_frame, Frame, RouteEntry, RowDescriptor};
    use mimo26_wire::{SourceKind, WireNaive};

    /// E4M3 code 0x38 is 1.0; scale byte 127 is 2^0 = 1.0.
    #[test]
    fn decode_hidden_one_is_one() {
        let payload = [0x38u8; HIDDEN];
        let scales = [127u8; HIDDEN / 32];
        let out = decode_hidden(&payload, &scales).expect("decode");
        assert_eq!(out.len(), HIDDEN);
        assert!(out.iter().all(|&v| (v - 1.0).abs() < 1e-6), "not all 1.0");
    }

    #[test]
    fn decode_hidden_rejects_bad_shape() {
        assert!(decode_hidden(&[0u8; HIDDEN + 1], &[127u8; HIDDEN / 32]).is_err());
        assert!(decode_hidden(&[0u8; HIDDEN], &[127u8; HIDDEN / 32 + 1]).is_err());
    }

    /// A tiny synthetic grouped image: `n` experts' quarter slices, each payload
    /// = 1.0 (E2M1 nibble 2) and scale 127, so the FFN oracle is deterministic.
    fn synthetic_grouped(n: usize) -> Vec<u8> {
        use mimo26_repack::geom::Proj;
        let mut bytes = vec![0u8; n * QUARTER_SLICE_BYTES];
        for e in 0..n {
            let base = e * QUARTER_SLICE_BYTES;
            for p in [Proj::Gate, Proj::Up, Proj::Down] {
                let poff = base + p.slice_payload_off();
                let plen = p.slice_payload_bytes();
                // E2M1 nibble 2 (1.0) in both halves of each byte.
                bytes[poff..poff + plen].fill(0x22);
                let soff = base + p.slice_scale_off();
                let slen = p.slice_scale_bytes();
                bytes[soff..soff + slen].fill(127);
            }
        }
        bytes
    }

    /// CPU oracle FFN + reduce (the device `ffn_route_reduce` drop-in in tests).
    /// Takes the grouped image length because the ffn closure only receives a
    /// raw pointer; the CPU test passes the host image pointer directly.
    fn cpu_ffn(
        grouped_len: usize,
    ) -> impl Fn(*const u8, &[f32], &RoutePlan) -> Result<Vec<u16>, String> {
        move |grouped: *const u8, hidden: &[f32], rp: &RoutePlan| {
            // SAFETY: the CPU test passes the host grouped image pointer; it is
            // valid for `grouped_len` bytes (one quarter slice per expert).
            let grouped = unsafe { std::slice::from_raw_parts(grouped, grouped_len) };
            let x = rp.replicate_x(hidden).map_err(|e| e.to_string())?;
            let out = mimo26_expert::grouped::expert_ffn_self_contained(grouped, &x, &rp.plan, NaiveBits::NONE)
                .map(|o| o.data)
                .map_err(|e| e.to_string())?;
            let (padded, weight) = rp.token_major_routes();
            Ok(crate::decode::cpu_reduce_bf16(&out, &padded, &weight, rp.tokens, HIDDEN))
        }
    }

    #[test]
    fn serve_round_trips_a_request() {
        // 1 token, 8 routes all to expert 0 (weight 1.0 on the first route).
        let routes: Vec<RouteEntry> = (0..8)
            .map(|s| RouteEntry {
                row_index: 0,
                expert_id: 0,
                gate_weight: if s == 0 { 1.0 } else { 0.0 },
            })
            .collect();
        let hidden = mimo26_wire::frame::HiddenRow {
            payload: vec![0x38u8; HIDDEN], // all 1.0
            scales: vec![127u8; HIDDEN / 32],
        };
        let request = RequestFrame {
            request_id: 9,
            placement_version: 1,
            layer_id: 5,
            executor_id: 1,
            source_kind: SourceKind::Decode,
            token_position: 0,
            flags: 0,
            seq: 0,
            rows: vec![RowDescriptor {
                row_id: 0,
                source_kind: SourceKind::Decode,
                source_request_id: 9,
                token_position: 0,
                route_offset: 0,
                route_count: 8,
            }],
            routes,
            hidden_rows: vec![hidden],
        };
        let grouped = synthetic_grouped(2); // 2 experts resident
        let cpu = cpu_ffn(grouped.len());

        let bytes = serve(&request, grouped.as_ptr(), &cpu, WireNaive::NONE).expect("serve");
        assert_eq!(bytes.len(), 128 + 8192, "1 token => 8,192 B return");

        match decode_frame(&bytes, WireNaive::NONE).expect("decode return") {
            Frame::Return(frame) => {
                assert_eq!(frame.rows.len(), 1);
                assert_eq!(frame.route_count, 8);
                // Recompute the expected pre-sum directly and compare BF16.
                let rp = RoutePlan::from_routes(&request.routes, 1).expect("plan");
                let hidden_f32 = decode_hidden(&request.hidden_rows[0].payload, &request.hidden_rows[0].scales).unwrap();
                let want = cpu(grouped.as_ptr(), &hidden_f32, &rp).expect("ffn");
                assert_eq!(frame.rows[0].codes, want);
            }
            Frame::Request(_) => panic!("expected return frame"),
        }
    }
}
