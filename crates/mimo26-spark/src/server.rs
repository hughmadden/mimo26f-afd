//! The Spark wire server loop: recv → L4 accept → serve → stamp → send.
//!
//! One [`SparkServer::step`] pulls a frame off the transport, applies the L4
//! sequence policy (`StreamReceiver`), serves the request (FFN injected), stamps
//! the return with the next sequence (`StreamSender`), and asserts the live 8 KB
//! per-token return. L4 faults surface as [`ServeError::L4`] for the window
//! protocol to retry/fail loud; a serve failure is a [`ServeError::Serve`].

use std::fmt;

use mimo26_wire::frame::Frame;
use mimo26_wire::l4::{StreamReceiver, StreamSender};
use mimo26_wire::{WireError, WireNaive};

use crate::route::RoutePlan;
use crate::transport::ByteTransport;

/// The Spark-side wire server.
pub struct SparkServer<T: ByteTransport> {
    transport: T,
    rx: StreamReceiver,
    tx: StreamSender,
    naive: WireNaive,
}

/// A server-loop failure: an L4 fault (retry/drop class), a serve failure, or a
/// transport I/O error.
#[derive(Debug)]
pub enum ServeError {
    L4(WireError),
    Serve(String),
    Io(std::io::Error),
}

impl fmt::Display for ServeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ServeError::L4(e) => write!(f, "L4: {e}"),
            ServeError::Serve(e) => write!(f, "serve: {e}"),
            ServeError::Io(e) => write!(f, "io: {e}"),
        }
    }
}

impl std::error::Error for ServeError {}

impl<T: ByteTransport> SparkServer<T> {
    pub fn new(transport: T, naive: WireNaive) -> Self {
        Self {
            transport,
            rx: StreamReceiver::new(naive),
            tx: StreamSender::new(naive),
            naive,
        }
    }

    /// One non-blocking step. `Ok(None)` when no frame is ready; `Ok(Some(seq))`
    /// after serving one request (the stamped return sequence); `Err` on a
    /// detected L4 fault, a serve failure, or a transport I/O error.
    pub fn step<F>(&mut self, grouped: *const u8, ffn: &F) -> Result<Option<u64>, ServeError>
    where
        F: Fn(*const u8, &[f32], &RoutePlan) -> Result<Vec<u16>, String>,
    {
        let Some(bytes) = self.transport.recv() else {
            return Ok(None);
        };
        let frame = self.rx.accept(&bytes).map_err(ServeError::L4)?;
        let req = match frame {
            Frame::Request(r) => r,
            Frame::Return(_) => {
                return Err(ServeError::L4(WireError::BadKind(
                    mimo26_wire::layout::KIND_RETURN,
                )))
            }
        };
        let ret = crate::serve::serve_return(&req, grouped, ffn, self.naive, &mut crate::serve::Timings::default())
            .map_err(ServeError::Serve)?;
        let stamped_seq = self.tx.next_seq();
        let stamped = self.tx.encode_return(&ret).map_err(ServeError::L4)?;
        // Live 8 KB-per-Spark-per-token assertion (the A6 return contract).
        assert_eq!(
            stamped.len(),
            mimo26_wire::HEADER_LEN + ret.rows.len() * mimo26_wire::RETURN_ROW_BYTES,
            "return frame is not 8,192 B per token"
        );
        self.transport.send(stamped).map_err(ServeError::Io)?;
        Ok(Some(stamped_seq))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mimo26_repack::geom::QUARTER_SLICE_BYTES;
    use mimo26_wire::frame::{RouteEntry, RowDescriptor};
    use mimo26_wire::l4::StreamReceiver;
    use mimo26_wire::{Frame, SourceKind};

    fn cpu_ffn(
        grouped_len: usize,
    ) -> impl Fn(*const u8, &[f32], &RoutePlan) -> Result<Vec<u16>, String> {
        move |grouped: *const u8, hidden: &[f32], rp: &RoutePlan| {
            // SAFETY: the CPU test passes the host grouped image pointer; valid
            // for `grouped_len` bytes.
            let grouped = unsafe { std::slice::from_raw_parts(grouped, grouped_len) };
            let x = rp.replicate_x(hidden).map_err(|e| e.to_string())?;
            let out = mimo26_expert::grouped::expert_ffn_self_contained(
                grouped,
                &x,
                &rp.plan,
                mimo26_expert::NaiveBits::NONE,
            )
            .map(|o| o.data)
            .map_err(|e| e.to_string())?;
            let (padded, weight) = rp.token_major_routes();
            Ok(crate::decode::cpu_reduce_bf16(&out, &padded, &weight, rp.tokens, mimo26_expert::slice::HIDDEN))
        }
    }

    /// A tiny 2-expert grouped image (each slice all-1.0 weights, scale 127).
    fn synthetic_grouped(n: usize) -> Vec<u8> {
        use mimo26_repack::geom::Proj;
        let mut bytes = vec![0u8; n * QUARTER_SLICE_BYTES];
        for e in 0..n {
            let base = e * QUARTER_SLICE_BYTES;
            for p in [Proj::Gate, Proj::Up, Proj::Down] {
                let poff = base + p.slice_payload_off();
                let plen = p.slice_payload_bytes();
                bytes[poff..poff + plen].fill(0x22); // E2M1 nibble 2 == 1.0
                let soff = base + p.slice_scale_off();
                let slen = p.slice_scale_bytes();
                bytes[soff..soff + slen].fill(127);
            }
        }
        bytes
    }

    fn request(id: u64) -> mimo26_wire::RequestFrame {
        mimo26_wire::RequestFrame {
            request_id: id,
            placement_version: 1,
            layer_id: 1,
            executor_id: 0,
            source_kind: SourceKind::Decode,
            token_position: 0,
            flags: 0,
            seq: 0,
            rows: vec![RowDescriptor {
                row_id: 0,
                source_kind: SourceKind::Decode,
                source_request_id: id,
                token_position: 0,
                route_offset: 0,
                route_count: 8,
            }],
            routes: (0..8)
                .map(|s| RouteEntry {
                    row_index: 0,
                    expert_id: 0,
                    gate_weight: if s == 0 { 1.0 } else { 0.0 },
                })
                .collect(),
            hidden_rows: vec![mimo26_wire::frame::HiddenRow {
                payload: vec![0x38u8; 4096],
                scales: vec![127u8; 128],
            }],
        }
    }

    /// The full server loop over the loopback: coordinator sends two requests,
    /// the server serves both, and the coordinator's receiver accepts the
    /// stamped returns in order.
    #[test]
    fn server_serves_two_requests_over_the_loopback() {
        let mut lb = crate::transport::Loopback::new();
        let mut coord_tx = mimo26_wire::l4::StreamSender::new(WireNaive::NONE);
        let mut coord_rx = StreamReceiver::new(WireNaive::NONE);
        let grouped = synthetic_grouped(2);
        let cpu = cpu_ffn(grouped.len());

        let mut server = SparkServer::new(&mut lb.b, WireNaive::NONE);

        // Send request 0.
        let r0 = coord_tx.encode_request(&request(0)).expect("encode req 0");
        lb.a.send(r0).expect("send req 0");
        // Send request 1 before stepping (buffered).
        let r1 = coord_tx.encode_request(&request(1)).expect("encode req 1");
        lb.a.send(r1).expect("send req 1");

        // Serve both.
        let seq0 = server.step(grouped.as_ptr(), &cpu).expect("step 0").expect("served");
        let seq1 = server.step(grouped.as_ptr(), &cpu).expect("step 1").expect("served");
        assert_eq!(seq0, 0);
        assert_eq!(seq1, 1);

        // Coordinator drains the two stamped returns in order.
        let ret0 = coord_rx.accept(&lb.a.recv().expect("ret 0")).expect("accept ret 0");
        let ret1 = coord_rx.accept(&lb.a.recv().expect("ret 1")).expect("accept ret 1");
        match (ret0, ret1) {
            (Frame::Return(r0), Frame::Return(r1)) => {
                assert_eq!(r0.request_id, 0);
                assert_eq!(r1.request_id, 1);
                assert_eq!(r0.seq, 0);
                assert_eq!(r1.seq, 1);
            }
            _ => panic!("expected return frames"),
        }
    }
}
