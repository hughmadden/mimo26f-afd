//! `mimo26-spark` — the Spark expert-rank serving daemon (I5 go-window).
//!
//! Bring-up order (go-window `runs/20260924-i5/packets/go-window-first-tokens.md`):
//! step 0 preflight (slice staging + sm_121a build + boot identity readback),
//! step 1 bring-up (listen on the TCP transport). This binary:
//!
//! 1. reads the resident slice directory's manifest and verifies every slice
//!    (sha256 + size + geometry) — the boot identity readback; refuses to serve
//!    on any mismatch ([`mimo26_spark::boot::readback`]);
//! 2. indexes this rank's slice files from the manifest
//!    ([`mimo26_spark::resident::load_manifest`]) and reads/uploads the
//!    grouped image on demand per layer;
//! 3. listens on the TCP transport and serves DS41RTE3 v3 request frames with
//!    the full L4 ladder + the live 8 KB-per-token return assertion, running the
//!    expert FFN on the device (R18c mixed-M decode for M<=8, frozen per-M
//!    prefill otherwise) via [`mimo26_spark::decode`].
//!
//! Args: `--rank <0..3> --dir <slice-dir> --listen <addr>`.

use std::io::Write;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use mimo26_repack::geom;
use mimo26_spark::{boot, resident, serve, timeline};
use mimo26_spark::decode;
#[cfg(feature = "cuda")]
use mimo26_spark::device::DeviceBuffer;
#[cfg(feature = "cuda")]
use mimo26_spark::ffi;
use mimo26_spark::transport::{ByteTransport, TcpTransport};
use mimo26_wire::frame::{Frame, ReturnFrame, ReturnRow};
use mimo26_wire::layout::Status;
use mimo26_wire::l4::{StreamReceiver, StreamSender};
use mimo26_wire::WireNaive;

struct Args {
    rank: usize,
    dir: PathBuf,
    listen: String,
}

fn parse_args() -> Result<Args, String> {
    let mut rank: Option<usize> = None;
    let mut dir: Option<PathBuf> = None;
    let mut listen = "0.0.0.0:8600".to_string();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--rank" => {
                let v = it.next().ok_or("--rank needs a value")?;
                rank = Some(v.parse().map_err(|_| format!("bad rank {v}"))?);
            }
            "--dir" => {
                let v = it.next().ok_or("--dir needs a value")?;
                dir = Some(PathBuf::from(v));
            }
            "--listen" => {
                let v = it.next().ok_or("--listen needs a value")?;
                listen = v;
            }
            other => return Err(format!("unknown arg {other}")),
        }
    }
    let rank = rank.ok_or("missing --rank")?;
    let dir = dir.ok_or("missing --dir")?;
    geom::check_rank(rank).map_err(|e| e.to_string())?;
    Ok(Args { rank, dir, listen })
}

/// FFN (CPU merge gate, no `cuda`): the pure-Rust oracle stands in so the daemon
/// binary still links and the wire/serve path stays CPU-testable. Returns the
/// per-token BF16 return rows (FFN + reduce + RNE BF16).
#[cfg(not(feature = "cuda"))]
fn ffn_cpu(
    grouped: *const u8,
    hidden: &[f32],
    rp: &mimo26_spark::route::RoutePlan,
) -> Result<Vec<u16>, String> {
    // SAFETY: the CPU-only daemon passes the host grouped image pointer (256
    // quarter slices resident); it is valid for the full residency length.
    let len = geom::EXPERTS_PER_LAYER * geom::QUARTER_SLICE_BYTES;
    let grouped = unsafe { std::slice::from_raw_parts(grouped, len) };
    let x = rp.replicate_x(hidden)?;
    let out = mimo26_expert::grouped::expert_ffn_self_contained(
        grouped,
        &x,
        &rp.plan,
        mimo26_expert::NaiveBits::NONE,
    )
    .map(|o| o.data)
    .map_err(|e| e.to_string())?;
    let (padded, weight) = rp.token_major_routes();
    Ok(decode::cpu_reduce_bf16(&out, &padded, &weight, rp.tokens, mimo26_expert::slice::HIDDEN))
}

/// A `Status::Error` return for a failed request: one zero BF16 row per token,
/// so the coordinator decodes it and raises `SparkReportedError` (fail the
/// request) instead of seeing a dropped connection.
fn error_return(req: &mimo26_wire::RequestFrame) -> ReturnFrame {
    let meta = serve::return_meta(req);
    error_return_for(&meta, req.rows.len())
}

/// An error Status return of `rows` rows for the request `meta` identifies.
fn error_return_for(meta: &ReturnFrame, rows: usize) -> ReturnFrame {
    let req = meta;
    let rows = (0..rows).map(|_| ReturnRow { codes: vec![0u16; mimo26_expert::slice::HIDDEN] }).collect();
    ReturnFrame {
        request_id: req.request_id,
        placement_version: req.placement_version,
        layer_id: req.layer_id,
        executor_id: req.executor_id,
        token_position: req.token_position,
        status: Status::Error,
        flags: 0,
        route_count: 8,
        seq: 0,
        rows,
    }
}

/// Prepare `layer`'s B1 pool on first touch (grouped image read, upload,
/// B1P1 prepare; the canonical device copy is freed). Returns the load time.
#[cfg(feature = "cuda")]
fn ensure_b1_layer(
    b1_layers: &mut [Option<mimo26_spark::b1::Layer>],
    resident: &resident::Resident,
    layer: usize,
    rank: u32,
) -> Result<f64, String> {
    if b1_layers[layer].is_some() {
        return Ok(0.0);
    }
    let t = Instant::now();
    let prepared = resident
        .grouped_image(layer, geom::EXPERTS_PER_LAYER)
        .map_err(|e| e.to_string())
        .and_then(|img| decode::upload_grouped(&img))
        .and_then(|d| mimo26_spark::b1::Layer::new(d.as_ptr() as *const u8, d.bytes(), rank))?;
    if !prepared.all_scales_normal() {
        eprintln!("b1 L{layer}: exceptional weight scales present; FC1/FC2 run the checked kernels");
    }
    b1_layers[layer] = Some(prepared);
    Ok(t.elapsed().as_secs_f64() * 1e3)
}

/// CUDA page-lock of the RDMA receive ring for one connection, released before
/// the transport frees the ring (declare after it: locals drop in reverse).
#[cfg(feature = "cuda")]
struct RingRegistration(Option<*mut u8>);

#[cfg(feature = "cuda")]
impl Drop for RingRegistration {
    fn drop(&mut self) {
        if let Some(p) = self.0 {
            // SAFETY: registered by this guard, still allocated (the transport outlives it).
            unsafe { mimo26_spark::cuda::cudaHostUnregister(p as *mut core::ffi::c_void) };
        }
    }
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("usage: mimo26-spark --rank <0..3> --dir <slice-dir> --listen <addr>");
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };

    // Step 0c: boot identity readback — refuse to serve on any mismatch.
    let receipt = match boot::readback(&args.dir) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("boot identity readback FAILED: {e}");
            std::process::exit(3);
        }
    };
    println!("boot readback: {}", receipt.summary);

    // Index this rank's resident slice files (no 40 GB read — the boot readback
    // verified them from disk; the grouped image is read on demand + uploaded).
    let resident = match resident::Resident::load_manifest(&args.dir, args.rank) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("load_manifest FAILED: {e}");
            std::process::exit(4);
        }
    };
    println!(
        "resident: rank={} slices={}",
        args.rank,
        resident.slices
    );

    // Startup AOT self-check: the manifest bake must match the live device.
    #[cfg(feature = "cuda")]
    if let Err(e) = decode::check_aot() {
        eprintln!("{e}");
        std::process::exit(6);
    }

    // Lazy per-layer grouped image cache (256 experts x 3,342,336 B each). On
    // the device the image is uploaded once per layer and the host copy freed;
    // on the CPU merge gate the host bytes are retained.
    #[cfg(feature = "cuda")]
    let mut device_cache: Vec<Option<DeviceBuffer>> = (0..=geom::MOE_LAYERS).map(|_| None).collect();
    #[cfg(not(feature = "cuda"))]
    let mut device_cache: Vec<Option<Vec<u8>>> = (0..=geom::MOE_LAYERS).map(|_| None).collect();

    // Perf reset R3: the B1 (E-W4A8-v1 tensor-core) expert path, opt-in with
    // MIMO26_SPARK_B1=1. Each layer's canonical image is prepared once into a B1P1
    // pool and the canonical device copy freed; requests skip the host decode.
    #[cfg(feature = "cuda")]
    let use_b1 = std::env::var("MIMO26_SPARK_B1").map(|v| v != "0").unwrap_or(false);
    #[cfg(feature = "cuda")]
    let mut b1_layers: Vec<Option<mimo26_spark::b1::Layer>> = (0..=geom::MOE_LAYERS).map(|_| None).collect();
    #[cfg(feature = "cuda")]
    let mut b1_scratch = if use_b1 {
        match mimo26_spark::b1::Scratch::new() {
            Ok(s) => Some(s),
            Err(e) => {
                eprintln!("b1 scratch FAILED: {e}");
                std::process::exit(7);
            }
        }
    } else {
        None
    };
    #[cfg(feature = "cuda")]
    println!("expert path: {}", if use_b1 { "B1 E-W4A8-v1 (tensor core)" } else { "B2 E-FP32" });

    // FFN closure: decode (R18c mixed-M) for M<=8, prefill (frozen per-M)
    // otherwise, then the device route reduce (multi-chunk). The grouped image is
    // device-resident (uploaded once per layer by the serve loop, host copy freed)
    // — the kernel reads cudaMalloc memory at full bandwidth (zero-copy was 3x
    // slower). The padded x is gathered on-device from the token-major hidden
    // (no pageable host replicate); x/scratch/out are pooled in a `Scratch`.
    #[cfg(feature = "cuda")]
    let ffn = {
        use std::cell::RefCell;
        let scratch = RefCell::new(decode::Scratch::new());
        move |d_grouped: *const u8,
              hidden: &[f32],
              rp: &mimo26_spark::route::RoutePlan|
              -> Result<Vec<u16>, String> {
            let mut scratch = scratch.borrow_mut();
            let mixed = rp.plan.total_tokens() <= 8;
            match decode::ffn_route_reduce(
                d_grouped, hidden, rp, ffi::OUT_F32, mixed, &mut scratch,
            ) {
                Ok(bf16) => Ok(bf16),
                Err(_) => {
                    // Fallback: host replicate + single-chunk device FFN + CPU reduce.
                    let x = rp.replicate_x(hidden)?;
                    let out = if mixed {
                        decode::decode_ffn(d_grouped, &x, &rp.plan, geom::EXPERTS_PER_LAYER, ffi::OUT_F32, &mut scratch)?
                    } else {
                        decode::prefill_ffn(d_grouped, &x, &rp.plan, geom::EXPERTS_PER_LAYER, ffi::OUT_F32, &mut scratch)?
                    };
                    let (padded, weight) = rp.token_major_routes();
                    Ok(decode::cpu_reduce_bf16(&out, &padded, &weight, rp.tokens, mimo26_expert::slice::HIDDEN))
                }
            }
        }
    };
    #[cfg(not(feature = "cuda"))]
    let ffn = ffn_cpu;

    let listener = match std::net::TcpListener::bind(&args.listen) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("bind {} failed: {e}", args.listen);
            std::process::exit(5);
        }
    };
    let addr = listener.local_addr().expect("local addr");
    println!("listening on {addr} (rank {})", args.rank);
    let _ = std::io::stdout().flush();

    let naive = WireNaive::NONE;
    // I5-R16: dump the first layer-1 request frame (raw wire bytes) so the real
    // skewed plan can be replayed offline. Env flag, off by default.
    let dump_frame = std::env::var("MIMO26_SPARK_DUMP_FRAME").is_ok();
    let mut dumped = false;
    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(e) => {
                eprintln!("accept failed: {e}");
                continue;
            }
        };
        // Inference runs only on the RDMA fabric: a coordinator that dialled this
        // daemon over a LAN/10G address (the API path) is refused, TCP or RDMA.
        match stream.local_addr().map_err(|e| e.to_string()).and_then(|a| mimo26_rdma::fabric_port(a.ip())) {
            Ok(Some((dev, port, _, gbps))) => eprintln!("connection on fabric {dev} port {port} at {gbps} Gb/s"),
            Ok(None) => eprintln!("connection on loopback or MIMO26_WIRE_ALLOW_LAN=1, fabric check skipped"),
            Err(e) => {
                eprintln!("refused connection from {:?}: {e}", stream.peer_addr());
                continue;
            }
        }
        // Perf reset R2 part 2: a coordinator that opens with the RDMA handshake
        // gets an RC queue pair; anything else stays on TCP.
        let rdma = match mimo26_spark::transport::RdmaTransport::accept(&stream) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("rdma handshake failed: {e}");
                continue;
            }
        };
        let mut transport: Box<dyn ByteTransport> = match rdma {
            Some(t) => Box::new(t),
            None => match TcpTransport::from_stream(stream) {
                Ok(t) => Box::new(t),
                Err(e) => {
                    eprintln!("transport setup failed: {e}");
                    continue;
                }
            },
        };
        // Perf reset (Spark zero-copy receive): with B1 over RDMA, requests are
        // served where the NIC lands them; page-lock the ring so the device
        // copies of the hidden rows run as DMA.
        #[cfg(feature = "cuda")]
        let zero_copy = use_b1 && transport.in_place_recv();
        #[cfg(feature = "cuda")]
        let _ring_reg = RingRegistration(if zero_copy {
            transport.recv_ring().and_then(|(p, n)| {
                // SAFETY: the live receive ring of this connection's transport.
                let rc = unsafe { mimo26_spark::cuda::cudaHostRegister(p as *mut core::ffi::c_void, n, 0) };
                if rc == mimo26_spark::cuda::SUCCESS {
                    Some(p)
                } else {
                    eprintln!("cudaHostRegister of the receive ring failed ({rc}); copies go unpinned");
                    None
                }
            })
        } else {
            None
        });
        let mut rx = StreamReceiver::new(naive);
        let mut tx = StreamSender::new(naive);
        let mut window = timeline::Window::default();
        'conn: loop {
            #[cfg(feature = "cuda")]
            if zero_copy {
                let recv_start = Instant::now();
                let Some((ptr, len)) = transport.recv_slot() else {
                    break 'conn; // peer closed
                };
                let recv_ms = recv_start.elapsed().as_secs_f64() * 1e3;
                // SAFETY: the slot stays valid, and the NIC cannot write it, until
                // release_slot below; `bytes` is not used after that.
                let bytes: &[u8] = unsafe { std::slice::from_raw_parts(ptr, len) };
                let view = match mimo26_wire::frame::RequestView::parse(bytes, naive) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("L4 accept failed: {e}");
                        break;
                    }
                };
                if let Err(e) = rx.accept_seq(view.seq) {
                    eprintln!("L4 accept failed: {e}");
                    break;
                }
                timeline::tl("crc_done", Some(view.layer_id));
                let layer = view.layer_id as usize;
                if layer == 0 || layer > geom::MOE_LAYERS {
                    eprintln!("request layer {layer} out of MoE range 1..={}", geom::MOE_LAYERS);
                    break;
                }
                if dump_frame && layer == 1 && !dumped {
                    let path = "/var/tmp/mimo26f-kernel/req-layer1.bin";
                    match std::fs::write(path, bytes) {
                        Ok(()) => {
                            eprintln!("dumped layer-1 request frame ({}) B", bytes.len());
                            dumped = true;
                        }
                        Err(e) => eprintln!("dump frame failed: {e}"),
                    }
                }
                let load_ms = match ensure_b1_layer(&mut b1_layers, &resident, layer, args.rank as u32) {
                    Ok(ms) => ms,
                    Err(e) => {
                        eprintln!("b1 prepare L{layer} failed: {e}");
                        break;
                    }
                };
                let tokens = view.rows;
                let meta = serve::return_meta_view(&view);
                let len = mimo26_wire::HEADER_LEN + tokens * mimo26_wire::RETURN_ROW_BYTES;
                let mut timings = serve::Timings::default();
                let computed = {
                    let l = b1_layers[layer].as_ref().expect("prepared b1 layer");
                    let scratch = b1_scratch.as_mut().expect("b1 scratch");
                    match transport.send_buffer() {
                        None => Err("no send buffer".to_string()),
                        Some(buf) if len > buf.len() => Err(format!("return frame {len} B exceeds the send buffer")),
                        Some(buf) => {
                            let body = &mut buf[mimo26_wire::HEADER_LEN..len];
                            // SAFETY: the registered buffer halves are page-aligned and
                            // HEADER_LEN (128) is even, so the body is 2-byte aligned.
                            let body16 = unsafe {
                                std::slice::from_raw_parts_mut(body.as_mut_ptr() as *mut u16, tokens * mimo26_wire::HIDDEN)
                            };
                            serve::serve_b1_view(&view, l, scratch, &mut timings, body16)
                        }
                    }
                };
                // The FFN synchronized its stream: the slot's rows are on the device.
                // Re-post it before this return goes out, so the coordinator's next
                // request on this lane always finds a posted slot. `view` borrows
                // the slot and ends here.
                drop(view);
                if let Err(e) = transport.release_slot() {
                    eprintln!("rdma: re-post failed: {e}");
                    break;
                }
                if let Err(e) = computed {
                    eprintln!("serve L{layer} failed: {e}");
                    if let Ok(stamped) = tx.encode_return(&error_return_for(&meta, tokens)) {
                        let _ = transport.send(stamped);
                    }
                    break;
                }
                let seq = tx.take_seq();
                match mimo26_wire::frame::return_header_seq(&meta, tokens, seq, naive) {
                    Ok(h) => {
                        let Some(buf) = transport.send_buffer() else {
                            eprintln!("rdma: send buffer lost");
                            break;
                        };
                        buf[..mimo26_wire::HEADER_LEN].copy_from_slice(&h);
                        mimo26_wire::frame::seal_in_place(&mut buf[..len], naive);
                    }
                    Err(e) => {
                        eprintln!("L4 header failed: {e}");
                        break;
                    }
                }
                timeline::tl("seal_done", Some(meta.layer_id));
                let send_start = Instant::now();
                if let Err(e) = transport.send_in_place(len) {
                    eprintln!("send failed: {e}");
                    break;
                }
                let send_ms = send_start.elapsed().as_secs_f64() * 1e3;
                window.add(tokens, timings.ffn_ms);
                if timeline::trace() {
                    let epoch_ms = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0);
                    eprintln!(
                        "timing t={epoch_ms} layer={layer} tokens={tokens} recv={recv_ms:.3} load={load_ms:.3} plan={:.3} ffn={:.3} reduce={:.3} send={send_ms:.3} ms",
                        timings.plan_ms, timings.ffn_ms, timings.reduce_ms,
                    );
                }
                continue;
            }
            // recv: a blocking read of one complete request frame (the header is
            // read first, then the body sized by wire_bytes, directly into the
            // frame buffer — no yield busy-poll between partial reads).
            let recv_start = Instant::now();
            let bytes = match transport.recv() {
                Some(b) => b,
                None => break 'conn, // peer closed the connection
            };
            let recv_ms = recv_start.elapsed().as_secs_f64() * 1e3;

            let frame = match rx.accept(&bytes) {
                Ok(f) => f,
                Err(e) => {
                    eprintln!("L4 accept failed: {e}");
                    break;
                }
            };
            let req = match frame {
                Frame::Request(r) => r,
                Frame::Return(_) => {
                    eprintln!("L4: unexpected return frame");
                    break;
                }
            };
            timeline::tl("crc_done", Some(req.layer_id));
            let layer = req.layer_id as usize;
            if layer == 0 || layer > geom::MOE_LAYERS {
                eprintln!("request layer {layer} out of MoE range 1..={}", geom::MOE_LAYERS);
                break;
            }
            if dump_frame && layer == 1 && !dumped {
                let path = "/var/tmp/mimo26f-kernel/req-layer1.bin";
                if let Err(e) = std::fs::write(path, &bytes) {
                    eprintln!("dump frame failed: {e}");
                } else {
                    eprintln!("dumped layer-1 request frame ({}) B", bytes.len());
                    dumped = true;
                }
            }
            // lazy grouped-image load + device upload: first touch per layer
            // only. The host copy is freed immediately after upload so the
            // footprint stays device-resident (~41 GB), not 80 GB.
            let mut load_ms = 0.0f64;
            #[cfg(feature = "cuda")]
            if use_b1 {
                match ensure_b1_layer(&mut b1_layers, &resident, layer, args.rank as u32) {
                    Ok(ms) => load_ms = ms,
                    Err(e) => {
                        eprintln!("b1 prepare L{layer} failed: {e}");
                        break;
                    }
                }
            }
            #[cfg(feature = "cuda")]
            let need_b2 = !use_b1;
            #[cfg(not(feature = "cuda"))]
            let need_b2 = true;
            if need_b2 && device_cache[layer].is_none() {
                let t = Instant::now();
                match resident.grouped_image(layer, geom::EXPERTS_PER_LAYER) {
                    Ok(img) => {
                        #[cfg(feature = "cuda")]
                        match decode::upload_grouped(&img) {
                            Ok(d) => device_cache[layer] = Some(d),
                            Err(e) => {
                                eprintln!("upload_grouped L{layer} failed: {e}");
                                break;
                            }
                        }
                        #[cfg(not(feature = "cuda"))]
                        {
                            device_cache[layer] = Some(img);
                        }
                    }
                    Err(e) => {
                        eprintln!("grouped_image L{layer} failed: {e}");
                        break;
                    }
                }
                load_ms = t.elapsed().as_secs_f64() * 1e3;
            }
            let mut timings = serve::Timings::default();
            // Perf reset R2 zero-copy return: with B1 and an in-place transport
            // (RDMA), the partial lands directly in the registered send buffer and
            // only the 128-B header is written; one SEND, no frame copies.
            #[cfg(feature = "cuda")]
            if use_b1 && transport.send_buffer().is_some() {
                let tokens = req.rows.len();
                let len = mimo26_wire::HEADER_LEN + tokens * mimo26_wire::RETURN_ROW_BYTES;
                let l = b1_layers[layer].as_ref().expect("prepared b1 layer");
                let scratch = b1_scratch.as_mut().expect("b1 scratch");
                let computed = {
                    let buf = transport.send_buffer().expect("send buffer");
                    if len > buf.len() {
                        Err(format!("return frame {len} B exceeds the send buffer"))
                    } else {
                        let body = &mut buf[mimo26_wire::HEADER_LEN..len];
                        // SAFETY: the registered buffer is page-aligned and HEADER_LEN
                        // (128) is even, so the body is 2-byte aligned; exact length.
                        let body16 = unsafe {
                            std::slice::from_raw_parts_mut(body.as_mut_ptr() as *mut u16, tokens * mimo26_wire::HIDDEN)
                        };
                        serve::serve_b1_into(&req, l, scratch, &mut timings, body16)
                    }
                };
                if let Err(e) = computed {
                    eprintln!("serve L{layer} failed: {e}");
                    if let Ok(stamped) = tx.encode_return(&error_return(&req)) {
                        let _ = transport.send(stamped);
                    }
                    break;
                }
                let seq = tx.take_seq();
                match mimo26_wire::frame::return_header_seq(&serve::return_meta(&req), tokens, seq, naive) {
                    Ok(h) => {
                        let buf = transport.send_buffer().expect("send buffer");
                        buf[..mimo26_wire::HEADER_LEN].copy_from_slice(&h);
                        mimo26_wire::frame::seal_in_place(&mut buf[..len], naive);
                    }
                    Err(e) => {
                        eprintln!("L4 header failed: {e}");
                        break;
                    }
                }
                timeline::tl("seal_done", Some(req.layer_id));
                let send_start = Instant::now();
                if let Err(e) = transport.send_in_place(len) {
                    eprintln!("send failed: {e}");
                    break;
                }
                let send_ms = send_start.elapsed().as_secs_f64() * 1e3;
                window.add(tokens, timings.ffn_ms);
                if timeline::trace() {
                    let epoch_ms = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0);
                    eprintln!(
                        "timing t={epoch_ms} layer={layer} tokens={tokens} recv={recv_ms:.3} load={load_ms:.3} plan={:.3} ffn={:.3} reduce={:.3} send={send_ms:.3} ms",
                        timings.plan_ms, timings.ffn_ms, timings.reduce_ms,
                    );
                }
                continue;
            }
            #[cfg(feature = "cuda")]
            let served = if use_b1 {
                let l = b1_layers[layer].as_ref().expect("prepared b1 layer");
                serve::serve_return_b1(&req, l, b1_scratch.as_mut().expect("b1 scratch"), &mut timings)
            } else {
                let grouped: *const u8 =
                    device_cache[layer].as_ref().expect("cached grouped image").as_ptr() as *const u8;
                serve::serve_return(&req, grouped, &ffn, naive, &mut timings)
            };
            #[cfg(not(feature = "cuda"))]
            let served = {
                let grouped: *const u8 = device_cache[layer].as_ref().expect("cached grouped image").as_ptr();
                serve::serve_return(&req, grouped, &ffn, naive, &mut timings)
            };
            let ret = match served {
                Ok(r) => r,
                Err(e) => {
                    // I5-R6 §1: fail the request, never the rank. Send an error
                    // Status return (the coordinator surfaces it, not Broken pipe),
                    // then close this connection and keep listening.
                    eprintln!("serve L{layer} failed: {e}");
                    if let Ok(stamped) = tx.encode_return(&error_return(&req)) {
                        let _ = transport.send(stamped);
                    }
                    break;
                }
            };
            let stamped = match tx.encode_return(&ret) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("L4 encode failed: {e}");
                    break;
                }
            };
            timeline::tl("seal_done", Some(req.layer_id));
            // Live 8 KB-per-Spark-per-token assertion (the A6 return contract).
            assert_eq!(
                stamped.len(),
                mimo26_wire::HEADER_LEN + ret.rows.len() * mimo26_wire::RETURN_ROW_BYTES,
                "return frame is not 8,192 B per token"
            );
            let send_start = Instant::now();
            timeline::tl("send_start", Some(req.layer_id));
            if let Err(e) = transport.send(stamped) {
                eprintln!("send failed: {e}");
                break;
            }
            timeline::tl("send_end", Some(req.layer_id));
            let send_ms = send_start.elapsed().as_secs_f64() * 1e3;

            // Per-request timing line (go-window step 4: the Spark half of the
            // prefill budget), epoch-ms timestamped for coordinator correlation.
            window.add(req.rows.len(), timings.ffn_ms);
            if timeline::trace() {
                let epoch_ms = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_millis())
                    .unwrap_or(0);
                eprintln!(
                    "timing t={epoch_ms} layer={layer} tokens={} recv={recv_ms:.3} load={load_ms:.3} plan={:.3} ffn={:.3} reduce={:.3} send={send_ms:.3} ms",
                    req.rows.len(),
                    timings.plan_ms,
                    timings.ffn_ms,
                    timings.reduce_ms,
                );
            }
        }
    }
}
