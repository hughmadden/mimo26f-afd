// Standalone linked integration probe: actual expert and wire libraries, no
// copied codec bodies and no Cargo.lock/dev-dependency changes. CPU only.
use mimo26_expert::{rank_sum::{rank_pre_sum, ROUTES}, NaiveBits};
use mimo26_wire::{bf16, decode_frame, encode_return, CoordinatorSum, Frame,
    ReturnFrame, ReturnRow, Status, WireNaive, FLAG_RETURN_REQUIRED, HIDDEN,
    HEADER_LEN, RETURN_ROW_BYTES};

fn presummed(route_values: [f32; ROUTES], naive: NaiveBits) -> Vec<f32> {
    let raw: Vec<_> = route_values.iter().flat_map(|&v| std::iter::repeat(v).take(HIDDEN)).collect();
    rank_pre_sum(&raw, &[0.125; ROUTES], 1, naive).unwrap()
}

fn decoded(rank: usize, row: &[f32], naive: WireNaive) -> ReturnFrame {
    let frame = ReturnFrame {
        request_id: 17, placement_version: 1, layer_id: 1, executor_id: rank as u64,
        token_position: 0, status: Status::Ok, flags: FLAG_RETURN_REQUIRED,
        route_count: ROUTES, seq: 0,
        rows: row.chunks_exact(HIDDEN).map(|chunk| ReturnRow {
            codes: chunk.iter().map(|&v| bf16::f32_to_bf16(v, naive)).collect()
        }).collect(),
    };
    let bytes = encode_return(&frame, WireNaive::NONE).unwrap();
    assert_eq!(row.len() % HIDDEN, 0);
    assert_eq!(bytes.len(), HEADER_LEN + row.len() / HIDDEN * RETURN_ROW_BYTES);
    match decode_frame(&bytes, WireNaive::NONE).unwrap() {
        Frame::Return(frame) => frame,
        _ => panic!("return decoded as request"),
    }
}

fn through_wire(rows: &[Vec<f32>; 4], order: [usize; 4], naive: WireNaive) -> Vec<f32> {
    let mut coordinator = CoordinatorSum::new(rows[0].len() / HIDDEN, HIDDEN, WireNaive::NONE);
    for rank in order { coordinator.accumulate(&decoded(rank, &rows[rank], naive)).unwrap(); }
    assert!(coordinator.is_complete());
    coordinator.result().unwrap().to_vec()
}

// Integration adapter: arrival is arbitrary, arithmetic order is not.
// Buffers actual decoded frames; this test does not claim the serving caller
// has already adopted the adapter.
fn buffered_wire(rows: &[Vec<f32>; 4], arrival: [usize; 4]) -> Result<Vec<f32>, &'static str> {
    let mut pending: [Option<ReturnFrame>; 4] = std::array::from_fn(|_| None);
    for rank in arrival {
        if rank >= 4 { return Err("rank out of range"); }
        let frame = decoded(rank, &rows[rank], WireNaive::NONE);
        let slot = frame.executor_id as usize;
        if pending[slot].is_some() { return Err("duplicate rank"); }
        pending[slot] = Some(frame);
    }
    let mut coordinator = CoordinatorSum::new(rows[0].len() / HIDDEN, HIDDEN, WireNaive::NONE);
    for frame in pending {
        coordinator.accumulate(&frame.ok_or("missing rank")?).map_err(|_| "coordinator refusal")?;
    }
    Ok(coordinator.result().map_err(|_| "incomplete ranks")?.to_vec())
}

fn selftest() {
    // A dyadic independent arithmetic oracle: sum (j+1)/8 = 4.5.
    let routes = [1.,2.,3.,4.,5.,6.,7.,8.];
    assert_eq!(presummed(routes, NaiveBits::NONE), vec![4.5; HIDDEN]);
    assert_ne!(presummed(routes, NaiveBits::ROUTE_WEIGHT_TWICE), vec![4.5; HIDDEN]);
    let rows = std::array::from_fn(|_| presummed(routes, NaiveBits::NONE));
    assert_eq!(through_wire(&rows, [0,1,2,3], WireNaive::NONE), vec![18.; HIDDEN]);
    println!("SEAM PASS weighted8-route pre-sum -> 8192-byte ReturnRow -> actual encode/decode -> fixed-rank CoordinatorSum");

    // Actual codec wrong implementation, not a test-side approximation.
    let rows = std::array::from_fn(|_| presummed([f32::from_bits(0x3f81_8000); ROUTES], NaiveBits::NONE));
    assert_ne!(through_wire(&rows, [0,1,2,3], WireNaive::NONE),
        through_wire(&rows, [0,1,2,3], WireNaive::of(WireNaive::BF16_TRUNCATE)));
    println!("SEAM NEGATIVE PASS route-weight-twice and actual BF16 truncation");

    // BF16 per-route returns lose the small residual before the rank pre-sum.
    let values = [8.03125,8.03125,-16.,0.,0.,0.,0.,0.];
    let correct = presummed(values, NaiveBits::NONE);
    assert_eq!(correct, vec![0.0078125; HIDDEN]);
    let wrong: f32 = values.iter().map(|&v| bf16::bf16_to_f32(bf16::f32_to_bf16_rne(v * 0.125))).sum();
    assert_eq!(wrong, 0.0);
    assert_ne!(wrong, bf16::bf16_to_f32(bf16::f32_to_bf16_rne(correct[0])));
    println!("SEAM NEGATIVE PASS round-each-route loses exact residual");

    // R8 (ADVISOR-I4:487): CoordinatorSum itself buffers the four rank rows
    // and sums in rank order 0→3, so the FP32 result is bit-reproducible for
    // every arrival order — the caller no longer has to enforce the order.
    let rows = [vec![16777216.; HIDDEN], vec![1.; HIDDEN], vec![-16777216.; HIDDEN], vec![1.; HIDDEN]];
    assert_eq!(through_wire(&rows, [0,1,2,3], WireNaive::NONE), vec![1.; HIDDEN]);
    assert_eq!(through_wire(&rows, [0,2,1,3], WireNaive::NONE), vec![1.; HIDDEN]);
    println!("SEAM ORDER PASS CoordinatorSum is rank-ordered: [0,1,2,3]=1 and [0,2,1,3]=1 (R8; no caller-side buffering required)");
    let mut permutations = 0;
    for a in 0..4 { for b in 0..4 { for c in 0..4 { for d in 0..4 {
        if (1usize << a) | (1 << b) | (1 << c) | (1 << d) != 15 { continue; }
        // Rank-ordering now holds directly through the codec, no adapter.
        assert_eq!(through_wire(&rows, [a,b,c,d], WireNaive::NONE), vec![1.; HIDDEN]);
        permutations += 1;
    }}}}
    assert_eq!(permutations, 24);
    assert!(buffered_wire(&rows, [0,1,2,2]).is_err());
    assert!(buffered_wire(&rows, [0,1,2,4]).is_err());
    println!("SEAM ORDER PASS: all 24 arrival permutations rank-ordered directly; duplicate/out-of-range ranks refused");
}

fn bound_probe() -> bool {
    // 1 + 2^-8 is exactly the midpoint between BF16 1 and 1+2^-7.
    // Eight dyadic weights yield this rank partial exactly, with zero compute error.
    let y = f32::from_bits(0x3f80_8000);
    assert_eq!(y, 1.00390625);
    let rows = std::array::from_fn(|_| presummed([y; ROUTES], NaiveBits::NONE));
    for row in &rows { assert_eq!(row, &vec![y; HIDDEN]); }
    let received = through_wire(&rows, [0,1,2,3], WireNaive::NONE);
    let reference = 4.0 * f64::from(y);
    let requested = 2.4e-6 + 4.0 * f64::from(y.abs()) * 2.0f64.powi(-9);
    let violations = received.iter().filter(|&&v| (f64::from(v)-reference).abs() > requested).count();
    assert_eq!(bf16::f32_to_bf16_rne(y), 0x3f80);
    assert_eq!(received, vec![4.; HIDDEN]);
    println!("COMPUTE ROW synthetic_exact_dyadic routes=8 ranks=4 hidden={HIDDEN} maxabs=0 bound=0.0000024 PASS");
    println!("WIRE ROW codec=DS41RTE3-v3 rank_partial={y:.12} prewire_sum={reference:.12} wire_sum={:.12} abs_error={:.12} legacy_bound={requested:.15} violations={violations}/{HIDDEN}", received[0], reference-f64::from(received[0]));
    if violations != 0 {
        println!("WIRE_BOUND_VIOLATION: legacy 2.4e-6 + sum_abs_rank*2^-9 does not bound ordinary BF16 RNE; historical failure retained");
    }
    violations != 0
}

fn corrected_bound_probe() {
    let y = f32::from_bits(0x3f80_8000);
    let rows = std::array::from_fn(|_| presummed([y; ROUTES], NaiveBits::NONE));
    let received = buffered_wire(&rows, [3,1,0,2]).unwrap();
    let decoded_abs = 4.0 * f64::from(bf16::bf16_to_f32(bf16::f32_to_bf16_rne(y)).abs());
    let reference = 4.0 * f64::from(y);
    let corrected = 2.4e-6 + reference.abs() * 2.0f64.powi(-8)
        + 3.0 * 2.0f64.powi(-24) * decoded_abs;
    let error = (reference - f64::from(received[0])).abs();
    for &v in &received { assert!((reference - f64::from(v)).abs() <= corrected); }
    println!("WIRE R8 PASS exact tie fixture: error={error:.15} corrected_bound={corrected:.15} coordinates={HIDDEN}; compute=0; decoded-rank FP32 sum term included");
}

fn load32(path: &std::path::Path, count: usize) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() != count * 4 { return Err("wrong f32 artifact length".into()); }
    let values: Vec<_> = bytes.chunks_exact(4).map(|v| f32::from_le_bytes(v.try_into().unwrap())).collect();
    if values.iter().any(|v| !v.is_finite()) { return Err("nonfinite f32 artifact".into()); }
    Ok(values)
}
fn load64(path: &std::path::Path, count: usize) -> Result<Vec<f64>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() != count * 8 { return Err("wrong f64 artifact length".into()); }
    let values: Vec<_> = bytes.chunks_exact(8).map(|v| f64::from_le_bytes(v.try_into().unwrap())).collect();
    if values.iter().any(|v| !v.is_finite()) { return Err("nonfinite f64 artifact".into()); }
    Ok(values)
}

fn real_probe(gpu: &std::path::Path, oracle: &std::path::Path, naive: u8) -> Result<bool, String> {
    let partial = load64(&oracle.join("wire-partial.f64"), 4 * 8 * 8 * HIDDEN)?;
    let full = load64(&oracle.join("wire-full.f64"), 8 * HIDDEN)?;
    let weights = load32(&oracle.join("wire-weights.f32"), 8)?;
    let mut total_coordinates = 0;
    for m in [1,2,4,8] {
        if naive != 0 && m != 1 { continue; }
        let prefix = if naive == 1 { "wire-naive-M" } else { "wire-M" };
        let raw = load32(&gpu.join(format!("{prefix}{m}.f32")), 4 * m * 8 * HIDDEN)?;
        let mut raw_max = 0.0f64;
        for rank in 0..4 { for token in 0..m { for slot in 0..8 { for h in 0..HIDDEN {
            let got = f64::from(raw[((rank * m + token) * 8 + slot) * HIDDEN + h]);
            let want = partial[((rank * 8 + token) * 8 + slot) * HIDDEN + h];
            let error = (got - want).abs(); raw_max = raw_max.max(error);
            if error > 1e-5 + 1e-5 * want.abs() {
                println!("ORACLE_MISMATCH real GPU FC2 M={m} rank={rank} token={token} route={slot} h={h} error={error}");
                return Ok(false);
            }
        }}}}
        let route_weights: Vec<_> = (0..m).flat_map(|_| weights.iter().copied()).collect();
        let rows: [Vec<f32>; 4] = std::array::from_fn(|rank| {
            let begin = rank * m * 8 * HIDDEN;
            let flags = if naive == 2 { NaiveBits::ROUTE_WEIGHT_TWICE } else { NaiveBits::NONE };
            rank_pre_sum(&raw[begin..begin + m * 8 * HIDDEN], &route_weights, m, flags).unwrap()
        });
        let returned = buffered_wire(&rows, [3,1,0,2]).map_err(str::to_string)?;
        let decoded_rows: [Vec<f32>; 4] = std::array::from_fn(|rank| {
            decoded(rank, &rows[rank], WireNaive::NONE).rows.iter()
                .flat_map(|row| row.codes.iter().map(|&v| bf16::bf16_to_f32(v))).collect()
        });
        let mut compute_max = 0.0f64; let mut wire_max = 0.0f64; let mut max_ratio = 0.0f64;
        for i in 0..m * HIDDEN {
            // Compute term excludes the separately budgeted coordinator FP32 sum.
            let mut prewire = 0.0f64;
            let mut rank_abs = 0.0f64; let mut decoded_abs = 0.0f64;
            for rank in 0..4 {
                prewire += f64::from(rows[rank][i]); rank_abs += f64::from(rows[rank][i]).abs();
                decoded_abs += f64::from(decoded_rows[rank][i]).abs();
            }
            let compute = (prewire - full[i]).abs();
            let wire = (f64::from(returned[i]) - full[i]).abs();
            let bound = 2.4e-6 + rank_abs * 2.0f64.powi(-8) + 3.0 * 2.0f64.powi(-24) * decoded_abs;
            compute_max = compute_max.max(compute); wire_max = wire_max.max(wire); max_ratio = max_ratio.max(wire / bound);
            if compute > 2.4e-6 || wire > bound {
                println!("ORACLE_MISMATCH real wire M={m} coordinate={i} compute={compute} wire={wire} R8_bound={bound}");
                return Ok(false);
            }
        }
        total_coordinates += m * HIDDEN;
        println!("ARTIFACT COMPUTE PASS lattice=E-FP32 M={m} raw_FC2_maxabs={raw_max:.12e} weighted_TP4_maxabs={compute_max:.12e} bound=2.4e-6");
        println!("ARTIFACT WIRE PASS codec=DS41RTE3-v3 M={m} routes=8 ranks=4 payload_per_token_rank=8192 maxabs={wire_max:.12e} max_R8_ratio={max_ratio:.9} buffered_rank_order=0,1,2,3");
    }
    println!("ARTIFACT SEAM PASS checked_coordinates={total_coordinates}; per-route f32 artifacts -> CPU rank pre-sum -> actual BF16 frames -> buffered CoordinatorSum; source provenance supplied by runner; not GPU route-reducer qualification");
    Ok(true)
}

fn main() {
    match std::env::args().nth(1).as_deref() {
        Some("--selftest") => { selftest(); assert!(bound_probe(), "known exact counterexample disappeared"); corrected_bound_probe(); },
        Some("--legacy-bound") => { if bound_probe() { std::process::exit(3); } },
        Some("--strict-bound") => corrected_bound_probe(),
        Some("--real") => {
            let args: Vec<_> = std::env::args().collect();
            if args.len() != 5 || !matches!(args[4].as_str(), "0" | "1" | "2") { std::process::exit(2); }
            match real_probe(std::path::Path::new(&args[2]), std::path::Path::new(&args[3]), args[4].parse().unwrap()) {
                Ok(true) => {}, Ok(false) => std::process::exit(3),
                Err(error) => { eprintln!("INPUT_FAILURE: {error}"); std::process::exit(2); }
            }
        },
        _ => { eprintln!("expected --selftest, --legacy-bound or --strict-bound"); std::process::exit(2); }
    }
}
