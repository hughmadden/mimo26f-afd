//! Explicit B1 prep cells, CPU execution / optional sm_89 compilation only.
//! Numerical mode: E-W4A8-v1. No runtime CUDA initialization.
//! Run through scripts/dev.sh test expert-unit --test b1_prep -- --ignored --nocapture.
use std::{path::PathBuf, process::Command, time::{SystemTime, UNIX_EPOCH}};

fn run(case: &str) -> PathBuf {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = PathBuf::from(std::env::var_os("MIMO26F_BUILD_ROOT")
        .expect("use scripts/dev.sh test expert-unit"));
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let stage = root.join(format!("b1-{case}-{}-{nonce}", std::process::id()));
    std::fs::create_dir_all(&stage).unwrap();
    let status = Command::new("python3")
        .arg(crate_dir.join("tests/b1/run.py"))
        .arg(case).arg(&stage)
        .env("CUDA_VISIBLE_DEVICES", "")
        .status().expect("start B1 CPU cell");
    assert!(status.success(), "B1 {case} failed; retained artifacts: {}", stage.display());
    stage
}

#[test]
#[ignore = "explicit CPU cell with real Rust rank images; no GPU"]
fn b1_prepare_roundtrip() { run("prepare"); }

#[test]
#[ignore = "explicit CPU bounds cell; optional compile-only CUDA"]
fn b1_stage_bounds() { run("staging"); }

#[test]
#[ignore = "explicit CPU plan replay; not a CUDA graph receipt"]
fn b1_plan_replay() { run("plan"); }

#[test]
#[ignore = "explicit independent quantizer CPU/reference and CUDA compile-only cell"]
fn b1_quantizer_v1() { run("quant"); }

// Compile the existing wire source unchanged into this integration harness:
// no Cargo.toml edits, copied wire implementation or replacement wire oracle.
#[allow(dead_code)] #[path = "../../mimo26-wire/src/bf16.rs"] pub mod bf16;
#[allow(dead_code)] #[path = "../../mimo26-wire/src/crc32c.rs"] pub mod crc32c;
#[allow(dead_code)] #[path = "../../mimo26-wire/src/error.rs"] pub mod error;
#[allow(dead_code)] #[path = "../../mimo26-wire/src/frame.rs"] pub mod frame;
#[allow(dead_code)] #[path = "../../mimo26-wire/src/l4.rs"] pub mod l4;
#[allow(dead_code)] #[path = "../../mimo26-wire/src/layout.rs"] pub mod layout;
#[allow(dead_code)] #[path = "../../mimo26-wire/src/naive.rs"] pub mod naive;
pub use error::WireError;

#[test]
#[ignore = "explicit CPU route scaffold -> real ReturnRow / CoordinatorSum"]
fn b1_route_weight_once() {
    let stage = run("route");
    let n = 3 * 4096;
    let mut reference = vec![0.0f32; n];
    let mut magnitude = vec![0.0f32; n];
    let mut exact_wire = vec![0.0f32; n];
    let mut decoded_magnitude = vec![0.0f64; n];
    let mut coordinator = l4::CoordinatorSum::new(3, 4096, naive::WireNaive::NONE);
    // CoordinatorSum currently adds in arrival order: integration must submit
    // ranks in this fixed order; this test does not assert it reorders arrivals.
    for rank in 0..4 {
        let raw = std::fs::read(stage.join(format!("rank{rank}.f32"))).unwrap();
        let encoded = std::fs::read(stage.join(format!("rank{rank}.bf16"))).unwrap();
        assert_eq!(raw.len(), n * 4);
        assert_eq!(encoded.len(), 3 * 8192);
        let partial: Vec<f32> = raw.chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
        let codes: Vec<u16> = encoded.chunks_exact(2)
            .map(|b| u16::from_le_bytes(b.try_into().unwrap())).collect();
        for i in 0..n {
            assert_eq!(codes[i], bf16::f32_to_bf16_rne(partial[i]), "wire RNE rank {rank} index {i}");
            reference[i] += partial[i];
            magnitude[i] += partial[i].abs();
            exact_wire[i] += bf16::bf16_to_f32(codes[i]);
            decoded_magnitude[i] += bf16::bf16_to_f32(codes[i]).abs() as f64;
        }
        let message = frame::ReturnFrame {
            request_id: 26, placement_version: 1, layer_id: 1, executor_id: rank,
            token_position: 0, status: layout::Status::Ok,
            flags: frame::FLAG_RETURN_REQUIRED, route_count: 24, seq: 0,
            rows: codes.chunks_exact(4096).map(|c| frame::ReturnRow { codes: c.to_vec() }).collect(),
        };
        let wire = frame::encode_return(&message, naive::WireNaive::NONE).unwrap();
        assert_eq!(wire.len(), layout::HEADER_LEN + 3 * 8192);
        let decoded = match frame::decode_frame(&wire, naive::WireNaive::NONE).unwrap() {
            frame::Frame::Return(value) => value,
            _ => panic!("wrong frame kind"),
        };
        coordinator.accumulate(&decoded).unwrap();
    }
    assert!(coordinator.is_complete());
    let mut adopted_bound_failures = 0;
    for (i, &got) in coordinator.result().unwrap().iter().enumerate() {
        assert_eq!(got.to_bits(), exact_wire[i].to_bits());
        let error = (got as f64 - reference[i] as f64).abs();
        let bound = 2.4e-6 + magnitude[i] as f64 / 512.0;
        if error > bound { adopted_bound_failures += 1; }
        // ADVISOR-I4 section 9 R8 explicitly supersedes the tighter R3 bound;
        // retain the original failure detector separately, not a silent change.
        let r8_bound = 2.4e-6 + magnitude[i] as f64 / 256.0
            + 3.0 * 2.0f64.powi(-24) * decoded_magnitude[i];
        assert!(error <= r8_bound, "R8 wire bound index {i}: {error} > {r8_bound}");
    }
    // Pin the old R3 counterexample alongside the explicitly ruled R8 bound.
    // The original failed receipt remains retained; this is CPU seam proof only.
    assert_eq!(adopted_bound_failures, 241);
    std::fs::write(stage.join("wire-bound.json"),
        "{\"scaffold\":\"PASS\",\"legacy_R3_wire_bound\":\"FAIL\",\"R8_wire_bound\":\"PASS\",\"violations\":241,\"positions\":12288,\"first_index\":8195,\"error\":0.001953125,\"bound\":0.00187160166015625}\n").unwrap();
    println!("B1 scaffold PASS: real wire types, 4 ranks x 3 rows, 8192 B per rank row, fixed order; E-W4A8-v1");
    println!("LEGACY R3 WIRE BOUND: FAIL on 241/12288 adversarial positions; corrected R8: PASS; CPU seam only");
}

#[test]
fn b1_wire_bound_counterexample() {
    let values = [-0.4443359375f32, -0.3076171875, -0.1708984375, -0.0341796875];
    // These partials are also realizable with normalized top8 weights: eight
    // equal raw route values, each weighted by exactly 1/8, sum to the same y_r.
    for &value in &values {
        let normalized = (0..8).fold(0.0f32, |sum, _| sum + 0.125f32 * value);
        assert_eq!(normalized.to_bits(), value.to_bits());
    }
    let reference: f32 = values.iter().sum();
    let wire: f32 = values.iter().map(|&v| bf16::bf16_to_f32(bf16::f32_to_bf16_rne(v))).sum();
    let error = (wire - reference).abs() as f64;
    let adopted_bound = 2.4e-6 + values.iter().map(|v| v.abs() as f64).sum::<f64>() / 512.0;
    assert_eq!(error, 0.001953125);
    assert!(error > adopted_bound, "counterexample unexpectedly disappeared");
}
