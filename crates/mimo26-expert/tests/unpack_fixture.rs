//! Honest separation: portable synthetic unit tests vs ignored real-byte CPU proof.
//! The old WIP reconstructed bytes from expected answers; that is NOT an oracle.
//! Real tests read raw checkpoint bytes with both SHA256s verified by a stdlib
//! Python harness. A CPU pass does not substitute for Spark hardware unpack proof.
mod common;
use common::*;
use mimo26_expert::fixture::{self, FixtureBlock};
use mimo26_expert::mxfp4;
use mimo26_expert::slice::Proj;
use mimo26_expert::NaiveBits;

fn synthetic() -> (FixtureBlock, Vec<u8>, Vec<u8>) {
    let block = FixtureBlock {
        name: "synthetic.gate_proj".into(), shape: [1, 64],
        weight_sha256: String::new(), scale_sha256: String::new(),
        positions: vec![(0, 0), (0, 1), (0, 32), (0, 33)],
        // Hand-derived: 0x71 => 0.5,6; second block scale 128 doubles them.
        expected_f32_bits: vec![0x3f000000, 0x40c00000, 0x3f800000, 0x41400000],
        scale_byte_min: 127, scale_byte_max: 128, scale_byte_255_count: 0, saturated_count: 0,
    };
    (block, vec![0x71; 32], vec![127, 128])
}

#[test]
fn unpack_matches_hand_derived_bits() {
    let (b, w, s) = synthetic();
    fixture::check_block(&b, &w, &s, naive_env()).unwrap(); // NEGATIVE
}

#[test]
fn check_block_accepts_and_rejects() {
    let (b, mut w, s) = synthetic();
    fixture::check_block(&b, &w, &s, NaiveBits::NONE).unwrap();
    w[0] ^= 15;
    let err = fixture::check_block(&b, &w, &s, NaiveBits::NONE).unwrap_err();
    assert!(err.to_string().contains("sample 0"));
    assert!(fixture::check_block(&b, &w[..31], &s, NaiveBits::NONE).is_err());
}

#[test]
fn count_matches_detects_each_mutation() {
    let (b, w, s) = synthetic();
    assert_eq!(fixture::count_matches(&b, &w, &s, NaiveBits::NONE), 4);
    for flag in [NaiveBits::NIBBLE_SWAP, NaiveBits::SCALE_OFF_BY_ONE] {
        assert!(fixture::count_matches(&b, &w, &s, flag) < 4);
    }
}

#[test]
fn fixture_blocks_are_the_real_checkpoint_geometry() {
    let fx = load_fixture().unwrap();
    for b in &fx.blocks {
        let p = b.proj().unwrap();
        assert_eq!(b.shape, [p.out_rows(), p.in_cols()]);
        assert!(fx.layers.contains(&b.layer()));
        assert!(fx.experts.contains(&b.expert()));
    }
    for p in Proj::ALL { assert_eq!(fx.blocks_for(p).len(), 9); }
}

#[test]
fn e8m0_mapping_is_pinned_on_the_real_scale_range() {
    for b in load_fixture().unwrap().blocks {
        assert!(b.scale_byte_min >= 118 && b.scale_byte_max <= 125);
        for byte in b.scale_byte_min..=b.scale_byte_max {
            assert_eq!(mxfp4::e8m0_scale(byte, NaiveBits::NONE), 2.0f64.powi(i32::from(byte) - 127));
        }
    }
}

#[test]
fn fixture_reference_names_the_non_naive_semantics() {
    let fx = load_fixture().unwrap();
    assert!(fx.reference.contains("naive=False"));
    assert!(fx.reference.contains("LOW nibble"));
    assert!(fx.reference.contains("255"));
    assert!(fx.source.contains("READ-ONLY"));
}

fn real_samples() -> (fixture::Fixture, Vec<u8>) {
    let weights = std::env::var_os("MIMO26_WEIGHTS_DIR").map(std::path::PathBuf::from)
        .unwrap_or_else(local_weights_dir);
    let output = std::process::Command::new("python3")
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/oracle_samples.py"))
        .arg(fixture_path()).arg(weights).output().expect("read real samples");
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let fx = load_fixture().unwrap();
    assert_eq!(output.stdout.len(), fx.total_samples() * 3);
    (fx, output.stdout)
}

#[test]
#[ignore = "reads local checkpoint, CPU proof only; run explicitly before GPU proof"]
fn real_checkpoint_samples_match_all_55296_oracle_bits() {
    let (fx, raw) = real_samples();
    let mut j = 0;
    for b in &fx.blocks {
        for (i, &(_, c)) in b.positions.iter().enumerate() {
            let sample = &raw[j..j+3]; j += 3;
            let got = mxfp4::unpack_element(&sample[..1], &sample[1..], c % 2, naive_env());
            assert_eq!(got.to_bits(), b.expected_f32_bits[i], "{} sample {i}", b.name);
        }
    }
    println!("CPU REAL UNPACK: 27 blocks, 55296/55296 bits; both tensor hashes verified");
}

#[test]
#[ignore = "reads local checkpoint, explicit per-trap detectors"]
fn real_checkpoint_each_mutation_is_detected() {
    let (fx, raw) = real_samples();
    for flag in [NaiveBits::NIBBLE_SWAP, NaiveBits::SCALE_OFF_BY_ONE] {
        let mut j = 0;
        for b in &fx.blocks {
            let mut different = 0;
            for (i, &(_, c)) in b.positions.iter().enumerate() {
                let sample = &raw[j..j+3]; j += 3;
                let got = mxfp4::unpack_element(&sample[..1], &sample[1..], c % 2, flag);
                different += usize::from(got.to_bits() != b.expected_f32_bits[i]);
            }
            assert!(different > 0, "{} has no detection power for {flag:?}", b.name);
        }
    }
}
