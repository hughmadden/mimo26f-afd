//! AOT capacity classes {256, 2048, 4096} on sm_121 — positive and negative
//! gates (ADVISOR-I4 §3.2 last item, §3.3).
//!
//! # Two-run classification
//!
//! NEGATIVES — FAIL under `MIMO26_SPIKE_NAIVE=1`, PASS on the correct impl:
//!   * `aot_arch_gate_rejects_counts_and_foreign_arch`
//!   * `aot_sm_count_gate_rejects_arch_values`
//!   * `aot_cross_pairs_rejected`
//!   * `aot_manifest_with_a_wrong_sm_refuses`
//!   * `aot_capacity_class_mismatch_refuses`
//!
//! BOTH RUNS: `aot_positive_gate_accepts_the_spark`,
//! `aot_capacity_classes_are_the_packet_set`, `aot_error_text_is_actionable`,
//! `detector_*`.
//!
//! The naive implementation is §8's mixed `aot_sm = 170 | 121` union on one
//! field ([`NaiveBits::AOT_MIXED_GATE`]) plus an ignored capacity class
//! ([`NaiveBits::AOT_CAPACITY_IGNORED`]).

mod common;

use common::*;
use mimo26_expert::aot::{
    self, check_aot, check_aot_arch, check_aot_sm_count, check_capacity_class, AotManifest, Role,
    CAPACITY_CLASSES, DEFAULT_CAPACITY_CLASS,
};
use mimo26_expert::NaiveBits;

/// NEGATIVE (env-default). Gate 1: only sm_121 (GB10) / sm_120 (coordinator) as
/// ARCH — 170 is an SM count and sm_89 (the dev host's 4090) is not an AOT target.
#[test]
fn aot_arch_gate_rejects_counts_and_foreign_arch() {
    for (role, bad) in [
        (Role::Gb10, 170u32),     // SM count in the arch slot (§8 mixing)
        (Role::Gb10, 120),        // coordinator arch on a Spark
        (Role::Gb10, 89),         // the dev host's RTX 4090 — correctness target, never AOT
        (Role::Coordinator, 121), // Spark arch on the coordinator
    ] {
        let r = check_aot_arch(role, bad, naive_env());
        assert!(r.is_err(), "aot_arch must reject {bad} for {role:?}, got {r:?}");
    }
}

/// NEGATIVE (env-default). Gate 2: only 48 (GB10) / 170,188 (coordinator) as SM
/// COUNT — 121 is an arch value and 48 never pairs with the coordinator.
#[test]
fn aot_sm_count_gate_rejects_arch_values() {
    for (role, bad) in [
        (Role::Gb10, 121u32),     // arch value in the count slot
        (Role::Gb10, 170),        // 5090 count on a Spark
        (Role::Coordinator, 48),  // Spark count on the coordinator
        (Role::Coordinator, 121), // arch value in the count slot
    ] {
        let r = check_aot_sm_count(role, bad, naive_env());
        assert!(r.is_err(), "aot_sm_count must reject {bad} for {role:?}, got {r:?}");
    }
}

/// NEGATIVE (env-default). Cross-pairs: both fields individually "hit the
/// union" but belong to different machines — the mixed §8 gate waves these
/// through, the split gates must not.
#[test]
fn aot_cross_pairs_rejected() {
    for (role, arch, count) in [
        (Role::Gb10, 121u32, 170u32), // Spark arch + 5090 count
        (Role::Coordinator, 120, 48), // coord arch + Spark count
        (Role::Coordinator, 121, 170), // Spark arch + 5090 count claimed for coord
    ] {
        let r = check_aot(role, arch, count, 2048, 2048, naive_env());
        assert!(
            r.is_err(),
            "check_aot must reject ({arch}, {count}) for {role:?}, got {r:?}"
        );
    }
}

/// NEGATIVE (env-default). **The negative gate the packet names**: a manifest
/// with a wrong SM must refuse.
#[test]
fn aot_manifest_with_a_wrong_sm_refuses() {
    let m = AotManifest::spark(DEFAULT_CAPACITY_CLASS, "test-build");
    // The real Spark: sm_121, 48 SMs — accepted.
    assert!(m.check(Role::Gb10, 121, 48, NaiveBits::NONE).is_ok());
    // A manifest served on the wrong device must refuse.
    for (arch, sm) in [
        (120u32, 48u32),  // coordinator arch
        (121, 170),       // 5090 SM count
        (89, 48),         // the dev host's 4090
        (121, 188),       // RTX PRO 6000 count
        (170, 48),        // an SM count in the arch slot
    ] {
        let r = m.check(Role::Gb10, arch, sm, naive_env());
        assert!(
            r.is_err(),
            "a manifest with arch={arch} sm={sm} must refuse on a Spark, got {r:?}"
        );
    }
}

/// NEGATIVE: corrupt artifact metadata on a correct live device (the actual
/// wrong-manifest gate, distinct from running a good manifest on a wrong device).
#[test]
fn aot_corrupt_manifest_refuses_on_correct_device() {
    for class in CAPACITY_CLASSES {
        let mut manifest = AotManifest::spark(class, "test-build");
        manifest.arch = 120;
        assert!(manifest.check(Role::Gb10, 121, 48, naive_env()).is_err());
        manifest.arch = 121;
        manifest.sm_count = 170;
        assert!(manifest.check(Role::Gb10, 121, 48, naive_env()).is_err());
    }
}

/// BOTH RUNS: two supported coordinator targets are not the same AOT artifact.
#[test]
fn aot_supported_but_different_device_refuses() {
    let manifest = AotManifest { arch: 120, sm_count: 170, capacity_class: 256, build_id: "test".into() };
    assert!(manifest.check(Role::Coordinator, 120, 170, NaiveBits::NONE).is_ok());
    assert!(manifest.check(Role::Coordinator, 120, 188, NaiveBits::NONE).is_err());
    assert!(check_aot(Role::Gb10, 121, 48, 256, 4096, NaiveBits::AOT_CAPACITY_IGNORED).is_ok());
}

/// NEGATIVE (env-default). The capacity class must match the bake: a 256-class
/// bake served a 4096-class manifest overflows, and vice versa.
#[test]
fn aot_capacity_class_mismatch_refuses() {
    let naive = naive_env();
    for baked in CAPACITY_CLASSES {
        for served in CAPACITY_CLASSES {
            let r = check_capacity_class(baked, served, naive);
            if baked == served {
                assert!(r.is_ok(), "class {baked} must accept itself, got {r:?}");
            } else {
                assert!(
                    r.is_err(),
                    "class {baked} must refuse a {served} manifest, got {r:?}"
                );
            }
        }
    }
    // An unknown class refuses even when it equals the bake.
    assert!(check_capacity_class(512, 512, naive).is_err(), "512 is not a class");
    assert!(check_capacity_class(0, 0, naive).is_err(), "0 is not a class");
}

/// BOTH RUNS. The positive gate: the real Spark target passes all three gates
/// at every capacity class.
#[test]
fn aot_positive_gate_accepts_the_spark() {
    for class in CAPACITY_CLASSES {
        let m = AotManifest::spark(class, "test-build");
        assert!(
            m.check(Role::Gb10, 121, 48, NaiveBits::NONE).is_ok(),
            "the Spark (sm_121, 48 SMs) must pass at class {class}"
        );
        assert!(
            m.check_served(Role::Gb10, 121, 48, class, NaiveBits::NONE).is_ok(),
            "serving class {class} from a class-{class} bake must pass"
        );
    }
    // The coordinator's own targets still pass their gates.
    assert!(check_aot(Role::Coordinator, 120, 170, 2048, 2048, NaiveBits::NONE).is_ok());
    assert!(check_aot(Role::Coordinator, 120, 188, 2048, 2048, NaiveBits::NONE).is_ok());
}

/// BOTH RUNS. The capacity classes are exactly the packet's set.
#[test]
fn aot_capacity_classes_are_the_packet_set() {
    assert_eq!(CAPACITY_CLASSES, [256, 2048, 4096]);
    assert_eq!(DEFAULT_CAPACITY_CLASS, 2048);
    assert_eq!(aot::parse_capacity_class("256").unwrap(), 256);
    assert_eq!(aot::parse_capacity_class("2048").unwrap(), 2048);
    assert_eq!(aot::parse_capacity_class("4096").unwrap(), 4096);
    assert!(aot::parse_capacity_class("512").is_err());
    assert!(aot::parse_capacity_class("").is_err());
    assert!(aot::parse_capacity_class("two thousand").is_err());
}

/// BOTH RUNS. The error text carries the role, the value and the allowlist.
#[test]
fn aot_error_text_is_actionable() {
    let msg = check_aot_arch(Role::Gb10, 89, NaiveBits::NONE).unwrap_err().to_string();
    assert!(msg.contains("89") && msg.contains("Gb10") && msg.contains("121"), "got: {msg}");
    let msg = check_aot_sm_count(Role::Gb10, 170, NaiveBits::NONE).unwrap_err().to_string();
    assert!(msg.contains("170") && msg.contains("Gb10") && msg.contains("48"), "got: {msg}");
    let msg = check_capacity_class(256, 4096, NaiveBits::NONE).unwrap_err().to_string();
    assert!(msg.contains("256") && msg.contains("4096"), "got: {msg}");
}

/// BOTH RUNS. The detectors: with the naive flags set, the wrong implementation
/// is accepted — so the negatives above have real detection power.
#[test]
fn detector_mixed_gate_and_ignored_capacity_are_detected() {
    // The mixed gate accepts a cross-pair.
    assert!(check_aot(Role::Gb10, 121, 170, 2048, 2048, NaiveBits::AOT_MIXED_GATE).is_ok());
    assert!(check_aot(Role::Gb10, 121, 170, 2048, 2048, NaiveBits::NONE).is_err());
    // The ignored capacity accepts any class.
    assert!(check_capacity_class(256, 4096, NaiveBits::AOT_CAPACITY_IGNORED).is_ok());
    assert!(check_capacity_class(256, 4096, NaiveBits::NONE).is_err());
}
