//! A10 AOT gates — both gates negative-tested (ARCHITECTURE.md §11.9:
//! "Each has its own negative test"), plus the cross-pair negative.
//!
//! # Two-run classification
//!
//! NEGATIVES — FAIL under `MIMO26_SPIKE_NAIVE=1`, PASS on the correct impl:
//!   * `aot_arch_gate_rejects_counts_and_foreign_arch`
//!   * `aot_sm_count_gate_rejects_arch_values`
//!   * `aot_cross_pairs_rejected`
//!
//! BOTH RUNS: `aot_happy_paths`, `aot_error_text_is_actionable`.
//!
//! The naive implementation is §8's mixed `aot_sm = 170 | 121` union on one
//! field ([`NaiveBits::AOT_MIXED_GATE`]): it accepts an SM count where an arch
//! belongs and `sm_121` where a count belongs.

mod common;

use mimo26_attn::aot::{check_aot, check_aot_arch, check_aot_sm_count, Role};
use mimo26_attn::{bits_from_env, NaiveBits};

/// NEGATIVE (env-default). Gate 1: only sm_120 (coord) / sm_121 (GB10) as
/// ARCH — 170 is an SM count and sm_89 (the dev host's 4090) is not an AOT target.
#[test]
fn aot_arch_gate_rejects_counts_and_foreign_arch() {
    for (role, bad) in [
        (Role::Coordinator, 170u32),  // SM count in the arch slot (§8 mixing)
        (Role::Coordinator, 121),     // Spark arch on the coordinator
        (Role::Coordinator, 89),      // the dev host's RTX 4090 — correctness target, never AOT
        (Role::Gb10, 120),            // coordinator arch on a Spark
    ] {
        let r = check_aot_arch(role, bad, bits_from_env());
        assert!(
            r.is_err(),
            "aot_arch must reject {bad} for {role:?}, got {r:?}"
        );
    }
}

/// NEGATIVE (env-default). Gate 2: only 170 / 188 / 48 as SM COUNT — 121 is an
/// arch value and 48 never pairs with the coordinator.
#[test]
fn aot_sm_count_gate_rejects_arch_values() {
    for (role, bad) in [
        (Role::Coordinator, 121u32), // arch value in the count slot
        (Role::Coordinator, 48),     // Spark count on the coordinator
        (Role::Gb10, 120),           // arch value in the count slot
        (Role::Gb10, 170),           // 5090 count on a Spark
    ] {
        let r = check_aot_sm_count(role, bad, bits_from_env());
        assert!(
            r.is_err(),
            "aot_sm_count must reject {bad} for {role:?}, got {r:?}"
        );
    }
}

/// NEGATIVE (env-default). Cross-pairs: both fields individually "hit the
/// union" but belong to different machines — the mixed §8 gate waves these
/// through, the split gates must not.
#[test]
fn aot_cross_pairs_rejected() {
    for (role, arch, count) in [
        (Role::Coordinator, 121u32, 170u32), // Spark arch + 5090 count
        (Role::Gb10, 121, 170),               // Spark arch + 5090 count claimed for GB10
        (Role::Coordinator, 120, 48),         // coord arch + Spark count
    ] {
        let r = check_aot(role, arch, count, bits_from_env());
        assert!(
            r.is_err(),
            "check_aot must reject ({arch}, {count}) for {role:?}, got {r:?}"
        );
    }
}

/// BOTH RUNS — the real targets pass both gates (5090 170 SMs, RTX PRO 6000
/// 188 SMs, GB10 48 SMs — ADVISOR-I3 §2).
#[test]
fn aot_happy_paths() {
    assert!(check_aot(Role::Coordinator, 120, 170, NaiveBits::NONE).is_ok());
    assert!(check_aot(Role::Coordinator, 120, 188, NaiveBits::NONE).is_ok());
    assert!(check_aot(Role::Gb10, 121, 48, NaiveBits::NONE).is_ok());
    // and the mixed naive gate is detectable on a cross-pair (attribution)
    assert!(check_aot(Role::Gb10, 121, 170, NaiveBits::AOT_MIXED_GATE).is_ok());
}

/// BOTH RUNS — the error text carries the role, the value and the allowlist.
#[test]
fn aot_error_text_is_actionable() {
    let msg = check_aot_arch(Role::Coordinator, 89, NaiveBits::NONE)
        .unwrap_err()
        .to_string();
    assert!(msg.contains("89") && msg.contains("Coordinator") && msg.contains("120"), "got: {msg}");
    let msg = check_aot_sm_count(Role::Gb10, 170, NaiveBits::NONE)
        .unwrap_err()
        .to_string();
    assert!(msg.contains("170") && msg.contains("Gb10") && msg.contains("48"), "got: {msg}");
}
