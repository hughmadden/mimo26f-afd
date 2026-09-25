//! A10 AOT gates — **two independent gates**, pure check logic (no CUDA calls;
//! the caller reads `cudaDeviceProp` and passes the fields).
//!
//! ARCHITECTURE.md §11.9 (supersedes §8's mixed `aot_sm = 170 | 121`):
//! * [`check_aot_arch`] — `sm_120` on the coordinator (5090 / RTX PRO 6000),
//!   `sm_121` on GB10;
//! * [`check_aot_sm_count`] — 170 (5090) / 188 (RTX PRO 6000) / 48 (GB10).
//!
//! Each gate has its own negative test (`tests/aot_gates.rs`), plus a cross-pair
//! negative: a coordinator arch with a Spark SM count (and vice versa) must
//! FAIL. The naive implementation is §8's union gate on one field
//! ([`NaiveBits::AOT_MIXED_GATE`]) — it happily accepts an SM *count* in the
//! *arch* slot (170) and `sm_121` as an SM count, which is exactly how a
//! mismatched AOT bake slips through. Rebuild, don't patch baked graphs.
//!
//! Note the dev host's RTX 4090 is **sm_89**: correct logic REJECTS it for either role
//! (portable-kernel correctness runs there unbaked; AOT bakes target 120/121).

use std::fmt;

use crate::NaiveBits;

/// Which AOT target is being baked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// 5090 first, then RTX PRO 6000 class — arch sm_120, 170/188 SMs.
    Coordinator,
    /// GB10 Spark — arch sm_121, 48 SMs.
    Gb10,
}

/// Arch gate allowlist (compute capability as `10*major + minor`).
pub const COORD_ARCHS: [u32; 1] = [120]; // sm_120 (and sm_120a — same cc 12.0)
pub const GB10_ARCHS: [u32; 1] = [121]; // sm_121
/// SM-count gate allowlist.
pub const COORD_SM_COUNTS: [u32; 2] = [170, 188]; // 5090 / RTX PRO 6000
pub const GB10_SM_COUNTS: [u32; 1] = [48]; // GB10

/// The §8 union the naive gate checks a single field against.
const NAIVE_UNION: [u32; 2] = [170, 121];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AotError {
    WrongArch { role: Role, arch: u32, allowed: Vec<u32> },
    WrongSmCount { role: Role, sm_count: u32, allowed: Vec<u32> },
}

impl fmt::Display for AotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AotError::WrongArch { role, arch, allowed } => write!(
                f,
                "aot_arch: {arch} is not an allowed arch for {role:?} (allowed {allowed:?}) — rebuild, don't patch"
            ),
            AotError::WrongSmCount { role, sm_count, allowed } => write!(
                f,
                "aot_sm_count: {sm_count} is not an allowed SM count for {role:?} (allowed {allowed:?}) — rebuild, don't patch"
            ),
        }
    }
}

impl std::error::Error for AotError {}

fn arch_allowed(role: Role) -> &'static [u32] {
    match role {
        Role::Coordinator => &COORD_ARCHS,
        Role::Gb10 => &GB10_ARCHS,
    }
}

fn sm_allowed(role: Role) -> &'static [u32] {
    match role {
        Role::Coordinator => &COORD_SM_COUNTS,
        Role::Gb10 => &GB10_SM_COUNTS,
    }
}

/// **Gate 1 — arch.** Naive misfeature [`NaiveBits::AOT_MIXED_GATE`]: accept the
/// field if it is in the §8 union `{170, 121}` regardless of role — i.e. an SM
/// count passes as an arch and `sm_121` passes for the coordinator.
pub fn check_aot_arch(role: Role, arch: u32, naive: NaiveBits) -> Result<(), AotError> {
    if naive.has(NaiveBits::AOT_MIXED_GATE) {
        return if NAIVE_UNION.contains(&arch) {
            Ok(())
        } else {
            Err(AotError::WrongArch { role, arch, allowed: NAIVE_UNION.to_vec() })
        };
    }
    let allowed = arch_allowed(role);
    if allowed.contains(&arch) {
        Ok(())
    } else {
        Err(AotError::WrongArch { role, arch, allowed: allowed.to_vec() })
    }
}

/// **Gate 2 — SM count.** Naive misfeature [`NaiveBits::AOT_MIXED_GATE`]: same
/// union on this field — `sm_121` (an arch) passes as an SM count and 170
/// passes on a Spark.
pub fn check_aot_sm_count(role: Role, sm_count: u32, naive: NaiveBits) -> Result<(), AotError> {
    if naive.has(NaiveBits::AOT_MIXED_GATE) {
        return if NAIVE_UNION.contains(&sm_count) {
            Ok(())
        } else {
            Err(AotError::WrongSmCount { role, sm_count, allowed: NAIVE_UNION.to_vec() })
        };
    }
    let allowed = sm_allowed(role);
    if allowed.contains(&sm_count) {
        Ok(())
    } else {
        Err(AotError::WrongSmCount { role, sm_count, allowed: allowed.to_vec() })
    }
}

/// Both gates must pass for the same role (§11.9). The naive mixed gate accepts
/// when ANY field hits `{170, 121}` — cross-pairs sail through.
pub fn check_aot(role: Role, arch: u32, sm_count: u32, naive: NaiveBits) -> Result<(), AotError> {
    if naive.has(NaiveBits::AOT_MIXED_GATE) {
        return if [arch, sm_count].iter().any(|v| NAIVE_UNION.contains(v)) {
            Ok(())
        } else {
            Err(AotError::WrongArch { role, arch, allowed: NAIVE_UNION.to_vec() })
        };
    }
    check_aot_arch(role, arch, NaiveBits::NONE)?;
    check_aot_sm_count(role, sm_count, NaiveBits::NONE)
}
