//! AOT capacity classes {256, 2048, 4096} on sm_121 — pure check logic.
//!
//! ADVISOR-I4 §3.2 (last item): "AOT capacity classes {256, 2048, 4096} on
//! sm_121, each with the two AOT gates from I3 extended to the Spark binary:
//! the positive gate on the real device; the negative gate on a manifest with a
//! wrong SM, which must refuse."
//!
//! # Shape (mirrors `crates/mimo26-attn/src/aot.rs`, READ-ONLY reference)
//!
//! No CUDA calls: the caller reads `cudaDeviceProp` (or a manifest) and passes
//! the fields. Two independent gates, each with its own negative test:
//!
//! * [`check_aot_arch`] — `sm_121` on GB10 (the Spark). `sm_120` is the
//!   coordinator's arch and `sm_89` is the dev host's 4090 (a correctness target, never
//!   an AOT target).
//! * [`check_aot_sm_count`] — 48 SMs on GB10.
//! * [`check_capacity_class`] — the manifest's declared capacity class must be
//!   one of {256, 2048, 4096} **and** must match the class the binary was baked
//!   for.
//!
//! The naive implementation is §8's union gate on one field
//! ([`NaiveBits::AOT_MIXED_GATE`]) plus an ignored capacity class
//! ([`NaiveBits::AOT_CAPACITY_IGNORED`]) — it happily accepts an SM *count* in
//! the *arch* slot (170) and `sm_121` as an SM count, which is exactly how a
//! mismatched AOT bake slips through. Rebuild, don't patch baked graphs.
//!
//! # Why capacity classes at all
//!
//! The Spark binary is baked ahead of time for a fixed maximum batch capacity
//! (tokens per expert / tokens per layer). A 4096-class bake served a 256-class
//! manifest wastes shared memory and occupancy; a 256-class bake served a
//! 4096-class manifest **overflows** — the failure mode is a silent wrong
//! answer or an illegal access, not a clean refusal. The class is therefore
//! part of the manifest and part of the gate.

use std::fmt;

use crate::NaiveBits;

/// Which AOT target is being baked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// GB10 Spark — arch sm_121, 48 SMs. The expert ranks.
    Gb10,
    /// 5090 / RTX PRO 6000 class — arch sm_120, 170/188 SMs. The coordinator.
    Coordinator,
}

/// Arch gate allowlist (compute capability as `10*major + minor`).
pub const GB10_ARCHS: [u32; 1] = [121]; // sm_121
pub const COORD_ARCHS: [u32; 1] = [120]; // sm_120 (and sm_120a — same cc 12.0)
/// SM-count gate allowlist.
pub const GB10_SM_COUNTS: [u32; 1] = [48]; // GB10
pub const COORD_SM_COUNTS: [u32; 2] = [170, 188]; // 5090 / RTX PRO 6000

/// The §8 union the naive gate checks a single field against.
const NAIVE_UNION: [u32; 2] = [170, 121];

/// The AOT capacity classes (ADVISOR-I4 §3.2).
pub const CAPACITY_CLASSES: [u32; 3] = [256, 2048, 4096];

/// The capacity class the Spark expert binary is baked for by default.
///
/// 2048 is the prefill chunk (§3.1 "Prefill chunk 2,048") and the class the
/// first Spark bake targets; 256 is the decode class (C16 x 8 = 256 unique
/// experts/layer) and 4096 is the headroom class.
pub const DEFAULT_CAPACITY_CLASS: u32 = 2048;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AotError {
    /// The device arch is not an allowed arch for the role.
    WrongArch { role: Role, arch: u32, allowed: Vec<u32> },
    /// The device SM count is not an allowed count for the role.
    WrongSmCount { role: Role, sm_count: u32, allowed: Vec<u32> },
    /// A supported device still differs from the device this artifact was baked for.
    DeviceMismatch { baked_arch: u32, baked_sms: u32, live_arch: u32, live_sms: u32 },
    /// The manifest's capacity class is not one of {256, 2048, 4096}.
    UnknownCapacityClass { class: u32, allowed: Vec<u32> },
    /// The manifest's capacity class is not the class the binary was baked for.
    CapacityClassMismatch { baked: u32, manifest: u32 },
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
            AotError::DeviceMismatch { baked_arch, baked_sms, live_arch, live_sms } => write!(
                f,
                "aot_device: baked sm_{baked_arch}/{baked_sms} SMs != live sm_{live_arch}/{live_sms} SMs — rebuild, don't patch"
            ),
            AotError::UnknownCapacityClass { class, allowed } => write!(
                f,
                "aot_capacity: {class} is not an AOT capacity class (allowed {allowed:?})"
            ),
            AotError::CapacityClassMismatch { baked, manifest } => write!(
                f,
                "aot_capacity: manifest declares class {manifest} but the binary is baked for {baked} — rebuild, don't patch"
            ),
        }
    }
}

impl std::error::Error for AotError {}

fn arch_allowed(role: Role) -> &'static [u32] {
    match role {
        Role::Gb10 => &GB10_ARCHS,
        Role::Coordinator => &COORD_ARCHS,
    }
}

fn sm_allowed(role: Role) -> &'static [u32] {
    match role {
        Role::Gb10 => &GB10_SM_COUNTS,
        Role::Coordinator => &COORD_SM_COUNTS,
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

/// **Gate 3 — capacity class.** The manifest's class must be a known class and
/// must equal the class the binary was baked for.
///
/// Naive misfeature [`NaiveBits::AOT_CAPACITY_IGNORED`]: the class is not
/// checked at all (any value passes), which is how a 256-class bake gets served
/// a 4096-class manifest.
pub fn check_capacity_class(
    baked: u32,
    manifest: u32,
    naive: NaiveBits,
) -> Result<(), AotError> {
    if naive.has(NaiveBits::AOT_CAPACITY_IGNORED) {
        return Ok(());
    }
    if !CAPACITY_CLASSES.contains(&manifest) {
        return Err(AotError::UnknownCapacityClass {
            class: manifest,
            allowed: CAPACITY_CLASSES.to_vec(),
        });
    }
    if manifest != baked {
        return Err(AotError::CapacityClassMismatch { baked, manifest });
    }
    Ok(())
}

/// All three gates must pass for the same role and the same capacity class.
///
/// The naive mixed gate accepts when ANY field hits `{170, 121}` — cross-pairs
/// sail through.
pub fn check_aot(
    role: Role,
    arch: u32,
    sm_count: u32,
    baked_class: u32,
    manifest_class: u32,
    naive: NaiveBits,
) -> Result<(), AotError> {
    if naive.has(NaiveBits::AOT_MIXED_GATE) {
        return if [arch, sm_count].iter().any(|v| NAIVE_UNION.contains(v)) {
            Ok(())
        } else {
            Err(AotError::WrongArch { role, arch, allowed: NAIVE_UNION.to_vec() })
        };
    }
    check_aot_arch(role, arch, NaiveBits::NONE)?;
    check_aot_sm_count(role, sm_count, NaiveBits::NONE)?;
    check_capacity_class(baked_class, manifest_class, naive)
}

/// The Spark expert binary's AOT manifest — the fields the gate reads.
///
/// This is the *shape* of the manifest the Spark reports at boot (I5 G0
/// identity readback); the repack crate owns the slice sha manifest, this owns
/// the bake identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AotManifest {
    /// `sm_121` on a Spark.
    pub arch: u32,
    /// 48 on GB10.
    pub sm_count: u32,
    /// The capacity class the binary was baked for.
    pub capacity_class: u32,
    /// Free-form build identity (commit, nvcc version) — carried, not gated.
    pub build_id: String,
}

impl AotManifest {
    /// The manifest for a Spark bake at `capacity_class`.
    pub fn spark(capacity_class: u32, build_id: &str) -> Self {
        Self {
            arch: GB10_ARCHS[0],
            sm_count: GB10_SM_COUNTS[0],
            capacity_class,
            build_id: build_id.to_string(),
        }
    }

    /// Run all three gates against the live device fields.
    pub fn check(
        &self,
        role: Role,
        live_arch: u32,
        live_sm_count: u32,
        naive: NaiveBits,
    ) -> Result<(), AotError> {
        self.check_served(role, live_arch, live_sm_count, self.capacity_class, naive)
    }

    /// Run the gates against a **served** manifest (a different class may be
    /// requested than the one baked).
    pub fn check_served(
        &self,
        role: Role,
        live_arch: u32,
        live_sm_count: u32,
        served_class: u32,
        naive: NaiveBits,
    ) -> Result<(), AotError> {
        if !naive.has(NaiveBits::AOT_MIXED_GATE) {
            // Validate the artifact fields, not only the live device allowlist.
            check_aot_arch(role, self.arch, NaiveBits::NONE)?;
            check_aot_sm_count(role, self.sm_count, NaiveBits::NONE)?;
            if self.arch != live_arch || self.sm_count != live_sm_count {
                return Err(AotError::DeviceMismatch {
                    baked_arch: self.arch, baked_sms: self.sm_count,
                    live_arch, live_sms: live_sm_count,
                });
            }
        }
        check_aot(
            role,
            live_arch,
            live_sm_count,
            self.capacity_class,
            served_class,
            naive,
        )
    }
}

/// Parse a capacity class from a manifest string (`"256"`, `"2048"`, `"4096"`).
///
/// A manifest with a wrong SM must refuse — and so must one with an
/// unparseable class. This is the string boundary the boot path uses.
pub fn parse_capacity_class(s: &str) -> Result<u32, AotError> {
    let class: u32 = s.trim().parse().map_err(|_| AotError::UnknownCapacityClass {
        class: u32::MAX,
        allowed: CAPACITY_CLASSES.to_vec(),
    })?;
    if CAPACITY_CLASSES.contains(&class) {
        Ok(class)
    } else {
        Err(AotError::UnknownCapacityClass {
            class,
            allowed: CAPACITY_CLASSES.to_vec(),
        })
    }
}
