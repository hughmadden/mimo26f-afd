//! Phase-contract v1.1 — KV page schema, arm registry, and the invariants
//! testable before the serving path exists. Normative spec (ADVISOR-I5 §9
//! I5-R1): `runs/20260923-i4/reviews/i5-phase-contract-v1.1-lead.md`
//! (`a8c3f60`), which supersedes v1 in full.
//!
//! The contract constrains **arithmetic provenance**: a token's KV may not
//! depend on how many other tokens chose the same expert (INV-2/INV-5, the
//! ADVISOR-I4:463 rule), on cache state (INV-5), or on batch shape (INV-7).
//! Every derivation path honours the position's phase tag at the current
//! `phi_version` (INV-4), pages are phase-homogeneous (INV-9), and nothing
//! outside the registry executes (INV-8). v1.1 deltas (MiMo `78b43a7`):
//! arities are *bodies* of the E-FP32 arm (not arms), `phi_version` epoch
//! split with `state=stale`, role-based provenance, digest-vs-numeric
//! instrument split, and phase-homogeneous pages (INV-9 / T10).

use std::collections::{HashMap, HashSet};

/// A position's intrinsic provenance, assigned once at token creation and
/// never changed (INV-1). v1.1 role-based definition: user- and system-role
/// prompt text is `CONTEXT`; assistant-role spans and draft outputs are
/// `GENERATED` *by role*, even when client-fabricated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phase {
    /// Arrived as user/system prompt text.
    Context,
    /// Assistant-role span or draft output — by role, not by lineage.
    Generated,
}

/// A conversation role, for the v1.1 role-based provenance rule (INV-1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    User,
    System,
    Assistant,
    /// A tool result (the `tool` role) — the model's input, so `CONTEXT`.
    Tool,
}

/// Role-based provenance (v1.1 §1): user/system text is `CONTEXT`; assistant
/// spans are `GENERATED` *by role* — a client-fabricated or edited assistant
/// span is still `GENERATED`, pinning the operational rule over prose lineage.
pub fn phase_for_role(role: Role) -> Phase {
    match role {
        Role::User | Role::System | Role::Tool => Phase::Context,
        Role::Assistant => Phase::Generated,
    }
}

/// Layer kind for a KV page: global attention vs sliding-window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerKind {
    Ga,
    Swa,
}

/// A derivation path — any computation that (re)produces KV for a position
/// (INV-4). v1.1 adds the history-divergence path (edit / regenerate).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DerivationPath {
    InitialPrefill,
    ChunkedPrefillContinuation,
    PrefixCacheHit,
    EvictionRebuild,
    PrefixResend,
    PreemptionResume,
    ContinueRequest,
    RejectedDraftRederivation,
    HistoryDivergence,
}

/// Lattice family for an arm (§4: cross-lattice equality is never claimed).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Lattice {
    EFp32,
    EBf16,
    EW4A8V1,
}

/// Arm identifier. v1 has a single arm (the current E-FP32 lattice) until a
/// Track P arm is adopted. Emulation arities are *bodies* of an arm, not arms
/// (MiMo F1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ArmId(pub &'static str);

/// The single v1 arm (E-FP32; "single arm until a Track P arm is adopted").
pub const SINGLE_ARM: ArmId = ArmId("e-fp32-v1");

/// The single v1 attention-projection arm (A-f32q). Φ is per model component
/// (§1: expert FFN and attention projection arithmetic each have their own
/// phase→arm map); the serving path instantiates one [`ArmRegistry`] per
/// component — this crate's registry is generic over the component.
pub const ATTENTION_ARM: ArmId = ArmId("a-f32q-v1");

/// A qualified arithmetic configuration with an arm version (INV-3 tags it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Arm {
    pub id: ArmId,
    pub version: u32,
    pub lattice: Lattice,
}

/// Equivalence class between two executions (§4): BITWISE for the same arm +
/// version (and bitwise-preserving version bumps), TOLERANCE for same-lattice
/// tolerance-class body pairs, NONE across lattices (equality never claimed).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EquivClass {
    Bitwise,
    Tolerance,
    None,
}

/// §4 class of a pair of arms.
pub fn class_for(a: &Arm, b: &Arm) -> EquivClass {
    if a.id == b.id {
        EquivClass::Bitwise
    } else if a.lattice == b.lattice {
        EquivClass::Tolerance
    } else {
        EquivClass::None
    }
}

/// Page state: `valid` until truncation poisons it (INV-6); `stale` marks a
/// page from an older `phi_version` epoch (v1.1 F2 — a planned upgrade, never
/// corruption).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PageState {
    Valid,
    Invalid,
    Stale,
}

/// Content-addressed conversation-prefix identity (v1.1 F3): any client-side
/// history divergence lands on a fresh key by construction.
pub type PrefixKey = u64;

/// Payload digest — a **bitwise instrument only** (v1.1 F4); cleared on
/// truncation (INV-6). Computed by the writer over the page payload.
pub type Digest = [u8; 32];

/// A KV page: a contiguous position range of one layer kind **and exactly one
/// phase** (INV-9). Carries its phase tag, arm id and `phi_version` (§3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvPage {
    pub prefix_key: PrefixKey,
    /// `[start, end)` contiguous positions.
    pub position_range: (u64, u64),
    pub layer_kind: LayerKind,
    pub phase_tag: Phase,
    pub arm_id: ArmId,
    pub arm_version: u32,
    pub phi_version: u32,
    pub derivation_path: DerivationPath,
    pub payload_digest: Digest,
    pub state: PageState,
}

/// Lookup key = `{prefix_key, position_range, phase_tag}` (§3). Phase is part
/// of the key, so byte-identical text served as CONTEXT and as GENERATED is
/// two entries that can never cross-serve. `arm_id` is an *output* of the tag
/// via Φ, never an input to lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PageKey {
    pub prefix_key: PrefixKey,
    pub position_range: (u64, u64),
    pub phase_tag: Phase,
}

impl KvPage {
    pub fn key(&self) -> PageKey {
        PageKey {
            prefix_key: self.prefix_key,
            position_range: self.position_range,
            phase_tag: self.phase_tag,
        }
    }
}

/// Registry failure modes — always fail loud (INV-8: no runtime default arm).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PhaseError {
    /// Φ has no arm for this phase (a phase the registry does not map).
    UnregisteredPhase(Phase),
    /// An arm id referenced by Φ or a path but never `register`ed.
    UnregisteredArm(ArmId),
    /// The (phi_version, phase, arm, path) quadruple is not in the registry.
    UnregisteredQuad { phi_version: u32, phase: Phase, arm: ArmId, path: DerivationPath },
    /// A same-epoch cache hit's stored tag disagrees with the request.
    PhaseTagMismatch { stored: Phase, requested: Phase },
    /// A same-epoch cache hit's stored arm is not Φ(phase).
    ArmMismatch { stored: ArmId, expected: ArmId },
    /// A position has neither a page nor a `phase_spans` entry (protocol error).
    MissingSpan { position: u64 },
}

/// Result of a §3 cache-hit validation (v1.1 F2 epoch-split rule).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HitVerdict {
    /// Same epoch, tags consistent — a valid hit.
    Valid,
    /// Older `phi_version` — invalidate and rebuild per INV-4, never corruption.
    Stale,
}

/// The versioned phase→arm plan of record Φ (INV-2: keys only on phase — never
/// on group size, batch, cache state, admission, chunk boundary or request
/// shape) plus the legal `(phi_version, phase, arm, path)` quadruples (INV-8).
#[derive(Debug, Clone)]
pub struct ArmRegistry {
    phi_version: u32,
    phi: HashMap<Phase, ArmId>,
    arms: HashMap<ArmId, Arm>,
    legal: HashSet<(u32, Phase, ArmId, DerivationPath)>,
}

impl Default for ArmRegistry {
    fn default() -> Self {
        ArmRegistry {
            phi_version: 1,
            phi: HashMap::new(),
            arms: HashMap::new(),
            legal: HashSet::new(),
        }
    }
}

impl ArmRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Current Φ epoch (v1.1 F2).
    pub fn phi_version(&self) -> u32 {
        self.phi_version
    }

    /// Bump the Φ epoch — a plan-of-record act (a new arm for a phase = the
    /// quality ladder). Existing pages from the old epoch classify `stale`.
    pub fn bump_phi_version(&mut self) -> u32 {
        self.phi_version += 1;
        self.phi_version
    }

    /// Register one arm definition. Nothing outside the registry executes
    /// (INV-8); `resolve` refuses an arm id that was never registered.
    pub fn register(&mut self, arm: Arm) {
        self.arms.insert(arm.id, arm);
    }

    /// Set Φ(phase) = arm under the current epoch. Keys only on phase (INV-2).
    pub fn set_phi(&mut self, phase: Phase, arm: ArmId) {
        self.phi.insert(phase, arm);
    }

    /// Declare a `(phase, arm, path)` triple legal under the current epoch.
    pub fn allow_path(&mut self, phase: Phase, arm: ArmId, path: DerivationPath) {
        self.legal.insert((self.phi_version, phase, arm, path));
    }

    /// Φ lookup — fail loud when the phase has no arm or the arm was never
    /// registered (INV-8: no default).
    pub fn arm_for(&self, phase: Phase) -> Result<&Arm, PhaseError> {
        let id = self.phi.get(&phase).ok_or(PhaseError::UnregisteredPhase(phase))?;
        self.arms.get(id).ok_or(PhaseError::UnregisteredArm(*id))
    }

    /// Resolve the arm a derivation path must execute for a phase: Φ(phase)
    /// validated against the registry's legal quadruples at the current epoch.
    /// Any unregistered quadruple is a fail-loud error (INV-8).
    pub fn resolve(&self, phase: Phase, path: DerivationPath) -> Result<&Arm, PhaseError> {
        let id = self.phi.get(&phase).ok_or(PhaseError::UnregisteredPhase(phase))?;
        let arm = self.arms.get(id).ok_or(PhaseError::UnregisteredArm(*id))?;
        if !self.legal.contains(&(self.phi_version, phase, *id, path)) {
            return Err(PhaseError::UnregisteredQuad {
                phi_version: self.phi_version,
                phase,
                arm: *id,
                path,
            });
        }
        Ok(arm)
    }
}

/// §3 hit validation, epoch-split (v1.1 F2): compare `phi_version` first.
/// * Different epoch → [`HitVerdict::Stale`] (invalidate + rebuild, never
///   corruption).
/// * Same epoch → a `phase_tag` mismatch or `arm_id ≠ Φ(phase)` is corruption
///   (fail loud); otherwise [`HitVerdict::Valid`].
pub fn validate_cache_hit(
    stored: &KvPage,
    requested_phase: Phase,
    registry: &ArmRegistry,
) -> Result<HitVerdict, PhaseError> {
    if stored.phi_version != registry.phi_version() {
        return Ok(HitVerdict::Stale);
    }
    if stored.phase_tag != requested_phase {
        return Err(PhaseError::PhaseTagMismatch {
            stored: stored.phase_tag,
            requested: requested_phase,
        });
    }
    let arm = registry.arm_for(stored.phase_tag)?;
    if stored.arm_id != arm.id {
        return Err(PhaseError::ArmMismatch { stored: stored.arm_id, expected: arm.id });
    }
    Ok(HitVerdict::Valid)
}

/// Run-length encode a position→phase stream into `(phase, start, end)` runs —
/// the phase-homogeneous packing basis (INV-9): a page never spans two phases,
/// so a mixed-phase conversation packs into pages split at every boundary.
pub fn run_length_encode(phases: &[Phase]) -> Vec<(Phase, u64, u64)> {
    let mut out = Vec::new();
    let Some((&first, rest)) = phases.split_first() else {
        return out;
    };
    let mut start = 0u64;
    let mut cur = first;
    for (i, &p) in rest.iter().enumerate() {
        if p != cur {
            out.push((cur, start, (i + 1) as u64));
            cur = p;
            start = (i + 1) as u64;
        }
    }
    out.push((cur, start, phases.len() as u64));
    out
}
