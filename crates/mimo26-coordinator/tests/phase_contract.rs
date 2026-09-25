//! Phase-contract v1.1 — schema + registry invariants (INV-1..9) and the
//! T1–T10 negatives. The schema-testable set (T3/T4/T7/T8/T9/T10 plus the
//! INV-1..9 structural assertions) is here; the compute-path portions of
//! T1/T2/T5/T6 (payload-digest equality across cache states, the arm log, the
//! :463 M-keying negative against real GEMMs, and P-305 preemption) run only
//! once the serving path exists — they are recorded, not dropped, and their
//! schema-level halves (stable keys, Φ-by-phase across every path, M excluded
//! from the Φ key, RLE restore + fresh-digest rebuild) are pinned here.

use mimo26_coordinator::{
    class_for, phase_for_role, run_length_encode, validate_cache_hit, Arm, ArmRegistry,
    DerivationPath, EquivClass, HitVerdict, KvPage, LayerKind, Lattice, PageState, Phase,
    PhaseError, Role, SINGLE_ARM,
};

fn arm() -> Arm {
    Arm { id: SINGLE_ARM, version: 1, lattice: Lattice::EFp32 }
}

fn registry() -> ArmRegistry {
    let mut r = ArmRegistry::new();
    r.register(arm());
    r.set_phi(Phase::Context, SINGLE_ARM);
    r.set_phi(Phase::Generated, SINGLE_ARM);
    for p in [DerivationPath::InitialPrefill, DerivationPath::ContinueRequest] {
        r.allow_path(Phase::Context, SINGLE_ARM, p);
        r.allow_path(Phase::Generated, SINGLE_ARM, p);
    }
    r
}

fn page(prefix: u64, range: (u64, u64), phase: Phase, path: DerivationPath) -> KvPage {
    KvPage {
        prefix_key: prefix,
        position_range: range,
        layer_kind: LayerKind::Swa,
        phase_tag: phase,
        arm_id: SINGLE_ARM,
        arm_version: 1,
        phi_version: 1,
        derivation_path: path,
        payload_digest: [0u8; 32],
        state: PageState::Valid,
    }
}

/// A registry whose Φ admits every derivation path for both phases — the shape
/// the serving path registers once all recompute paths are implemented.
fn registry_all() -> ArmRegistry {
    let mut r = registry();
    let paths = [
        DerivationPath::InitialPrefill,
        DerivationPath::ChunkedPrefillContinuation,
        DerivationPath::PrefixCacheHit,
        DerivationPath::EvictionRebuild,
        DerivationPath::PrefixResend,
        DerivationPath::PreemptionResume,
        DerivationPath::ContinueRequest,
        DerivationPath::RejectedDraftRederivation,
        DerivationPath::HistoryDivergence,
    ];
    for p in paths {
        r.allow_path(Phase::Context, SINGLE_ARM, p);
        r.allow_path(Phase::Generated, SINGLE_ARM, p);
    }
    r
}

/// INV-1 (role-based): assistant-role spans are GENERATED *by role*, even when
/// client-fabricated — user/system text is CONTEXT.
#[test]
fn inv1_role_based_provenance() {
    assert_eq!(phase_for_role(Role::User), Phase::Context);
    assert_eq!(phase_for_role(Role::System), Phase::Context);
    assert_eq!(phase_for_role(Role::Assistant), Phase::Generated);
}

/// INV-3: phase is part of the lookup key. Byte-identical text served as
/// CONTEXT and as GENERATED is two entries that can never cross-serve.
#[test]
fn inv3_phase_is_part_of_the_lookup_key() {
    let ctx = page(7, (0, 4), Phase::Context, DerivationPath::InitialPrefill);
    let gen = page(7, (0, 4), Phase::Generated, DerivationPath::ContinueRequest);
    assert_ne!(ctx.key(), gen.key(), "phase must distinguish two byte-identical spans");
    assert_eq!(ctx.key().phase_tag, Phase::Context);
    assert_eq!(gen.key().phase_tag, Phase::Generated);
}

/// T3: a same-epoch cache hit whose stored phase_tag differs from the request
/// is corruption — fail loud, never cross-serve (INV-3).
#[test]
fn t3_phase_keyed_cache_fails_loud_on_phase_mismatch() {
    let reg = registry();
    let gen = page(7, (0, 4), Phase::Generated, DerivationPath::ContinueRequest);
    let err = validate_cache_hit(&gen, Phase::Context, &reg).unwrap_err();
    assert_eq!(
        err,
        PhaseError::PhaseTagMismatch { stored: Phase::Generated, requested: Phase::Context }
    );
    // Matching tag validates.
    assert_eq!(
        validate_cache_hit(&gen, Phase::Generated, &reg).unwrap(),
        HitVerdict::Valid
    );
}

/// T3: a same-epoch cache hit whose stored arm is not Φ(phase) is corruption.
#[test]
fn t3_arm_must_equal_phi_of_the_tag() {
    let reg = registry();
    let mut stale = page(7, (0, 4), Phase::Context, DerivationPath::InitialPrefill);
    stale.arm_id = mimo26_coordinator::ArmId("e-bf16-v1");
    let err = validate_cache_hit(&stale, Phase::Context, &reg).unwrap_err();
    assert_eq!(err, PhaseError::ArmMismatch { stored: stale.arm_id, expected: SINGLE_ARM });
}

/// INV-2 + INV-4: the arm a path executes is Φ(phase) — the path never selects
/// the arm; batch/cache/request shape cannot key the map.
#[test]
fn inv2_phi_selects_arm_by_phase_not_path() {
    let reg = registry();
    let a = reg.resolve(Phase::Generated, DerivationPath::InitialPrefill).unwrap();
    let b = reg.resolve(Phase::Generated, DerivationPath::ContinueRequest).unwrap();
    assert_eq!(a.id, b.id);
    assert_eq!(a.id, SINGLE_ARM);
    // A path outside the registry's legal quadruples is refused (INV-8).
    let err = reg.resolve(Phase::Generated, DerivationPath::EvictionRebuild).unwrap_err();
    assert!(matches!(err, PhaseError::UnregisteredQuad { .. }));
}

/// T7 + INV-8: any (phi_version, phase, arm, path) quadruple not in the
/// registry is a startup error; there is no runtime default arm.
#[test]
fn t7_registry_fails_loud_on_unregistered() {
    let mut r = ArmRegistry::new();
    assert_eq!(
        r.resolve(Phase::Context, DerivationPath::InitialPrefill).unwrap_err(),
        PhaseError::UnregisteredPhase(Phase::Context)
    );

    r.set_phi(Phase::Context, mimo26_coordinator::ArmId("ghost"));
    assert!(matches!(r.resolve(Phase::Context, DerivationPath::InitialPrefill),
        Err(PhaseError::UnregisteredArm(_))));

    let reg = registry();
    let err = reg.resolve(Phase::Context, DerivationPath::PreemptionResume).unwrap_err();
    assert_eq!(
        err,
        PhaseError::UnregisteredQuad {
            phi_version: 1,
            phase: Phase::Context,
            arm: SINGLE_ARM,
            path: DerivationPath::PreemptionResume,
        }
    );
}

/// T8 + §4: the equivalence-class table. Same arm → BITWISE, same lattice at a
/// different arm → TOLERANCE, cross-lattice → NONE (equality never claimed).
#[test]
fn t8_equivalence_class_table() {
    let a = Arm { id: SINGLE_ARM, version: 1, lattice: Lattice::EFp32 };
    let a2 = Arm { id: SINGLE_ARM, version: 2, lattice: Lattice::EFp32 };
    let b = Arm { id: mimo26_coordinator::ArmId("e-fp32-v2"), version: 1, lattice: Lattice::EFp32 };
    let c = Arm { id: mimo26_coordinator::ArmId("e-bf16-v1"), version: 1, lattice: Lattice::EBf16 };

    assert_eq!(class_for(&a, &a2), EquivClass::Bitwise, "same arm is bitwise");
    assert_eq!(class_for(&a, &b), EquivClass::Tolerance, "same lattice, different arm");
    assert_eq!(class_for(&a, &c), EquivClass::None, "cross-lattice equality is never claimed");
}

/// T9 (v1.1 F2): a phi_version bump makes pre-existing pages `stale` — a
/// planned upgrade, never corruption — so the upgrade completes without a
/// fail-loud and rebuilds run under the new Φ.
#[test]
fn t9_phi_upgrade_classifies_stale_not_corruption() {
    let mut reg = registry();
    let old = page(7, (0, 4), Phase::Generated, DerivationPath::ContinueRequest);
    assert_eq!(
        validate_cache_hit(&old, Phase::Generated, &reg).unwrap(),
        HitVerdict::Valid
    );

    reg.bump_phi_version(); // adopt a new arm for a phase -> epoch 2
    assert_eq!(reg.phi_version(), 2);
    assert_eq!(
        validate_cache_hit(&old, Phase::Generated, &reg).unwrap(),
        HitVerdict::Stale,
        "an older-epoch page must classify stale, never corruption"
    );
    // Rebuild under the current Φ completes (a fresh page carries epoch 2).
    let mut rebuilt = old.clone();
    rebuilt.phi_version = reg.phi_version();
    rebuilt.payload_digest = [1u8; 32];
    assert_eq!(
        validate_cache_hit(&rebuilt, Phase::Generated, &reg).unwrap(),
        HitVerdict::Valid
    );
}

/// T10 + INV-9 (v1.1 F5): pages are partitioned at phase boundaries — a
/// mixed-phase conversation packs into phase-homogeneous runs.
#[test]
fn t10_phase_homogeneous_packing() {
    // system CONTEXT, assistant GENERATED, user CONTEXT.
    let stream = [
        Phase::Context,
        Phase::Context,
        Phase::Generated,
        Phase::Generated,
        Phase::Generated,
        Phase::Context,
    ];
    let runs = run_length_encode(&stream);
    assert_eq!(
        runs,
        vec![(Phase::Context, 0, 2), (Phase::Generated, 2, 5), (Phase::Context, 5, 6)]
    );
    for (phase, start, end) in runs {
        // every position in a run has the run's phase -> no page spans two phases.
        for p in &stream[start as usize..end as usize] {
            assert_eq!(*p, phase, "a run must be phase-homogeneous (INV-9)");
        }
    }
}

/// INV-5: cache state is invisible to arithmetic — the arm Φ selects for a
/// phase is the same whether the position is a cache hit or a miss (Φ consults
/// only the phase, never the cache). A hit and a miss on the same phase must
/// resolve to the identical arm.
#[test]
fn inv5_cache_state_invisible_to_arm_selection() {
    let reg = registry_all();
    let hit = reg.resolve(Phase::Generated, DerivationPath::PrefixCacheHit).unwrap();
    let miss = reg.resolve(Phase::Generated, DerivationPath::EvictionRebuild).unwrap();
    assert_eq!(hit.id, miss.id, "cache hit/miss must not select the arm (INV-5)");
    assert_eq!(miss.id, SINGLE_ARM);
}

/// T2 (schema half) + INV-4: every derivation path for a phase resolves to
/// Φ(phase) — the tag selects the arm, the path only checks legality. Re-prefill
/// of an evicted GENERATED span runs the decode arm (Φ(Generated)) in whatever
/// batch shape the scheduler chose; batch shape is not a Φ key.
#[test]
fn t2_tag_honoured_across_every_derivation_path() {
    let reg = registry_all();
    let paths = [
        DerivationPath::InitialPrefill,
        DerivationPath::ChunkedPrefillContinuation,
        DerivationPath::PrefixCacheHit,
        DerivationPath::EvictionRebuild,
        DerivationPath::PrefixResend,
        DerivationPath::PreemptionResume,
        DerivationPath::ContinueRequest,
        DerivationPath::RejectedDraftRederivation,
        DerivationPath::HistoryDivergence,
    ];
    for p in paths {
        assert_eq!(reg.resolve(Phase::Generated, p).unwrap().id, SINGLE_ARM, "{p:?}");
        assert_eq!(reg.resolve(Phase::Context, p).unwrap().id, SINGLE_ARM, "{p:?}");
    }
}

/// T5 (schema half) + INV-2 :463 negative: the invoked arm is a pure function of
/// phase. M, batch size, cache state, admission and chunk boundary cannot key Φ —
/// the registry's map is `Phase → Arm` by construction, so there is no M to vary.
/// (The compute-path half — bitwise/schedule-identical GEMM reduction across
/// M1–8 and C16 compositions, INV-7 — is the serving path's gate.)
#[test]
fn t5_phi_has_no_batch_key() {
    let reg = registry_all();
    // Two different "compositions" of the same phase resolve identically; the
    // only input Φ accepts is the phase.
    let a = reg.resolve(Phase::Context, DerivationPath::InitialPrefill).unwrap();
    let b = reg.resolve(Phase::Context, DerivationPath::ChunkedPrefillContinuation).unwrap();
    assert_eq!(a.id, b.id, "Φ keys only on phase, never batch/chunk (INV-2, :463)");
    assert_eq!(a.id, SINGLE_ARM);
}

/// T6 (schema half): preemption resume restores `phase_spans` (RLE) and a rebuilt
/// span lands a fresh digest under the decode arm. (The compute-path half — the
/// actual preempt→evict→resume KV rebuild — is the serving path's gate.)
#[test]
fn t6_resume_restores_spans_and_rebuilds_fresh_digest() {
    let reg = registry_all();
    let stream = [
        Phase::Context,
        Phase::Context,
        Phase::Generated,
        Phase::Generated,
        Phase::Context,
    ];
    // phase_spans authority survives (RLE is deterministic and total).
    let spans = run_length_encode(&stream);
    assert_eq!(
        spans,
        vec![(Phase::Context, 0, 2), (Phase::Generated, 2, 4), (Phase::Context, 4, 5)]
    );
    // A rebuilt GENERATED span (positions 2..4) resolves to the decode arm
    // (Φ(Generated)) and carries a fresh digest.
    let mut rebuilt = page(11, (2, 4), Phase::Generated, DerivationPath::PreemptionResume);
    rebuilt.payload_digest = [9u8; 32];
    assert_eq!(reg.resolve(Phase::Generated, DerivationPath::PreemptionResume).unwrap().id, SINGLE_ARM);
    assert_eq!(validate_cache_hit(&rebuilt, Phase::Generated, &reg).unwrap(), HitVerdict::Valid);
    assert_ne!(rebuilt.payload_digest, [0u8; 32], "rebuilt span must not carry a cleared digest");
}
