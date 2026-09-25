//! A1 F1 "KV pool and paging" — the page table ties the phase-contract KvPage
//! to the byte accounting (allocate → reserve; evict → invalidate + digest
//! clear + release; lookup → validate against the registry).

use mimo26_coordinator::{
    Arm, ArmRegistry, DerivationPath, HitVerdict, KvPage, KvPool, LayerKind, Lattice, Pager,
    PagerError, PageState, Phase, SINGLE_ARM,
};

fn registry() -> ArmRegistry {
    let mut r = ArmRegistry::new();
    r.register(Arm { id: SINGLE_ARM, version: 1, lattice: Lattice::EFp32 });
    r.set_phi(Phase::Context, SINGLE_ARM);
    r.set_phi(Phase::Generated, SINGLE_ARM);
    r.allow_path(Phase::Context, SINGLE_ARM, DerivationPath::InitialPrefill);
    r.allow_path(Phase::Generated, SINGLE_ARM, DerivationPath::ContinueRequest);
    r
}

fn page(prefix: u64, range: (u64, u64), phase: Phase) -> KvPage {
    KvPage {
        prefix_key: prefix,
        position_range: range,
        layer_kind: LayerKind::Swa,
        phase_tag: phase,
        arm_id: SINGLE_ARM,
        arm_version: 1,
        phi_version: 1,
        derivation_path: DerivationPath::InitialPrefill,
        payload_digest: [7u8; 32],
        state: PageState::Valid,
    }
}

#[test]
fn allocate_reserves_and_lookup_validates() {
    let mut pager = Pager::new(KvPool::new(1_000_000));
    let p = page(1, (0, 4), Phase::Context);
    pager.allocate(p.clone(), 400).unwrap();
    assert_eq!(pager.used_bytes(), 400);
    assert_eq!(pager.lookup(&p.key(), Phase::Context, &registry()).unwrap(), HitVerdict::Valid);
    // Cross-phase lookup of a CONTEXT page is corruption.
    let err = pager.lookup(&p.key(), Phase::Generated, &registry()).unwrap_err();
    assert!(matches!(err, PagerError::Phase(_)));
}

#[test]
fn evict_invalidates_clears_digest_and_releases_bytes() {
    let mut pager = Pager::new(KvPool::new(1_000_000));
    let p = page(1, (0, 4), Phase::Generated);
    pager.allocate(p.clone(), 400).unwrap();
    let evicted = pager.evict(&p.key()).expect("evict");
    assert_eq!(evicted.state, PageState::Invalid, "INV-6: invalidated");
    assert_eq!(evicted.payload_digest, [0u8; 32], "INV-6: digest cleared");
    assert_eq!(pager.used_bytes(), 0, "bytes released");
    assert!(pager.get(&p.key()).is_none(), "excluded from lookup");
    assert_eq!(pager.lookup(&p.key(), Phase::Generated, &registry()).unwrap_err(), PagerError::NotFound);
}

#[test]
fn allocate_rejects_when_the_pool_is_full() {
    let mut pager = Pager::new(KvPool::new(100));
    let p = page(1, (0, 4), Phase::Context);
    assert_eq!(pager.allocate(p.clone(), 200).unwrap_err(), PagerError::PoolRejected);
    assert!(pager.is_empty(), "rejected page must not be inserted");
}

#[test]
fn stale_epoch_surfaces_as_stale_not_not_found() {
    let mut pager = Pager::new(KvPool::new(1_000_000));
    let mut reg = registry();
    let p = page(1, (0, 4), Phase::Generated);
    pager.allocate(p.clone(), 400).unwrap();
    reg.bump_phi_version(); // epoch 2; the page is from epoch 1
    assert_eq!(pager.lookup(&p.key(), Phase::Generated, &reg).unwrap(), HitVerdict::Stale);
}

/// T4 (phase contract §5): truncation is total — after a rejected-draft span is
/// evicted, no valid page covers its positions, the digest is cleared, and
/// re-derivation lands a FRESH digest (never a stale/readable remnant).
#[test]
fn t4_truncation_totality() {
    let reg = registry();
    let mut pager = Pager::new(KvPool::new(1_000_000));

    // A rejected-draft (Generated) span derived with a non-zero digest.
    let mut draft = page(9, (0, 4), Phase::Generated);
    draft.derivation_path = DerivationPath::RejectedDraftRederivation;
    draft.payload_digest = [0xAA; 32];
    pager.allocate(draft.clone(), 400).unwrap();

    // Truncate (INV-6): invalidated, digest cleared, excluded from lookup.
    let evicted = pager.evict(&draft.key()).expect("evict");
    assert_eq!(evicted.state, PageState::Invalid, "no valid page may cover the positions");
    assert_eq!(evicted.payload_digest, [0u8; 32], "stale digest must be cleared");
    assert!(pager.get(&draft.key()).is_none(), "no readable remnant in lookup");
    assert_eq!(
        pager.lookup(&draft.key(), Phase::Generated, &reg).unwrap_err(),
        PagerError::NotFound
    );

    // Re-derivation under the same tag lands a FRESH digest via the decode arm.
    let mut redone = page(9, (0, 4), Phase::Generated);
    redone.derivation_path = DerivationPath::RejectedDraftRederivation;
    redone.payload_digest = [0xBB; 32];
    pager.allocate(redone.clone(), 400).unwrap();
    let got = pager.get(&redone.key()).unwrap();
    assert_eq!(got.state, PageState::Valid);
    assert_eq!(got.payload_digest, [0xBB; 32], "re-derivation must carry a fresh digest");
    assert_ne!(got.payload_digest, evicted.payload_digest, "old digest gone, fresh digest present");
}
