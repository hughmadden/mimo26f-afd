//! KV paging — the page table that ties the phase-contract `KvPage` schema
//! (phase.rs) to the byte accounting (kv_pool.rs). A1 F1: "KV pool and
//! paging" is coordinator-side.
//!
//! A page is the unit of cache storage (a contiguous position range of one
//! layer kind and one phase). Allocating a page reserves its bytes in the pool;
//! eviction (INV-6 truncation) invalidates it, clears the digest (no readable
//! remnant identity) and releases the bytes. Lookup validates the phase tag
//! and arm against the registry (phase.rs `validate_cache_hit`).

use std::collections::HashMap;

use crate::kv_pool::KvPool;
use crate::phase::{
    validate_cache_hit, ArmRegistry, Digest, HitVerdict, KvPage, PageKey, Phase, PhaseError,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PagerError {
    /// The pool cannot hold the page (429-class on admission).
    PoolRejected,
    /// The page key is not present.
    NotFound,
    /// Phase/arm/epoch validation failed (phase.rs).
    Phase(PhaseError),
}

impl From<PhaseError> for PagerError {
    fn from(e: PhaseError) -> Self {
        PagerError::Phase(e)
    }
}

/// A paged KV cache: `PageKey → (page, reserved bytes)` with boot-measured
/// accounting.
#[derive(Debug, Clone)]
pub struct Pager {
    pool: KvPool,
    pages: HashMap<PageKey, (KvPage, u64)>,
}

impl Pager {
    pub fn new(pool: KvPool) -> Self {
        Pager { pool, pages: HashMap::new() }
    }

    pub fn used_bytes(&self) -> u64 {
        self.pool.used()
    }
    pub fn total_bytes(&self) -> u64 {
        self.pool.total()
    }
    pub fn len(&self) -> usize {
        self.pages.len()
    }
    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    /// Reserve `bytes` for a page and insert it. Rejects (does not insert) when
    /// the pool cannot hold it — admission gates NEW work only.
    pub fn allocate(&mut self, page: KvPage, bytes: u64) -> Result<(), PagerError> {
        let key = page.key();
        if !self.pool.reserve_bytes(bytes) {
            return Err(PagerError::PoolRejected);
        }
        self.pages.insert(key, (page, bytes));
        Ok(())
    }

    /// Lookup a page and validate it against the registry (phase tag + arm +
    /// epoch). A stale epoch or a phase/arm mismatch surfaces, never silently
    /// cross-serves.
    pub fn lookup(
        &self,
        key: &PageKey,
        requested_phase: Phase,
        registry: &ArmRegistry,
    ) -> Result<HitVerdict, PagerError> {
        let (page, _) = self.pages.get(key).ok_or(PagerError::NotFound)?;
        Ok(validate_cache_hit(page, requested_phase, registry)?)
    }

    pub fn get(&self, key: &PageKey) -> Option<&KvPage> {
        self.pages.get(key).map(|(p, _)| p)
    }

    /// INV-6 truncation: invalidate the page, clear its digest, and release the
    /// reserved bytes. Returns the evicted page (state = Invalid, digest zeroed).
    pub fn evict(&mut self, key: &PageKey) -> Option<KvPage> {
        let (mut page, bytes) = self.pages.remove(key)?;
        page.state = crate::phase::PageState::Invalid;
        page.payload_digest = [0u8; 32];
        self.pool.release(bytes);
        Some(page)
    }
}

/// Re-export the digest clear helper (INV-6) for the serving path.
pub fn cleared_digest() -> Digest {
    [0u8; 32]
}
