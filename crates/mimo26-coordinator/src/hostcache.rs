//! KV host tier 1 (ARCHITECTURE §11.2–11.3; the DS41RT host snapshot cache v3,
//! `dsv41-flash-tp4-engram/research/afd-hostcache-design.md`, in its `on-evict`
//! store mode): the RAM home of retained snapshots the device had to evict.
//!
//! Retained snapshots live on the device first (perf reset K3, `api.rs`: a
//! snapshot is a *point* in a slot's history with its position state saved
//! device-side): with no pressure nothing is copied to RAM. When the device
//! evicts a point (a bank over its size, or a slot or its memory needed), the
//! scheduler stores it here; a returning conversation that misses the device
//! restores from here instead of prefilling. RAM eviction deletes: prompt
//! snapshots before turn snapshots, oldest first (the engine's
//! `Retention::evict_one` order).
//!
//! A snapshot is a request's KV state at an exact position:
//! - the GA rows of the 9 full-attention layers, stored as 256-token pages
//!   ([`KV_PAGE_ROWS`], 2,949,120 B) shared between snapshots. The device KV is
//!   not paged here, so a page is identified by a hash chain over the token
//!   prefix (a page's KV depends only on the tokens up to its end) where the
//!   design uses the device page identity;
//! - the 39 SWA layers' visible rows (the last `window - 1`);
//! - the DFlash draft rings;
//! - the next token (greedy, so the first token after an exact restore needs
//!   no forward).
//!
//! A prompt is restored only from a snapshot whose tokens are a prefix of it (no
//! replay of an empty SWA window, trap T17); the rest is prefilled. Everything
//! runs on the scheduler thread. `MIMO26_HOST_CACHE_GB=0` turns it off (nothing
//! is allocated or copied).

use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use mimo26_attn::cuda;

use mimo26_attn::device::DeviceBuffer;

use crate::dforward::{DeviceKv, KV_PAGE_ROWS};

/// Shortest snapshot worth caching (the design's `--host-cache-min-tokens`).
const MIN_TOKENS: usize = 512;

/// Which retention bank a snapshot came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// At prompt end (the multi-turn fallback point).
    Prompt,
    /// At completion end.
    Turn,
}

/// Page-locked slab size (whole slots are carved from each).
const SLAB_BYTES: usize = 1 << 30;

/// Fixed-size slots carved from page-locked slabs.
struct Arena {
    slabs: Vec<Vec<u8>>,
    slot_bytes: usize,
    free: Vec<(usize, usize)>,
}

impl Arena {
    fn new(slot_bytes: usize, slots: usize) -> Result<Self, String> {
        let per_slab = (SLAB_BYTES / slot_bytes).max(1);
        let mut slabs = Vec::new();
        let mut free = Vec::with_capacity(slots);
        let mut left = slots;
        while left > 0 {
            let n = left.min(per_slab);
            let mut slab = vec![0u8; n * slot_bytes];
            // SAFETY: a live allocation of the given size, registered once for the
            // process lifetime (unregistered in Drop).
            let rc = unsafe { cuda::cudaHostRegister(slab.as_mut_ptr() as _, slab.len(), 0) };
            if rc != cuda::SUCCESS {
                return Err(format!("hostcache: cudaHostRegister {} B: {}", slab.len(), cuda::error_string(rc)));
            }
            let s = slabs.len();
            free.extend((0..n).rev().map(|i| (s, i * slot_bytes)));
            slabs.push(slab);
            left -= n;
        }
        Ok(Self { slabs, slot_bytes, free })
    }

    fn get(&self, slot: (usize, usize)) -> &[u8] {
        &self.slabs[slot.0][slot.1..slot.1 + self.slot_bytes]
    }

    fn get_mut(&mut self, slot: (usize, usize)) -> &mut [u8] {
        let n = self.slot_bytes;
        &mut self.slabs[slot.0][slot.1..slot.1 + n]
    }
}

impl Drop for Arena {
    fn drop(&mut self) {
        for s in &mut self.slabs {
            // SAFETY: registered in `new`.
            unsafe { cuda::cudaHostUnregister(s.as_mut_ptr() as _) };
        }
    }
}

struct Page {
    slot: (usize, usize),
    refs: u32,
}

struct Snapshot {
    tokens: Vec<usize>,
    /// Hash-chain ids of the full pages, in order.
    pages: Vec<u128>,
    /// The rows past the last full page (fewer than 256), in a page slot.
    tail: Option<(usize, usize)>,
    /// SWA state then draft rings.
    state: (usize, usize),
    swa_rows: usize,
    has_draft: bool,
    next: usize,
    kind: Kind,
    last_use: u64,
}

/// Hash chain over 256-token blocks: `ids[i]` identifies the prefix
/// `tokens[..256 * (i + 1)]` (two SipHash lanes, 128 bits).
fn page_chain(tokens: &[usize]) -> Vec<u128> {
    let mut prev = 0u128;
    tokens
        .chunks_exact(KV_PAGE_ROWS)
        .map(|block| {
            let lane = |seed: u64| {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                seed.hash(&mut h);
                prev.hash(&mut h);
                block.hash(&mut h);
                h.finish()
            };
            prev = (u128::from(lane(0x6d69_6d6f)) << 64) | u128::from(lane(0x3236_6b76));
            prev
        })
        .collect()
}

/// Counters for the log line and receipts.
#[derive(Default, Debug, Clone, Copy)]
pub struct Stats {
    pub captures: u64,
    pub restores: u64,
    pub restored_tokens: u64,
    pub evicted: u64,
    pub pages_written: u64,
}

/// The host KV tier: page and state arenas, the shared page map, the snapshots.
pub struct HostCache {
    pages: Arena,
    states: Arena,
    page_map: HashMap<u128, Page>,
    snaps: Vec<Snapshot>,
    clock: u64,
    swa_bytes: usize,
    pub stats: Stats,
}

impl HostCache {
    /// `MIMO26_HOST_CACHE_GB` of page-locked RAM (0 = off), sized from `kv`'s
    /// layout: 10% for state slots (SWA + draft), the rest for GA pages. Unset,
    /// the default adapts to the host: min(32, 40% of `MemAvailable` at boot)
    /// GiB, so the recipe carries to coordinator hosts with less RAM.
    pub fn from_env(kv: &DeviceKv) -> Result<Option<Self>, String> {
        let gb: f64 = match std::env::var("MIMO26_HOST_CACHE_GB").ok().and_then(|v| v.parse().ok()) {
            Some(g) => g,
            None => {
                let avail_kb: f64 = std::fs::read_to_string("/proc/meminfo")
                    .ok()
                    .and_then(|m| m.lines().find(|l| l.starts_with("MemAvailable:")).map(str::to_string))
                    .and_then(|l| l.split_whitespace().nth(1).and_then(|v| v.parse().ok()))
                    .unwrap_or(0.0);
                (0.4 * avail_kb / (1u64 << 20) as f64).min(32.0)
            }
        };
        if gb <= 0.0 {
            return Ok(None);
        }
        let budget = (gb * (1u64 << 30) as f64) as usize;
        let page_bytes = kv.ga_page_bytes();
        let swa_bytes = kv.swa_state_bytes();
        let state_bytes = swa_bytes + kv.draft_state_bytes();
        let n_states = (budget / 10 / state_bytes).clamp(4, 256);
        let n_pages = (budget.saturating_sub(n_states * state_bytes) / page_bytes).max(16);
        let t0 = std::time::Instant::now();
        let me = Self {
            pages: Arena::new(page_bytes, n_pages)?,
            states: Arena::new(state_bytes, n_states)?,
            page_map: HashMap::new(),
            snaps: Vec::new(),
            clock: 0,
            swa_bytes,
            stats: Stats::default(),
        };
        eprintln!(
            "[hostcache] tier 1: {n_pages} GA pages of {page_bytes} B ({} tokens) + {n_states} states of {state_bytes} B, \
             {:.1} GiB page-locked in {:.1} s",
            n_pages * KV_PAGE_ROWS,
            (n_pages * page_bytes + n_states * state_bytes) as f64 / (1u64 << 30) as f64,
            t0.elapsed().as_secs_f64()
        );
        Ok(Some(me))
    }

    /// Evict one snapshot: the oldest prompt snapshot, else the oldest turn
    /// snapshot; false when none is left.
    fn evict_one(&mut self) -> bool {
        let Some(i) = (0..self.snaps.len()).min_by_key(|&i| (self.snaps[i].kind == Kind::Turn, self.snaps[i].last_use))
        else {
            return false;
        };
        let s = self.snaps.swap_remove(i);
        self.release(s);
        self.stats.evicted += 1;
        true
    }

    fn release(&mut self, s: Snapshot) {
        self.states.free.push(s.state);
        if let Some(t) = s.tail {
            self.pages.free.push(t);
        }
        for id in s.pages {
            if let Some(p) = self.page_map.get_mut(&id) {
                p.refs -= 1;
                if p.refs == 0 {
                    let slot = p.slot;
                    self.page_map.remove(&id);
                    self.pages.free.push(slot);
                }
            }
        }
    }

    fn alloc_page(&mut self) -> Option<(usize, usize)> {
        loop {
            if let Some(s) = self.pages.free.pop() {
                return Some(s);
            }
            if !self.evict_one() {
                return None;
            }
        }
    }

    fn alloc_state(&mut self) -> Option<(usize, usize)> {
        loop {
            if let Some(s) = self.states.free.pop() {
                return Some(s);
            }
            if !self.evict_one() {
                return None;
            }
        }
    }

    /// Store a snapshot of `tokens` with `next` as the token after it: GA rows
    /// `[0, tokens.len())` from `kv` (which holds at least that many), the
    /// position state from the device copy `state` (`DeviceKv::save_state_dev`,
    /// `swa_rows` SWA rows per layer). Pages already held are shared, not copied.
    /// On exhaustion the capture is dropped (the request is unaffected).
    pub fn capture(&mut self, kv: &DeviceKv, tokens: &[usize], next: usize, kind: Kind, state: &DeviceBuffer,
        swa_rows: usize) -> Result<(), String> {
        let n = tokens.len();
        if n < MIN_TOKENS || kv.tokens() < n {
            return Ok(());
        }
        self.clock += 1;
        if let Some(s) = self.snaps.iter_mut().find(|s| s.tokens == tokens) {
            s.last_use = self.clock;
            return Ok(());
        }
        let t0 = std::time::Instant::now();
        let chain = page_chain(tokens);
        let mut held: Vec<u128> = Vec::with_capacity(chain.len());
        let mut written = 0u64;
        let abort = |me: &mut Self, held: Vec<u128>| {
            // Drop the references this capture took; pages it wrote alone go free.
            for id in held {
                if let Some(p) = me.page_map.get_mut(&id) {
                    p.refs -= 1;
                    if p.refs == 0 {
                        let slot = p.slot;
                        me.page_map.remove(&id);
                        me.pages.free.push(slot);
                    }
                }
            }
        };
        for (i, &id) in chain.iter().enumerate() {
            if let Some(p) = self.page_map.get_mut(&id) {
                p.refs += 1;
                held.push(id);
                continue;
            }
            let Some(slot) = self.alloc_page() else {
                abort(self, held);
                return Ok(());
            };
            kv.export_ga_page(i * KV_PAGE_ROWS, KV_PAGE_ROWS, self.pages.get_mut(slot))?;
            self.page_map.insert(id, Page { slot, refs: 1 });
            held.push(id);
            written += 1;
        }
        let rem = n % KV_PAGE_ROWS;
        let tail = if rem > 0 {
            let Some(slot) = self.alloc_page() else {
                abort(self, held);
                return Ok(());
            };
            kv.export_ga_page(n - rem, rem, self.pages.get_mut(slot))?;
            Some(slot)
        } else {
            None
        };
        kv.sync_export()?;
        let Some(state_slot) = self.alloc_state() else {
            if let Some(t) = tail {
                self.pages.free.push(t);
            }
            abort(self, held);
            return Ok(());
        };
        let has_draft = kv.draft_state_bytes() > 0;
        let slot = state_slot;
        let buf = self.states.get_mut(slot);
        let len = kv.state_bytes().min(buf.len()).min(state.bytes());
        // SAFETY: `buf` is a live page-locked slot of at least `len` bytes, `state` a
        // live device buffer of at least `len` bytes.
        let rc = unsafe { cuda::cudaMemcpy(buf.as_mut_ptr() as _, state.as_ptr() as _, len, cuda::D2H) };
        if rc != cuda::SUCCESS {
            self.states.free.push(slot);
            if let Some(t) = tail {
                self.pages.free.push(t);
            }
            abort(self, held);
            return Err(format!("hostcache: state copy: {}", cuda::error_string(rc)));
        }
        self.snaps.push(Snapshot {
            tokens: tokens.to_vec(),
            pages: held,
            tail,
            state: slot,
            swa_rows,
            has_draft,
            next,
            kind,
            last_use: self.clock,
        });
        self.stats.captures += 1;
        self.stats.pages_written += written;
        eprintln!("[hostcache] store {kind:?} snapshot {n} tokens: {written} new of {} pages, {:.1} ms ({} snapshots, \
            {} pages held)", chain.len(), t0.elapsed().as_secs_f64() * 1e3, self.snaps.len(), self.page_map.len());
        Ok(())
    }

    /// The longest snapshot whose tokens are a prefix of `prompt`: `(index, tokens)`.
    pub fn lookup(&self, prompt: &[usize]) -> Option<(usize, usize)> {
        let chain = page_chain(prompt);
        let mut best: Option<usize> = None;
        for (i, s) in self.snaps.iter().enumerate() {
            let n = s.tokens.len();
            if n > prompt.len() || best.is_some_and(|b| self.snaps[b].tokens.len() >= n) {
                continue;
            }
            let full = n / KV_PAGE_ROWS;
            if s.pages[..] != chain[..full] || s.tokens[full * KV_PAGE_ROWS..] != prompt[full * KV_PAGE_ROWS..n] {
                continue;
            }
            best = Some(i);
        }
        best.map(|i| (i, self.snaps[i].tokens.len()))
    }

    /// Load snapshot `idx` into the fresh cache `kv`. Returns `(tokens restored,
    /// the snapshot's next token, its bank)`.
    pub fn restore(&mut self, idx: usize, kv: &mut DeviceKv) -> Result<(usize, usize, Kind), String> {
        let t0 = std::time::Instant::now();
        self.clock += 1;
        let s = &mut self.snaps[idx];
        s.last_use = self.clock;
        let n = s.tokens.len();
        kv.reset();
        kv.reserve_ga(n)?;
        for id in &s.pages {
            let p = self.page_map.get(id).ok_or("hostcache: snapshot page missing")?;
            kv.import_ga_page(KV_PAGE_ROWS, self.pages.get(p.slot))?;
        }
        if let Some(t) = s.tail {
            kv.import_ga_page(n % KV_PAGE_ROWS, self.pages.get(t))?;
        }
        kv.finish_ga_import()?;
        let buf = self.states.get(s.state);
        kv.import_swa(s.swa_rows, &buf[..self.swa_bytes])?;
        if s.has_draft {
            kv.import_draft(&buf[self.swa_bytes..])?;
        }
        kv.set_tokens(n);
        let (next, kind) = (s.next, s.kind);
        self.stats.restores += 1;
        self.stats.restored_tokens += n as u64;
        eprintln!("[hostcache] restore {n} tokens in {:.1} ms ({} restores, {} tokens total)",
            t0.elapsed().as_secs_f64() * 1e3, self.stats.restores, self.stats.restored_tokens);
        Ok((n, next, kind))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_chain_identifies_prefixes() {
        let a: Vec<usize> = (0..1000).collect();
        let mut b = a.clone();
        b[700] = 5;
        let (ca, cb) = (page_chain(&a), page_chain(&b));
        assert_eq!(ca.len(), 3);
        assert_eq!(ca[..2], cb[..2]);
        assert_ne!(ca[2], cb[2]);
        // A page's id depends on everything before it, not just its own block.
        let mut c = a.clone();
        c[3] = 9;
        let cc = page_chain(&c);
        assert!(cc.iter().zip(&ca).all(|(x, y)| x != y));
        assert_eq!(page_chain(&a[..255]).len(), 0);
    }
}
