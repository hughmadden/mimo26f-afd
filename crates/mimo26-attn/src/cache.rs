//! KV storage: paged GA KV (256-token pages) + SWA rings (ADVISOR-I3 §3 A5,
//! ARCHITECTURE.md §11.7).
//!
//! Traps pinned here:
//! * **T18** — `v_scale` (0.707) is applied **BEFORE** caching
//!   (`modeling_mimo_v2.py:301-302` caches `V×0.707`). [`RowStore::append`]
//!   scales on the way IN; reads never rescale. The naive store keeps raw V and
//!   scales on read ([`NaiveBits::VSCALE_AFTER_STORE`] /
//!   [`NaiveBits::VSCALE_ON_READ`]) — equivalent in F32, **different FP8
//!   codes** (that is the trap), pinned byte-exact in `tests/cache_t18_t8.rs`.
//! * **T8** — SWA eviction is `min(batch_pos) − window + 1`, NOT "keep the last
//!   window": a batch of `T` queries needs the last `window + T` rows or the
//!   chunk's early queries lose visible keys (oracle `kv.py:77-90` ported
//!   verbatim; `drop_keep` ∨ `drop_trim`).
//! * **T3** — GA layers never evict and never window (`AttnSpec::window_gated`).
//! * **A2** — the GA page is `PAGE_TOKENS` (256) tokens; pages are read through
//!   a page table (`GaPaged::page_rows`), never by assumption of contiguity.

use crate::fp8kv::{self, EncodedKv};
use crate::geom::{ScaleMode, PAGE_TOKENS};
use crate::{AttnError, NaiveBits};

/// How one layer's rows are held.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoreMode {
    /// f32 rows, V stored post-`v_scale` (the oracle-parity path).
    F32,
    /// FP8 E4M3, unit scale (§11.12 default — 11,520 B/token at real dims).
    Fp8Unit,
    /// FP8 E4M3, per-token×head scales, K/V separate planes (T20 extension).
    Fp8PerTokenHead,
}

impl StoreMode {
    pub fn scale_mode(self) -> Option<ScaleMode> {
        match self {
            StoreMode::F32 => None,
            StoreMode::Fp8Unit => Some(ScaleMode::Unit),
            StoreMode::Fp8PerTokenHead => Some(ScaleMode::PerTokenHead),
        }
    }
}

/// Append-order row buffer for one layer (a GA page slot or a SWA ring).
/// K is stored as given; V is stored **scaled** (`value_scale`, T18).
#[derive(Clone, Debug)]
pub struct RowStore {
    mode: StoreMode,
    n_kv: usize,
    d_qk: usize,
    d_v: usize,
    value_scale: f32,
    f32_k: Vec<f32>,
    f32_v: Vec<f32>,
    enc: Option<EncodedKv>,
    n_rows: usize,
}

impl RowStore {
    pub fn new(mode: StoreMode, n_kv: usize, d_qk: usize, d_v: usize, value_scale: f32) -> Self {
        RowStore {
            mode,
            n_kv,
            d_qk,
            d_v,
            value_scale,
            f32_k: Vec::new(),
            f32_v: Vec::new(),
            enc: None,
            n_rows: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.n_rows
    }

    pub fn is_empty(&self) -> bool {
        self.n_rows == 0
    }

    pub fn mode(&self) -> StoreMode {
        self.mode
    }

    pub fn n_kv(&self) -> usize {
        self.n_kv
    }

    pub fn d_qk(&self) -> usize {
        self.d_qk
    }

    pub fn d_v(&self) -> usize {
        self.d_v
    }

    pub fn value_scale(&self) -> f32 {
        self.value_scale
    }

    /// Append `n` rows of RAW (unscaled) K and V.
    ///
    /// T18: the correct path multiplies V by `value_scale` HERE — before any
    /// storage or quantization. `NaiveBits::VSCALE_AFTER_STORE` stores raw V
    /// (the read path then applies the scale — [`Self::rows`]).
    pub fn append(&mut self, k_raw: &[f32], v_raw: &[f32], n: usize, naive: NaiveBits) -> Result<(), AttnError> {
        if k_raw.len() != n * self.n_kv * self.d_qk || v_raw.len() != n * self.n_kv * self.d_v {
            return Err(AttnError::ShapeMismatch {
                what: format!(
                    "RowStore::append: k {} / v {} for n={} n_kv={} d_qk={} d_v={}",
                    k_raw.len(),
                    v_raw.len(),
                    n,
                    self.n_kv,
                    self.d_qk,
                    self.d_v
                ),
            });
        }
        if n == 0 {
            return Ok(());
        }
        let s = if naive.has(NaiveBits::VSCALE_AFTER_STORE) {
            1.0
        } else {
            self.value_scale
        };
        let v_stored: Vec<f32> = v_raw.iter().map(|&x| x * s).collect();
        match self.mode {
            StoreMode::F32 => {
                self.f32_k.extend_from_slice(k_raw);
                self.f32_v.extend_from_slice(&v_stored);
            }
            StoreMode::Fp8Unit | StoreMode::Fp8PerTokenHead => {
                let mode = self.mode.scale_mode().expect("fp8 mode");
                let chunk = fp8kv::encode_kv(
                    k_raw,
                    &v_stored,
                    n,
                    self.n_kv,
                    self.d_qk,
                    self.d_v,
                    mode,
                    naive,
                )?;
                match &mut self.enc {
                    None => self.enc = Some(chunk),
                    Some(dst) => merge_encoded(dst, chunk),
                }
            }
        }
        self.n_rows += n;
        Ok(())
    }

    /// All rows as f32: `(k, v)` with V in **cached (scaled)** form. FP8 rows
    /// round-trip through the codec. `NaiveBits::VSCALE_ON_READ` re-applies
    /// `value_scale` here (the double-scale bug).
    pub fn rows(&self, naive: NaiveBits) -> (Vec<f32>, Vec<f32>) {
        let read_scale = if naive.has(NaiveBits::VSCALE_ON_READ) {
            self.value_scale
        } else {
            1.0
        };
        match self.mode {
            StoreMode::F32 => {
                let v = if read_scale == 1.0 {
                    self.f32_v.clone()
                } else {
                    self.f32_v.iter().map(|&x| x * read_scale).collect()
                };
                (self.f32_k.clone(), v)
            }
            StoreMode::Fp8Unit | StoreMode::Fp8PerTokenHead => {
                let enc = self.enc.as_ref().expect("fp8 rows present");
                fp8kv::decode_kv(enc, read_scale)
            }
        }
    }

    /// Rows `[lo, hi)` as f32 `(k, v)` in cached form.
    pub fn rows_range(&self, lo: usize, hi: usize, naive: NaiveBits) -> (Vec<f32>, Vec<f32>) {
        assert!(hi <= self.n_rows && lo <= hi, "rows_range out of bounds");
        let (k, v) = self.rows(naive);
        let klen = self.n_kv * self.d_qk;
        let vlen = self.n_kv * self.d_v;
        (k[lo * klen..hi * klen].to_vec(), v[lo * vlen..hi * vlen].to_vec())
    }

    /// T18 pin accessor: the FP8 codes/scales as stored (None in F32 mode).
    pub fn encoded(&self) -> Option<&EncodedKv> {
        self.enc.as_ref()
    }

    /// F32-mode pin accessor: V exactly as stored (post-`v_scale` on the
    /// correct path).
    pub fn stored_v_f32(&self) -> &[f32] {
        &self.f32_v
    }

    /// Drop `n` leading rows (SWA eviction). O(rows) — the twin is CPU.
    pub fn drop_front(&mut self, n: usize) {
        assert!(n <= self.n_rows, "drop_front out of bounds");
        if n == 0 {
            return;
        }
        match self.mode {
            StoreMode::F32 => {
                let klen = self.n_kv * self.d_qk;
                let vlen = self.n_kv * self.d_v;
                self.f32_k.drain(..n * klen);
                self.f32_v.drain(..n * vlen);
            }
            StoreMode::Fp8Unit | StoreMode::Fp8PerTokenHead => {
                let enc = self.enc.as_mut().expect("fp8 rows present");
                let klen = self.n_kv * self.d_qk;
                let vlen = self.n_kv * self.d_v;
                enc.k_codes.drain(..n * klen);
                enc.v_codes.drain(..n * vlen);
                if enc.layout == fp8kv::Layout::PerTokenHead {
                    enc.k_scales.drain(..n * self.n_kv);
                    enc.v_scales.drain(..n * self.n_kv);
                } else if enc.layout == fp8kv::Layout::Block128Shared {
                    let blocks = fp8kv::shared_blocks_per_row(self.n_kv, self.d_qk, self.d_v);
                    enc.shared_scales.drain(..n * blocks);
                }
                enc.n_tok = enc.n_tok.saturating_sub(n);
            }
        }
        self.n_rows -= n;
    }
}

fn merge_encoded(dst: &mut EncodedKv, src: EncodedKv) {
    assert_eq!(dst.layout, src.layout, "encoded layout must match on merge");
    assert_eq!(dst.n_kv, src.n_kv, "encoded n_kv must match on merge");
    dst.k_codes.extend_from_slice(&src.k_codes);
    dst.v_codes.extend_from_slice(&src.v_codes);
    dst.k_scales.extend_from_slice(&src.k_scales);
    dst.v_scales.extend_from_slice(&src.v_scales);
    dst.shared_scales.extend_from_slice(&src.shared_scales);
    dst.clip_count += src.clip_count;
    dst.n_tok += src.n_tok;
}

// ---------------------------------------------------------------------------
// SWA ring — T8 eviction (oracle kv.py:56-95 ported)
// ---------------------------------------------------------------------------

/// One SWA layer's ring for one slot: `window`-bounded rows with absolute
/// positions, evicting only rows no query can ever see again.
#[derive(Clone, Debug)]
pub struct SwaRing {
    store: RowStore,
    pos: Vec<i64>,
    window: usize,
}

impl SwaRing {
    pub fn new(mode: StoreMode, n_kv: usize, d_qk: usize, d_v: usize, window: usize, value_scale: f32) -> Self {
        SwaRing {
            store: RowStore::new(mode, n_kv, d_qk, d_v, value_scale),
            pos: Vec::new(),
            window,
        }
    }

    pub fn len(&self) -> usize {
        self.store.len()
    }

    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Append rows at absolute `positions` (must be non-decreasing — oracle
    /// kv.py:67-68 fail-loud), then evict (T8).
    pub fn append(
        &mut self,
        k_raw: &[f32],
        v_raw: &[f32],
        positions: &[i64],
        naive: NaiveBits,
    ) -> Result<(), AttnError> {
        let n = positions.len();
        if n == 0 {
            return Ok(());
        }
        if let Some(&last) = self.pos.last() {
            if positions[0] < last {
                return Err(AttnError::KvLayout {
                    what: format!(
                        "SwaRing::append: positions must be non-decreasing ({} after {}) — oracle kv.py:67-68",
                        positions[0], last
                    ),
                });
            }
        }
        self.store.append(k_raw, v_raw, n, naive)?;
        self.pos.extend_from_slice(positions);
        let drop = self.evict_count(n, naive);
        if drop > 0 {
            self.store.drop_front(drop);
            self.pos.drain(..drop);
        }
        Ok(())
    }

    /// T8 eviction count — oracle `kv.py:77-90` verbatim semantics:
    /// `keep_from = min(batch_pos) − window + 1`;
    /// `drop_keep = #{rows with pos < keep_from}`;
    /// `drop_trim = max(0, n − (window + T))`;
    /// `drop = min(n, max(drop_keep, drop_trim))`.
    ///
    /// `NaiveBits::EVICT_KEEP_LAST` is the wrong "keep the last window"
    /// (`drop = max(0, n − window)`), which eats rows the early queries of a
    /// multi-row batch still need.
    fn evict_count(&self, batch_rows: usize, naive: NaiveBits) -> usize {
        let n = self.pos.len();
        if naive.has(NaiveBits::EVICT_KEEP_LAST) {
            return n.saturating_sub(self.window);
        }
        let batch_min = self.pos[n - batch_rows..].iter().copied().min().unwrap_or(0);
        let keep_from = batch_min - self.window as i64 + 1;
        let drop_keep = lower_bound_i64(&self.pos, keep_from);
        let drop_trim = n.saturating_sub(self.window + batch_rows);
        n.min(drop_keep.max(drop_trim))
    }

    /// `(k, v, positions)` currently visible (V in cached/scaled form).
    pub fn get(&self, naive: NaiveBits) -> (Vec<f32>, Vec<f32>, Vec<i64>) {
        let (k, v) = self.store.rows(naive);
        (k, v, self.pos.clone())
    }
}

fn lower_bound_i64(v: &[i64], x: i64) -> usize {
    let (mut lo, mut hi) = (0usize, v.len());
    while lo < hi {
        let mid = (lo + hi) / 2;
        if v[mid] < x {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

// ---------------------------------------------------------------------------
// Paged GA KV — 256-token pages + page table (A2/A5)
// ---------------------------------------------------------------------------

/// One GA layer's paged KV for one slot. Rows live in page slots reached
/// through [`GaPaged::page_table`] — the kernels use the same
/// `(logical page, offset)` arithmetic (`fp8kv::k_code_off` within a page).
#[derive(Clone, Debug)]
pub struct GaPaged {
    store_template: RowStore,
    page_tokens: usize,
    page_table: Vec<usize>, // logical page -> slot index in `slots`
    slots: Vec<RowStore>,   // physical page slots
    n_rows: usize,
}

impl GaPaged {
    pub fn new(mode: StoreMode, n_kv: usize, d_qk: usize, d_v: usize, value_scale: f32) -> Self {
        GaPaged {
            store_template: RowStore::new(mode, n_kv, d_qk, d_v, value_scale),
            page_tokens: PAGE_TOKENS,
            page_table: Vec::new(),
            slots: Vec::new(),
            n_rows: 0,
        }
    }

    pub fn with_page_tokens(mut self, page_tokens: usize) -> Self {
        assert!(page_tokens >= 1);
        self.page_tokens = page_tokens;
        self
    }

    pub fn n_tokens(&self) -> usize {
        self.n_rows
    }

    pub fn page_tokens(&self) -> usize {
        self.page_tokens
    }

    /// `(logical page, offset in page)` for a token — the A2 page arithmetic.
    pub fn page_of(&self, token: usize) -> (usize, usize) {
        (token / self.page_tokens, token % self.page_tokens)
    }

    /// Physical slot for a logical page (the indirection the page table adds).
    pub fn slot_of(&self, logical_page: usize) -> usize {
        self.page_table[logical_page]
    }

    pub fn allocated_pages(&self) -> usize {
        self.slots.len()
    }

    /// Pool bytes committed for this layer's GA KV at its storage mode.
    pub fn pool_bytes(&self, naive: NaiveBits) -> usize {
        let (n_kv, d_qk, d_v) = self.kv_dims();
        let mode = self.store_template.mode();
        match mode.scale_mode() {
            // f32 twin storage: 4 bytes per element.
            None => self.allocated_pages() * self.page_tokens * n_kv * (d_qk + d_v) * 4,
            Some(m) => {
                let layout = fp8kv::Layout::of(m, naive);
                self.allocated_pages() * self.page_tokens * fp8kv::kv_bytes_per_token(layout, n_kv, d_qk, d_v)
            }
        }
    }

    /// Append `n` rows (raw K/V; V scaled before storage, T18), spilling into
    /// new pages as needed. GA never evicts (T3).
    pub fn append(&mut self, k_raw: &[f32], v_raw: &[f32], n: usize, naive: NaiveBits) -> Result<(), AttnError> {
        let mut done = 0usize;
        let (n_kv, d_qk, d_v) = self.kv_dims();
        let klen = n_kv * d_qk;
        let vlen = n_kv * d_v;
        while done < n {
            let logical_page = self.n_rows / self.page_tokens;
            let off = self.n_rows % self.page_tokens;
            let take = (self.page_tokens - off).min(n - done);
            if logical_page == self.page_table.len() {
                let slot = self.slots.len();
                let mode = self.store_template.mode();
                let vs = self.store_template.value_scale();
                self.slots.push(RowStore::new(mode, n_kv, d_qk, d_v, vs));
                self.page_table.push(slot);
            }
            let slot = self.page_table[logical_page];
            let k = &k_raw[done * klen..(done + take) * klen];
            let v = &v_raw[done * vlen..(done + take) * vlen];
            self.slots[slot].append(k, v, take, naive)?;
            self.n_rows += take;
            done += take;
        }
        Ok(())
    }

    /// Read rows `[lo, hi)` (cached form) through the page table.
    pub fn rows_range(&self, lo: usize, hi: usize, naive: NaiveBits) -> (Vec<f32>, Vec<f32>) {
        assert!(hi <= self.n_rows && lo <= hi, "rows_range out of bounds");
        let klen = self.kv_dims().0 * self.kv_dims().1;
        let vlen = self.kv_dims().0 * self.kv_dims().2;
        let mut k = Vec::with_capacity((hi - lo) * klen);
        let mut v = Vec::with_capacity((hi - lo) * vlen);
        let mut t = lo;
        while t < hi {
            let (page, off) = self.page_of(t);
            let take = (self.page_tokens - off).min(hi - t);
            let slot = self.slot_of(page);
            let (pk, pv) = self.slots[slot].rows_range(off, off + take, naive);
            k.extend_from_slice(&pk);
            v.extend_from_slice(&pv);
            t += take;
        }
        (k, v)
    }

    /// One page's rows as stored (the unit of chunked prefill / tier copy).
    pub fn page_rows(&self, logical_page: usize, naive: NaiveBits) -> (Vec<f32>, Vec<f32>) {
        let slot = self.slot_of(logical_page);
        let lo = logical_page * self.page_tokens;
        let hi = (lo + self.page_tokens).min(self.n_rows);
        assert!(hi > lo, "page_rows: empty page");
        self.slots[slot].rows_range(0, hi - lo, naive)
    }

    fn kv_dims(&self) -> (usize, usize, usize) {
        (self.store_template.n_kv(), self.store_template.d_qk(), self.store_template.d_v())
    }
}
