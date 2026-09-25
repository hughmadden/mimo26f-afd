//! GQA-packed attention math + the kernel decompositions (A5).
//!
//! [`attention`] is the **two-pass** reference (explicit max, then softmax sum —
//! the same shape as `oracle/mimo26/nn/layers.py:69-126`, computed in f64 and
//! returned f32). The decomposition paths — [`decode_split_kv`] (flash-decoding:
//! per-split `(m, l, o)` partials + [`reduce_partials`]) and
//! [`attention_chunked`] (chunked prefill over pages with running state) — use a
//! **one-pass online** accumulator instead, so `tests/splitkv_prefill_equiv.rs`
//! cross-checks two independent algorithms AND `tests/oracle_parity.rs` locks
//! both to the numpy oracle (I-Gold; never self-consistency alone).
//!
//! Contract details (all trap-pinned):
//! * **GQA packing** — KV head `kvh` serves Q heads `kvh·n_rep … (kvh+1)·n_rep−1`
//!   (`layers.py:110-113` folds per KV head; never materialize repeated KV).
//! * **QK 192 / V 128 (T4/c2)** — the logit scale is `1/√d_qk`, never `1/√d_v`
//!   [`attn_scale`]; V rows are `d_v` wide and read as such.
//! * **Sink (T6/c1)** — an extra softmax COLUMN with per-Q-head logit `[64]` and
//!   **zero value**; it absorbs mass and contributes nothing to the output.
//!   SWA layers only: a GA layer with a sink argument is **bitwise identical**
//!   to one without (c1) — the sink is ignored, never applied.
//! * **Window (T3)** — `AttnSpec::window_gated()`: SWA sees `0 ≤ q_pos−k_pos <
//!   128`; GA sees all past keys.
//! * **T18** — V arrives in CACHED form (already ×0.707; see `cache::RowStore`).
//!   [`NaiveBits::VSCALE_ON_READ`] re-scales on read (double-scale bug).
//! * **T9** — absolute `q_pos`/`k_pos` are honored [`NaiveBits::POS_ZEROED`]
//!   zeroes them (the silent post-step-1 garbage).
//! * **Split-KV sink** — the sink column exists ONCE per query, so it is added
//!   once at [`reduce_partials`], never once per split
//!   [`NaiveBits::SINK_PER_SPLIT`] (that counts it `n_splits` times).

use crate::cache::GaPaged;
use crate::geom::AttnSpec;
use crate::{AttnError, Family, NaiveBits};

/// Logit scale — `1/√d_qk` (oracle `layers.py:115` divides by `√Dqk`).
/// c2/T4: the naive port scales by the V width (`d_v`) — the QK/V mixup.
pub fn attn_scale(d_qk: usize, d_v: usize, naive: NaiveBits) -> f64 {
    if naive.has(NaiveBits::SCALE_BY_DV) {
        1.0 / (d_v as f64).sqrt()
    } else {
        1.0 / (d_qk as f64).sqrt()
    }
}

/// Sink logit for Q head `h`. T6: the bias is **per Q head** (`[64]` on the real
/// model, `oracle/mimo26/model.py:152-153`). `NaiveBits::SINK_PER_KV` indexes
/// per KV head (`h / n_rep`) — wrong the moment the bias differs within a GQA
/// group.
fn sink_logit(sink: &[f32], h: usize, n_rep: usize, naive: NaiveBits) -> f64 {
    if naive.has(NaiveBits::SINK_PER_KV) {
        f64::from(sink[h / n_rep])
    } else {
        f64::from(sink[h])
    }
}

/// Causal + optional window visibility (oracle `layers.py:99-102`):
/// `k_pos <= q_pos` and, when windowed, `q_pos - k_pos < window`.
fn is_visible(q_pos: i64, k_pos: i64, window: Option<usize>) -> bool {
    if k_pos > q_pos {
        return false;
    }
    match window {
        None => true,
        Some(w) => q_pos - k_pos < w as i64,
    }
}

/// Whether the sink column is live for this spec+call (c1 family gate).
fn sink_active(spec: &AttnSpec, sink: Option<&[f32]>, naive: NaiveBits) -> bool {
    match sink {
        None => false,
        Some(_) => spec.sink_allowed() || naive.has(NaiveBits::SINK_ON_GA),
    }
}

fn check_sink(spec: &AttnSpec, sink: Option<&[f32]>) -> Result<(), AttnError> {
    if let Some(s) = sink {
        if s.len() != spec.n_q {
            return Err(AttnError::SinkLength { got: s.len(), expected: spec.n_q });
        }
    }
    Ok(())
}

/// Fail-loud entry: a sink on a GA layer is a caller bug — error instead of the
/// bitwise-ignore that [`attention`] does (c1 has both pins).
pub fn attention_checked(
    spec: &AttnSpec,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_pos: &[i64],
    k_pos: &[i64],
    sink: Option<&[f32]>,
    naive: NaiveBits,
) -> Result<Vec<f32>, AttnError> {
    if sink.is_some() && !spec.sink_allowed() {
        return Err(AttnError::SinkNotSupported {
            family: format!("{:?}", spec.family),
        });
    }
    attention(spec, q, k, v, q_pos, k_pos, sink, naive)
}

/// V element `(j, kvh, i)` — correct reads the `d_v`-wide row.
/// `NaiveBits::V_BROADCAST` is the bug oracle that assumes `d_v == d_qk` and
/// reads at the QK stride (on a real buffer that is out-of-bounds memory; the
/// twin clamps to model the silent misread).
fn v_at(v: &[f32], j: usize, kvh: usize, i: usize, d_v: usize, d_qk: usize, n_kv: usize, naive: NaiveBits) -> f64 {
    if naive.has(NaiveBits::V_BROADCAST) {
        let flat = (j * n_kv + kvh) * d_qk + i;
        f64::from(v[flat.min(v.len() - 1)])
    } else {
        f64::from(v[(j * n_kv + kvh) * d_v + i])
    }
}

/// Two-pass attention reference. `v` is CACHED V (post-`v_scale`, T18).
/// Returns `[T][n_q][d_v]` (f32).
///
/// A query with no visible key gets an all-zero output row (oracle
/// `layers.py:123-124`) — including when a sink exists (its value is zero).
pub fn attention(
    spec: &AttnSpec,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_pos: &[i64],
    k_pos: &[i64],
    sink: Option<&[f32]>,
    naive: NaiveBits,
) -> Result<Vec<f32>, AttnError> {
    let AttnSpec { n_q, n_kv, d_qk, d_v, .. } = *spec;
    let t_len = q_pos.len();
    let s_len = k_pos.len();
    if q.len() != t_len * n_q * d_qk {
        return Err(AttnError::ShapeMismatch { what: "q length".into() });
    }
    if k.len() != s_len * n_kv * d_qk {
        return Err(AttnError::ShapeMismatch { what: "k length (T4: QK width is d_qk)".into() });
    }
    if v.len() != s_len * n_kv * d_v {
        return Err(AttnError::ShapeMismatch { what: "v length (T4: V width is d_v, no broadcast)".into() });
    }
    check_sink(spec, sink)?;
    let scale = attn_scale(d_qk, d_v, naive);
    let window = if spec.family == Family::Ga && !naive.has(NaiveBits::GA_WINDOWED) {
        None
    } else {
        spec.window_gated()
    };
    let use_sink = sink_active(spec, sink, naive);
    let read_scale = if naive.has(NaiveBits::VSCALE_ON_READ) {
        f64::from(spec.value_scale)
    } else {
        1.0
    };
    let n_rep = spec.n_rep();
    let mut out = vec![0f32; t_len * n_q * d_v];
    for t in 0..t_len {
        let qp = if naive.has(NaiveBits::POS_ZEROED) { 0 } else { q_pos[t] };
        for h in 0..n_q {
            let kvh = h / n_rep;
            let qoff = (t * n_q + h) * d_qk;
            // pass 1: max over visible keys (+ the sink column, layers.py:117-120)
            let mut m = f64::NEG_INFINITY;
            let mut any = false;
            let mut visible = vec![false; s_len];
            for j in 0..s_len {
                let kp = if naive.has(NaiveBits::POS_ZEROED) { 0 } else { k_pos[j] };
                if is_visible(qp, kp, window) {
                    visible[j] = true;
                    any = true;
                    let mut s = 0.0f64;
                    for i in 0..d_qk {
                        s += f64::from(q[qoff + i]) * f64::from(k[(j * n_kv + kvh) * d_qk + i]);
                    }
                    s *= scale;
                    m = m.max(s);
                }
            }
            if !any {
                continue; // zero row (layers.py:123-124)
            }
            if use_sink {
                let s = sink_logit(sink.expect("checked"), h, n_rep, naive);
                m = m.max(s);
            }
            // pass 2: softmax weights + value sum (the sink contributes 0 to o)
            let mut l = 0.0f64;
            let mut acc = vec![0.0f64; d_v];
            for j in 0..s_len {
                if !visible[j] {
                    continue;
                }
                let mut s = 0.0f64;
                for i in 0..d_qk {
                    s += f64::from(q[qoff + i]) * f64::from(k[(j * n_kv + kvh) * d_qk + i]);
                }
                s *= scale;
                let p = (s - m).exp();
                l += p;
                for i in 0..d_v {
                    acc[i] += p * v_at(v, j, kvh, i, d_v, d_qk, n_kv, naive) * read_scale;
                }
            }
            if use_sink {
                l += (sink_logit(sink.expect("checked"), h, n_rep, naive) - m).exp();
            }
            let lo = (t * n_q + h) * d_v;
            if l > 0.0 {
                for i in 0..d_v {
                    out[lo + i] = (acc[i] / l) as f32;
                }
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// online-softmax decomposition (split-KV decode + chunked prefill)
// ---------------------------------------------------------------------------

/// One `(query, Q head)` accumulator: running max `m`, denominator `l`, and
/// value numerator `o` (f64 — the reduce must be exact enough at 1M keys).
#[derive(Clone, Debug)]
pub struct Partial {
    pub m: f64,
    pub l: f64,
    pub o: Vec<f64>,
}

impl Partial {
    pub fn new(d_v: usize) -> Self {
        Partial { m: f64::NEG_INFINITY, l: 0.0, o: vec![0.0; d_v] }
    }

    /// Fold one key `(logit s, value row v)` in, rescaling by `exp(old_m−m)`.
    /// `NaiveBits::NO_RUNNING_RESCALE` skips the rescale — the classic broken
    /// online softmax (wrong whenever a later chunk holds a larger logit).
    pub fn fold(&mut self, s: f64, val: &[f64], naive: NaiveBits) {
        let m_new = self.m.max(s);
        if naive.has(NaiveBits::NO_RUNNING_RESCALE) {
            let p = (s - m_new).exp();
            self.l += p;
            for i in 0..self.o.len() {
                self.o[i] += p * val[i];
            }
            self.m = m_new;
            return;
        }
        let rescale = if self.m.is_finite() { (self.m - m_new).exp() } else { 0.0 };
        self.l *= rescale;
        for x in self.o.iter_mut() {
            *x *= rescale;
        }
        let p = (s - m_new).exp();
        self.l += p;
        for i in 0..self.o.len() {
            self.o[i] += p * val[i];
        }
        self.m = m_new;
    }
}

/// Fold key range `[j_lo, j_hi)` of layer rows into `p` for `(q_row, kvh)`.
#[allow(clippy::too_many_arguments)]
pub fn fold_range_kvh(
    p: &mut Partial,
    spec: &AttnSpec,
    kvh: usize,
    q_row: &[f32],
    k: &[f32],
    v: &[f32],
    j_lo: usize,
    j_hi: usize,
    q_pos: i64,
    k_pos: &[i64],
    window: Option<usize>,
    naive: NaiveBits,
) {
    let AttnSpec { n_kv, d_qk, d_v, .. } = *spec;
    let scale = attn_scale(d_qk, d_v, naive);
    let read_scale = if naive.has(NaiveBits::VSCALE_ON_READ) {
        f64::from(spec.value_scale)
    } else {
        1.0
    };
    for j in j_lo..j_hi {
        let kp = if naive.has(NaiveBits::POS_ZEROED) { 0 } else { k_pos[j] };
        if !is_visible(q_pos, kp, window) {
            continue;
        }
        let mut s = 0.0f64;
        for i in 0..d_qk {
            s += f64::from(q_row[i]) * f64::from(k[(j * n_kv + kvh) * d_qk + i]);
        }
        s *= scale;
        let val: Vec<f64> = (0..d_v)
            .map(|i| v_at(v, j, kvh, i, d_v, d_qk, n_kv, naive) * read_scale)
            .collect();
        p.fold(s, &val, naive);
    }
}

/// Reduce per-split partials to `(denominator l, numerator o)` for one `(t, h)`.
///
/// The sink column exists **once per query**: it enters the denominator here,
/// once, with `m = max(max_i m_i, sink)` (its value is zero, so `o` is
/// untouched). `NaiveBits::SINK_PER_SPLIT` folds a sink into EVERY split's
/// `(m, l)` first — counting the column `n_splits` times.
pub fn reduce_partials(partials: &[Partial], sink_logit_v: Option<f64>, naive: NaiveBits) -> (f64, Vec<f64>) {
    let d_v = partials.first().map(|p| p.o.len()).unwrap_or(0);
    let per_split_sink = naive.has(NaiveBits::SINK_PER_SPLIT);
    let mut m = f64::NEG_INFINITY;
    for p in partials {
        let pm = if per_split_sink {
            match sink_logit_v {
                Some(b) => p.m.max(b),
                None => p.m,
            }
        } else {
            p.m
        };
        m = m.max(pm);
    }
    if !per_split_sink {
        if let Some(b) = sink_logit_v {
            m = m.max(b);
        }
    }
    let mut l = 0.0f64;
    let mut o = vec![0.0f64; d_v];
    for p in partials {
        let (pm, pl) = if per_split_sink {
            // bug: this split carried its own sink column
            match sink_logit_v {
                Some(b) => {
                    let pm = p.m.max(b);
                    let mut pl = if p.m.is_finite() { p.l * (p.m - pm).exp() } else { 0.0 };
                    pl += (b - pm).exp();
                    (pm, pl)
                }
                None => (p.m, p.l),
            }
        } else {
            (p.m, p.l)
        };
        if !pm.is_finite() && pl == 0.0 {
            continue;
        }
        let w = (pm - m).exp();
        l += w * pl;
        for i in 0..d_v {
            o[i] += w * p.o[i];
        }
    }
    if !per_split_sink {
        if let Some(b) = sink_logit_v {
            l += (b - m).exp();
        }
    }
    (l, o)
}

fn window_for(spec: &AttnSpec, naive: NaiveBits) -> Option<usize> {
    if spec.family == Family::Ga && !naive.has(NaiveBits::GA_WINDOWED) {
        None
    } else {
        spec.window_gated()
    }
}

/// Flash-decoding decomposition: split the S cached rows into `n_splits`
/// contiguous ranges, fold each into a [`Partial`], then [`reduce_partials`].
/// Mathematically [`attention`] (pinned in `tests/splitkv_prefill_equiv.rs`);
/// this is what `kernels/attn_decode_splitkv.cu` + `attn_reduce.cu` compute.
#[allow(clippy::too_many_arguments)]
pub fn decode_split_kv(
    spec: &AttnSpec,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_pos: &[i64],
    k_pos: &[i64],
    sink: Option<&[f32]>,
    n_splits: usize,
    naive: NaiveBits,
) -> Result<Vec<f32>, AttnError> {
    let AttnSpec { n_q, n_kv, d_qk, d_v, .. } = *spec;
    let t_len = q_pos.len();
    let s_len = k_pos.len();
    if q.len() != t_len * n_q * d_qk || k.len() != s_len * n_kv * d_qk || v.len() != s_len * n_kv * d_v {
        return Err(AttnError::ShapeMismatch { what: "decode_split_kv shapes (T4)".into() });
    }
    check_sink(spec, sink)?;
    let n_rep = spec.n_rep();
    let window = window_for(spec, naive);
    let use_sink = sink_active(spec, sink, naive);
    let n_splits = n_splits.max(1);
    let mut out = vec![0f32; t_len * n_q * d_v];
    for t in 0..t_len {
        let qp = if naive.has(NaiveBits::POS_ZEROED) { 0 } else { q_pos[t] };
        for h in 0..n_q {
            let kvh = h / n_rep;
            let q_row = &q[(t * n_q + h) * d_qk..(t * n_q + h) * d_qk + d_qk];
            let mut partials = Vec::with_capacity(n_splits);
            for sp in 0..n_splits {
                let lo = s_len * sp / n_splits;
                let hi = s_len * (sp + 1) / n_splits;
                let mut p = Partial::new(d_v);
                fold_range_kvh(&mut p, spec, kvh, q_row, k, v, lo, hi, qp, k_pos, window, naive);
                partials.push(p);
            }
            let sl = if use_sink {
                Some(sink_logit(sink.expect("checked"), h, n_rep, naive))
            } else {
                None
            };
            let (l, o) = reduce_partials(&partials, sl, naive);
            let lo = (t * n_q + h) * d_v;
            if l > 0.0 {
                for i in 0..d_v {
                    out[lo + i] = (o[i] / l) as f32;
                }
            }
        }
    }
    Ok(out)
}

/// Chunked prefill: fold keys in `keys_per_chunk` chunks with running state,
/// then one reduce (the `kernels/attn_prefill_chunk.cu` schedule). Equals
/// [`attention`]; `NaiveBits::NO_RUNNING_RESCALE` / `SINK_PER_SPLIT` are the
/// two ways this decomposition goes wrong.
#[allow(clippy::too_many_arguments)]
pub fn attention_chunked(
    spec: &AttnSpec,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_pos: &[i64],
    k_pos: &[i64],
    sink: Option<&[f32]>,
    keys_per_chunk: usize,
    naive: NaiveBits,
) -> Result<Vec<f32>, AttnError> {
    let AttnSpec { n_q, n_kv, d_qk, d_v, .. } = *spec;
    let t_len = q_pos.len();
    let s_len = k_pos.len();
    if q.len() != t_len * n_q * d_qk || k.len() != s_len * n_kv * d_qk || v.len() != s_len * n_kv * d_v {
        return Err(AttnError::ShapeMismatch { what: "attention_chunked shapes (T4)".into() });
    }
    check_sink(spec, sink)?;
    let n_rep = spec.n_rep();
    let window = window_for(spec, naive);
    let use_sink = sink_active(spec, sink, naive);
    let keys_per_chunk = keys_per_chunk.max(1);
    let mut out = vec![0f32; t_len * n_q * d_v];
    for t in 0..t_len {
        let qp = if naive.has(NaiveBits::POS_ZEROED) { 0 } else { q_pos[t] };
        for h in 0..n_q {
            let kvh = h / n_rep;
            let q_row = &q[(t * n_q + h) * d_qk..(t * n_q + h) * d_qk + d_qk];
            let mut p = Partial::new(d_v);
            let mut lo = 0usize;
            while lo < s_len {
                let hi = (lo + keys_per_chunk).min(s_len);
                fold_range_kvh(&mut p, spec, kvh, q_row, k, v, lo, hi, qp, k_pos, window, naive);
                lo = hi;
            }
            let sl = if use_sink {
                Some(sink_logit(sink.expect("checked"), h, n_rep, naive))
            } else {
                None
            };
            let (l, o) = reduce_partials(&[p], sl, naive);
            let lo = (t * n_q + h) * d_v;
            if l > 0.0 {
                for i in 0..d_v {
                    out[lo + i] = (o[i] / l) as f32;
                }
            }
        }
    }
    Ok(out)
}

/// Paged GA prefill/decode: read through the page table (256-token pages) and
/// fold page by page — the A5 "chunked prefill over pages" path. Equals
/// [`attention`] over the same rows with the caller's **absolute** `k_pos`
/// (`len == ga.n_tokens()`), honored for causal/window visibility. A paged cache
/// that starts mid-sequence hands back rows at non-0-based positions, so `k_pos`
/// is **not** implied to be `0..n_tokens`.
/// `NaiveBits::PAGED_KPOS_ZERO_BASED` reproduces the 0-based hardcode.
#[allow(clippy::too_many_arguments)]
pub fn attention_paged(
    spec: &AttnSpec,
    q: &[f32],
    q_pos: &[i64],
    k_pos: &[i64],
    sink: Option<&[f32]>,
    ga: &GaPaged,
    pages_per_chunk: usize,
    naive: NaiveBits,
) -> Result<Vec<f32>, AttnError> {
    let AttnSpec { n_q, d_qk, d_v, .. } = *spec;
    let t_len = q_pos.len();
    let s_len = ga.n_tokens();
    if q.len() != t_len * n_q * d_qk {
        return Err(AttnError::ShapeMismatch { what: "attention_paged q".into() });
    }
    if k_pos.len() != s_len {
        return Err(AttnError::ShapeMismatch {
            what: format!("attention_paged k_pos {} != n_tokens {}", k_pos.len(), s_len),
        });
    }
    check_sink(spec, sink)?;
    let n_rep = spec.n_rep();
    let window = window_for(spec, naive);
    let use_sink = sink_active(spec, sink, naive);
    let page_tokens = ga.page_tokens();
    let pages_per_chunk = pages_per_chunk.max(1);
    // T9 (paged): honor the caller's absolute key positions. The naive port
    // hardcodes 0-based row indices (`k_pos = 0..n_tokens`) — wrong for any
    // cache that doesn't start at position 0.
    let kpos_eff: Vec<i64> = if naive.has(NaiveBits::PAGED_KPOS_ZERO_BASED) {
        (0..s_len as i64).collect()
    } else {
        k_pos.to_vec()
    };
    let mut out = vec![0f32; t_len * n_q * d_v];
    for t in 0..t_len {
        let qp = if naive.has(NaiveBits::POS_ZEROED) { 0 } else { q_pos[t] };
        for h in 0..n_q {
            let kvh = h / n_rep;
            let q_row = &q[(t * n_q + h) * d_qk..(t * n_q + h) * d_qk + d_qk];
            let mut p = Partial::new(d_v);
            let n_pages = (s_len + page_tokens - 1) / page_tokens;
            let mut pg = 0usize;
            while pg < n_pages {
                let pg_hi = (pg + pages_per_chunk).min(n_pages);
                let lo = pg * page_tokens;
                let hi = (pg_hi * page_tokens).min(s_len);
                let (k_rows, v_rows) = ga.rows_range(lo, hi, naive);
                fold_range_kvh(&mut p, spec, kvh, q_row, &k_rows, &v_rows, 0, hi - lo, qp, &kpos_eff[lo..hi], window, naive);
                pg = pg_hi;
            }
            let sl = if use_sink {
                Some(sink_logit(sink.expect("checked"), h, n_rep, naive))
            } else {
                None
            };
            let (l, o) = reduce_partials(&[p], sl, naive);
            let lo = (t * n_q + h) * d_v;
            if l > 0.0 {
                for i in 0..d_v {
                    out[lo + i] = (o[i] / l) as f32;
                }
            }
        }
    }
    Ok(out)
}
