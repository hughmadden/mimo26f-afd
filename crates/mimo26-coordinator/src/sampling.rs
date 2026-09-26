//! Served sampling (perf reset V3): DS41RT v15's target-sampling contract.
//!
//! - **Greedy is the default.** A request without `temperature`, or with `temperature < 1e-5` or
//!   `top_k == 1`, takes the argmax exactly as before (the checkpoint's `generation_config` has
//!   `do_sample: false`). Once a request samples, filters it leaves out are off (`top_p = 1`, no
//!   `top_k`, `min_p = 0`), never inherited.
//! - **Filters in vLLM's order:** temperature, `min_p`, `top_k`, `top_p`, then an exact
//!   categorical draw (`kernels/sample.cu`). Ties at a top-k or top-p boundary are kept (vLLM;
//!   DS41RT keeps exactly `k`, lowest id first).
//! - **Draws are a function of `(seed, position)`**, `position` being the index of the emitted
//!   token (0 for the first token after the prompt). DS41RT's SplitMix64 with its target-sampling
//!   domain; the full 64-bit value scales the kept weight. So batching, speculation and the cache
//!   state cannot change a token, and a request without `seed` gets a random one.
//! - **Speculation stays exact** (DS41RT's sample-and-match): every verify row draws its own
//!   target token, a draft is accepted while it equals the draw, and the first mismatch emits the
//!   draw. With single-token drafts that is speculative sampling's accept-with-probability-p(d),
//!   resample-from-the-rest rule, and the emitted token is always the target's own draw.
//! - Only token ids the tokenizer knows are drawn (151,675 of the 152,576 lm_head rows for MiMo):
//!   the padding rows and the drafter's mask id can never come out.
//!
//! The older `sampler.rs` (T23: repetition penalty, temperature 1.0 / top_p 0.95 defaults) is the
//! port's CPU reference from before serving existed; the server never used it.

use std::sync::Arc;

/// Temperatures below this are greedy (vLLM's `_SAMPLING_EPS`, DS41RT's `GREEDY_TEMPERATURE_EPS`).
pub const GREEDY_EPS: f32 = 1.0e-5;
/// The largest accepted temperature (the OpenAI range, as DS41RT).
pub const MAX_TEMPERATURE: f32 = 2.0;
/// Below `m - FLOOR` a token's fixed-point weight is 0 (`exp(-28) * 2^40 < 1`).
const FLOOR: f32 = 28.0;
const Q_SCALE: f32 = 1_099_511_627_776.0; // 2^40

/// A sampled request's parameters (greedy requests have none).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampling {
    pub temperature: f32,
    /// `>= 1`: off.
    pub top_p: f32,
    /// 0: off.
    pub top_k: usize,
    /// 0: off.
    pub min_p: f32,
    pub seed: u64,
}

impl Sampling {
    /// The request's sampling, or `None` for greedy (`temperature < 1e-5` or `top_k == 1`).
    /// `top_k` 0 is off; `seed` `None` draws one. Errors name the offending parameter.
    pub fn new(temperature: f32, top_p: f32, top_k: usize, min_p: f32, seed: Option<u64>)
        -> Result<Option<Sampling>, String> {
        if !temperature.is_finite() || !(0.0..=MAX_TEMPERATURE).contains(&temperature) {
            return Err("temperature must be a number between 0 and 2".into());
        }
        if !top_p.is_finite() || !(top_p > 0.0 && top_p <= 1.0) {
            return Err("top_p must be greater than 0 and at most 1".into());
        }
        if !min_p.is_finite() || !(0.0..=1.0).contains(&min_p) {
            return Err("min_p must be a number between 0 and 1".into());
        }
        if temperature < GREEDY_EPS || top_k == 1 {
            return Ok(None);
        }
        Ok(Some(Sampling { temperature, top_p, top_k, min_p, seed: seed.unwrap_or_else(random_seed) }))
    }

    /// The 64-bit draw for emitted-token `position`: DS41RT v15's `random_uniform` mix (SplitMix64
    /// over its target-sampling domain, the seed and the position), all 64 bits.
    pub fn draw(&self, position: u64) -> u64 {
        const DOMAIN: u64 = 0x7f4a_7c15_9e37_79b9;
        let mut x = self
            .seed
            .wrapping_add(DOMAIN)
            .wrapping_add(position.wrapping_mul(0x9e37_79b9_7f4a_7c15))
            .wrapping_add(0x9e37_79b9_7f4a_7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    }

    /// The kernel's work item: logit row `row` of the batch, drawn for emitted-token `position`.
    pub fn row(&self, row: usize, position: u64) -> DeviceRow {
        DeviceRow {
            row: row as i32,
            inv_t: 1.0 / self.temperature,
            top_p: self.top_p,
            ln_min_p: if self.min_p > 0.0 { self.min_p.ln() } else { f32::NEG_INFINITY },
            top_k: self.top_k.min(i32::MAX as usize) as i32,
            pad: 0,
            rnd: self.draw(position),
        }
    }
}

/// A seed for a request that gave none: the clock mixed with a process counter (DS41RT's).
fn random_seed() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_nanos() as u64;
    now ^ NEXT.fetch_add(0x9e37_79b9_7f4a_7c15, Ordering::Relaxed)
}

/// One sampled row for `m26c_sample_rows` (`kernels/sample.cu` `Row`, 32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeviceRow {
    pub row: i32,
    pub inv_t: f32,
    pub top_p: f32,
    pub ln_min_p: f32,
    pub top_k: i32,
    pub pad: i32,
    pub rnd: u64,
}

/// What a snapshot knows about the token after it. `greedy` is the argmax there when known: always
/// for a prompt snapshot, for a turn snapshot only when its request was greedy (a sampled
/// request's last token is a draw). `logits` is the last row's logits, kept for prompt snapshots so
/// a sampled request resuming exactly there draws its first token without a forward.
#[derive(Clone, Debug, Default)]
pub struct After {
    pub greedy: Option<usize>,
    pub logits: Option<Arc<Vec<f32>>>,
}

impl After {
    /// From a prompt's last logit row: the argmax, and the row itself when `keep`.
    pub fn from_logits(logits: &[f32], keep: bool) -> After {
        After { greedy: Some(crate::sampler::greedy(logits)), logits: keep.then(|| Arc::new(logits.to_vec())) }
    }

    /// Whether a request resuming exactly here gets its first token without a forward.
    pub fn serves(&self, sampled: bool) -> bool {
        if sampled { self.logits.is_some() } else { self.greedy.is_some() }
    }
}

/// Order-preserving key (the kernel's `okey`): larger float, larger key; -0 == +0; NaN is 0.
fn okey(s: f32) -> u32 {
    if s.is_nan() {
        return 0;
    }
    let b = (if s == 0.0 { 0.0f32 } else { s }).to_bits();
    if b & 0x8000_0000 != 0 { !b } else { b | 0x8000_0000 }
}

fn weight(s: f32, m: f32) -> u64 {
    // The kernel truncates toward zero (`__float2ull_rz`); `as` does the same.
    ((s - m).exp() * Q_SCALE) as u64
}

/// The largest key `t` with `sum(w_i : key_i >= t) >= target` (keys and weights of the candidates).
fn threshold(mut kw: Vec<(u32, u64)>, target: u64) -> u32 {
    kw.sort_unstable_by(|a, b| b.0.cmp(&a.0));
    let mut cum = 0u64;
    let mut i = 0;
    while i < kw.len() {
        let k = kw[i].0;
        while i < kw.len() && kw[i].0 == k {
            cum += kw[i].1;
            i += 1;
        }
        if cum >= target {
            return k;
        }
    }
    kw.last().map_or(0, |x| x.0)
}

/// The CPU reference of `m26c_sample_rows` for one row: the drawn id among `logits[..vocab]`, or
/// `None` for a non-finite row (the kernel then leaves the argmax). Integer arithmetic throughout
/// after the maximum, as the kernel; only `exp` may differ in its last bit from the GPU's `expf`.
pub fn select(logits: &[f32], vocab: usize, r: &DeviceRow) -> Option<usize> {
    let l = &logits[..vocab];
    let s: Vec<f32> = l.iter().map(|&x| x * r.inv_t).collect();
    let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !m.is_finite() {
        return None;
    }
    let smin = if r.ln_min_p > f32::NEG_INFINITY { m + r.ln_min_p } else { f32::NEG_INFINITY };
    let kfloor = okey(smin.max(m - FLOOR));
    let keys: Vec<u32> = s.iter().map(|&x| okey(x)).collect();
    let mut kk = kfloor;
    if r.top_k > 0 {
        let n = keys.iter().filter(|&&k| k >= kfloor).count();
        if n > r.top_k as usize {
            kk = threshold(keys.iter().filter(|&&k| k >= kfloor).map(|&k| (k, 1)).collect(), r.top_k as u64);
        }
    }
    let mut kp = kk;
    if r.top_p < 1.0 {
        let kw: Vec<(u32, u64)> = (0..vocab).filter(|&i| keys[i] >= kk).map(|i| (keys[i], weight(s[i], m))).collect();
        let q: u64 = kw.iter().map(|x| x.1).sum();
        let target = ((r.top_p as f64 * q as f64).ceil() as u64).clamp(1, q);
        kp = threshold(kw, target);
    }
    let z: u64 = (0..vocab).filter(|&i| keys[i] >= kp).map(|i| weight(s[i], m)).sum();
    let draw = ((u128::from(r.rnd) * u128::from(z)) >> 64) as u64;
    let mut acc = 0u64;
    for i in 0..vocab {
        if keys[i] >= kp {
            acc += weight(s[i], m);
            if draw < acc {
                return Some(i);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(t: f32, top_p: f32, top_k: usize, min_p: f32, seed: u64, pos: u64) -> DeviceRow {
        Sampling::new(t, top_p, top_k, min_p, Some(seed)).unwrap().expect("sampled").row(0, pos)
    }

    #[test]
    fn greedy_and_validation() {
        assert_eq!(Sampling::new(0.0, 1.0, 0, 0.0, None), Ok(None));
        assert_eq!(Sampling::new(0.9, 1.0, 1, 0.0, None), Ok(None));
        assert_eq!(Sampling::new(5e-6, 0.5, 0, 0.0, None), Ok(None));
        assert!(Sampling::new(2.5, 1.0, 0, 0.0, None).is_err());
        assert!(Sampling::new(-0.1, 1.0, 0, 0.0, None).is_err());
        assert!(Sampling::new(1.0, 0.0, 0, 0.0, None).is_err());
        assert!(Sampling::new(1.0, 1.5, 0, 0.0, None).is_err());
        assert!(Sampling::new(1.0, 1.0, 0, 1.5, None).is_err());
        assert!(Sampling::new(f32::NAN, 1.0, 0, 0.0, None).is_err());
        let s = Sampling::new(0.7, 0.9, 40, 0.05, Some(7)).unwrap().unwrap();
        assert_eq!((s.top_k, s.seed), (40, 7));
    }

    #[test]
    fn draw_matches_ds41rt_uniform() {
        // DS41RT v15 `random_uniform` is `(mix >> 40) / 2^24` of the same mix.
        let s = Sampling::new(1.0, 1.0, 0, 0.0, Some(0x1234)).unwrap().unwrap();
        for pos in [0u64, 1, 2, 1000] {
            let u = (s.draw(pos) >> 40) as f32 * (1.0 / 16_777_216.0);
            assert!((0.0..1.0).contains(&u));
        }
        assert_ne!(s.draw(0), s.draw(1));
        assert_eq!(s.draw(5), Sampling { seed: 0x1234, ..s }.draw(5));
    }

    #[test]
    fn a_snapshot_serves_what_it_knows() {
        let prompt = After::from_logits(&[0.5, 2.0, 1.0], true);
        assert_eq!(prompt.greedy, Some(1));
        assert!(prompt.serves(false) && prompt.serves(true));
        let short = After::from_logits(&[0.5, 2.0, 1.0], false);
        assert!(short.serves(false) && !short.serves(true));
        // A sampled request's turn snapshot: its last token was a draw, not the argmax.
        let sampled_turn = After { greedy: None, logits: None };
        assert!(!sampled_turn.serves(false) && !sampled_turn.serves(true));
    }

    #[test]
    fn keys_order_floats() {
        let xs = [f32::NEG_INFINITY, -3.5, -1e-30, -0.0, 0.0, 1e-30, 2.0, f32::INFINITY];
        for w in xs.windows(2) {
            assert!(okey(w[0]) <= okey(w[1]), "{} {}", w[0], w[1]);
        }
        assert_eq!(okey(-0.0), okey(0.0));
        assert_eq!(okey(f32::NAN), 0);
    }

    /// Frequencies of 200,000 draws against the exact filtered distribution.
    fn check_distribution(logits: &[f32], r: DeviceRow, want: &[f64]) {
        let n = 200_000u64;
        let mut hits = vec![0u64; logits.len()];
        let s = Sampling { temperature: 1.0 / r.inv_t, top_p: r.top_p, top_k: r.top_k as usize,
            min_p: 0.0, seed: 99 };
        for pos in 0..n {
            let x = DeviceRow { rnd: s.draw(pos), ..r };
            hits[select(logits, logits.len(), &x).unwrap()] += 1;
        }
        for (i, (&h, &p)) in hits.iter().zip(want).enumerate() {
            let f = h as f64 / n as f64;
            let sd = (p * (1.0 - p) / n as f64).sqrt().max(1e-9);
            assert!((f - p).abs() <= 5.0 * sd + 1e-12, "token {i}: {f} vs {p}");
            if p == 0.0 {
                assert_eq!(h, 0, "token {i} filtered out but drawn");
            }
        }
    }

    fn softmax(l: &[f64]) -> Vec<f64> {
        let m = l.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = l.iter().map(|x| (x - m).exp()).collect();
        let z: f64 = e.iter().sum();
        e.iter().map(|x| x / z).collect()
    }

    #[test]
    fn draws_follow_the_filtered_distribution() {
        let logits = [2.0f32, 1.0, 0.5, 0.0, -1.0, -3.0, 1.0, -30.0];
        let l64: Vec<f64> = logits.iter().map(|&x| f64::from(x)).collect();
        // Temperature only (0.5): softmax(2 * logits); -30 * 2 is far below the floor.
        let mut want = softmax(&l64.iter().map(|x| 2.0 * x).collect::<Vec<_>>());
        want[7] = 0.0;
        check_distribution(&logits, row(0.5, 1.0, 0, 0.0, 1, 0), &want);
        // top_k 2: the two 1.0s tie for second place, so both are kept (three tokens).
        let z3: f64 = [0usize, 1, 6].iter().map(|&i| l64[i].exp()).sum();
        let mut want = vec![0.0; 8];
        for i in [0usize, 1, 6] {
            want[i] = l64[i].exp() / z3;
        }
        check_distribution(&logits, row(1.0, 1.0, 2, 0.0, 1, 0), &want);
        // top_k 4: 2.0, both 1.0s and 0.5.
        let z4: f64 = [0usize, 1, 6, 2].iter().map(|&i| l64[i].exp()).sum();
        let mut want = vec![0.0; 8];
        for i in [0usize, 1, 6, 2] {
            want[i] = l64[i].exp() / z4;
        }
        check_distribution(&logits, row(1.0, 1.0, 4, 0.0, 1, 0), &want);
        // top_p 0.6: p = softmax(logits); 2.0 holds 0.465, the tied 1.0s take it past 0.6.
        let p = softmax(&l64);
        let z: f64 = [0usize, 1, 6].iter().map(|&i| p[i]).sum();
        let mut want = vec![0.0; 8];
        for i in [0usize, 1, 6] {
            want[i] = p[i] / z;
        }
        check_distribution(&logits, row(1.0, 0.6, 0, 0.0, 1, 0), &want);
    }

    #[test]
    fn min_p_keeps_tokens_near_the_top() {
        // min_p 0.3 at T 1: keep p_i >= 0.3 p_max, i.e. logit >= 2 + ln 0.3 = 0.796.
        let logits = [2.0f32, 1.0, 0.5, 0.0, 1.0];
        let r = row(1.0, 1.0, 0, 0.3, 3, 0);
        for pos in 0..2000 {
            let x = DeviceRow { rnd: Sampling { temperature: 1.0, top_p: 1.0, top_k: 0, min_p: 0.3, seed: 3 }.draw(pos), ..r };
            assert!([0, 1, 4].contains(&select(&logits, 5, &x).unwrap()));
        }
    }

    #[test]
    fn vocab_bound_is_respected_and_rows_are_deterministic() {
        let mut logits = vec![0.0f32; 16];
        logits[15] = 50.0; // past the served vocabulary
        let r = row(1.0, 1.0, 0, 0.0, 11, 4);
        let a = select(&logits, 12, &r).unwrap();
        assert!(a < 12);
        assert_eq!(select(&logits, 12, &r), Some(a));
        let inf = [f32::INFINITY, 0.0];
        assert_eq!(select(&inf, 2, &r), None);
    }
}
