//! Served sampling kernel check (perf reset V3): `m26c_sample_rows` against the CPU reference
//! `sampling::select` on synthetic logit rows at MiMo's width (152,576 lm_head rows, 151,675
//! sampleable ids), and the kernel's time at serving shapes.
//!
//!   sample_check [rows (default 2048)]
//!
//! Rows: realistic (a few peaks over a noisy floor), flat (a huge nucleus), tied (logits in steps of
//! 0.5), damaged (NaN and -inf entries), one-hot, each with a strong peak in the padding rows that
//! must never be drawn. Parameters: random temperature / top_p / top_k / min_p per row.
//!
//! PASS: every draw equals the reference's and lies below 151,675. The two differ only in `exp`'s
//! last bit, which can move a draw only when it lands within ~1e-7 of a boundary.

use mimo26_coordinator::dforward::select_rows_host;
use mimo26_coordinator::sampling::{select, Sampling};

const LD: usize = 152_576;
const VOCAB: usize = 151_675;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f(&mut self) -> f32 {
        (self.next() >> 40) as f32 / 16_777_216.0
    }
    fn normal(&mut self) -> f32 {
        let (a, b) = (self.f().max(1e-7), self.f());
        (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos()
    }
    fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[(self.next() % xs.len() as u64) as usize]
    }
}

fn logits(rng: &mut Rng, kind: usize) -> Vec<f32> {
    let mut l: Vec<f32> = match kind {
        1 => (0..LD).map(|_| 0.1 * rng.normal()).collect(),
        2 => (0..LD).map(|_| (2.0 * (3.0 * rng.normal() - 4.0)).round() * 0.5).collect(),
        4 => vec![-40.0; LD],
        _ => (0..LD).map(|_| 3.0 * rng.normal() - 5.0).collect(),
    };
    let peaks = 1 + (rng.next() % 20) as usize;
    for _ in 0..peaks {
        let i = (rng.next() % VOCAB as u64) as usize;
        l[i] = if kind == 4 { 40.0 } else { 10.0 + 15.0 * rng.f() };
        if kind == 2 {
            l[i] = l[i].round();
        }
    }
    if kind == 3 {
        for _ in 0..50 {
            let i = (rng.next() % VOCAB as u64) as usize;
            l[i] = if rng.next() % 2 == 0 { f32::NAN } else { f32::NEG_INFINITY };
        }
    }
    // A peak in the padding rows: the argmax may take it, a draw never.
    l[VOCAB + (rng.next() % (LD - VOCAB) as u64) as usize] = 60.0;
    l
}

fn main() {
    let n: usize = std::env::args().nth(1).and_then(|v| v.parse().ok()).unwrap_or(2048);
    let mut rng = Rng(0x5eed_1234_abcd_ef01);
    let (mut rows_done, mut bad, mut kernel_ms, mut chunks) = (0usize, 0usize, 0f64, 0usize);
    while rows_done < n {
        let m = (n - rows_done).min(128);
        let mut x = Vec::with_capacity(m * LD);
        let mut sampled = Vec::with_capacity(m);
        for r in 0..m {
            x.extend(logits(&mut rng, r % 5));
            let s = Sampling::new(rng.pick(&[0.2, 0.6, 1.0, 1.4, 2.0]), rng.pick(&[1.0, 0.95, 0.9, 0.5, 0.05]),
                rng.pick(&[0usize, 0, 5, 50, 400]), rng.pick(&[0.0, 0.0, 0.02, 0.3]), Some(rng.next()))
                .expect("valid")
                .expect("sampled");
            sampled.push(s.row(r, rng.next() % 4096));
        }
        let (got, ms) = select_rows_host(&x, LD, VOCAB, &sampled).expect("kernel");
        kernel_ms += ms;
        chunks += 1;
        for (r, d) in sampled.iter().enumerate() {
            let row = &x[r * LD..(r + 1) * LD];
            let want = select(row, VOCAB, d);
            let g = got[r];
            if want != Some(g) || g >= VOCAB {
                bad += 1;
                eprintln!("row {}: kind {} T {:.2} top_p {} top_k {} ln_min_p {}: gpu {g} cpu {want:?}", rows_done + r,
                    r % 5, 1.0 / d.inv_t, d.top_p, d.top_k, d.ln_min_p);
            }
        }
        rows_done += m;
    }
    println!("{n} rows: {bad} mismatches; kernel {:.3} ms per 128-row batch", kernel_ms / chunks as f64);
    // Serving shapes: a C1 speculative step (8 rows) and C16 (128 rows), T 0.7 / top_p 0.95.
    for m in [1usize, 8, 128] {
        let mut x = Vec::with_capacity(m * LD);
        let mut sampled = Vec::with_capacity(m);
        let s = Sampling::new(0.7, 0.95, 0, 0.0, Some(7)).unwrap().unwrap();
        for r in 0..m {
            x.extend(logits(&mut rng, 0));
            sampled.push(s.row(r, r as u64));
        }
        let mut best = f64::MAX;
        for _ in 0..5 {
            best = best.min(select_rows_host(&x, LD, VOCAB, &sampled).expect("kernel").1);
        }
        println!("  {m:3} rows, T 0.7 top_p 0.95: {best:.3} ms");
    }
    println!("RESULT: {}", if bad == 0 { "PASS" } else { "FAIL" });
    std::process::exit(if bad == 0 { 0 } else { 1 });
}
