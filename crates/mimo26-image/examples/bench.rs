//! Timing for decode + preprocess on real files (single thread).
//!
//! cargo run --release --offline -p mimo26-image --example bench -- <image> [<image> ...]
//!
//! Prints, per file, the median and best of N runs for decode(),
//! preprocess() and the two together, plus the resulting grid.
//! `BENCH_DUMP=<dir>` also writes `<file name>.rgb` (decoded RGB8) and
//! `<file name>.pv.f32` (pixel_values, little-endian f32) for comparison with
//! the Python oracle in tests/gen_goldens.py.

use mimo26_image::{decode, preprocess};
use std::time::Instant;

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn stats(mut v: Vec<f64>) -> (f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[v.len() / 2], v[0])
}

fn main() {
    let runs: usize = std::env::var("BENCH_RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);
    for path in std::env::args().skip(1) {
        let bytes = std::fs::read(&path).expect("read input");
        let (mut td, mut tp, mut tt) = (Vec::new(), Vec::new(), Vec::new());
        let mut summary = String::new();
        for _ in 0..runs {
            let t0 = Instant::now();
            let img = decode(&bytes).expect("decode");
            let d = ms(t0);
            let t1 = Instant::now();
            let p = preprocess(&img).expect("preprocess");
            let q = ms(t1);
            td.push(d);
            tp.push(q);
            tt.push(d + q);
            summary = format!(
                "{}x{} -> grid {}x{}x{} ({} rows, {} merged tokens, {:.1} MB f32)",
                img.width,
                img.height,
                p.grid_t,
                p.grid_h,
                p.grid_w,
                p.rows(),
                p.merged_tokens(),
                p.data.len() as f64 * 4.0 / 1e6
            );
        }
        if let Ok(dir) = std::env::var("BENCH_DUMP") {
            let img = decode(&bytes).expect("decode");
            let p = preprocess(&img).expect("preprocess");
            let name = std::path::Path::new(&path)
                .file_name()
                .unwrap()
                .to_string_lossy()
                .to_string();
            let dir = std::path::Path::new(&dir);
            std::fs::write(dir.join(format!("{name}.rgb")), &img.data).expect("write rgb");
            let pv: Vec<u8> = p.data.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(dir.join(format!("{name}.pv.f32")), pv).expect("write pv");
        }
        let ((dm, db), (pm, pb), (tm, tb)) = (stats(td), stats(tp), stats(tt));
        println!("{path} ({} bytes): {summary}", bytes.len());
        println!(
            "  decode {dm:.1} ms (best {db:.1}) | preprocess {pm:.1} ms (best {pb:.1}) | total {tm:.1} ms (best {tb:.1}) over {runs} runs"
        );
    }
}
