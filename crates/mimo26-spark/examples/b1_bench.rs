//! B1 expert FFN bench for one Spark rank: replay a dumped request frame
//! (`MIMO26_SPARK_DUMP_FRAME=1`) against one real layer's prepared B1 pool and
//! report median per-phase GPU times. `--out` writes the BF16 partial so kernel
//! changes can be compared against the current output.
//!
//! ```text
//! b1_bench --dir /var/tmp/mimo26f-kernel/slices --rank 0 --layer 1 \
//!   --frame /var/tmp/mimo26f-kernel/req-layer1.bin [--iters 20] [--rows N] [--repeat R] [--out out.bf16]
//! ```
//!
//! `--repeat R` tiles the frame's rows R times (e.g. a 4,096-row request from a
//! 2,048-row dump: every expert gets R times the rows, the same group/chunk
//! structure scaled) to time larger expert chunks.

use std::path::Path;

use mimo26_spark::resident::Resident;
use mimo26_spark::{b1, decode, serve};
use mimo26_wire::frame::Frame;
use mimo26_wire::l4::StreamReceiver;
use mimo26_wire::WireNaive;

fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut dir, mut rank, mut layer, mut frame) = (String::new(), 0usize, 1usize, String::new());
    let (mut iters, mut rows_cap, mut out, mut repeat) = (20usize, usize::MAX, None::<String>, 1usize);
    while let Some(a) = args.next() {
        let mut v = || args.next().unwrap_or_else(|| panic!("{a} needs a value"));
        match a.as_str() {
            "--dir" => dir = v(),
            "--rank" => rank = v().parse().expect("--rank"),
            "--layer" => layer = v().parse().expect("--layer"),
            "--frame" => frame = v(),
            "--iters" => iters = v().parse().expect("--iters"),
            "--rows" => rows_cap = v().parse().expect("--rows"),
            "--out" => out = Some(v()),
            "--repeat" => repeat = v().parse().expect("--repeat"),
            other => panic!("unknown argument {other}"),
        }
    }
    let resident = Resident::load_manifest(Path::new(&dir), rank).expect("load_manifest");
    let img = resident.grouped_image(layer, 256).expect("grouped_image");
    let canonical = decode::upload_grouped(&img).expect("upload");
    drop(img);
    let l = b1::Layer::new(canonical.as_ptr() as *const u8, canonical.bytes(), rank as u32).expect("b1 prepare");
    drop(canonical);

    let bytes = std::fs::read(&frame).unwrap_or_else(|e| panic!("read {frame}: {e}"));
    let req = match StreamReceiver::new(WireNaive::NONE).accept(&bytes).expect("decode frame") {
        Frame::Request(r) => r,
        Frame::Return(_) => panic!("expected a request frame"),
    };
    let (mut payload, mut scales, mut ids, mut weights) = serve::b1_inputs(&req).expect("b1 inputs");
    let rows = req.rows.len().min(rows_cap);
    payload.truncate(rows * 4096);
    scales.truncate(rows * 128);
    ids.truncate(rows * 8);
    weights.truncate(rows * 8);
    let (payload, scales, ids, weights) = (payload.repeat(repeat), scales.repeat(repeat), ids.repeat(repeat), weights.repeat(repeat));
    let rows = rows * repeat;

    let mut s = b1::Scratch::new().expect("scratch");
    let mut y = vec![0u16; rows * 4096];
    for _ in 0..3 {
        b1::ffn(&l, &mut s, &payload, &scales, &ids, &weights, rows, &mut y).expect("warmup ffn");
    }
    let mut st = Vec::with_capacity(iters);
    for _ in 0..iters {
        st.push(b1::ffn(&l, &mut s, &payload, &scales, &ids, &weights, rows, &mut y).expect("ffn"));
    }
    let m = |f: &dyn Fn(&b1::FfnStats) -> f32| median(st.iter().map(f).collect());
    println!(
        "b1_bench layer={layer} rank={rank} rows={rows} groups={} iters={iters} median ms: gpu={:.3} \
         [fc1={:.3} q={:.3} fc2={:.3} red={:.3}] upload_plan={:.3} tail={:.3}",
        st[0].groups,
        m(&|x| x.gpu_ms),
        m(&|x| x.phase_ms[0]),
        m(&|x| x.phase_ms[1]),
        m(&|x| x.phase_ms[2]),
        m(&|x| x.phase_ms[3]),
        m(&|x| x.upload_plan_ms),
        m(&|x| x.tail_ms)
    );
    if let Some(o) = out {
        let b: Vec<u8> = y.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&o, b).unwrap_or_else(|e| panic!("write {o}: {e}"));
        println!("wrote {o}");
    }
}
