//! Vision tower parity (perf reset V2): the device encoder against the checkpoint's own module.
//!
//! Goldens come from `harness/vision_ref.py` (prep: HF preprocessing; vit: the reference
//! `MiMoVisionTransformer` in FP32 with `--dump`). For each case prefix P it reads
//! `P.prep.pixel_values.npy`, `P.prep.grid.npy`, `P.ref.embeds.npy` and, when present,
//! `P.ref.blocks.npy`, and prints the error per block and for the final embeddings.
//!
//!   vision_check <visual.safetensors> <config.json> <case prefix>...
//!
//! PASS: every case's embeddings within relative L2 3e-2 of the FP32 reference and every
//! token's cosine similarity >= 0.98. For scale, the reference module itself run in BF16 (how the
//! model is normally served) is at relative L2 7.1e-2 / worst-token cosine 0.933 (640x480) and
//! 9.3e-2 / 0.800 (1280x960) from its FP32 run; when `P.refbf16.embeds.npy` exists its numbers are
//! printed alongside.

use std::path::Path;

use mimo26_coordinator::vision::VisionTower;

fn npy(path: &str) -> Result<(Vec<usize>, Vec<u8>, String), String> {
    let b = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err(format!("{path}: not an npy file"));
    }
    let (hlen, start) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else {
        (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12)
    };
    let hdr = std::str::from_utf8(&b[start..start + hlen]).map_err(|e| e.to_string())?;
    let descr = hdr.split("'descr':").nth(1).and_then(|s| s.split('\'').nth(1)).ok_or("descr")?.to_string();
    let shape_s = hdr.split("'shape':").nth(1).and_then(|s| s.split('(').nth(1)).and_then(|s| s.split(')').next())
        .ok_or("shape")?;
    let shape: Vec<usize> = shape_s.split(',').filter(|x| !x.trim().is_empty()).map(|x| x.trim().parse().unwrap()).collect();
    if hdr.contains("'fortran_order': True") {
        return Err(format!("{path}: fortran order"));
    }
    Ok((shape, b[start + hlen..].to_vec(), descr))
}

fn f32s(path: &str) -> Result<(Vec<usize>, Vec<f32>), String> {
    let (shape, raw, descr) = npy(path)?;
    if descr != "<f4" {
        return Err(format!("{path}: dtype {descr}"));
    }
    Ok((shape, raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()))
}

fn compare(a: &[f32], b: &[f32], dim: usize) -> (f64, f64, f64) {
    let (mut d2, mut n2, mut maxd) = (0f64, 0f64, 0f64);
    let mut min_cos = 1f64;
    for (ra, rb) in a.chunks(dim).zip(b.chunks(dim)) {
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (&x, &y) in ra.iter().zip(rb) {
            let (x, y) = (x as f64, y as f64);
            d2 += (x - y) * (x - y);
            n2 += y * y;
            maxd = maxd.max((x - y).abs());
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
        if na > 0.0 && nb > 0.0 {
            min_cos = min_cos.min(dot / (na.sqrt() * nb.sqrt()));
        }
    }
    ((d2 / n2.max(1e-30)).sqrt(), min_cos, maxd)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: vision_check <visual.safetensors> <config.json> <case prefix>...");
        std::process::exit(2);
    }
    let cfg = std::fs::read_to_string(&args[2]).expect("config.json");
    let t = std::time::Instant::now();
    let tower = VisionTower::load(Path::new(&args[1]), &cfg).unwrap_or_else(|e| panic!("load: {e}"));
    println!("loaded vision tower: {:.2} GB in {:.1} s", tower.weight_bytes() as f64 / 1e9, t.elapsed().as_secs_f64());
    let dev = tower.upload().unwrap_or_else(|e| panic!("upload: {e}"));
    let mut fails = 0;
    for p in &args[3..] {
        let (_, pix) = f32s(&format!("{p}.prep.pixel_values.npy")).unwrap();
        let (gshape, graw, gd) = npy(&format!("{p}.prep.grid.npy")).unwrap();
        assert!(gd == "<i8" && gshape == vec![3], "grid {gd} {gshape:?}");
        let grid: Vec<i64> = graw.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();
        let (gh, gw) = (grid[1] as usize, grid[2] as usize);
        let (_, want) = f32s(&format!("{p}.ref.embeds.npy")).unwrap();
        let blocks = f32s(&format!("{p}.ref.blocks.npy")).ok();
        let t = std::time::Instant::now();
        let (emb, dumps) = dev.encode(&pix, gh, gw, blocks.is_some()).unwrap_or_else(|e| panic!("{p}: {e}"));
        let ms = t.elapsed().as_secs_f64() * 1e3;
        if let Some((bshape, bref)) = &blocks {
            let per = bshape[1] * bshape[2];
            for (i, d) in dumps.iter().enumerate() {
                let (rel, cos, maxd) = compare(d, &bref[i * per..(i + 1) * per], bshape[2]);
                if i < 3 || i % 5 == 4 || i + 1 == dumps.len() || rel > 3e-2 {
                    println!("  {p} block {i:2}: rel L2 {rel:.2e}  min cos {cos:.5}  max |d| {maxd:.3e}");
                }
            }
        }
        let (rel, cos, maxd) = compare(&emb, &want, 4096);
        let ok = rel <= 3e-2 && cos >= 0.98 && emb.len() == want.len();
        if let Ok((_, b16)) = f32s(&format!("{p}.refbf16.embeds.npy")) {
            let (r, c, _) = compare(&b16, &want, 4096);
            println!("  {p}: the reference in BF16 vs FP32: rel L2 {r:.3e}, min cos {c:.5}");
        }
        fails += !ok as usize;
        println!("{p}: grid {gh}x{gw}, {} tokens, {ms:.1} ms: rel L2 {rel:.3e}, min cos {cos:.5}, max |d| {maxd:.3e} {}",
            emb.len() / 4096, if ok { "PASS" } else { "FAIL" });
    }
    println!("RESULT: {}", if fails == 0 { "PASS" } else { "FAIL" });
    std::process::exit(if fails == 0 { 0 } else { 1 });
}
