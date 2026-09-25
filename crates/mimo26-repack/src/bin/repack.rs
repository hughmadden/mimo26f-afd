//! `mimo26-repack` — the layout-v2 quarter-slice staging driver (I5 go-window
//! step 0a). Reads the checkpoint shards on the local host, permutes each
//! expert's six MXFP4 tensors into the pinned quarter-slice layout for ONE rank,
//! writes the slice files plus a sha256 manifest, and is silent on identity
//! (the daemon's boot readback verifies the manifest on serve).
//!
//! Runs on each Spark with `--rank r` (rank r on spark(r+1)); it reads the
//! checkpoint copy already resident on that Spark and never moves weights
//! between hosts. Pure byte permutation — no dequantization.
//!
//! Args: `--rank <0..3> --checkpoint <dir> --out <slice-dir>`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use mimo26_repack::geom;
use mimo26_repack::manifest::{self, Manifest, SliceEntry};
use mimo26_repack::repack::{build_slice, read_expert};
use mimo26_repack::safetensors::SafetensorsHeader;
use mimo26_repack::sha256;
use mimo26_repack::{Mxfp4Naive, RepackError};

struct Args {
    rank: usize,
    checkpoint: PathBuf,
    out: PathBuf,
}

fn parse_args() -> Result<Args, String> {
    let mut rank: Option<usize> = None;
    let mut checkpoint: Option<PathBuf> = None;
    let mut out: Option<PathBuf> = None;
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--rank" => rank = Some(it.next().ok_or("--rank needs a value")?.parse().map_err(|_| "bad rank")?),
            "--checkpoint" => checkpoint = Some(PathBuf::from(it.next().ok_or("--checkpoint needs a value")?)),
            "--out" => out = Some(PathBuf::from(it.next().ok_or("--out needs a value")?)),
            other => return Err(format!("unknown arg {other}")),
        }
    }
    let rank = rank.ok_or("missing --rank")?;
    let checkpoint = checkpoint.ok_or("missing --checkpoint")?;
    let out = out.ok_or("missing --out")?;
    geom::check_rank(rank).map_err(|e| e.to_string())?;
    Ok(Args { rank, checkpoint, out })
}

fn shard_headers(checkpoint: &Path) -> Result<HashMap<String, SafetensorsHeader>, RepackError> {
    let mut map = HashMap::new();
    for expert in 0..geom::EXPERTS_PER_LAYER {
        let name = geom::shard_file(expert);
        if map.contains_key(&name) {
            continue;
        }
        let header = SafetensorsHeader::read(&checkpoint.join(&name))?;
        map.insert(name, header);
    }
    Ok(map)
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("usage: mimo26-repack --rank <0..3> --checkpoint <dir> --out <slice-dir>");
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };
    if !args.checkpoint.is_dir() {
        eprintln!("checkpoint dir not found: {}", args.checkpoint.display());
        std::process::exit(2);
    }
    std::fs::create_dir_all(&args.out).expect("create slice dir");

    let headers = match shard_headers(&args.checkpoint) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("shard header read failed: {e}");
            std::process::exit(3);
        }
    };
    println!(
        "repack: rank={} checkpoint={} out={} shards={} layers={} experts={}",
        args.rank,
        args.checkpoint.display(),
        args.out.display(),
        headers.len(),
        geom::MOE_LAYERS,
        geom::EXPERTS_PER_LAYER
    );

    let mut manifest = Manifest::new();
    let mut total_bytes: u64 = 0;
    for layer in geom::FIRST_MOE_LAYER..geom::FIRST_MOE_LAYER + geom::MOE_LAYERS {
        for expert in 0..geom::EXPERTS_PER_LAYER {
            let shard = geom::shard_file(expert);
            let shard_path = args.checkpoint.join(&shard);
            let header = headers.get(&shard).expect("cached shard header");
            let t = match read_expert(header, &shard_path, layer, expert) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("read_expert L{layer} E{expert} failed: {e}");
                    std::process::exit(4);
                }
            };
            let slice = match build_slice(&t, args.rank, Mxfp4Naive::NONE) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("build_slice L{layer} E{expert} failed: {e}");
                    std::process::exit(4);
                }
            };
            let file = geom::slice_file_name(layer, expert, args.rank);
            let hex = sha256::hex(&sha256::sha256(&slice));
            let path = args.out.join(&file);
            std::fs::write(&path, &slice).expect("write slice");
            total_bytes += slice.len() as u64;
            manifest.push(SliceEntry {
                file,
                sha256: hex,
                bytes: slice.len() as u64,
                layer,
                expert,
                rank: args.rank,
                shard,
                tensors: manifest::source_tensors(layer, expert),
            });
        }
        println!("repack: layer {layer}/47 done");
    }

    manifest.write(&args.out.join("manifest.json")).expect("write manifest");
    println!(
        "repack complete: rank={} slices={} bytes={} manifest={}",
        args.rank,
        manifest.slices.len(),
        total_bytes,
        args.out.join("manifest.json").display()
    );
}
