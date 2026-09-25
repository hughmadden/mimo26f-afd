//! Producer half of the cross-crate, real-weight TP4 identity cell.
//! `scripts/dev.sh test gemm tp4-cpu` creates the independent oracle directory,
//! runs this repacker, then runs the actual expert library on these byte images.
//! Default ignored: requires the explicit cell's private staging directory.
use std::path::Path;
use mimo26_repack::{geom, manifest, identity, repack, sha256, Manifest, SliceEntry, Mxfp4Naive};

#[test]
#[ignore = "staged real-weight integration cell; scripts/dev.sh test gemm tp4-cpu"]
fn emit_real_v2_slices_for_expert_identity() {
    let root = std::path::PathBuf::from(std::env::var_os("MIMO26_TP4_DIR").expect("private oracle staging"));
    for layer in [1,24,46] {
        for expert in [0,7,255] {
            let tag = format!("L{layer:02}_E{expert:03}");
            let read = |p: &str, suffix: &str| std::fs::read(root.join(format!("{tag}.{p}.{suffix}"))).unwrap();
            let tensors = repack::ExpertTensors {
                gate_w: read("gate_proj", "w"), gate_s: read("gate_proj", "s"),
                up_w: read("up_proj", "w"), up_s: read("up_proj", "s"),
                down_w: read("down_proj", "w"), down_s: read("down_proj", "s"),
            };
            let mut manifest = Manifest::new();
            for rank in 0..4 {
                let bytes = repack::build_slice(&tensors, rank, Mxfp4Naive::NONE).unwrap();
                let file = geom::slice_file_name(layer, expert, rank);
                std::fs::write(root.join(&file), &bytes).unwrap();
                manifest.push(SliceEntry {
                    file, sha256: sha256::hex(&sha256::sha256(&bytes)), bytes: bytes.len() as u64,
                    layer, expert, rank, shard: geom::shard_file(expert),
                    tensors: manifest::source_tensors(layer, expert),
                });
            }
            manifest.write(&root.join(format!("{tag}.manifest.json"))).unwrap();
            for entry in &manifest.slices {
                let checked = identity::load_slice(Path::new(&root), &manifest, &entry.file).unwrap();
                assert_eq!(checked.len(), 3_342_336);
            }
            println!("REPACK v2 {tag}: 4 slices, checked version + size + SHA256");
        }
    }
    std::fs::write(root.join("layout.version"), format!("{}\n", manifest::VERSION)).unwrap();
}
