//! Consumer half of the real checkpoint TP4 integration cell.
//! Actual repack output -> actual expert CPU implementation -> independent
//! full-expert NumPy/FP64 reference. No source inclusion or circular oracle.
use mimo26_expert::{grouped::{self, GroupedPlan}, slice, bits_from_env};

fn floats(path: &std::path::Path) -> Vec<f32> {
    let raw = std::fs::read(path).unwrap();
    assert_eq!(raw.len() % 4, 0);
    raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

#[test]
#[ignore = "scripts/dev.sh test gemm tp4-cpu stages the independent full FFN oracle"]
fn real_tp4_partials_sum_to_full_expert_ffn() {
    let root = std::path::PathBuf::from(std::env::var_os("MIMO26_TP4_DIR").expect("private oracle staging"));
    let version: u32 = std::fs::read_to_string(root.join("layout.version")).unwrap().trim().parse().unwrap();
    slice::check_layout_version(version).unwrap();
    let x = floats(&root.join("x.f32"));
    assert_eq!(x.len(), 64 * 4096);
    let mut cases = 0;
    let mut overall_max = 0.0f64;
    for layer in [1,24,46] {
        for expert in [0,7,255] {
            let tag = format!("L{layer:02}_E{expert:03}");
            let want = floats(&root.join(format!("{tag}.y.f32")));
            assert_eq!(want.len(), 64 * 4096);
            let ranks: Vec<_> = (0..4).map(|rank| {
                std::fs::read(root.join(slice::slice_file_name(layer,expert,rank))).unwrap()
            }).collect();
            for m in [1,8,64] {
                let plan = GroupedPlan::uniform(1,m,0);
                let mut sum = vec![0.0f32; m * 4096];
                for image in &ranks {
                    let result = grouped::expert_ffn_self_contained(image, &x[..m*4096], &plan, bits_from_env()).unwrap();
                    assert_eq!(result.out_rows, 4096);
                    assert_eq!(result.data.len(), sum.len());
                    for (dst, value) in sum.iter_mut().zip(result.data) { *dst += value; }
                }
                let mut max_abs = 0.0f64;
                for (i, (&got, &expected)) in sum.iter().zip(&want).enumerate() {
                    assert!(got.is_finite() && expected.is_finite(), "{tag} M={m} nonfinite at {i}");
                    let error = (got as f64 - expected as f64).abs();
                    max_abs = max_abs.max(error);
                    assert!(error <= 1e-5 + 1e-5 * (expected as f64).abs(),
                        "{tag} M={m} output {i}: sum {got} oracle {expected} abs {error}");
                }
                overall_max = overall_max.max(max_abs);
                println!("TP4 CPU {tag} M={m}: {}/{} outputs PASS max_abs={max_abs:.9e}", sum.len(), sum.len());
                cases += 1;
            }
        }
    }
    assert_eq!(cases, 27);
    println!("TP4 CPU IDENTITY: 27/27 cases, 9 real experts, M=1/8/64; max_abs={overall_max:.9e}");
}
