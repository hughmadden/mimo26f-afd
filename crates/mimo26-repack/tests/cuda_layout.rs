//! Mechanically pin the device layout against the sole repack writer.
use std::collections::BTreeMap;
use mimo26_repack::{geom,manifest};
fn expected() -> BTreeMap<String,usize> {
    let mut m=BTreeMap::from([
        ("M26X_LAYOUT_VERSION".into(),manifest::VERSION as usize),
        ("M26X_HIDDEN".into(),geom::HIDDEN),
        ("M26X_INTERMEDIATE".into(),geom::INTERMEDIATE),
        ("M26X_EP_RANKS".into(),geom::EP_RANKS),
        ("M26X_QUARTER_SLICE_BYTES".into(),geom::QUARTER_SLICE_BYTES),
    ]);
    for (name,p) in [("GATE",geom::Proj::Gate),("UP",geom::Proj::Up),("DOWN",geom::Proj::Down)] {
        for (field,value) in [("ROWS",p.slice_rows()),("COLS",p.slice_in_cols()),
            ("PAYLOAD_OFF",p.slice_payload_off()),("SCALE_OFF",p.slice_scale_off())] {
            m.insert(format!("M26X_{name}_{field}"),value);
        }
    }
    m
}
fn matches(text:&str) -> bool {
    let mut actual=BTreeMap::new();
    for line in text.lines().filter(|l|l.starts_with("#define ")) {
        let fields:Vec<_>=line.split_whitespace().collect();
        if fields.len()!=3 { return false; }
        let Ok(value)=fields[2].parse::<usize>() else {return false;};
        if actual.insert(fields[1].to_string(),value).is_some() {return false;}
    }
    actual==expected()
}
fn header() -> String {
    std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"),
        "/../mimo26-expert/kernels/include/mimo26_slice_layout.h")).unwrap()
}
fn kernel() -> String {
    std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"),
        "/../mimo26-expert/kernels/expert_gemm.cu")).unwrap()
}
// V2-F2: pin the actual CUDA projection table as well as its included header.
// Replacing a checked macro with a copied literal must fail even if the header
// remains correct. Whitespace changes are harmless; semantic table drift is not.
fn uses_pinned_projection_table(source: &str) -> bool {
    let compact: String = source.chars().filter(|c| !c.is_whitespace()).collect();
    let table = concat!(
        "__host____device__Geomgeometry(intp){",
        "if(p==M26X_PROJ_GATE)return{M26X_GATE_ROWS,M26X_GATE_COLS,M26X_GATE_PAYLOAD_OFF,M26X_GATE_SCALE_OFF};",
        "if(p==M26X_PROJ_UP)return{M26X_UP_ROWS,M26X_UP_COLS,M26X_UP_PAYLOAD_OFF,M26X_UP_SCALE_OFF};",
        "return{M26X_DOWN_ROWS,M26X_DOWN_COLS,M26X_DOWN_PAYLOAD_OFF,M26X_DOWN_SCALE_OFF};}"
    );
    let pinned = expected();
    let overrides = source.lines().any(|line| {
        let mut words = line.split_whitespace();
        words.next() == Some("#define") && words.next().is_some_and(|key| pinned.contains_key(key))
    });
    compact.contains(table) && !overrides
}
#[test]
fn device_layout_equals_repack_geometry() { assert!(matches(&header())); }
#[test]
fn actual_cuda_projection_table_uses_only_pinned_geometry() {
    assert!(uses_pinned_projection_table(&kernel()));
}
#[test]
fn cuda_table_drift_fails_even_with_an_unchanged_header() {
    let source = kernel();
    assert!(matches(&header()));
    assert!(!uses_pinned_projection_table(&source.replace("M26X_DOWN_COLS", "M26X_GATE_COLS")));
    assert!(!uses_pinned_projection_table(&source.replace("M26X_GATE_SCALE_OFF", "524288")));
    assert!(!uses_pinned_projection_table(&(source + "\n#define M26X_GATE_ROWS 256\n")));
}
#[test]
fn all_payload_tiles_and_scale_words_have_required_alignment() {
    for p in [geom::Proj::Gate, geom::Proj::Up, geom::Proj::Down] {
        let cols = p.slice_in_cols();
        for row in 0..p.slice_rows() {
            for k in (0..cols).step_by(256) {
                assert_eq!((p.slice_payload_off() + row * cols / 2 + k / 2) % 128, 0);
                assert_eq!((p.slice_scale_off() + row * cols / 32 + k / 32) % 8, 0);
            }
        }
    }
}
#[test]
fn stale_offset_and_old_version_are_rejected() {
    let h=header();
    assert!(!matches(&h.replace("M26X_GATE_SCALE_OFF 1048576","M26X_GATE_SCALE_OFF 524288")));
    assert!(!matches(&h.replace("M26X_LAYOUT_VERSION 2","M26X_LAYOUT_VERSION 1")));
    assert!(!matches(&(h+"\n#define M26X_GATE_COLS 4096\n")));
}
