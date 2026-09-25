//! BOTH-RUNS cross-language pins. Repack also pins this header independently.
use mimo26_expert::{slice,grouped,NaiveBits};
fn pin(text:&str,key:&str,value:impl std::fmt::Display) {
    let prefix=format!("#define {key} ");
    let got:Vec<_>=text.lines().filter(|line|line.starts_with(&prefix)).collect();
    assert_eq!(got,vec![format!("{prefix}{value}")],"{key} drift");
}
#[test]
fn device_layout_matches_expert_consumer() {
    let h=include_str!("../kernels/include/mimo26_slice_layout.h");
    for (key,value) in [("LAYOUT_VERSION",slice::LAYOUT_VERSION as usize),("HIDDEN",slice::HIDDEN),
        ("INTERMEDIATE",slice::INTERMEDIATE),("EP_RANKS",slice::EP_RANKS),("QUARTER_SLICE_BYTES",slice::QUARTER_SLICE_BYTES)] {
        pin(h,&format!("M26X_{key}"),value);
    }
    for (name,p) in [("GATE",slice::Proj::Gate),("UP",slice::Proj::Up),("DOWN",slice::Proj::Down)] {
        for (field,value) in [("ROWS",p.slice_rows()),("COLS",p.slice_in_cols()),
            ("PAYLOAD_OFF",p.slice_payload_off()),("SCALE_OFF",p.slice_scale_off())] {
            pin(h,&format!("M26X_{name}_{field}"),value);
        }
    }
}
#[test]
fn device_flags_and_launch_constants_match_rust() {
    let h=include_str!("../kernels/include/mimo26_expert_bits.h");
    for (name,flag) in [("NIBBLE_SWAP",NaiveBits::NIBBLE_SWAP),("E8M0_NO_CLAMP",NaiveBits::E8M0_NO_CLAMP),
        ("SCALE_OFF_BY_ONE",NaiveBits::SCALE_OFF_BY_ONE),("PAD_ROW_READ",NaiveBits::PAD_ROW_READ),
        ("AOT_MIXED_GATE",NaiveBits::AOT_MIXED_GATE),("AOT_CAPACITY_IGNORED",NaiveBits::AOT_CAPACITY_IGNORED),
        ("BF16_ACCUM",NaiveBits::BF16_ACCUM),("SCALE_ONE",NaiveBits::SCALE_ONE)] {
        pin(h,&format!("M26X_NAIVE_{name}"),format!("{}u",flag.0));
    }
    for (name,value) in [("THREADS",grouped::THREADS),("ROWS_PER_BLOCK",grouped::ROWS_PER_BLOCK),
        ("LANES_PER_ROW",grouped::LANES_PER_ROW),("K_TILE",grouped::K_TILE),("BLOCK",grouped::K_STEP)] {
        pin(h,&format!("M26X_{name}"),value);
    }
}
#[test]
fn force_scale_one_is_detected() {
    for byte in [0,118,125,126,128,254,255] {
        let want=2.0f64.powi(i32::from(byte.min(254))-127);
        assert_eq!(mimo26_expert::mxfp4::e8m0_scale(byte,NaiveBits::NONE),want);
        assert_ne!(mimo26_expert::mxfp4::e8m0_scale(byte,NaiveBits::SCALE_ONE),want);
    }
}
/// NEGATIVE: the literal documented T10 misfeature fails this real entry point.
#[test]
fn e8m0_scales_are_not_forced_to_one() {
    assert_eq!(mimo26_expert::mxfp4::e8m0_scale(125,mimo26_expert::bits_from_env()),0.25);
}
