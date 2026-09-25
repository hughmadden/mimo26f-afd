//! §3.1 arithmetic reproduction (ADVISOR-I4) — all MODEL. Each test names the
//! wrong implementation it kills.

use mimo26_lanesim::model::*;
use mimo26_lanesim::ModelGeom;

const G: ModelGeom = ModelGeom::REAL;

/// Kills: plain-0.5-B-per-param MXFP4 (the E8M0-32 scale byte forgotten),
/// wrong expert shape (not 3 x 2048 x 4096), wrong quarter split.
#[test]
fn expert_is_1337_mb_and_quarter_slice_334_mb() {
    assert_eq!(G.expert_bytes(), 13_369_344); // 3 x 2048 x 4096 x 0.53125 B
    assert!((G.expert_bytes() as f64 / 1e6 - 13.37).abs() < 0.005);
    assert_eq!(G.quarter_slice_bytes(), 3_342_336);
    assert!((G.quarter_slice_bytes() as f64 / 1e6 - 3.34).abs() < 0.005);
    assert_eq!(G.quarter_slice_bytes() * 4, G.expert_bytes());
}

/// Kills: request rows that forget the 40-B descriptor or the UE8M0 scales,
/// and any return sizing that is not H x 2 BF16 (§3.3: return bytes/token must
/// be asserted, not commented).
#[test]
fn request_row_4360_b_and_compact_return_8192_b() {
    assert_eq!(G.request_row_bytes(), 4_360);
    assert_eq!(40 + 8 * 12 + 4096 + 4096 / 32, 4_360); // REQ_ROW breakdown
    assert_eq!(G.return_bytes_per_token(), 8_192);
    assert_eq!(G.return_bytes_per_token(), G.hidden * 2);
}

/// Kills: the old ds41rt per-route FP32 return — the 24.6 MB/token wall the
/// compact pre-summed return exists to avoid (never per-route FP32).
#[test]
fn per_route_fp32_return_is_the_24_6_mb_wall() {
    assert_eq!(G.per_route_fp32_wall_per_token(), 24_641_536);
    assert!((G.per_route_fp32_wall_per_token() as f64 / 1e6 - 24.6).abs() < 0.05);
    // the compact path (4 ranks x 47 layers x 8 KB/token) is 16x under the wall
    assert!(G.return_bytes_per_token() as u64 * 4 * 47 < G.per_route_fp32_wall_per_token());
}

/// Kills: 46/48-layer bookkeeping, or counting full experts (13.37 MB) where
/// the table counts quarter slices (3.34 MB) — resident is 256 x 47 x 3.34 MB.
#[test]
fn resident_is_40_2_gb_per_spark() {
    assert_eq!(G.resident_bytes_per_spark(), 40_214_986_752);
    assert!((G.resident_bytes_per_spark() as f64 / 1e9 - 40.2).abs() < 0.05);
}

/// Kills: independent single-draw unique-expert math, full-expert streaming
/// (instead of quarter slices), 273 GB/s applied as a multiplier, and kernel
/// efficiency that multiplies instead of divides. Reproduces the ADVISOR-I4
/// §3.1 table at 100% / 75% / 60% within its rounding (0.1 GB / 1 ms).
#[test]
fn section_31_table_reproduced_within_rounding() {
    // (rows/step, unique experts/layer, GB per Spark per step, [100%, 75%, 60%] ms)
    let want: [(usize, f64, f64, [f64; 3]); 4] = [
        (8, 57.4, 9.0, [33.0, 44.0, 55.0]),
        (48, 200.0, 31.5, [115.0, 154.0, 192.0]),
        (128, 252.0, 39.5, [145.0, 193.0, 241.0]),
        (2048, 256.0, 40.2, [147.0, 196.0, 246.0]),
    ];
    for (case, (rows, u, gb, ms)) in table_cases().iter().zip(want.iter()) {
        assert_eq!(case.rows_per_step, *rows, "{} rows", case.name);
        assert!((case.unique_experts - u).abs() <= 1.0, "{} unique experts", case.name);
        assert!(
            (case.bytes_per_spark_per_step as f64 / 1e9 - gb).abs() <= 0.1,
            "{} bytes: {} GB vs {} GB",
            case.name,
            case.bytes_per_spark_per_step as f64 / 1e9,
            gb
        );
        for (m, w) in case.floor_ms.iter().zip(ms.iter()) {
            assert!((m - w).abs() <= 1.0, "{} floor {} ms vs {} ms", case.name, m, w);
        }
        assert_eq!(case.label, "MODEL");
    }
}

/// Kills: dropping the 47 x (40 + 80) us layer-boundary term, or computing
/// tok/s from drafted positions (8/step) instead of the MEASURED mean DFlash
/// acceptance (4.46 accepted tok/step).
#[test]
fn afd_c1_step_model_54_65_76_ms_and_83_69_59_toks() {
    for (eff, want_ms, want_tps) in [(0.75, 54.0, 83.0), (0.60, 65.0, 69.0), (0.50, 76.0, 59.0)] {
        let ms = afd_c1_step_ms(eff);
        assert!((ms - want_ms).abs() <= 1.0, "eff {eff}: {ms} ms vs {want_ms} ms");
        let tps = toks_per_s(ms);
        assert!((tps - want_tps).abs() <= 1.0, "eff {eff}: {tps} tok/s vs {want_tps} tok/s");
    }
    // the MEASURED bar is carried next to the model, never conflated with it
    assert_eq!(D7_STEP_MS, 62.0);
    assert!((D7_TOK_PER_S - toks_per_s(D7_STEP_MS)).abs() < 1.0);
    assert_eq!(MEAN_DFLASH_ACCEPTANCE, 4.46);
}

/// Kills: kernel efficiency applied in the wrong direction at the model level.
#[test]
fn stream_floor_scales_inverse_with_efficiency() {
    let bytes = 9_000_000_000u64;
    assert!((stream_floor_ms(bytes, 0.5) / stream_floor_ms(bytes, 1.0) - 2.0).abs() < 1e-9);
    assert!((stream_floor_ms(bytes, 0.75) - 44.0).abs() < 1.0);
    assert_eq!(TABLE_EFFICIENCIES, [1.0, 0.75, 0.6]);
}
