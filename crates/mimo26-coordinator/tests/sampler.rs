//! Sampler semantics + T23 server defaults (ADVISOR-I3 §10.2).

use mimo26_coordinator::{
    apply_repetition_penalty, apply_temperature, apply_top_k, apply_top_p, greedy, sample,
    softmax, SamplingParams, SeenSet,
};

#[test]
fn t23_defaults_match_advisor_i3_10_2() {
    let p = SamplingParams::default();
    assert_eq!(p.temperature, 1.0, "default temperature");
    assert_eq!(p.top_p, 0.95, "default top_p");
    assert_eq!(p.top_k, 0, "default top_k (off)");
    assert_eq!(p.repetition_penalty, 1.05, "default repetition_penalty");
    assert_eq!(p.max_tokens, 65_536, "default max_tokens must be 65,536, never 2048 (T23)");
}

#[test]
fn repetition_penalty_divides_positive_multiplies_negative() {
    let mut seen = SeenSet::new(4);
    seen.mark(0);
    seen.mark(1);
    let mut l = [2.0f32, -2.0, 0.0, 1.0];
    apply_repetition_penalty(&mut l, &seen, 1.05);
    assert!((l[0] - 2.0 / 1.05).abs() < 1e-6, "positive seen logit /1.05, got {}", l[0]);
    assert!((l[1] - (-2.0 * 1.05)).abs() < 1e-6, "negative seen logit *1.05, got {}", l[1]);
    assert_eq!(l[2], 0.0, "zero logit unchanged");
    assert_eq!(l[3], 1.0, "unseen logit unchanged");
}

#[test]
fn temperature_scales_logits() {
    let mut l = [2.0f32, 4.0];
    apply_temperature(&mut l, 2.0);
    assert_eq!(l, [1.0, 2.0]);
}

#[test]
fn top_k_masks_below_the_kth_largest() {
    let mut l = [1.0f32, 5.0, 4.0, 3.0, 2.0];
    apply_top_k(&mut l, 2);
    assert_eq!(l[1], 5.0);
    assert_eq!(l[2], 4.0);
    for &i in &[0usize, 3, 4] {
        assert_eq!(l[i], f32::NEG_INFINITY, "token {i} below top-k must be -inf");
    }
}

#[test]
fn top_p_keeps_smallest_nucleus() {
    // softmax([2,1,0]) ~ [0.665, 0.2445, 0.090]; top_p 0.75 keeps {0,1}.
    let mut l = [2.0f32, 1.0, 0.0];
    apply_top_p(&mut l, 0.75);
    assert!(l[0] > 0.0 && l[1] > 0.0, "nucleus keeps the top two");
    assert_eq!(l[2], f32::NEG_INFINITY, "low-prob token masked");
}

#[test]
fn greedy_is_argmax_with_lowest_index_on_ties() {
    assert_eq!(greedy(&[1.0, 5.0, 3.0, 2.0]), 1);
    assert_eq!(greedy(&[3.0, 3.0, 1.0]), 0, "tie resolves to the lowest id");
}

#[test]
fn softmax_sums_to_one_and_is_stable() {
    let p = softmax(&[10.0, 0.0, -10.0]);
    let s: f32 = p.iter().sum();
    assert!((s - 1.0).abs() < 1e-5, "softmax sums to 1, got {s}");
    assert!(p[0] > 0.99, "dominant mass on the max logit");
}

#[test]
fn sample_with_temperature_zero_is_greedy_regardless_of_u() {
    let params = SamplingParams { temperature: 0.0, ..Default::default() };
    let seen = SeenSet::new(4);
    let logits = [1.0f32, 5.0, 3.0, 2.0];
    assert_eq!(sample(&logits, &params, &seen, 0.0), 1);
    assert_eq!(sample(&logits, &params, &seen, 0.99), 1);
}

#[test]
fn sample_is_a_deterministic_multinomial_given_u() {
    let params = SamplingParams {
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        max_tokens: 65_536,
    };
    let seen = SeenSet::new(2);
    // Uniform logits -> p = [0.5, 0.5].
    assert_eq!(sample(&[0.0f32, 0.0], &params, &seen, 0.3), 0);
    assert_eq!(sample(&[0.0f32, 0.0], &params, &seen, 0.7), 1);
}
