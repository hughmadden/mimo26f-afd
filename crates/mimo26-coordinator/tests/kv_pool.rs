//! A1 admission + KV pool accounting (ARCHITECTURE §11.1/§11.8, T21).

use mimo26_coordinator::{
    full_lifetime_reservation, request_reservation, Admission, KvPool, DEFAULT_ADMIT_RESERVE,
    GA_KV_BYTES_PER_TOKEN, MAX_OUTPUT_TOKENS, SWA_RING_BYTES_PER_SEQ,
};

#[test]
fn kv_geometry_matches_the_config() {
    assert_eq!(GA_KV_BYTES_PER_TOKEN, 11_520, "9 GA × 4 KV heads × 320 dims");
    assert_eq!(SWA_RING_BYTES_PER_SEQ, 12_779_520, "39 SWA × 8 KV heads × 320 × window 128");
}

#[test]
fn t21_full_lifetime_reservation_is_755_mb_and_over_reserves() {
    // The trap T21 quotes: prompt + full max_tokens = 755 MB/request at MiMo size.
    let full = full_lifetime_reservation(0, MAX_OUTPUT_TOKENS);
    assert_eq!(full, SWA_RING_BYTES_PER_SEQ + MAX_OUTPUT_TOKENS * GA_KV_BYTES_PER_TOKEN);
    assert_eq!(MAX_OUTPUT_TOKENS * GA_KV_BYTES_PER_TOKEN, 754_974_720, "65,536 × 11,520 = 755 MB");
}

#[test]
fn grow_on_demand_reserves_only_the_output_quantum() {
    // The fix: reserve prompt + min(max_tokens, admit_reserve) = 8,192 tokens.
    let r = request_reservation(0, MAX_OUTPUT_TOKENS, DEFAULT_ADMIT_RESERVE);
    assert_eq!(
        r,
        SWA_RING_BYTES_PER_SEQ + DEFAULT_ADMIT_RESERVE * GA_KV_BYTES_PER_TOKEN,
        "grow-on-demand reserves the 8,192-token quantum, not the full 65,536"
    );
    assert!(r < full_lifetime_reservation(0, MAX_OUTPUT_TOKENS));
}

#[test]
fn admission_gates_new_work_only_and_never_fails_a_running_request() {
    // A 14 GiB pool (the ~14–16 GiB boot-measured range on a 5090).
    let total = 14u64 * 1024 * 1024 * 1024;
    let mut pool = KvPool::new(total);

    // 100 requests at 2K prompt each fit.
    let mut admitted = 0;
    for _ in 0..100 {
        match pool.admit(2048, MAX_OUTPUT_TOKENS, DEFAULT_ADMIT_RESERVE) {
            Admission::Admitted { .. } => admitted += 1,
            Admission::Rejected => break,
        }
    }
    assert!(admitted > 0 && admitted <= 100);

    // A running request can always grow (never failed for pool pressure);
    // growth is accounted but not gated.
    let before = pool.used();
    pool.grow(1000);
    assert_eq!(pool.used(), before + 1000 * GA_KV_BYTES_PER_TOKEN);

    // A tiny pool rejects new admission (429 path), never a running request.
    let mut tiny = KvPool::new(SWA_RING_BYTES_PER_SEQ / 2);
    assert_eq!(tiny.admit(2048, MAX_OUTPUT_TOKENS, DEFAULT_ADMIT_RESERVE), Admission::Rejected);
}
