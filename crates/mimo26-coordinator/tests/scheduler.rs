//! A4 scheduler (ARCHITECTURE §11.1/§11.4): chunked prefill, long-context lane,
//! admission integration, and the prefill/decode lifecycle.

use mimo26_coordinator::{KvPool, Request, Scheduler, SchedulerConfig, Step, SubmitOutcome};

// ~200 MB — enough for several small requests (each reserves the SWA ring
// 12,779,520 B plus prompt + output quantum).
fn sched() -> Scheduler {
    Scheduler::new(SchedulerConfig::default(), KvPool::new(200_000_000))
}

#[test]
fn prefill_chunks_at_2048() {
    let s = sched();
    assert_eq!(s.chunks(5000), vec![(0, 2048), (2048, 2048), (4096, 904)]);
    assert_eq!(s.chunks(2048), vec![(0, 2048)]);
    assert_eq!(s.chunks(0), Vec::<(u64, u64)>::new());
}

#[test]
fn long_context_gets_a_concurrency_1_lane() {
    let s = sched();
    assert!(!s.is_long_context(200_000));
    assert!(s.is_long_context(600_000));
    assert_eq!(s.concurrency_for(600_000), 1, "long-context lane runs alone");
    assert_eq!(s.concurrency_for(2_000), 8, "normal request shares the lanes");
}

#[test]
fn admission_gates_new_work_and_reserves_kv() {
    let mut s = Scheduler::new(SchedulerConfig::default(), KvPool::new(100_000_000));
    assert_eq!(s.submit(Request::new(1, 1000, 4096)), SubmitOutcome::Accepted);
    assert!(s.used_bytes() > 0, "admission reserves KV bytes");

    // A tiny pool rejects (429) and never queues the request.
    let mut s2 = Scheduler::new(SchedulerConfig::default(), KvPool::new(100));
    assert_eq!(s2.submit(Request::new(2, 1000, 4096)), SubmitOutcome::Rejected);
    assert_eq!(s2.pending_len(), 0);
}

#[test]
fn lifecycle_pending_prefill_decode_done() {
    let mut s = sched();
    s.submit(Request::new(7, 5000, 64));
    // Prefill in 3 chunks, then the request joins the decode lanes.
    let mut prefilled = 0u64;
    for _ in 0..3 {
        match s.next_step() {
            Step::PrefillChunk { request_id, len, .. } => {
                assert_eq!(request_id, 7);
                prefilled += len;
            }
            other => panic!("expected a prefill chunk, got {other:?}"),
        }
    }
    assert_eq!(prefilled, 5000);
    assert_eq!(s.decoding_len(), 1, "request joins decode after its last chunk");
    match s.next_step() {
        Step::DecodeRound { request_ids } => assert_eq!(request_ids, vec![7]),
        other => panic!("expected a decode round, got {other:?}"),
    }
    let done = s.finish(7).expect("finish");
    assert_eq!(done.state, mimo26_coordinator::RequestState::Done);
    assert_eq!(s.decoding_len(), 0);
}

#[test]
fn preempt_lowest_priority_releases_its_reservation() {
    let mut s = sched();
    s.submit(Request::new(1, 100, 16));
    s.submit(Request::new(2, 100, 16));
    // Prefill both into decode.
    for _ in 0..2 {
        assert!(matches!(s.next_step(), Step::PrefillChunk { .. }));
    }
    assert_eq!(s.decoding_len(), 2);
    let used_before = s.used_bytes();
    let preempted = s.preempt_lowest_priority().expect("preempt");
    assert_eq!(preempted.id, 2, "most recent lane is lowest priority");
    assert!(s.used_bytes() < used_before, "preemption releases the reservation");
    assert_eq!(s.decoding_len(), 1);
}

#[test]
fn mixing_heuristic_time_slices_under_half_expert_touch() {
    assert!(Scheduler::should_time_slice(0.3));
    assert!(!Scheduler::should_time_slice(0.6));
}
