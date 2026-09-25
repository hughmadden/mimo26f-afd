//! A4 scheduler — request lifecycle, chunked prefill, lane management, and
//! admission integration (ARCHITECTURE.md §11.1/§11.4). The coordinator owns
//! the scheduler (A1 F1 spine: "KV pool and paging; config"; §6 "A4 scheduler").
//!
//! Policy captured here (the specified parts; the §11.4 prefill/decode mixing
//! heuristic is a serving-time knob, see `should_time_slice`):
//! * a standalone prefill chunk is 2,048 tokens (4,096 on 96 GB coordinators);
//! * a long prefill is preemptible at chunk boundaries;
//! * requests over `long_context_threshold` (512K) run in a long-context lane
//!   at concurrency 1;
//! * admission gates NEW work only (A1, grow-on-demand); a running request is
//!   never failed for pool pressure.

use std::collections::VecDeque;

use crate::kv_pool::{Admission, KvPool, DEFAULT_ADMIT_RESERVE};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestState {
    Pending,
    Prefilling,
    Decoding,
    Done,
}

/// One request slot. `reservation` is the KV bytes admitted for it (released
/// on `finish`).
#[derive(Debug, Clone)]
pub struct Request {
    pub id: u64,
    pub prompt_tokens: u64,
    pub max_tokens: u64,
    pub state: RequestState,
    /// Tokens prefilled so far (0 until prefilling starts).
    pub prefill_pos: u64,
    reservation: u64,
}

impl Request {
    pub fn new(id: u64, prompt_tokens: u64, max_tokens: u64) -> Self {
        Request {
            id,
            prompt_tokens,
            max_tokens,
            state: RequestState::Pending,
            prefill_pos: 0,
            reservation: 0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SchedulerConfig {
    /// Standalone prefill chunk (2,048; 4,096 on 96 GB coordinators).
    pub prefill_chunk: u64,
    /// Requests over this get a long-context lane at concurrency 1.
    pub long_context_threshold: u64,
    pub max_concurrency: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        SchedulerConfig { prefill_chunk: 2048, long_context_threshold: 524_288, max_concurrency: 8 }
    }
}

/// One schedule step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Step {
    /// Prefill the next chunk of a pending request `[start, start+len)`.
    PrefillChunk { request_id: u64, start: u64, len: u64 },
    /// Run one decode round over the active decode lanes.
    DecodeRound { request_ids: Vec<u64> },
    Idle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitOutcome {
    Accepted,
    /// 429 Retry-After (A1 pressure-ladder step 3).
    Rejected,
}

/// A4 scheduler: admitted requests prefill in chunks, then decode in lanes.
pub struct Scheduler {
    config: SchedulerConfig,
    pool: KvPool,
    pending: VecDeque<Request>,
    decoding: Vec<Request>,
}

impl Scheduler {
    pub fn new(config: SchedulerConfig, pool: KvPool) -> Self {
        Scheduler { config, pool, pending: VecDeque::new(), decoding: Vec::new() }
    }

    pub fn used_bytes(&self) -> u64 {
        self.pool.used()
    }

    /// Long-context lane rule: requests over the threshold run alone.
    pub fn is_long_context(&self, prompt_tokens: u64) -> bool {
        prompt_tokens > self.config.long_context_threshold
    }

    /// Effective decode concurrency for a request (1 for long context).
    pub fn concurrency_for(&self, prompt_tokens: u64) -> usize {
        if self.is_long_context(prompt_tokens) {
            1
        } else {
            self.config.max_concurrency
        }
    }

    /// Split a prefill into `prefill_chunk`-token pieces `[(start, len), …]`.
    pub fn chunks(&self, prompt_tokens: u64) -> Vec<(u64, u64)> {
        let mut out = Vec::new();
        let mut pos = 0u64;
        while pos < prompt_tokens {
            let len = (prompt_tokens - pos).min(self.config.prefill_chunk);
            out.push((pos, len));
            pos += len;
        }
        out
    }

    /// Admit a NEW request (A1, grow-on-demand) and enqueue it for prefill.
    /// Over-budget admission is rejected (429); it never touches the queue.
    pub fn submit(&mut self, mut req: Request) -> SubmitOutcome {
        match self.pool.admit(req.prompt_tokens, req.max_tokens, DEFAULT_ADMIT_RESERVE) {
            Admission::Admitted { reservation } => {
                req.reservation = reservation;
                req.state = RequestState::Pending;
                self.pending.push_back(req);
                SubmitOutcome::Accepted
            }
            Admission::Rejected => SubmitOutcome::Rejected,
        }
    }

    /// Next schedule step: a prefill chunk if any request is prefilling,
    /// otherwise a decode round over the lanes (or Idle).
    pub fn next_step(&mut self) -> Step {
        // Advance the head pending request into prefilling.
        if self.decoding.is_empty() && self.pending.is_empty() {
            return Step::Idle;
        }
        if let Some(req) = self.pending.front_mut() {
            if req.state == RequestState::Pending {
                req.state = RequestState::Prefilling;
            }
            let start = req.prefill_pos;
            let remaining = req.prompt_tokens.saturating_sub(start);
            if remaining > 0 {
                let len = remaining.min(self.config.prefill_chunk);
                req.prefill_pos += len;
                if req.prefill_pos >= req.prompt_tokens {
                    req.state = RequestState::Decoding;
                    let req = self.pending.pop_front().unwrap();
                    let id = req.id;
                    self.decoding.push(req);
                    return Step::PrefillChunk { request_id: id, start, len };
                }
                return Step::PrefillChunk { request_id: req.id, start, len };
            }
        }
        if !self.decoding.is_empty() {
            let ids = self.decoding.iter().map(|r| r.id).collect();
            return Step::DecodeRound { request_ids: ids };
        }
        Step::Idle
    }

    /// Preempt the lowest-priority active request (pressure-ladder step 2).
    /// Long-context lanes and later-arrived requests are lower priority.
    pub fn preempt_lowest_priority(&mut self) -> Option<Request> {
        // Simple v1 policy: preempt the last (most recent) decode lane.
        let idx = self.decoding.len().checked_sub(1)?;
        let req = self.decoding.remove(idx);
        self.release_reservation(&req);
        Some(req)
    }

    /// Finish a decode request and release its KV reservation.
    pub fn finish(&mut self, id: u64) -> Option<Request> {
        let idx = self.decoding.iter().position(|r| r.id == id)?;
        let mut req = self.decoding.remove(idx);
        req.state = RequestState::Done;
        self.release_reservation(&req);
        Some(req)
    }

    pub fn decoding_len(&self) -> usize {
        self.decoding.len()
    }
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// §11.4 mixing heuristic: time-slice (a decode round, then a prefill chunk)
    /// when decode rounds touch under ~50% of experts; otherwise merge prefill
    /// slices into the decode rounds. `expert_touch_fraction` is measured by the
    /// serving loop.
    pub fn should_time_slice(expert_touch_fraction: f32) -> bool {
        expert_touch_fraction < 0.5
    }

    fn release_reservation(&mut self, req: &Request) {
        self.pool.release(req.reservation);
    }
}
