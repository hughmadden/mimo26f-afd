//! KV pool accounting + A1 admission (ARCHITECTURE.md §11.1, §11.8, §11.11).
//!
//! The coordinator owns the KV pool (A1 F1: "KV pool and paging; config").
//! Geometry is the MiMo-V2.6 config (oracle `MiMoConfig`): 9 GA layers × 4 KV
//! heads × (QK 192 + V 128) = **11,520 B/token** FP8, and 39 SWA rings × 8 KV
//! heads × 320 × window 128 = **12,779,520 B/seq**.
//!
//! T21 (ADVISOR-I4): the pool is **measured at boot**, never pinned from a
//! planner's `VRAM × 0.97 − weights` (that omits workspace, headroom and the
//! CUDA context — the first attempt's formula over-reserved an 18.13 GiB pool).
//! Admission reserves the prompt plus `min(max_tokens, admit_reserve)` and
//! **grows per round**; the lifetime `prompt + full max_tokens` reservation is
//! the 755 MB/request trap the grow-on-demand rule avoids. A running request is
//! never failed for pool pressure — only new admission is gated.

/// GA KV bytes per token (FP8, scales not counted here): 9 × 4 × (192 + 128).
pub const GA_KV_BYTES_PER_TOKEN: u64 = 9 * 4 * (192 + 128);
/// SWA ring bytes per sequence: 39 × 8 × (192 + 128) × 128.
pub const SWA_RING_BYTES_PER_SEQ: u64 = 39 * 8 * (192 + 128) * 128;
/// Output quantum reserved at admission (grows per round) — ARCHITECTURE §11.11.
pub const DEFAULT_ADMIT_RESERVE: u64 = 8192;
/// Default output cap when the client omits `max_tokens` (T23).
pub const MAX_OUTPUT_TOKENS: u64 = 65_536;

/// Grow-on-demand reservation: prompt KV + one SWA ring + `min(max_tokens,
/// admit_reserve)` output quantum.
pub fn request_reservation(prompt_tokens: u64, max_tokens: u64, admit_reserve: u64) -> u64 {
    let output_reserve = max_tokens.min(admit_reserve);
    prompt_tokens * GA_KV_BYTES_PER_TOKEN + SWA_RING_BYTES_PER_SEQ + output_reserve * GA_KV_BYTES_PER_TOKEN
}

/// The T21 trap shape: prompt + **full** `max_tokens` lifetime reservation.
/// At `max_tokens = 65,536` this is 65,536 × 11,520 = 754,974,720 B ≈ 755 MB
/// per request — the over-reservation the grow-on-demand rule avoids.
pub fn full_lifetime_reservation(prompt_tokens: u64, max_tokens: u64) -> u64 {
    prompt_tokens * GA_KV_BYTES_PER_TOKEN + SWA_RING_BYTES_PER_SEQ + max_tokens * GA_KV_BYTES_PER_TOKEN
}

/// Admission decision for a new request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Admission {
    /// Admitted with this reservation (bytes).
    Admitted { reservation: u64 },
    /// No room — 429 with `Retry-After` (pressure-ladder step 3).
    Rejected,
}

/// A boot-measured pool with grow-on-demand accounting.
#[derive(Debug, Clone, Copy)]
pub struct KvPool {
    total: u64,
    used: u64,
}

impl KvPool {
    /// `measured_bytes` comes from the device at boot (never a planner pin).
    pub fn new(measured_bytes: u64) -> Self {
        KvPool { total: measured_bytes, used: 0 }
    }

    pub fn total(&self) -> u64 {
        self.total
    }
    pub fn used(&self) -> u64 {
        self.used
    }
    pub fn free(&self) -> u64 {
        self.total.saturating_sub(self.used)
    }

    /// Gate NEW admission on `min(max_tokens, admit_reserve)`; reject with 429
    /// when the reservation does not fit. A running request is never failed
    /// here — only new work is gated (ARCHITECTURE §11.1).
    pub fn admit(&mut self, prompt_tokens: u64, max_tokens: u64, admit_reserve: u64) -> Admission {
        let reservation = request_reservation(prompt_tokens, max_tokens, admit_reserve);
        if self.used + reservation > self.total {
            return Admission::Rejected;
        }
        self.used += reservation;
        Admission::Admitted { reservation }
    }

    /// Grow a running request's reservation by `extra_tokens`. This never
    /// rejects: pressure on a running request is handled by the ladder
    /// (drop snapshots → preempt lowest priority), never by failing the request.
    pub fn grow(&mut self, extra_tokens: u64) -> u64 {
        let extra = extra_tokens * GA_KV_BYTES_PER_TOKEN;
        self.used += extra;
        extra
    }

    /// Reserve raw bytes at page granularity — returns false when it does not
    /// fit (the pager's admission gate for NEW page allocation).
    pub fn reserve_bytes(&mut self, bytes: u64) -> bool {
        if self.used + bytes > self.total {
            return false;
        }
        self.used += bytes;
        true
    }

    pub fn release(&mut self, reservation: u64) {
        self.used = self.used.saturating_sub(reservation);
    }
}
