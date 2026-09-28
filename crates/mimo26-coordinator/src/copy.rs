//! Copy windows (TensorFold import, 28 Sep 2026): drafts copied from a request's own context.
//!
//! Agent output often repeats its context: a file written back with one name changed, an edit call
//! quoting the lines it replaces, a function quoted verbatim. When the last [`MATCH`] tokens of a
//! request's context (prompt, then its output so far) occurred earlier in it, the tokens that
//! followed that occurrence are the round's drafts instead of DFlash's (up to 7, the verify block).
//! TensorFold (`ashhart/TensorFold`, MIT) measured such a match in ~25% of agent rounds, its next
//! token right 94% of the time, and set the entry at 8 tokens: shorter coincidental matches in fresh
//! code (indentation, `self.`) failed 56 of 70 copied tokens. Among several earlier occurrences the
//! one matching furthest back wins (up to [`EXTEND`] tokens), ties to the most recent; a copy that
//! runs into its own continuation repeats it (a periodic run). The target verifies every copied
//! token as it verifies a draft, so a copy changes which rows are verified, never the output.
//!
//! Greedy requests only (`api.rs`): a sampled request's draw leaves copied text more often, a failed
//! copy costs the drafter's round, and its coupled DFlash drafts already follow a copied span where
//! the target is near-certain. `MIMO26_COPY=0` turns copy windows off.

use std::collections::HashMap;

/// Context tokens that must match before a copy is proposed.
pub const MATCH: usize = 8;
/// Backward match length counted up to this when choosing among occurrences.
const EXTEND: usize = 64;
/// Earlier occurrences tried per proposal, newest first.
const CANDIDATES: usize = 64;
const NONE: u32 = u32::MAX;

/// A request's index of the [`MATCH`]-grams in its context, extended as the context grows.
#[derive(Default)]
pub struct CopyIndex {
    /// The last position (the gram's final token) of each gram seen.
    head: HashMap<[u32; MATCH], u32>,
    /// `prev[p]`: the previous position with the gram ending at `p`, or [`NONE`].
    prev: Vec<u32>,
}

impl CopyIndex {
    /// Up to `k` tokens to copy after `ctx`, or none when `k < 2` or the last [`MATCH`] tokens of
    /// `ctx` did not occur earlier. Positions before `ctx.len() - 1` are indexed first, so the
    /// context must only grow between calls (a request's history does).
    pub fn propose(&mut self, ctx: &[usize], k: usize) -> Vec<usize> {
        let n = ctx.len();
        if k < 2 || n <= MATCH {
            return Vec::new();
        }
        // Index every gram with at least one token after it.
        for p in self.prev.len()..n - 1 {
            if p + 1 < MATCH {
                self.prev.push(NONE);
                continue;
            }
            let g = gram(&ctx[p + 1 - MATCH..=p]);
            self.prev.push(self.head.insert(g, p as u32).unwrap_or(NONE));
        }
        let Some(&newest) = self.head.get(&gram(&ctx[n - MATCH..])) else {
            return Vec::new();
        };
        // The occurrence matching furthest back (ties: the most recent).
        let (mut best, mut best_len, mut p, mut tried) = (NONE, 0usize, newest, 0usize);
        while p != NONE && tried < CANDIDATES {
            let q = p as usize;
            let mut len = MATCH;
            while len < EXTEND && len <= q && ctx[q - len] == ctx[n - 1 - len] {
                len += 1;
            }
            if len > best_len {
                (best, best_len) = (p, len);
                if len == EXTEND {
                    break;
                }
            }
            p = self.prev[q];
            tried += 1;
        }
        let from = best as usize + 1;
        let mut out = Vec::with_capacity(k);
        for j in 0..k {
            let t = if from + j < n { ctx[from + j] } else { out[from + j - n] };
            out.push(t);
        }
        out
    }
}

fn gram(t: &[usize]) -> [u32; MATCH] {
    std::array::from_fn(|i| t[i] as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(s: &str) -> Vec<usize> {
        s.bytes().map(usize::from).collect()
    }

    #[test]
    fn copies_what_followed_an_earlier_match() {
        let mut c = CopyIndex::default();
        let ctx = ids("fn alpha(x) { return x + 1; }\n// again: fn alpha(x) { ret");
        assert_eq!(c.propose(&ctx, 7), ids("urn x +"));
        // Short matches do not count.
        let mut c = CopyIndex::default();
        assert!(c.propose(&ids("abcdefg xyz abcdefg"), 7).is_empty());
        // Nothing when k < 2 or the context is short.
        let mut c = CopyIndex::default();
        assert!(c.propose(&ctx, 1).is_empty());
        assert!(c.propose(&ctx[..MATCH], 7).is_empty());
    }

    #[test]
    fn longest_backward_match_wins_then_most_recent() {
        // "12345678" occurs twice; only the first occurrence is preceded by "XY", as the tail is.
        let mut c = CopyIndex::default();
        let ctx = ids("XY12345678AB..zz12345678CD..XY12345678");
        assert_eq!(c.propose(&ctx, 2), ids("AB"));
        // Equal evidence: the most recent occurrence.
        let mut c = CopyIndex::default();
        let ctx = ids("..12345678AB..12345678CD..12345678");
        assert_eq!(c.propose(&ctx, 2), ids("CD"));
    }

    #[test]
    fn a_growing_context_is_indexed_incrementally() {
        let mut c = CopyIndex::default();
        let mut ctx = ids("the quick brown fox jumps over the lazy dog; ");
        for t in ids("the quick") {
            ctx.push(t);
            let got = c.propose(&ctx, 7);
            let full = CopyIndex::default().propose(&ctx, 7);
            assert_eq!(got, full);
        }
        assert_eq!(c.propose(&ctx, 7), ids(" brown "));
    }

    #[test]
    fn a_periodic_run_repeats() {
        let mut c = CopyIndex::default();
        let ctx = ids("start: ab-ab-ab-ab-ab");
        assert_eq!(c.propose(&ctx, 7), ids("-ab-ab-"));
    }
}
