//! Token embedding lookup — the coordinator owns `embed_tokens` (ARCHITECTURE
//! §3). A pure row gather from `[vocab, hidden]`; no arithmetic, so the twin
//! is bit-exact by construction.

/// Gather embedding rows for `ids`. `table` is `[vocab, hidden]` row-major.
pub fn embed(ids: &[usize], table: &[f32], vocab: usize, hidden: usize) -> Vec<f32> {
    assert_eq!(table.len(), vocab * hidden, "table shape must be [vocab, hidden]");
    let mut out = vec![0.0f32; ids.len() * hidden];
    for (t, &id) in ids.iter().enumerate() {
        assert!(id < vocab, "token id {id} out of vocab {vocab}");
        out[t * hidden..(t + 1) * hidden]
            .copy_from_slice(&table[id * hidden..(id + 1) * hidden]);
    }
    out
}
