//! MoE dispatch/combine — the CPU twin of `oracle/mimo26/nn/layers.py::moe_forward`
//! (the arithmetic the Spark lane and the R8 coordinator sum jointly replace in
//! the engine; this is the golden-reference path).

/// Combine routed expert outputs: for every (row, slot) route, add
/// `expert_fn(expert, x_row) * weight` into the row. Equivalent to the oracle's
/// grouped dispatch (`out[rows] += y * wts[rows]`); per-(row, slot) here is the
/// same arithmetic, just ungrouped.
///
/// `x` is `[rows, hidden]`; `idx`/`wts` are `[rows, top_k]` flattened.
pub fn moe_forward<F: FnMut(usize, &[f32]) -> Vec<f32>>(
    x: &[f32],
    idx: &[usize],
    wts: &[f32],
    hidden: usize,
    top_k: usize,
    mut expert_fn: F,
) -> Vec<f32> {
    assert_eq!(x.len() % hidden, 0, "x length must be a multiple of hidden");
    let rows = x.len() / hidden;
    assert_eq!(idx.len(), rows * top_k, "idx shape [rows, top_k]");
    assert_eq!(wts.len(), rows * top_k, "wts shape [rows, top_k]");
    let mut out = vec![0.0f32; rows * hidden];
    for r in 0..rows {
        let xrow = &x[r * hidden..(r + 1) * hidden];
        for j in 0..top_k {
            let e = idx[r * top_k + j];
            let w = wts[r * top_k + j];
            let y = expert_fn(e, xrow);
            let orow = &mut out[r * hidden..(r + 1) * hidden];
            for i in 0..hidden {
                orow[i] += y[i] * w;
            }
        }
    }
    out
}
