//! CPU reference for the rank-local wire boundary; not a GPU implementation.
//! Input is unweighted FC2 output in [token, route-slot, hidden] order.
use crate::{slice::HIDDEN, NaiveBits};

pub const ROUTES: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankSumError {
    SizeOverflow,
    Shape,
    NonFinite,
}

/// Apply each route weight once, then add in slot order 0..7 in FP32.
/// Do not round individual routes or the rank sum here: BF16 belongs to the
/// subsequent ReturnRow encoder. A malformed/nonfinite input fails closed.
pub fn rank_pre_sum(
    raw: &[f32], weights: &[f32], tokens: usize, naive: NaiveBits,
) -> Result<Vec<f32>, RankSumError> {
    let routes = tokens.checked_mul(ROUTES).ok_or(RankSumError::SizeOverflow)?;
    let elements = routes.checked_mul(HIDDEN).ok_or(RankSumError::SizeOverflow)?;
    if weights.len() != routes || raw.len() != elements {
        return Err(RankSumError::Shape);
    }
    if raw.iter().chain(weights).any(|v| !v.is_finite()) {
        return Err(RankSumError::NonFinite);
    }
    let mut out = vec![0.0f32; tokens * HIDDEN];
    for token in 0..tokens {
        for slot in 0..ROUTES {
            let route = token * ROUTES + slot;
            let w = weights[route];
            for h in 0..HIDDEN {
                let mut weighted = raw[route * HIDDEN + h] * w;
                if naive.has(NaiveBits::ROUTE_WEIGHT_TWICE) { weighted *= w; }
                out[token * HIDDEN + h] += weighted;
                if !out[token * HIDDEN + h].is_finite() {
                    return Err(RankSumError::NonFinite);
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn weight_once_and_shape_failures() {
        let raw = vec![2.0; ROUTES * HIDDEN];
        let weights = vec![0.125; ROUTES];
        assert_eq!(rank_pre_sum(&raw, &weights, 1, NaiveBits::NONE).unwrap(), vec![2.0; HIDDEN]);
        assert_eq!(rank_pre_sum(&raw, &weights, 1, NaiveBits::ROUTE_WEIGHT_TWICE).unwrap(), vec![0.25; HIDDEN]);
        assert_eq!(rank_pre_sum(&raw[1..], &weights, 1, NaiveBits::NONE), Err(RankSumError::Shape));
        assert_eq!(rank_pre_sum(&[], &[], usize::MAX, NaiveBits::NONE), Err(RankSumError::SizeOverflow));
        assert_eq!(rank_pre_sum(&[], &[], 0, NaiveBits::NONE).unwrap(), Vec::<f32>::new());
    }
    #[test]
    fn rejects_nonfinite_inputs_and_accumulation() {
        let mut raw = vec![2.0; ROUTES * HIDDEN];
        let weights = vec![1.0; ROUTES];
        raw[0] = f32::NAN;
        assert_eq!(rank_pre_sum(&raw, &weights, 1, NaiveBits::NONE), Err(RankSumError::NonFinite));
        raw.fill(f32::MAX);
        assert_eq!(rank_pre_sum(&raw, &weights, 1, NaiveBits::NONE), Err(RankSumError::NonFinite));
    }
}
