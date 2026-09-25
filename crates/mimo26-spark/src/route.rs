//! Route reduction — the Spark's token→expert routing table and the grouped-FFN
//! → per-token pre-sum.
//!
//! A request carries `tokens x 8` route entries (token-major, slot-minor). The
//! Spark groups the routed rows by expert to build the FFN's [`GroupedPlan`],
//! replicates each token's hidden row once per route into the FFN input, then
//! reduces the grouped FFN output back to one weighted, pre-summed `[token,4096]`
//! partial. This is the CPU twin of the reduction the daemon performs between the
//! kernel and the wire return.

use std::collections::BTreeMap;

use mimo26_expert::grouped::{Group, GroupedPlan};
use mimo26_expert::slice::HIDDEN;
use mimo26_wire::RouteEntry;

/// Decompose one expert's real token count into compiled per-M tiles: greedy
/// 256, then 64, then 16, with only the last 16-tile padded. This caps the
/// waste at <16 rows per group (vs the next-size padding's up-to-4x), so the
/// measured ~1.735x waste factor drops below the 1.3x ladder threshold.
fn tile_m(real: usize) -> Result<Vec<usize>, String> {
    let mut tiles = Vec::new();
    let mut r = real;
    while r >= 256 {
        tiles.push(256);
        r -= 256;
    }
    while r >= 64 {
        tiles.push(64);
        r -= 64;
    }
    while r >= 16 {
        tiles.push(16);
        r -= 16;
    }
    if r > 0 {
        tiles.push(16); // the last 16-tile, padded (r in [1, 15])
    }
    Ok(tiles)
}

/// A resolved routing table for one request.
pub struct RoutePlan {
    /// The FFN grouped plan (groups ordered by ascending expert id), with each
    /// group's M padded up to the next compiled per-M size.
    pub plan: GroupedPlan,
    /// Per routed row (in the FFN's group order): the token it belongs to.
    pub row_token: Vec<usize>,
    /// Per routed row: the gate weight (route weight, applied once).
    pub row_weight: Vec<f32>,
    /// Per routed row: its index in the padded FFN input/output matrix.
    pub row_padded: Vec<usize>,
    /// Token count.
    pub tokens: usize,
}

impl RoutePlan {
    /// Build the routing table from `tokens x 8` route entries.
    ///
    /// The routes must be token-major, slot-minor (8 per token), with
    /// `row_index == token`. Any other shape is a protocol error.
    pub fn from_routes(routes: &[RouteEntry], tokens: usize) -> Result<Self, String> {
        if routes.len() != tokens * 8 {
            return Err(format!(
                "route: {} routes for {tokens} tokens (expected {})",
                routes.len(),
                tokens * 8
            ));
        }
        for (slot, r) in routes.iter().enumerate() {
            let token = slot / 8;
            if r.row_index as usize != token {
                return Err(format!(
                    "route: entry {slot} row_index {} != token {token}",
                    r.row_index
                ));
            }
        }

        // Group routed rows by expert id (ascending).
        let mut by_expert: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
        for (slot, r) in routes.iter().enumerate() {
            by_expert.entry(r.expert_id).or_default().push(slot);
        }

        let mut groups = Vec::with_capacity(by_expert.len());
        let mut row_token = Vec::with_capacity(routes.len());
        let mut row_weight = Vec::with_capacity(routes.len());
        let mut row_padded = Vec::with_capacity(routes.len());
        let mut running = 0usize; // padded row offset
        for (&expert, slots) in &by_expert {
            let real = slots.len();
            let tiles = tile_m(real)?;
            let group_start = running;
            for &tile in &tiles {
                groups.push(Group {
                    expert: expert as usize,
                    tokens: tile,
                    token_offset: running,
                });
                running += tile;
            }
            for (j, &slot) in slots.iter().enumerate() {
                row_token.push(slot / 8);
                row_weight.push(routes[slot].gate_weight);
                // The real routes fill the tiles contiguously from group_start.
                row_padded.push(group_start + j);
            }
        }
        Ok(Self {
            plan: GroupedPlan { groups, padded: 0 },
            row_token,
            row_weight,
            row_padded,
            tokens,
        })
    }

    /// The number of routed rows (`tokens x 8`).
    pub fn routed_rows(&self) -> usize {
        self.row_token.len()
    }

    /// The padded FFN input/output row count (sum of the per-group padded M).
    pub fn padded_rows(&self) -> usize {
        self.plan.total_tokens()
    }

    /// The M-padding waste factor `sum(padded M) / sum(real M)`.
    pub fn waste_factor(&self) -> f64 {
        self.padded_rows() as f64 / self.routed_rows() as f64
    }

    /// Replicate each token's hidden row once per route, in group order, padded
    /// to the per-expert compiled M, to form the FFN input `[padded_rows, 4096]`.
    /// Padding rows are zero.
    pub fn replicate_x(&self, hidden: &[f32]) -> Result<Vec<f32>, String> {
        if hidden.len() != self.tokens * HIDDEN {
            return Err(format!(
                "route: hidden has {} elements, expected {} tokens x {HIDDEN}",
                hidden.len(),
                self.tokens
            ));
        }
        let mut x = vec![0.0f32; self.padded_rows() * HIDDEN];
        for (&token, &padded) in self.row_token.iter().zip(self.row_padded.iter()) {
            let src = &hidden[token * HIDDEN..(token + 1) * HIDDEN];
            x[padded * HIDDEN..(padded + 1) * HIDDEN].copy_from_slice(src);
        }
        Ok(x)
    }

    /// Reduce the grouped FFN output `[padded_rows, 4096]` (group order, padded)
    /// into the per-token pre-summed `[tokens, 4096]`, applying each route weight
    /// once and skipping the padding rows.
    pub fn reduce(&self, ffn_out: &[f32]) -> Result<Vec<f32>, String> {
        if ffn_out.len() != self.padded_rows() * HIDDEN {
            return Err(format!(
                "route: ffn_out has {} elements, expected {} padded rows x {HIDDEN}",
                ffn_out.len(),
                self.padded_rows()
            ));
        }
        let mut out = vec![0.0f32; self.tokens * HIDDEN];
        for (token, w, padded) in self
            .row_token
            .iter()
            .zip(self.row_weight.iter())
            .zip(self.row_padded.iter())
            .map(|((t, w), p)| (t, w, p))
        {
            for h in 0..HIDDEN {
                out[token * HIDDEN + h] += ffn_out[padded * HIDDEN + h] * w;
            }
        }
        Ok(out)
    }

    /// The route table in token-major order (token 0's 8 routes, then token 1's,
    /// …), each token's routes in the exact order `reduce` iterates them (stable
    /// sort of the expert-grouped order by token). For the device route reduce,
    /// which must accumulate in the same order to be bitwise-identical. Returns
    /// `(padded, weight)` where `padded[i]`/`weight[i]` are token-major; token
    /// `t` occupies `[t*8, t*8+8)`.
    pub fn token_major_routes(&self) -> (Vec<i32>, Vec<f32>) {
        let mut order: Vec<usize> = (0..self.row_token.len()).collect();
        order.sort_by_key(|&i| self.row_token[i]); // stable
        let padded = order.iter().map(|&i| self.row_padded[i] as i32).collect();
        let weight = order.iter().map(|&i| self.row_weight[i]).collect();
        (padded, weight)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn route(row_index: u32, expert_id: u32, gate_weight: f32) -> RouteEntry {
        RouteEntry { row_index, expert_id, gate_weight }
    }

    #[test]
    fn groups_by_expert_and_is_a_bijection() {
        // 2 tokens, 8 routes each, overlapping experts.
        let routes = vec![
            route(0, 0, 0.5), route(0, 1, 0.25), route(0, 0, 0.125), route(0, 2, 0.0625),
            route(0, 3, 0.03125), route(0, 1, 0.015625), route(0, 0, 0.0078125), route(0, 4, 0.0078125),
            route(1, 1, 0.5), route(1, 0, 0.25), route(1, 4, 0.125), route(1, 2, 0.0625),
            route(1, 3, 0.03125), route(1, 0, 0.015625), route(1, 2, 0.0078125), route(1, 1, 0.00390625),
        ];
        let rp = RoutePlan::from_routes(&routes, 2).expect("plan");
        assert_eq!(rp.routed_rows(), 16);
        assert_eq!(rp.plan.groups.len(), 5); // experts 0..4

        // Every (token, slot) appears exactly once in group order.
        let mut seen = std::collections::HashSet::new();
        for &token in &rp.row_token {
            seen.insert(token);
        }
        assert_eq!(seen, [0usize, 1].into_iter().collect());
        assert_eq!(rp.row_token.iter().filter(|&&t| t == 0).count(), 8);
        assert_eq!(rp.row_token.iter().filter(|&&t| t == 1).count(), 8);

        // Weights are preserved (sum of weights per token == 1.0 for this
        // synthetic router: 0.5+0.25+0.125+0.0625+0.03125+0.015625+0.0078125+0.00390625).
        let s0: f32 = rp.row_weight.iter().zip(&rp.row_token).filter(|(_, &t)| t == 0).map(|(w, _)| w).sum();
        assert!((s0 - 1.0).abs() < 1e-6, "token 0 weights sum {s0}");
    }

    #[test]
    fn replicate_and_reduce_match_direct() {
        let routes = vec![
            route(0, 0, 0.5), route(0, 1, 0.5), route(0, 0, 0.0), route(0, 1, 0.0),
            route(0, 2, 0.0), route(0, 2, 0.0), route(0, 2, 0.0), route(0, 2, 0.0),
            route(1, 1, 0.25), route(1, 0, 0.75), route(1, 0, 0.0), route(1, 1, 0.0),
            route(1, 1, 0.0), route(1, 2, 0.0), route(1, 2, 0.0), route(1, 2, 0.0),
        ];
        let rp = RoutePlan::from_routes(&routes, 2).expect("plan");

        let hidden: Vec<f32> = (0..2 * HIDDEN).map(|i| i as f32 + 1.0).collect();
        let x = rp.replicate_x(&hidden).expect("replicate");
        assert_eq!(x.len(), rp.padded_rows() * HIDDEN);

        // Synthetic FFN output: ffn_out[padded_row][h] = (padded_row + 1) as f32.
        let ffn_out: Vec<f32> = (0..rp.padded_rows() * HIDDEN).map(|i| ((i / HIDDEN) + 1) as f32).collect();
        let got = rp.reduce(&ffn_out).expect("reduce");

        // Direct: for each token, sum over routes of (padded_row+1) * weight.
        for token in 0..2 {
            let mut want = 0.0f32;
            for (row, (&t, &w)) in rp.row_token.iter().zip(&rp.row_weight).enumerate() {
                if t == token {
                    want += (rp.row_padded[row] as f32 + 1.0) * w;
                }
            }
            for h in 0..HIDDEN {
                assert!(
                    (got[token * HIDDEN + h] - want).abs() < 1e-4,
                    "token {token} h {h}: {} != {want}",
                    got[token * HIDDEN + h]
                );
            }
        }
    }

    #[test]
    fn rejects_bad_route_shape() {
        let routes = vec![route(1, 0, 1.0)]; // 1 route for 1 token is wrong (need 8)
        assert!(RoutePlan::from_routes(&routes, 1).is_err());
        let bad = vec![route(1, 0, 1.0), route(0, 0, 1.0), route(0, 0, 1.0), route(0, 0, 1.0),
                       route(0, 0, 1.0), route(0, 0, 1.0), route(0, 0, 1.0), route(0, 0, 1.0)];
        assert!(RoutePlan::from_routes(&bad, 1).is_err()); // row_index 1 out of range
    }

    #[test]
    fn tile_m_decomposes_over_256() {
        assert_eq!(tile_m(300).unwrap(), vec![256, 16, 16, 16]);
        assert_eq!(tile_m(512).unwrap(), vec![256, 256]);
        assert_eq!(tile_m(600).unwrap(), vec![256, 256, 64, 16, 16]);
        assert_eq!(tile_m(25).unwrap(), vec![16, 16]);
        assert_eq!(tile_m(64).unwrap(), vec![64]);
        assert_eq!(tile_m(7).unwrap(), vec![16]);
    }

    /// A skewed expert with M > 256 splits into sub-tiles and the padded
    /// replicate/reduce still recombines to the direct weighted sum.
    #[test]
    fn skew_over_256_recombines() {
        // 40 tokens, 8 routes each, all to expert 0 => real M = 320 > 256.
        let mut routes = Vec::new();
        for t in 0..40u32 {
            for _ in 0..8 {
                routes.push(route(t, 0, 1.0 / 8.0));
            }
        }
        let rp = RoutePlan::from_routes(&routes, 40).expect("plan");
        // 320 real -> [256, 64] tiles (320 padded, no waste here).
        assert_eq!(rp.plan.groups.len(), 2);
        assert_eq!(rp.padded_rows(), 320);
        assert_eq!(rp.routed_rows(), 320);

        let hidden: Vec<f32> = (0..40 * HIDDEN).map(|i| i as f32 + 1.0).collect();
        let x = rp.replicate_x(&hidden).expect("replicate");
        assert_eq!(x.len(), 320 * HIDDEN);
        let ffn_out: Vec<f32> = (0..320 * HIDDEN).map(|i| ((i / HIDDEN) + 1) as f32).collect();
        let got = rp.reduce(&ffn_out).expect("reduce");

        // Direct: token t sums (padded_row+1) * (1/8) over its 8 routes.
        for t in 0..40 {
            let want: f32 = (0..320).filter(|&row| rp.row_token[row] == t)
                .map(|row| (rp.row_padded[row] as f32 + 1.0) * rp.row_weight[row])
                .sum();
            for h in 0..HIDDEN {
                assert!((got[t * HIDDEN + h] - want).abs() < 1e-3, "token {t} h {h}");
            }
        }
    }
}
