//! Replay a dumped DS41RTE3 request frame offline and print its route-plan
//! structure (the real skewed plan's tile-M histogram + launch count). Reads
//! `/var/tmp/mimo26f-kernel/req-layer1.bin` (dumped by the daemon under
//! `MIMO26_SPARK_DUMP_FRAME=1`). CPU-only — no resident, no device.
//!
//! ```text
//! cargo run -p mimo26-spark --example replay_frame -- /var/tmp/mimo26f-kernel/req-layer1.bin
//! ```

use std::collections::BTreeMap;

use mimo26_spark::route::RoutePlan;
use mimo26_wire::frame::Frame;
use mimo26_wire::l4::StreamReceiver;
use mimo26_wire::WireNaive;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "/var/tmp/mimo26f-kernel/req-layer1.bin".to_string());
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let mut rx = StreamReceiver::new(WireNaive::NONE);
    let frame = rx.accept(&bytes).expect("decode frame");
    let req = match frame {
        Frame::Request(r) => r,
        Frame::Return(_) => panic!("expected a request frame"),
    };
    let tokens = req.rows.len();
    let rp = RoutePlan::from_routes(&req.routes, tokens).expect("route plan");

    let mut hist: BTreeMap<usize, usize> = BTreeMap::new();
    for g in &rp.plan.groups {
        *hist.entry(g.tokens).or_default() += 1;
    }
    let hist_s: Vec<String> = hist.iter().map(|(m, c)| format!("M{m}:{c}")).collect();
    let capacity = 2048usize;
    let max_total = 8 * capacity;
    let max_groups = 256usize;
    // Count launches the same way `ffn_route_reduce` chunks.
    let mut n_launch = 0usize;
    let mut gi = 0usize;
    while gi < rp.plan.groups.len() {
        let mut ng = 0usize;
        let mut nt = 0usize;
        while gi < rp.plan.groups.len() && ng < max_groups && nt + rp.plan.groups[gi].tokens <= max_total {
            nt += rp.plan.groups[gi].tokens;
            ng += 1;
            gi += 1;
        }
        n_launch += 1;
    }

    println!("frame {} B, layer {}", bytes.len(), req.layer_id);
    println!("tokens={} routed_rows={} padded_rows={} waste={:.3}x",
        tokens, rp.routed_rows(), rp.padded_rows(), rp.waste_factor());
    println!("groups={} rows={} launches={}", rp.plan.groups.len(), rp.padded_rows(), n_launch);
    println!("tile-M histogram: {}", hist_s.join(" "));
}
