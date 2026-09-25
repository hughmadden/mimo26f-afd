//! `mimo26-coordinator` — the coordinator serving binary (I5 first tokens).
//!
//! Under the `cuda` feature this is the X1a bring-up: load the coordinator
//! weights, upload the dense GEMMs to the 5090, connect the four Spark daemons,
//! render + tokenize the X1a needle prompt, prefill the 4,027 tokens (GPU
//! attention + GPU dense + wire MoE), then greedy-decode to `KESTREL-41` + EOS.
//! Without `cuda` it runs the CPU smoke (tiny config + LCG weights).

#[cfg(feature = "cuda")]
mod serving_main {
    use std::sync::Arc;

    use mimo26_coordinator::api::CoordinatorEngine;
    use mimo26_coordinator::config::Config;
    use mimo26_coordinator::dforward::DeviceForward;
    use mimo26_coordinator::load::{load_coordinator_weights, weights_dir};
    use mimo26_coordinator::serving::ServingModel;
    use mimo26_coordinator::tokenizer::from_tokenizer_json;
    use mimo26_coordinator::wire::WireClient;

    pub fn run() {
        let cfg = Config::real();
        let dir = weights_dir();

        eprintln!("[coordinator] loading weights from {}", dir.display());
        let w = load_coordinator_weights(&dir, &cfg);

        let host_path = std::env::var_os("MIMO26_HOST_FORWARD").is_some();
        let max_chunk: usize = std::env::var("MIMO26_PREFILL_CHUNK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096);
        let kv_tokens: usize = std::env::var("MIMO26_KV_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096);
        let (model, dev) = if host_path {
            eprintln!("[coordinator] building HOST serving model (upload dense GEMMs)");
            (Some(ServingModel::new(cfg.clone(), w).expect("ServingModel::new")), None)
        } else {
            eprintln!("[coordinator] building DEVICE forward (chunk {max_chunk}, GA KV {kv_tokens} tokens)");
            let mut fwd = DeviceForward::new(cfg.clone(), w, max_chunk).expect("DeviceForward::new");
            // Perf reset S1: the checkpoint's DFlash drafter, unless MIMO26_DFLASH=0.
            let ddir = dir.join("dflash");
            if std::env::var("MIMO26_DFLASH").map(|v| v != "0").unwrap_or(true) && ddir.join("config.json").exists() {
                fwd.load_dflash(&ddir).expect("load DFlash drafter");
            }
            let kv = fwd.new_kv(kv_tokens).expect("DeviceKv");
            (None, Some((fwd, kv)))
        };

        let addrs: Vec<String> = std::env::var("MIMO26_SPARK_ADDRS")
            .map(|s| s.split(',').map(str::to_string).collect())
            .expect("MIMO26_SPARK_ADDRS: the four Sparks' RDMA-fabric addresses (host:port, rank order); no LAN default");
        eprintln!("[coordinator] connecting Sparks {:?}", addrs);
        let mut wire = WireClient::connect(&addrs).expect("WireClient::connect");

        if let Some((fwd, _)) = dev.as_ref() {
            fwd.register_plane_buffers(&wire.plane_buffers()).expect("register plane rings");
            fwd.register_plane_buffers(&wire.send_buffers()).expect("register request body");
        }
        let tok = from_tokenizer_json(dir.join("tokenizer.json").to_str().expect("utf8"))
            .expect("tokenizer");

        let engine = Arc::new(match (model, dev) {
            (Some(model), _) => CoordinatorEngine::new(cfg, model, tok, wire),
            (None, Some((fwd, kv))) => CoordinatorEngine::new_device(cfg, fwd, kv, tok, wire),
            (None, None) => unreachable!(),
        });
        let addr = std::env::var("MIMO26_API_ADDR").unwrap_or_else(|_| "0.0.0.0:8100".to_string());
        eprintln!("[coordinator] serving A8 API on {addr}");
        mimo26_api::serve(&addr, engine).expect("serve");
    }
}

#[cfg(not(feature = "cuda"))]
use mimo26_coordinator::{
    gen_weights, greedy, Config, LayerCache, Model, SamplingParams, StopConfig,
};

fn main() {
    #[cfg(feature = "cuda")]
    {
        serving_main::run();
        return;
    }
    #[cfg(not(feature = "cuda"))]
    {
        let real = Config::real();
        println!(
            "mimo26-coordinator: vocab={} hidden={} layers={} ({} GA / {} SWA) max_pos={}",
            real.vocab_size,
            real.hidden_size,
            real.num_hidden_layers,
            real.n_ga(),
            real.n_swa(),
            real.max_position_embeddings
        );
        // CPU smoke: tiny config + LCG weights + composed forward + greedy + stop.
        let cfg = Config::tiny();
        let model = Model::new(cfg.clone(), gen_weights(&cfg, 42, 0.02));
        let mut caches: Vec<LayerCache> =
            (0..cfg.num_hidden_layers).map(|l| LayerCache::for_layer(&cfg, l)).collect();
        let (logits, _hidden) = model.forward(&[5, 13, 77], &mut caches);
        let vocab = cfg.vocab_size;
        let last: Vec<f32> = logits[logits.len() - vocab..].to_vec();
        let token = greedy(&last);
        let stop = StopConfig::default();
        let reason = stop.check(token as u32, 1, "");
        let params = SamplingParams::default();
        println!(
            "CPU smoke: first token {token} stop={:?} (sampler defaults temp={} top_p={} max_tokens={})",
            reason, params.temperature, params.top_p, params.max_tokens
        );
        println!("CPU smoke PASS: composed forward + greedy + stop semantics wired.");
    }
}
