//! X1a gate (direct, not the API): load weights → ServingModel → WireClient →
//! render/tokenize the needle prompt → prefill 4,027 → greedy decode to
//! `KESTREL-41`+EOS. Run on the coordinator (sm_120):
//! `cargo run --features cuda --example x1a_run`.

use mimo26_coordinator::chat::{render_user_prompt, ChatOptions};
use mimo26_coordinator::config::Config;
use mimo26_coordinator::dforward::{DeviceForward, DeviceKv};
use mimo26_coordinator::load::{load_coordinator_weights, weights_dir};
use mimo26_coordinator::needle::place;
use mimo26_coordinator::serving::{Fp8KvCache, ServingModel};
use mimo26_coordinator::tokenizer::from_tokenizer_json;
use mimo26_coordinator::wire::WireClient;
use mimo26_coordinator::greedy;

fn main() {
    let cfg = Config::real();
    let dir = weights_dir();

    eprintln!("[x1a] loading weights from {}", dir.display());
    let w = load_coordinator_weights(&dir, &cfg);
    let device = std::env::var_os("MIMO26_HOST_FORWARD").is_none();
    let max_chunk: usize = std::env::var("MIMO26_PREFILL_CHUNK").ok().and_then(|v| v.parse().ok()).unwrap_or(4096);
    let (model, mut dev) = if device {
        eprintln!("[x1a] DEVICE forward (perf reset R1), chunk {max_chunk}");
        let fwd = DeviceForward::new(cfg.clone(), w, max_chunk).expect("DeviceForward::new");
        let kv = fwd.new_kv(131072).expect("DeviceKv");
        (None, Some((fwd, kv)))
    } else {
        eprintln!("[x1a] HOST forward");
        (Some(ServingModel::new(cfg.clone(), w).expect("ServingModel::new")), None)
    };

    let addrs: Vec<String> = std::env::var("MIMO26_SPARK_ADDRS")
        .map(|s| s.split(',').map(str::to_string).collect())
        .expect("MIMO26_SPARK_ADDRS: the four Sparks' RDMA-fabric addresses (host:port, rank order); no LAN default");
    let mut wire = WireClient::connect(&addrs).expect("WireClient::connect");

    if let Some((f, _)) = dev.as_ref() {
        f.register_plane_buffers(&wire.plane_buffers()).expect("register plane rings");
        f.register_plane_buffers(&wire.send_buffers()).expect("register request body");
    }
    let tok = from_tokenizer_json(dir.join("tokenizer.json").to_str().expect("utf8"))
        .expect("tokenizer");
    let target: usize = std::env::var("MIMO26_PROMPT_TARGET")
        .map(|s| s.parse().expect("MIMO26_PROMPT_TARGET"))
        .unwrap_or(4000);
    let (text, _) = place(|s| tok.encode(s).len(), 100, target, 48).expect("place");
    let rendered = render_user_prompt(&text, &ChatOptions::default());
    let sha = mimo26_repack::sha256::hex(&mimo26_repack::sha256::sha256(rendered.as_bytes()));
    eprintln!("[x1a] rendered sha256 {sha}");
    let mut prompt: Vec<usize> = tok.encode(&rendered).iter().map(|&x| x as usize).collect();
    if let Ok(n) = std::env::var("MIMO26_PROMPT_TOKENS") {
        let n: usize = n.parse().expect("MIMO26_PROMPT_TOKENS");
        prompt.truncate(n);
    }
    eprintln!("[x1a] prompt tokens {}", prompt.len());

    let mut caches: Vec<Fp8KvCache> =
        (0..cfg.num_hidden_layers).map(|l| Fp8KvCache::for_layer(&cfg, l)).collect();
    let vocab = cfg.vocab_size;
    let eos: Vec<usize> = cfg.eos_token_ids.iter().map(|&x| x as usize).collect();
    let mut gen_ids: Vec<usize> = Vec::new();

    let t_prefill = std::time::Instant::now();
    let logits: Vec<f32> = match (&model, &mut dev) {
        (Some(m), _) => m.forward(&prompt, &mut caches, Some(&mut wire)).0,
        (None, Some((f, kv))) => f.forward(&prompt, kv, &mut wire).expect("device forward"),
        _ => unreachable!(),
    };
    let prefill_s = t_prefill.elapsed().as_secs_f64();
    eprintln!("[x1a] prefill {} tokens in {:.3} s = {:.1} tok/s", prompt.len(), prefill_s, prompt.len() as f64 / prefill_s);
    // Production chunk check (builder step 3): dump the last-row logits so the
    // 4096/2048/1024 chunk runs can be diffed.
    if let Ok(path) = std::env::var("MIMO26_DUMP_LOGITS") {
        let last = &logits[logits.len() - vocab..];
        let text = last.iter().map(|f| f.to_string()).collect::<Vec<_>>().join("\n");
        std::fs::write(&path, text).expect("dump logits");
        eprintln!("[x1a] dumped last-row logits ({}) to {path}", last.len());
    }
    let mut next = greedy(&logits[logits.len() - vocab..]);
    gen_ids.push(next);
    eprintln!("[x1a] prefill done, first token {next}");

    let t_decode = std::time::Instant::now();
    while !eos.contains(&next) && gen_ids.len() < 32 {
        let logits: Vec<f32> = match (&model, &mut dev) {
            (Some(m), _) => m.forward(&[next], &mut caches, Some(&mut wire)).0,
            (None, Some((f, kv))) => f.forward(&[next], kv, &mut wire).expect("device decode"),
            _ => unreachable!(),
        };
        next = greedy(&logits[logits.len() - vocab..]);
        gen_ids.push(next);
    }
    let steps = gen_ids.len().saturating_sub(1).max(1);
    let dec_s = t_decode.elapsed().as_secs_f64();
    eprintln!("[x1a] decode {} steps in {:.3} s = {:.1} ms/token", steps, dec_s, dec_s * 1e3 / steps as f64);

    // Batched serving check (perf reset W4): MIMO26_X1A_BATCH=N prefills N
    // needle prompts of different lengths into their own caches, decodes them
    // together with decode_batch, and requires each row to equal that prompt's
    // single-request greedy decode.
    if let (Some((f, _)), Ok(nb)) = (dev.as_mut(), std::env::var("MIMO26_X1A_BATCH")) {
        let nb: usize = nb.parse().expect("MIMO26_X1A_BATCH");
        let prompts: Vec<Vec<usize>> = (0..nb)
            .map(|i| {
                let (text, _) = place(|s| tok.encode(s).len(), 100, target.saturating_sub(500 * i).max(600), 48)
                    .expect("place");
                tok.encode(&render_user_prompt(&text, &ChatOptions::default())).iter().map(|&x| x as usize).collect()
            })
            .collect();
        let mut kv_ref = f.new_kv(8192).expect("DeviceKv");
        let mut refs: Vec<Vec<usize>> = Vec::new();
        for p in &prompts {
            kv_ref.reset();
            let l = f.forward(p, &mut kv_ref, &mut wire).expect("ref prefill");
            let mut g = vec![greedy(&l[l.len() - vocab..])];
            while !eos.contains(g.last().unwrap()) && g.len() < 32 {
                let l = f.forward(&[*g.last().unwrap()], &mut kv_ref, &mut wire).expect("ref decode");
                g.push(greedy(&l[l.len() - vocab..]));
            }
            refs.push(g);
        }
        let mut kvs: Vec<DeviceKv> = Vec::new();
        let mut gens: Vec<Vec<usize>> = Vec::new();
        for p in &prompts {
            let mut kv = f.new_kv(8192).expect("DeviceKv");
            let l = f.forward(p, &mut kv, &mut wire).expect("batch prefill");
            gens.push(vec![greedy(&l[l.len() - vocab..])]);
            kvs.push(kv);
        }
        let t = std::time::Instant::now();
        let (mut steps, mut toks) = (0usize, 0usize);
        loop {
            let live: Vec<usize> = (0..nb)
                .filter(|&i| !eos.contains(gens[i].last().unwrap()) && gens[i].len() < 32)
                .collect();
            if live.is_empty() {
                break;
            }
            let ids: Vec<usize> = live.iter().map(|&i| *gens[i].last().unwrap()).collect();
            let mut rows: Vec<&mut DeviceKv> =
                kvs.iter_mut().enumerate().filter(|(i, _)| live.contains(i)).map(|(_, k)| k).collect();
            let l = f.decode_batch(&ids, &mut rows, &mut wire).expect("decode_batch");
            for (j, &i) in live.iter().enumerate() {
                gens[i].push(greedy(&l[j * vocab..(j + 1) * vocab]));
            }
            steps += 1;
            toks += live.len();
        }
        let s = t.elapsed().as_secs_f64();
        eprintln!("[x1a] batch {nb}: {steps} decode steps, {toks} tokens in {s:.3} s = {:.1} ms/step, {:.1} tok/s aggregate",
            s * 1e3 / steps.max(1) as f64, toks as f64 / s);
        let mut ok = true;
        for i in 0..nb {
            let m = gens[i] == refs[i];
            ok &= m;
            eprintln!("[x1a] batch row {i}: prompt {} tokens, gen {:?} {}", prompts[i].len(), gens[i],
                if m { "== single" } else { "!= single" });
            if !m {
                eprintln!("[x1a]   single {:?}", refs[i]);
            }
        }
        println!("X1A batch {nb}: {}", if ok { "PASS (every row equals its single-request decode)" } else { "FAIL" });
    }

    // DFlash check (perf reset S1): MIMO26_X1A_SPEC=1 loads the drafter and, for a
    // few of the bench's prompts, compares speculative greedy decoding (one
    // request, then all together) with the plain greedy decode token by token,
    // and reports acceptance and speed.
    if let (Some((f, _)), Ok(_)) = (dev.as_mut(), std::env::var("MIMO26_X1A_SPEC")) {
        f.load_dflash(&dir.join("dflash")).expect("load dflash");
        let cats: [(&str, &str, usize); 4] = [
            ("coding", "Write a Python function merge_intervals(intervals) that merges overlapping intervals and returns them sorted. Include a one-line docstring and two example calls.", 200),
            ("json", "Return only a JSON object describing a fictional bookstore with keys: name (string), city (string), founded (integer year), genres (array of 5 strings), staff (array of 3 objects, each with name and role). No prose.", 200),
            ("prose", "Explain in about 120 words how a refrigerator keeps food cold, for a curious 12-year-old.", 200),
            ("count", "Count from 1 to 80, comma separated, nothing else.", 256),
        ];
        let prompts: Vec<Vec<usize>> = cats
            .iter()
            .map(|(_, p, _)| tok.encode(&render_user_prompt(p, &ChatOptions::default())).iter().map(|&x| x as usize).collect())
            .collect();
        let mut kv = f.new_kv(8192).expect("DeviceKv");
        let mut refs: Vec<Vec<usize>> = Vec::new();
        let mut ok = true;
        for (ci, &(name, _, max)) in cats.iter().enumerate() {
            kv.reset();
            let l = f.forward(&prompts[ci], &mut kv, &mut wire).expect("ref prefill");
            let mut r = vec![greedy(&l[l.len() - vocab..])];
            let t = std::time::Instant::now();
            while !eos.contains(r.last().unwrap()) && r.len() < max {
                let l = f.forward(&[*r.last().unwrap()], &mut kv, &mut wire).expect("ref decode");
                r.push(greedy(&l[l.len() - vocab..]));
            }
            let ref_s = t.elapsed().as_secs_f64();
            kv.reset();
            let l = f.forward(&prompts[ci], &mut kv, &mut wire).expect("spec prefill");
            let mut g = vec![greedy(&l[l.len() - vocab..])];
            let t = std::time::Instant::now();
            let (mut steps, mut accepted) = (0usize, 0usize);
            while !eos.contains(g.last().unwrap()) && g.len() < max {
                let k = (max - g.len() - 1).min(7);
                let out = f.spec_step(&mut [&mut kv], &[*g.last().unwrap()], &[k], &[None], &mut wire).expect("spec_step");
                steps += 1;
                accepted += out[0].len() - 1;
                for &x in &out[0] {
                    g.push(x);
                    if eos.contains(&x) || g.len() >= max {
                        break;
                    }
                }
            }
            let spec_s = t.elapsed().as_secs_f64();
            let same = g == r;
            ok &= same;
            let div = g.iter().zip(&r).position(|(a, b)| a != b);
            eprintln!("[x1a spec] {name:7} prompt {} tok: plain {} tok {:.1} tok/s | spec {} tok in {steps} steps, \
                accepted {:.2}/step, {:.1} tok/s ({:.1} ms/step) | {}",
                prompts[ci].len(), r.len(), (r.len() - 1) as f64 / ref_s, g.len(), accepted as f64 / steps.max(1) as f64,
                (g.len() - 1) as f64 / spec_s, spec_s * 1e3 / steps.max(1) as f64,
                if same { "identical".to_string() } else { format!("DIVERGES at token {div:?}") });
            if !same {
                eprintln!("[x1a spec]   plain {:?}", &r[..r.len().min(48)]);
                eprintln!("[x1a spec]   spec  {:?}", &g[..g.len().min(48)]);
            }
            refs.push(r);
        }
        // All four requests together.
        let n = cats.len();
        let mut kvs: Vec<DeviceKv> = (0..n).map(|_| f.new_kv(8192).expect("DeviceKv")).collect();
        let mut gens: Vec<Vec<usize>> = Vec::new();
        for i in 0..n {
            let l = f.forward(&prompts[i], &mut kvs[i], &mut wire).expect("batch prefill");
            gens.push(vec![greedy(&l[l.len() - vocab..])]);
        }
        let t = std::time::Instant::now();
        let (mut steps, mut toks) = (0usize, 0usize);
        loop {
            let live: Vec<usize> =
                (0..n).filter(|&i| !eos.contains(gens[i].last().unwrap()) && gens[i].len() < cats[i].2).collect();
            if live.is_empty() {
                break;
            }
            let lasts: Vec<usize> = live.iter().map(|&i| *gens[i].last().unwrap()).collect();
            let ks: Vec<usize> = live.iter().map(|&i| (cats[i].2 - gens[i].len() - 1).min(7)).collect();
            let mut rows: Vec<&mut DeviceKv> =
                kvs.iter_mut().enumerate().filter(|(i, _)| live.contains(i)).map(|(_, k)| k).collect();
            let out = f.spec_step(&mut rows, &lasts, &ks, &vec![None; lasts.len()], &mut wire).expect("batch spec_step");
            for (j, &i) in live.iter().enumerate() {
                for &x in &out[j] {
                    gens[i].push(x);
                    toks += 1;
                    if eos.contains(&x) || gens[i].len() >= cats[i].2 {
                        break;
                    }
                }
            }
            steps += 1;
        }
        let s = t.elapsed().as_secs_f64();
        let mut batch_ok = true;
        for i in 0..n {
            let same = gens[i] == refs[i];
            batch_ok &= same;
            if !same {
                let div = gens[i].iter().zip(&refs[i]).position(|(a, b)| a != b);
                eprintln!("[x1a spec] batch row {i} ({}) diverges from plain at token {div:?}", cats[i].0);
            }
        }
        eprintln!("[x1a spec] batch {n}: {steps} steps, {toks} tokens in {s:.3} s = {:.1} tok/s aggregate, {:.1} ms/step",
            toks as f64 / s, s * 1e3 / steps.max(1) as f64);
        println!("X1A spec: single {} / batch {}", if ok { "PASS (identical to plain greedy)" } else { "DIVERGES" },
            if batch_ok { "PASS" } else { "DIVERGES" });
    }

    let expected: Vec<usize> = vec![3390, 784, 49051, 12, 19, 16, 151645];
    println!("X1A gen_ids={gen_ids:?}");
    if gen_ids == expected {
        println!("RESULT: PASS X1a chat needle retrieved verbatim: KESTREL-41 (EOS honoured)");
    } else {
        let first_div = gen_ids.iter().zip(expected.iter()).position(|(&g, &e)| g != e);
        let at = first_div.unwrap_or(gen_ids.len().min(expected.len()));
        eprintln!("RESULT: FAIL X1a first divergence at token {at}: got {:?} want {:?}",
            gen_ids.get(at), expected.get(at));
        std::process::exit(1);
    }
}
