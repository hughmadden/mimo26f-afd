//! X1a prompt golden: build + render the needle prompt and pin it to the spike
//! receipt — rendered sha256 `95e240d3…` and 4,027 rendered tokens. CPU-only
//! (tokenizer + chat template on the host). Run on the dev host or the coordinator (the weights dir
//! must hold `tokenizer.json` + `chat_template.jinja`).

use mimo26_coordinator::chat::{render_user_prompt, ChatOptions};
use mimo26_coordinator::load::weights_dir;
use mimo26_coordinator::needle::place;
use mimo26_coordinator::tokenizer::from_tokenizer_json;

fn main() {
    let dir = weights_dir();
    let tok_path = dir.join("tokenizer.json");
    let tok = from_tokenizer_json(tok_path.to_str().expect("utf8 path")).expect("tokenizer");

    let (text, meta) = place(|s| tok.encode(s).len(), 100, 4000, 48).expect("place");
    let raw_tokens = tok.encode(&text).len();
    let rendered = render_user_prompt(&text, &ChatOptions::default());
    let ids = tok.encode(&rendered);
    let sha = mimo26_repack::sha256::hex(&mimo26_repack::sha256::sha256(rendered.as_bytes()));

    println!(
        "X1A_PROMPT raw_tokens={raw_tokens} rendered_tokens={} rendered_sha256={sha} head_docs={} body_docs={} fact_start={} question_start={}",
        ids.len(), meta.head_docs, meta.body_docs, meta.fact_start, meta.question_start
    );
    let want_sha = "95e240d35c2ec69d2a0f1356327cb8a96a165ac8d3f4e61cdb7583c2b3f3cd38";
    if sha != want_sha {
        eprintln!("X1A_PROMPT FAIL rendered_sha256 {sha} != {want_sha}");
        std::process::exit(1);
    }
    if ids.len() != 4027 {
        eprintln!("X1A_PROMPT FAIL rendered_tokens {} != 4027", ids.len());
        std::process::exit(1);
    }
    println!("RESULT: PASS X1a prompt golden (rendered sha + 4,027 tokens)");
}
