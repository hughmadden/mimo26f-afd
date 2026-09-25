//! A2 F1 + A1 F1 spine: config, token ids, stop/length semantics (X1a gate),
//! and chat-template rendering — all CPU, all against the served artifacts.

use mimo26_coordinator::{
    check_output, render_chat, ChatMessage, ChatToolCall, Config, AttnLayerKind, StopConfig,
    StopReason, BOS_TOKEN_ID, EOS_TOKEN_IDS, GA_KV_BYTES_PER_TOKEN, IM_END, IM_START, MASK_TOKEN_ID,
    PAD_TOKEN_ID, SWA_RING_BYTES_PER_SEQ, THINK, THINK_CLOSE,
};
use mimo26_coordinator::tool_call::{parse_tool_calls, ParamType, ToolSchema};
use mimo26_coordinator::JsonValue;

#[test]
fn config_real_matches_oracle_and_kv_geometry() {
    let c = Config::real();
    assert_eq!(c.vocab_size, 152_576);
    assert_eq!(c.hidden_size, 4096);
    assert_eq!(c.intermediate_size, 16_384);
    assert_eq!(c.num_hidden_layers, 48);
    assert_eq!(c.num_attention_heads, 64);
    assert_eq!(c.num_key_value_heads, 4);
    assert_eq!(c.swa_num_key_value_heads, 8);
    assert_eq!(c.head_dim, 192);
    assert_eq!(c.v_head_dim, 128);
    assert_eq!(c.sliding_window, 128);
    assert_eq!(c.n_routed_experts, 256);
    assert_eq!(c.num_experts_per_tok, 8);
    assert_eq!(c.n_ga(), 9);
    assert_eq!(c.n_swa(), 39);
    assert_eq!(c.layer_kind(0), AttnLayerKind::Ga);
    assert_eq!(c.layer_kind(5), AttnLayerKind::Ga);
    assert_eq!(c.layer_kind(1), AttnLayerKind::Swa);
    assert_eq!(c.is_moe_layer(0), false);
    assert_eq!(c.is_moe_layer(1), true);
    assert_eq!(c.attn_dims(AttnLayerKind::Ga), (12_288, 768, 512, 8192));
    // KV geometry agrees with the kv_pool accounting (no drift).
    assert_eq!(c.ga_kv_bytes_per_token(), GA_KV_BYTES_PER_TOKEN);
    assert_eq!(c.swa_ring_bytes_per_seq(), SWA_RING_BYTES_PER_SEQ);
}

#[test]
fn config_tiny_is_internally_consistent() {
    let c = Config::tiny();
    assert_eq!(c.num_hidden_layers, 4);
    assert_eq!(c.n_ga(), 1); // tiny pattern has one GA layer
    assert_eq!(c.layer_kind(0), AttnLayerKind::Ga);
    assert_eq!(c.layer_kind(1), AttnLayerKind::Swa);
}

#[test]
fn token_ids_and_eos_set() {
    assert_eq!(BOS_TOKEN_ID, 151_643);
    assert_eq!(PAD_TOKEN_ID, 151_643);
    assert_eq!(MASK_TOKEN_ID, 151_675);
    assert_eq!(EOS_TOKEN_IDS, [151_643, 151_645, 151_672]);
    assert!(mimo26_coordinator::is_eos(151_645));
    assert!(mimo26_coordinator::is_eos(151_643));
    assert!(mimo26_coordinator::is_eos(151_672));
    assert!(!mimo26_coordinator::is_eos(5));
}

#[test]
fn stop_config_default_is_the_served_eos_and_cap() {
    let cfg = StopConfig::default();
    assert_eq!(cfg.eos, vec![151_643, 151_645, 151_672]);
    assert_eq!(cfg.max_tokens, 65_536, "T23: never the checkpoint's 2048");
}

#[test]
fn stop_checks_eos_then_length_then_string() {
    let mut cfg = StopConfig::default();
    cfg.max_tokens = 10;
    cfg.stop_strings = vec!["STOP".to_string()];
    assert_eq!(cfg.check(151_645, 1, "x"), Some(StopReason::Eos(151_645)));
    assert_eq!(cfg.check(7, 10, "x"), Some(StopReason::MaxTokens));
    assert_eq!(cfg.check(7, 9, "xx STOP"), Some(StopReason::StopString("STOP".into())));
    assert_eq!(cfg.check(7, 9, "xx"), None);
}

#[test]
fn t28_output_checks_flag_raw_mode_artifacts() {
    assert_eq!(check_output("KESTREL-41"), Vec::<String>::new());
    let bad = check_output("Answer:KESTREL-41REWARD:True");
    assert!(bad.iter().any(|v| v.contains("REWARD")), "{bad:?}");
    let bad = check_output(&format!("hi {IM_START}user"));
    assert!(bad.iter().any(|v| v.contains("role-less assistant marker")), "{bad:?}");
}

#[test]
fn chat_render_user_turn_matches_the_served_template() {
    let rendered = render_chat(&[ChatMessage::user("What is the access code?")], &[], &Default::default());
    let want = format!("{IM_START}user\nWhat is the access code?{IM_END}{IM_START}assistant\n{THINK}{THINK_CLOSE}");
    assert_eq!(rendered, want);
}

#[test]
fn chat_render_tool_call_roundtrips_the_parser() {
    let call = ChatToolCall {
        name: "fetch".to_string(),
        arguments: vec![
            ("url".to_string(), mimo26_api::json::Json::Str("https://x/y".to_string())),
            ("retries".to_string(), mimo26_api::json::Json::Num(5.0)),
        ],
    };
    let mut msg = ChatMessage::assistant("ok");
    msg.tool_calls = Some(vec![call]);
    let rendered = render_chat(&[msg], &[], &Default::default());

    let schemas = [ToolSchema {
        name: "fetch".to_string(),
        params: [("url".to_string(), ParamType::String), ("retries".to_string(), ParamType::Integer)]
            .into_iter()
            .collect(),
    }];
    let out = parse_tool_calls(&rendered, &schemas);
    assert_eq!(out.calls.len(), 1, "renderer output must parse back: {rendered:?}");
    assert_eq!(out.calls[0].name, "fetch");
    assert_eq!(out.calls[0].arguments.get("url"), Some(&JsonValue::Str("https://x/y".to_string())));
    assert_eq!(out.calls[0].arguments.get("retries"), Some(&JsonValue::Int(5)));
}
