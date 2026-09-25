//! A8 tool-call parser + T24 cap gate — ports the committed golden corpus
//! `harness/goldens/t29_parser_goldens.json` (generator `harness/t29_goldens.py`)
//! to the Rust parser. T29 hygiene: the tags are built from parts here; the
//! literals never appear in source.

use std::collections::BTreeMap;

use mimo26_coordinator::tool_call::{
    check_tool_calls, parse_tool_calls, CapPolicy, FinishReason, ParamType, ToolSchema,
};
use mimo26_coordinator::JsonValue;

const LT: char = '\x3c';
const GT: char = '\x3e';

fn t(name: &str, close: bool) -> String {
    format!("{}{}{}{}", LT, if close { "/" } else { "" }, name, GT)
}
fn fn_open(name: &str) -> String {
    format!("{}function={}{}", LT, name, GT)
}
fn pa_open(key: &str) -> String {
    format!("{}parameter={}{}", LT, key, GT)
}
fn im_end() -> String {
    format!("{}|im_end|{}", LT, GT)
}
fn tc() -> String {
    t("tool_call", false)
}
fn tc_close() -> String {
    t("tool_call", true)
}
fn pa_close() -> String {
    t("parameter", true)
}
fn fn_close() -> String {
    t("function", true)
}
fn think() -> String {
    t("think", false)
}
fn think_close() -> String {
    t("think", true)
}

fn call(name: &str, params: &[(&str, &str)]) -> String {
    let mut s = tc() + &fn_open(name);
    for (k, v) in params {
        s.push_str(&pa_open(k));
        s.push_str(v);
        s.push_str(&pa_close());
    }
    s.push_str(&fn_close());
    s.push_str(&tc_close());
    s
}

fn call_no_close(name: &str, params: &[(&str, &str)]) -> String {
    let mut s = tc() + &fn_open(name);
    for (k, v) in params {
        s.push_str(&pa_open(k));
        s.push_str(v);
        s.push_str(&pa_close());
    }
    s.push_str(&fn_close());
    s
}

fn schema(name: &str, params: &[(&str, ParamType)]) -> ToolSchema {
    ToolSchema {
        name: name.to_string(),
        params: params.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
    }
}

fn s(v: &str) -> JsonValue {
    JsonValue::Str(v.to_string())
}
fn i(v: i64) -> JsonValue {
    JsonValue::Int(v)
}

fn args(pairs: &[(&str, JsonValue)]) -> BTreeMap<String, JsonValue> {
    pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
}

#[test]
fn golden_simple_two_params_schema_coercion() {
    let schemas = [schema("fetch", &[("url", ParamType::String), ("retries", ParamType::Integer)])];
    let text = call("fetch", &[("url", "https://x/y"), ("retries", "5")]);
    let out = parse_tool_calls(&text, &schemas);
    assert_eq!(out.calls.len(), 1);
    assert_eq!(out.calls[0].name, "fetch");
    assert_eq!(out.calls[0].arguments, args(&[("url", s("https://x/y")), ("retries", i(5))]));
    assert!(out.error.is_none());
}

#[test]
fn golden_value_contains_closing_parameter_tag() {
    let schemas = [schema("echo", &[("v", ParamType::String)])];
    let text = call("echo", &[("v", &("abc".to_string() + &pa_close() + "def"))]);
    let out = parse_tool_calls(&text, &schemas);
    assert_eq!(out.calls.len(), 1);
    assert_eq!(out.calls[0].arguments, args(&[("v", s("abc"))]));
    assert!(
        out.reports.iter().any(|r| r.contains("unparsed tail")),
        "embedded closing tag tail must be reported, not silently dropped: {:?}",
        out.reports
    );
}

#[test]
fn golden_value_contains_chat_eos_special_token() {
    let schemas = [schema("echo", &[("v", ParamType::String)])];
    let text = call("echo", &[("v", &("abc".to_string() + &im_end() + "def"))]);
    let out = parse_tool_calls(&text, &schemas);
    assert_eq!(out.calls.len(), 1);
    // EOS inside an argument is plain text to the parser.
    assert_eq!(out.calls[0].arguments, args(&[("v", s(&("abc".to_string() + &im_end() + "def")))]));
    assert!(out.reports.is_empty(), "EOS-in-value must not be reported as a loss");
}

#[test]
fn golden_missing_closing_tool_call_tag_is_lost() {
    let schemas = [schema("fetch", &[("url", ParamType::String)])];
    let text = call_no_close("fetch", &[("url", "https://x/y")]);
    let out = parse_tool_calls(&text, &schemas);
    assert!(out.calls.is_empty(), "a call without its closing tag is LOST");
    assert!(out.reports.iter().any(|r| r.contains("lost call")));
}

#[test]
fn golden_nameless_call_is_an_error() {
    let schemas: [ToolSchema; 0] = [];
    let text = tc() + &pa_open("url") + "x" + &pa_close() + &tc_close();
    let out = parse_tool_calls(&text, &schemas);
    assert!(out.calls.is_empty());
    assert_eq!(out.error.as_deref(), Some("nameless tool call"));
    assert!(out.reports.iter().any(|r| r.contains("nameless")));
}

#[test]
fn golden_non_string_renders_with_tojson() {
    let schemas = [schema("set", &[("n", ParamType::Integer), ("s", ParamType::String)])];
    let text = call("set", &[("n", "5"), ("s", "\"5\"")]);
    let out = parse_tool_calls(&text, &schemas);
    assert_eq!(out.calls.len(), 1);
    assert_eq!(out.calls[0].arguments, args(&[("n", i(5)), ("s", s("5"))]));
}

#[test]
fn golden_think_block_then_call() {
    let schemas = [schema("fetch", &[("url", ParamType::String)])];
    let text = think() + "reasoning" + &think_close() + &call("fetch", &[("url", "https://x/y")]);
    let out = parse_tool_calls(&text, &schemas);
    assert_eq!(out.reasoning, vec!["reasoning".to_string()]);
    assert_eq!(out.calls.len(), 1);
    assert_eq!(out.calls[0].name, "fetch");
}

#[test]
fn golden_two_calls_one_message_in_order() {
    let schemas = [
        schema("a", &[("k", ParamType::Integer)]),
        schema("b", &[("k", ParamType::Integer)]),
    ];
    let text = call("a", &[("k", "1")]) + &call("b", &[("k", "2")]);
    let out = parse_tool_calls(&text, &schemas);
    assert_eq!(out.calls.len(), 2);
    assert_eq!(out.calls[0].name, "a");
    assert_eq!(out.calls[0].arguments, args(&[("k", i(1))]));
    assert_eq!(out.calls[1].name, "b");
    assert_eq!(out.calls[1].arguments, args(&[("k", i(2))]));
}

#[test]
fn t24_cap_truncates_at_cap_plus_one() {
    let calls = (0..7).map(|n| mimo26_coordinator::tool_call::ParsedCall {
        name: format!("f{n}"),
        arguments: BTreeMap::new(),
    }).collect();
    let check = check_tool_calls(calls, &CapPolicy { cap: 6, parallel: true }).unwrap();
    assert_eq!(check.calls.len(), 6);
    assert_eq!(check.finish_reason, FinishReason::ToolCalls);
    assert!(check.reports.iter().any(|r| r.contains("dropped")));
}

#[test]
fn t24_parallel_false_forces_cap_one() {
    let calls = (0..3).map(|n| mimo26_coordinator::tool_call::ParsedCall {
        name: format!("f{n}"),
        arguments: BTreeMap::new(),
    }).collect();
    let check = check_tool_calls(calls, &CapPolicy { cap: 6, parallel: false }).unwrap();
    assert_eq!(check.calls.len(), 1);
    assert_eq!(check.finish_reason, FinishReason::ToolCalls);
}

#[test]
fn t24_noname_is_an_error() {
    let calls = vec![mimo26_coordinator::tool_call::ParsedCall {
        name: String::new(),
        arguments: BTreeMap::new(),
    }];
    let err = check_tool_calls(calls, &CapPolicy { cap: 6, parallel: true }).unwrap_err();
    assert!(err.contains("nameless"));
}
