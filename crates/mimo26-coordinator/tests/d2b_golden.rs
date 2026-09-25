//! D2b golden: the tools block and history tool calls must render byte-identical
//! to the checkpoint `chat_template.jinja` applied the way transformers/vLLM
//! apply it (the `tojson` filter = `json.dumps(..., ensure_ascii=False,
//! sort_keys=False)`, default ", " / ": " separators, client key order,
//! description kept). The expected bytes come from an actual Jinja2 render
//! (harness/d2b_golden.py), committed HEX-ENCODED so no MiMo tool/chat tag
//! literal enters the repo (T29).

use mimo26_api::json::Json;
use mimo26_coordinator::{render_chat, ChatMessage, ChatOptions, ChatToolCall, Role};

const GOLDEN: &str = include_str!("data/d2b_golden.txt");

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn expected(name: &str) -> Vec<u8> {
    for line in GOLDEN.lines() {
        let (k, hx) = line.split_once(' ').expect("golden line");
        if k == name {
            let mut out = Vec::with_capacity(hx.len() / 2);
            for i in (0..hx.len()).step_by(2) {
                out.push(u8::from_str_radix(&hx[i..i + 2], 16).expect("hex"));
            }
            return out;
        }
    }
    panic!("golden case {name} not found");
}

fn assert_case(name: &str, messages: &[ChatMessage], tools: &[Json]) {
    let rendered = render_chat(messages, tools, &ChatOptions::default());
    let want = expected(name);
    assert_eq!(
        hex_encode(rendered.as_bytes()),
        hex_encode(&want),
        "D2b golden case {name} mismatch:\n--- got ---\n{rendered}\n--- want ---\n{}",
        String::from_utf8_lossy(&want)
    );
}

/// The four stress-corrupt tools (the T24 fixture), with descriptions, as the
/// client's whole tool objects in the client's key order.
fn tools() -> Vec<Json> {
    [
        r#"{"type": "function", "function": {"name": "grep", "description": "Search file contents for a regex pattern.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern", "path"]}}}"#,
        r#"{"type": "function", "function": {"name": "read", "description": "Read a file and return its contents with line numbers.", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}}, "required": ["file_path"]}}}"#,
        r#"{"type": "function", "function": {"name": "write", "description": "Write content to a file, replacing it entirely.", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}, "required": ["file_path", "content"]}}}"#,
        r#"{"type": "function", "function": {"name": "todo_write", "description": "Replace the todo list.", "parameters": {"type": "object", "properties": {"todos": {"type": "array", "items": {"type": "object", "properties": {"content": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}}, "required": ["content", "status"]}}}, "required": ["todos"]}}}"#,
    ]
    .iter()
    .map(|s| mimo26_api::json::parse(s).expect("tool json"))
    .collect()
}

/// Case (a): the T24 request — four tools with descriptions, a prior `read`
/// call and its tool result (history arguments in the client's JSON order).
#[test]
fn d2b_golden_t24() {
    let mut read_call = ChatMessage::assistant("");
    read_call.tool_calls = Some(vec![ChatToolCall {
        name: "read".into(),
        arguments: vec![("file_path".into(), Json::Str("src/theme/palette.js".into()))],
    }]);
    let tool_result = ChatMessage {
        role: Role::Tool,
        content: "216:   amber: 0xf2a900,\n217:   honey: 0xe8b923,\n218:   gold: 0xf6c council,\n219:   sand: 0xd9c19c,\n220:   ochre: 0xcc7722,\n".into(),
        reasoning: None,
        tool_calls: None,
    };
    let messages = vec![
        ChatMessage::system("You are a coding assistant working in the repository /work/app. Use the provided tools to make changes."),
        ChatMessage::user("There is a typo on line 218 of src/theme/palette.js, the gold entry is broken. Please fix it."),
        read_call,
        tool_result,
    ];
    assert_case("t24", &messages, &tools());
}

/// Case (b): a client system message plus tools — the tools system turn must
/// come BEFORE the client's own system turn.
#[test]
fn d2b_golden_system_tools() {
    let messages = vec![ChatMessage::system("You are a helpful assistant.")];
    assert_case("system_tools", &messages, &tools());
}

/// Case (c): a history call whose arguments carry non-string values (int, bool,
/// nested object) — rendered via `tojson` in the client's JSON order.
#[test]
fn d2b_golden_nonstring_arg() {
    let mut call = ChatMessage::assistant("");
    call.tool_calls = Some(vec![ChatToolCall {
        name: "grep".into(),
        arguments: vec![
            ("pattern".into(), Json::Str("foo".into())),
            ("max_count".into(), Json::Num(3.0)),
            ("case_sensitive".into(), Json::Bool(false)),
            ("options".into(), Json::Object(vec![
                ("multiline".into(), Json::Bool(true)),
                ("invert".into(), Json::Bool(false)),
            ])),
        ],
    }]);
    assert_case("nonstring_arg", &[call], &[]);
}

/// Case (d): no tools — unchanged from the no-tools render.
#[test]
fn d2b_golden_no_tools() {
    let messages = vec![ChatMessage::user("Hello")];
    assert_case("no_tools", &messages, &[]);
}
