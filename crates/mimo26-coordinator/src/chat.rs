//! Chat-template rendering (A2 F1; X1a needle gate). Rust twin of the served
//! `chat_template.jinja` (sha `853650be…`) for the v1 message surface: string
//! content, tool calls, and the assistant generation prompt with thinking off.
//!
//! The render is byte-exact for the cases v1 serves (verified against the
//! actual template output; see the D2b golden). T29 hygiene: the special tags
//! come from [`crate::token`] and the tool-call tags below are built from parts
//! — no literal tags in this source.

use mimo26_api::json::Json;

use crate::phase::Role;
use crate::token::{IM_END, IM_START, THINK, THINK_CLOSE};

// Tool-call tags (mirror the parser in crate::tool_call; built from parts).
const LT: char = '\x3c';
const GT: char = '\x3e';
fn tool_call_open() -> String {
    format!("{LT}tool_call{GT}")
}
fn tool_call_close() -> String {
    format!("{LT}/tool_call{GT}")
}
fn function_open(name: &str) -> String {
    format!("{LT}function={name}{GT}")
}
fn function_close() -> String {
    format!("{LT}/function{GT}")
}
fn parameter_open(key: &str) -> String {
    format!("{LT}parameter={key}{GT}")
}
fn parameter_close() -> String {
    format!("{LT}/parameter{GT}")
}
fn tools_open() -> String {
    format!("{LT}tools{GT}")
}
fn tools_close() -> String {
    format!("{LT}/tools{GT}")
}

/// One assistant tool call to render (the renderer's inverse of the parser).
#[derive(Debug, Clone, PartialEq)]
pub struct ChatToolCall {
    pub name: String,
    /// Arguments in the CLIENT's JSON order (D2b item 3: vLLM `json.loads` the
    /// arguments string before templating, so the template iterates the client's
    /// insertion order — never the tool schema's declared order).
    pub arguments: Vec<(String, Json)>,
}

/// One message in the conversation.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
    /// Assistant reasoning (rendered inside the think block; empty = none).
    pub reasoning: Option<String>,
    pub tool_calls: Option<Vec<ChatToolCall>>,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        ChatMessage { role: Role::User, content: content.into(), reasoning: None, tool_calls: None }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        ChatMessage {
            role: Role::Assistant,
            content: content.into(),
            reasoning: None,
            tool_calls: None,
        }
    }
    pub fn system(content: impl Into<String>) -> Self {
        ChatMessage { role: Role::System, content: content.into(), reasoning: None, tool_calls: None }
    }
}

/// Render options.
#[derive(Debug, Clone, Copy)]
pub struct ChatOptions {
    pub add_generation_prompt: bool,
    /// The served default is `false` (thinking off → empty think block).
    pub enable_thinking: bool,
}

impl Default for ChatOptions {
    fn default() -> Self {
        ChatOptions { add_generation_prompt: true, enable_thinking: false }
    }
}

fn role_str(role: Role) -> &'static str {
    match role {
        Role::User => "user",
        Role::System => "system",
        Role::Assistant => "assistant",
        Role::Tool => "tool",
    }
}

/// `tojson(ensure_ascii=False)` from the served template (transformers' filter):
/// `json.dumps(v, ensure_ascii=False, sort_keys=False)` — the default ", " and
/// ": " separators, insertion order, non-ASCII unescaped. Mirrors
/// `mimo26_api::json::serialize` but with the template's separator spacing.
fn tojson(v: &Json) -> String {
    let mut out = String::new();
    write_tojson(v, &mut out);
    out
}

fn write_tojson(v: &Json, out: &mut String) {
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Num(n) => {
            if n.fract() == 0.0 && n.abs() < 9.0e15 {
                out.push_str(&(*n as i64).to_string());
            } else {
                out.push_str(&format!("{n}"));
            }
        }
        Json::Str(s) => write_tojson_string(s, out),
        Json::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_tojson(item, out);
            }
            out.push(']');
        }
        Json::Object(pairs) => {
            out.push('{');
            for (i, (k, val)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_tojson_string(k, out);
                out.push_str(": ");
                write_tojson(val, out);
            }
            out.push('}');
        }
    }
}

fn write_tojson_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `render_value` from the template: strings as-is, non-strings via `tojson`.
fn render_value(v: &Json) -> String {
    match v {
        Json::Str(s) => s.clone(),
        other => tojson(other),
    }
}

/// `render_tool_calls` from the template.
fn render_tool_calls(calls: &[ChatToolCall]) -> String {
    let mut out = String::new();
    for call in calls {
        out.push_str(&tool_call_open());
        out.push_str(&function_open(&call.name));
        for (key, value) in &call.arguments {
            out.push_str(&parameter_open(key));
            out.push_str(&render_value(value));
            out.push_str(&parameter_close());
        }
        out.push_str(&function_close());
        out.push_str(&tool_call_close());
    }
    out
}

/// `render_tools` from the template: the wrapper text and each tool's JSON
/// serialised with the `tojson` filter (client's key order + description).
fn render_tools(tools: &[Json]) -> String {
    let mut out = String::from("You are provided with the following tools:\n\n");
    out.push_str(&tools_open());
    for t in tools {
        out.push('\n');
        out.push_str(&tojson(t));
    }
    out.push('\n');
    out.push_str(&tools_close());
    out
}

/// Render a message list exactly as the served template does (string content +
/// tool calls; multimodal content is a v1 non-goal — media returns 400). When
/// `tools` is non-empty, the template's system-side tools block is prepended
/// (its own system turn, before every message including a client system one).
pub fn render_chat(messages: &[ChatMessage], tools: &[Json], opts: &ChatOptions) -> String {
    let mut out = String::new();
    if !tools.is_empty() {
        out.push_str(IM_START);
        out.push_str("system\n");
        out.push_str(&render_tools(tools));
        out.push_str(IM_END);
    }
    for m in messages {
        match m.role {
            Role::Assistant => {
                out.push_str(IM_START);
                out.push_str("assistant\n");
                out.push_str(THINK);
                out.push_str(m.reasoning.as_deref().unwrap_or(""));
                out.push_str(THINK_CLOSE);
                out.push_str(&m.content);
                if let Some(calls) = &m.tool_calls {
                    if !calls.is_empty() {
                        out.push_str(&render_tool_calls(calls));
                    }
                }
                out.push_str(IM_END);
            }
            other => {
                out.push_str(IM_START);
                out.push_str(role_str(other));
                out.push('\n');
                out.push_str(&m.content);
                out.push_str(IM_END);
            }
        }
    }
    if opts.add_generation_prompt {
        out.push_str(IM_START);
        out.push_str("assistant\n");
        if !opts.enable_thinking {
            out.push_str(THINK);
            out.push_str(THINK_CLOSE);
        }
    }
    out
}

/// Convenience: the token ids the render would produce — an injectable
/// tokenizer is the real encoder; this is the pure-text surface.
pub fn render_user_prompt(content: &str, opts: &ChatOptions) -> String {
    render_chat(&[ChatMessage::user(content)], &[], opts)
}
