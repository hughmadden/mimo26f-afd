//! Request/response types for the chat completions API, and the JSON mapping.
//!
//! The request is decoded from the generic [`crate::json::Json`] value with
//! fail-loud validation (media content parts and a constrained `tool_choice`
//! are rejected before any generation). The response is assembled as `Json`
//! directly so the wire shape is explicit.

use crate::json::{Json, serialize};

/// An API error that maps to an HTTP status and an OpenAI-style error body.
#[derive(Debug, Clone)]
pub struct ApiError {
    pub status: u16,
    pub message: String,
    pub code: String,
}

impl ApiError {
    pub fn bad_request(msg: impl Into<String>) -> Self {
        ApiError { status: 400, message: msg.into(), code: "invalid_request_error".into() }
    }
    pub fn not_found(msg: impl Into<String>) -> Self {
        ApiError { status: 404, message: msg.into(), code: "not_found_error".into() }
    }
    pub fn internal(msg: impl Into<String>) -> Self {
        ApiError { status: 500, message: msg.into(), code: "server_error".into() }
    }
    pub fn body(&self) -> Json {
        Json::Object(vec![
            ("error".into(), Json::Object(vec![
                ("message".into(), Json::Str(self.message.clone())),
                ("type".into(), Json::Str(self.code.clone())),
                ("code".into(), Json::Str(self.code.clone())),
            ])),
        ])
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.status, self.message)
    }
}

/// One tool's function descriptor.
#[derive(Debug, Clone)]
pub struct ToolFunction {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<Json>,
}

#[derive(Debug, Clone)]
pub struct Tool {
    pub r#type: String,
    pub function: ToolFunction,
    /// The client's whole tool object, verbatim, in the client's key order
    /// (`type` + `function{name, description, parameters}` and any extra keys the
    /// client sent). The renderer serialises this with a transformers-style
    /// `tojson` (`json.dumps(..., ensure_ascii=False, sort_keys=False)`, the
    /// default ", " / ": " separators), so the client's key order and the
    /// description must survive the API parse (D2b item 1).
    pub raw: Json,
}

impl Tool {
    /// The JSON-schema type of a top-level property, if declared.
    pub fn property_schema(&self, key: &str) -> Option<String> {
        let params = self.function.parameters.as_ref()?;
        let props = params.get("properties")?.as_object()?;
        let entry = props.iter().find(|(k, _)| k == key)?;
        entry.1.get("type")?.as_str().map(|s| s.to_string())
    }
}

/// An OpenAI-format tool call (the wire shape).
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub r#type: String,
    pub name: String,
    pub arguments: String,
}

impl ToolCall {
    pub fn to_json(&self) -> Json {
        Json::Object(vec![
            ("id".into(), Json::Str(self.id.clone())),
            ("type".into(), Json::Str(self.r#type.clone())),
            ("function".into(), Json::Object(vec![
                ("name".into(), Json::Str(self.name.clone())),
                ("arguments".into(), Json::Str(self.arguments.clone())),
            ])),
        ])
    }
}

/// A chat message (request or assistant-history form).
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    /// Flattened text content (after media validation).
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
}

/// Sampling/control parameters decoded from the request.
#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub tools: Vec<Tool>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<u64>,
    pub stop: Vec<String>,
    pub stream: bool,
    pub include_usage: bool,
    pub enable_thinking: bool,
    pub parallel_tool_calls: bool,
    /// Decoded images in prompt order (perf reset V2); each stands in a message as an image marker.
    pub images: Vec<std::sync::Arc<crate::engine::ImageInput>>,
}

/// The model id this server serves (A8; the vendored tools send it).
pub const MODEL_ID: &str = "mimo-v2.6-flash";

const IMAGE_TYPES: &[&str] = &["image_url", "input_image", "image"];
const MEDIA_TYPES: &[&str] = &["input_audio", "audio", "input_video", "video"];

/// Images kept per request (perf reset V2; ADVISOR-I3 §10.2 item 6): the newest ones, the older
/// ones replaced by a note, so a long conversation with many images keeps working.
pub const MAX_IMAGES: usize = 16;

/// Image state while a request's messages are parsed.
struct Images {
    seen: usize,
    keep_from: usize,
    out: Vec<std::sync::Arc<crate::engine::ImageInput>>,
}

/// Client text without the image-marker noncharacters (only the API places markers).
fn clean(s: &str) -> String {
    s.chars().filter(|&c| c != crate::engine::IMAGE_OPEN && c != crate::engine::IMAGE_CLOSE).collect()
}

/// The URL of an image part: `image_url: {url}` (Chat Completions), `image_url: "…"` (Responses
/// `input_image`) or `image: "…"`.
fn image_url(part: &Json) -> Option<&str> {
    let v = part.get("image_url").or_else(|| part.get("image"))?;
    v.as_str().or_else(|| v.get("url").and_then(|u| u.as_str()))
}

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3))
}

fn decode_image(url: &str) -> Result<crate::engine::ImageInput, ApiError> {
    if !url.starts_with("data:") {
        return Err(ApiError::bad_request(
            "image URLs are not fetched; send the image inline as a data URL (data:image/png;base64,...)",
        ));
    }
    let img = mimo26_image::decode_data_url(url).map_err(|e| ApiError::bad_request(format!("image: {e}")))?;
    let tokens = mimo26_image::merged_tokens(img.height, img.width).map_err(|e| ApiError::bad_request(format!("image: {e}")))?;
    Ok(crate::engine::ImageInput { hash: fnv1a(url.as_bytes()), tokens, width: img.width, height: img.height, rgb: img.data })
}

fn flatten_content(content: &Json, images: &mut Images) -> Result<String, ApiError> {
    match content {
        Json::Str(s) => Ok(clean(s)),
        Json::Array(parts) => {
            let mut text = String::new();
            for part in parts {
                let ty = part.get("type").and_then(|t| t.as_str()).unwrap_or("");
                if MEDIA_TYPES.contains(&ty) {
                    return Err(ApiError::bad_request(format!(
                        "content part type \"{ty}\" is not supported (audio and video are rejected)"
                    )));
                }
                if IMAGE_TYPES.contains(&ty) {
                    let i = images.seen;
                    images.seen += 1;
                    if i < images.keep_from {
                        text.push_str(&format!("[image omitted: only the {MAX_IMAGES} most recent images are sent]"));
                        continue;
                    }
                    let url = image_url(part).ok_or_else(|| ApiError::bad_request("an image part has no URL"))?;
                    let img = decode_image(url)?;
                    text.push_str(&crate::engine::image_marker(&img));
                    images.out.push(std::sync::Arc::new(img));
                    continue;
                }
                if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                    text.push_str(&clean(t));
                }
            }
            Ok(text)
        }
        Json::Null => Ok(String::new()),
        _ => Err(ApiError::bad_request("message content must be a string or an array of parts")),
    }
}

/// Image parts in a request's messages (counted before parsing, to keep the newest).
fn count_images(messages: &[Json]) -> usize {
    messages.iter().filter_map(|m| m.get("content").and_then(|c| c.as_array())).flatten()
        .filter(|p| IMAGE_TYPES.contains(&p.get("type").and_then(|t| t.as_str()).unwrap_or(""))).count()
}

fn parse_message(v: &Json, images: &mut Images) -> Result<ChatMessage, ApiError> {
    let role = v.get("role").and_then(|r| r.as_str()).unwrap_or("").to_string();
    let content = match v.get("content") {
        Some(c) => flatten_content(c, images)?,
        None => String::new(),
    };
    let mut tool_calls = Vec::new();
    if let Some(tcs) = v.get("tool_calls").and_then(|t| t.as_array()) {
        for tc in tcs {
            let fun = tc.get("function").and_then(|f| f.as_object()).ok_or_else(|| {
                ApiError::bad_request("a message tool_call is missing its function object")
            })?;
            let name = fun.iter().find(|(k, _)| k == "name").and_then(|(_, n)| n.as_str());
            let name = match name {
                Some(n) if !n.is_empty() => n.to_string(),
                _ => return Err(ApiError::bad_request(
                    "messages[].tool_calls[] is missing a function name".to_string(),
                )),
            };
            let arguments = fun.iter().find(|(k, _)| k == "arguments")
                .and_then(|(_, a)| a.as_str()).unwrap_or("").to_string();
            let id = tc.get("id").and_then(|i| i.as_str()).unwrap_or("").to_string();
            tool_calls.push(ToolCall { id, r#type: "function".into(), name, arguments });
        }
    }
    Ok(ChatMessage { role, content, tool_calls })
}

fn parse_tool(v: &Json) -> Result<Tool, ApiError> {
    let r#type = v.get("type").and_then(|t| t.as_str()).unwrap_or("function").to_string();
    let fun = v.get("function").and_then(|f| f.as_object()).ok_or_else(|| {
        ApiError::bad_request("a tool is missing its function object")
    })?;
    let name = fun.iter().find(|(k, _)| k == "name").and_then(|(_, n)| n.as_str())
        .unwrap_or("").to_string();
    let description = fun.iter().find(|(k, _)| k == "description")
        .and_then(|(_, d)| d.as_str()).map(|s| s.to_string());
    let parameters = fun.iter().find(|(k, _)| k == "parameters").map(|(_, p)| p.clone());
    // Keep the raw tool object (client's key order) for the renderer's tojson.
    Ok(Tool {
        r#type,
        function: ToolFunction { name, description, parameters },
        raw: v.clone(),
    })
}

impl ChatRequest {
    /// Decode and validate a parsed request body.
    pub fn parse(body: &Json) -> Result<ChatRequest, ApiError> {
        let model = body.get("model").and_then(|m| m.as_str()).unwrap_or("").to_string();
        let mut images = Images { seen: 0, keep_from: 0, out: Vec::new() };
        let messages = match body.get("messages").and_then(|m| m.as_array()) {
            Some(ms) => {
                images.keep_from = count_images(ms).saturating_sub(MAX_IMAGES);
                ms.iter().map(|m| parse_message(m, &mut images)).collect::<Result<Vec<_>, _>>()?
            }
            None => return Err(ApiError::bad_request("missing messages array")),
        };
        if messages.is_empty() {
            return Err(ApiError::bad_request("messages must not be empty"));
        }
        let tools = match body.get("tools") {
            Some(Json::Array(ts)) => ts.iter().map(parse_tool).collect::<Result<Vec<_>, _>>()?,
            _ => Vec::new(),
        };
        // A constrained tool_choice is not yet served; refuse, never ignore.
        if let Some(tc) = body.get("tool_choice") {
            match tc {
                Json::Str(s) if s == "auto" || s == "none" => {}
                Json::Null => {}
                _ => return Err(ApiError::bad_request(
                    "constrained tool_choice is not yet supported".to_string(),
                )),
            }
        }
        let temperature = body.get("temperature").and_then(|t| t.as_f64());
        let max_tokens = body.get("max_tokens").and_then(|t| t.as_f64()).map(|n| n as u64);
        let stop = match body.get("stop") {
            Some(Json::Str(s)) => vec![s.clone()],
            Some(Json::Array(a)) => a.iter().filter_map(|x| x.as_str().map(|s| s.to_string())).collect(),
            _ => Vec::new(),
        };
        let stream = body.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);
        let include_usage = body.get("stream_options")
            .and_then(|so| so.get("include_usage")).and_then(|u| u.as_bool()).unwrap_or(false);
        let enable_thinking = body.get("chat_template_kwargs")
            .and_then(|k| k.get("enable_thinking")).and_then(|t| t.as_bool()).unwrap_or(false);
        let parallel_tool_calls = body.get("parallel_tool_calls").and_then(|p| p.as_bool()).unwrap_or(true);

        Ok(ChatRequest {
            model,
            messages,
            tools,
            temperature,
            max_tokens,
            stop,
            stream,
            include_usage,
            enable_thinking,
            parallel_tool_calls,
            images: images.out,
        })
    }
}

/// Serialize a usage block (the shared prompt/completion/total shape).
pub fn usage_json(prompt_tokens: u64, completion_tokens: u64) -> Json {
    Json::Object(vec![
        ("prompt_tokens".into(), Json::Num(prompt_tokens as f64)),
        ("completion_tokens".into(), Json::Num(completion_tokens as f64)),
        ("total_tokens".into(), Json::Num((prompt_tokens + completion_tokens) as f64)),
    ])
}

/// A convenient object builder.
pub fn obj(pairs: Vec<(&str, Json)>) -> Json {
    Json::Object(pairs.into_iter().map(|(k, v)| (k.to_string(), v)).collect())
}

pub fn arr(items: Vec<Json>) -> Json {
    Json::Array(items)
}

pub fn s(v: &str) -> Json {
    Json::Str(v.to_string())
}

pub fn to_string(v: &Json) -> String {
    serialize(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json;

    /// D2b item 1: the tool parse must keep the client's whole tool object in the
    /// client's key order (so the renderer can `tojson` it byte-identically) and
    /// must not drop the function description.
    #[test]
    fn tool_parse_keeps_raw_order_and_description() {
        // Keys in a deliberately non-alphabetical order, with a description.
        let doc = r#"{"type":"function","function":{"description":"reads a file","name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}}"#;
        let v = json::parse(doc).unwrap();
        let t = parse_tool(&v).unwrap();
        assert_eq!(t.function.name, "read_file");
        assert_eq!(t.function.description.as_deref(), Some("reads a file"));
        assert!(t.function.parameters.is_some());
        // The raw object round-trips in the client's order: description before name.
        assert_eq!(serialize(&t.raw), doc);
        // The raw object preserves the description even though the typed field
        // also carries it (belt and braces for the renderer).
        let fun = t.raw.get("function").and_then(|f| f.as_object()).unwrap();
        assert_eq!(fun.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(), vec!["description", "name", "parameters"]);
    }

    /// A tool without a description must parse with `description: None` and keep
    /// its raw object too.
    #[test]
    fn tool_parse_without_description() {
        let doc = r#"{"type":"function","function":{"name":"tool0","parameters":{}}}"#;
        let v = json::parse(doc).unwrap();
        let t = parse_tool(&v).unwrap();
        assert_eq!(t.function.description, None);
        assert_eq!(serialize(&t.raw), doc);
    }
}
