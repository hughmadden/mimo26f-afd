//! The `Engine` trait — the seam the coordinator implements and the API tests
//! stub. The API owns the HTTP surface and the tool-call parser; the engine
//! owns tokenization, the chat template and generation.

use crate::types::{ChatMessage, Tool};

/// Sampling/control parameters handed to [`Engine::generate`].
#[derive(Debug, Clone)]
pub struct GenerateParams {
    pub max_tokens: usize,
    pub temperature: f64,
    pub stop: Vec<String>,
    pub thinking: bool,
    /// Set by the API once the client is gone (a failed write): the engine stops
    /// generating and frees the request (perf reset Q2).
    pub cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// The request's images, in prompt order (perf reset V2). Each stands in the rendered prompt
    /// as an [`image_marker`].
    pub images: Vec<std::sync::Arc<ImageInput>>,
}

impl Default for GenerateParams {
    fn default() -> Self {
        GenerateParams { max_tokens: 65_536, temperature: 1.0, stop: Vec::new(), thinking: false, cancel: None,
            images: Vec::new() }
    }
}

/// A decoded image of a chat request (perf reset V2): RGB8 pixels, the number of language-model
/// positions it takes (`tokens`, its merged 2x2 patch grid) and a hash of its encoded bytes.
#[derive(Debug)]
pub struct ImageInput {
    pub hash: u64,
    pub tokens: usize,
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
}

/// The reserved characters that delimit an image in rendered message text. U+FDD0 and U+FDD1 are
/// Unicode noncharacters; the API strips them from client text, so only the API can place one.
pub const IMAGE_OPEN: char = '\u{FDD0}';
pub const IMAGE_CLOSE: char = '\u{FDD1}';

/// The text an image stands as in a message (where the chat template renders
/// `<|vision_start|><|image_pad|><|vision_end|>`): its hash and token count, which the engine's
/// tokenizer turns into the image's token span.
pub fn image_marker(img: &ImageInput) -> String {
    format!("{IMAGE_OPEN}{:016x}:{}{IMAGE_CLOSE}", img.hash, img.tokens)
}

/// The result of one generation.
#[derive(Debug, Clone)]
pub struct GenerateOutcome {
    /// The full completion text (which the API parses for tool calls and think
    /// blocks).
    pub text: String,
    /// `stop` (a stop sequence or EOS), `length` (max_tokens), or `tool_calls`
    /// (the tool-call cap fired).
    pub finish_reason: String,
    /// Completion token count (for `usage.completion_tokens`).
    pub completion_tokens: usize,
}

/// A model backend. All methods are `&self` so a single engine serves many
/// concurrent connections without interior synchronization.
pub trait Engine {
    /// Token count of the prompt (for `usage.prompt_tokens`).
    fn tokenize(&self, messages: &[ChatMessage], tools: &[Tool], thinking: bool) -> usize;

    /// Render the chat into the model's native input (chat template).
    fn render_chat(&self, messages: &[ChatMessage], tools: &[Tool], thinking: bool) -> String;

    /// The longest request (prompt plus output) this deployment can hold, when it
    /// is bounded (the coordinator's KV pool); prompts at or over it are refused
    /// with 400 before generation starts.
    fn max_context(&self) -> Option<usize> {
        None
    }

    /// Whether this engine encodes images (perf reset V2). Without it the API refuses image
    /// parts with a 400.
    fn vision(&self) -> bool {
        false
    }

    /// Generate the completion. `on_delta` is called with each incremental text
    /// delta (the API forwards it as an SSE content delta); the returned text is
    /// the full completion the API parses for tool calls and think blocks.
    fn generate(
        &self,
        prompt: &str,
        params: &GenerateParams,
        on_delta: &mut dyn FnMut(&str),
    ) -> Result<GenerateOutcome, String>;
}
