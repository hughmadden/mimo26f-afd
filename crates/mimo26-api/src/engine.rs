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
}

impl Default for GenerateParams {
    fn default() -> Self {
        GenerateParams { max_tokens: 65_536, temperature: 1.0, stop: Vec::new(), thinking: false, cancel: None }
    }
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
