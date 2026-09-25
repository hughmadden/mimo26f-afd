//! `mimo26-coordinator` — coordinator-side model-forward glue (I5 Track S).
//!
//! The golden-locked Rust twin of [`oracle/mimo26/nn/layers.py`] (consumed
//! read-only, never imported — I-Gold). Every function mirrors the oracle's
//! arithmetic at its documented precision: RMSNorm and SiLU run their shared
//! math in FP64 (the oracle upcasts the input, computes in FP64, casts back to
//! FP32), `linear`/`dense_ffn` accumulate in FP32, and the router computes FP32
//! logits then FP64 sigmoid/selection (T22: never a BF16 bias — a near-tie
//! that flips under a BF16 bias must not flip here).
//!
//! Layout convention: weight matrices are row-major `[out, in]`;
//! `linear(x, w) == x @ w.T`. Dimensions are explicit parameters so this CPU
//! twin is independent of the loader's quantized layouts.

pub mod chat;
pub mod config;
pub mod embeddings;
pub mod forward;
pub mod json;
pub mod kv_pool;
pub mod linear;
pub mod load;
pub mod moe;
pub mod needle;
pub mod norm;
pub mod paging;
pub mod phase;
pub mod router;
pub mod sampler;
pub mod scheduler;
pub mod stop;
pub mod streaming;
pub mod token;
pub mod tokenizer;
pub mod tool_call;
pub mod wire;

/// cuBLAS FFI for the GPU dense path (I5-R8a) — only compiled under `cuda`.
#[cfg(feature = "cuda")]
pub mod cublas;
/// The GPU dense device (FP32 weights + SGEMM) — only compiled under `cuda`.
#[cfg(feature = "cuda")]
pub mod gpu_dense;
/// The serving forward (dense on GPU, attention/MoE on CPU) — `cuda` only.
#[cfg(feature = "cuda")]
pub mod serving;

#[cfg(feature = "cuda")]
pub mod dforward;
/// The A8 `Engine` implementation — `cuda` only (generate needs the serving path).
#[cfg(feature = "cuda")]
pub mod api;

#[cfg(feature = "cuda")]
pub mod hostcache;
/// The image encoder (perf reset V2) — `cuda` only.
#[cfg(feature = "cuda")]
pub mod vision;

pub use chat::{render_chat, ChatMessage, ChatOptions, ChatToolCall};
pub use config::{Config, LayerKind as AttnLayerKind, DENSE_LAYER_IDS, GA_LAYER_IDS};
pub use embeddings::embed;
pub use json::{parse as json_parse, serialize as json_serialize, JsonValue};
pub use kv_pool::{
    full_lifetime_reservation, request_reservation, Admission, KvPool, DEFAULT_ADMIT_RESERVE,
    GA_KV_BYTES_PER_TOKEN, MAX_OUTPUT_TOKENS, SWA_RING_BYTES_PER_SEQ,
};
pub use linear::{dense_ffn, linear};
pub use load::{bf16_to_f32, load_coordinator_weights, weights_dir};
pub use moe::moe_forward;
pub use needle::{build_text, filler_doc, place, NeedleMeta, FACT, FACT_CODE, QUESTION};
pub use paging::{Pager, PagerError};
pub use stop::{check_output, StopConfig, StopReason};
pub use token::{is_eos, BOS_TOKEN_ID, EOS_TOKEN_IDS, IM_END, IM_START, MASK_TOKEN_ID, PAD_TOKEN_ID,
    THINK, THINK_CLOSE};
pub use tokenizer::{bytes_to_unicode, from_tokenizer_json, gpt2_pretokenize, BpeTokenizer};
pub use norm::{rmsnorm, silu};
pub use phase::{
    class_for, phase_for_role, run_length_encode, validate_cache_hit, Arm, ArmId, ArmRegistry,
    DerivationPath, Digest, EquivClass, HitVerdict, KvPage, LayerKind, Lattice, PageKey, PageState,
    Phase, PhaseError, PrefixKey, Role, ATTENTION_ARM, SINGLE_ARM,
};
pub use router::{router, router_from_logits};
pub use forward::{gen_weights, Lcg, LayerCache, Model, w_name as weight_name};
pub use sampler::{
    apply_repetition_penalty, apply_temperature, apply_top_k, apply_top_p, greedy, sample,
    softmax, SamplingParams, SeenSet,
};
pub use scheduler::{
    Request, RequestState, Scheduler, SchedulerConfig, Step, SubmitOutcome,
};
pub use tool_call::{
    check_tool_calls, parse_tool_calls, CapPolicy, FinishReason, ParamType, ParsedCall,
    ParseOutcome, ToolCallCheck, ToolSchema,
};
pub use wire::{quantize_hidden, WireClient};

/// Frozen R7 componentwise tolerance: `2e-5 * max(1, max|v|)` against the
/// oracle reference. Exposed so tests and later GPU kernels share one bar.
pub fn r7_tolerance(reference: &[f32]) -> f32 {
    let m = reference.iter().fold(1.0f32, |a, &v| a.max(v.abs()));
    2e-5 * m
}
