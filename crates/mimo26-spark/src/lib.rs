//! `mimo26-spark` — the Spark expert-rank daemon (I5 Track S).
//!
//! Owns the serving side of the attention/FFN split on a Spark rank:
//!
//! 1. [`ffi`] — the Rust→CUDA FFI surface for the layout-v2 expert kernels
//!    (`mimo26_expert_kernels.h`): the `m26x_plan` metadata struct and the
//!    decode/prefill/unpack/identity entry points. Declarations only; symbol
//!    resolution is a separate feature-gated build step so the CPU merge gate
//!    (`cargo test --workspace`) needs no nvcc.
//! 2. (next) the daemon: layout-v2 slice residency + boot identity readback
//!    (sha256, refuse on mismatch), decode via the R18c mixed-M dispatch,
//!    prefill via B2 E-FP32, and the DS41RTE3 v3 wire server with the full
//!    network L4 ladder (bitflip / reorder / truncation / duplicate) plus a
//!    live 8 KB-per-Spark-per-token return assertion.
//!
//! The seam stays at the attention/FFN boundary: this crate serves routed
//! experts only — it never touches embeddings, attention, KV, the router or the
//! sampler (ARCHITECTURE.md §3).

#[cfg(feature = "cuda")]
pub mod b1;
pub mod boot;
pub mod cuda;
pub mod decode;
pub mod device;
pub mod ffi;
pub mod resident;
pub mod route;
pub mod serve;
pub mod server;
pub mod timeline;
pub mod transport;
pub mod wire;
