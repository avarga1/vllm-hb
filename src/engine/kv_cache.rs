//! Per-sequence KV cache types.
//!
//! Each active sequence owns one `PerSeqCache` that travels with it through
//! the scheduler loop.  The worker holds a `HashMap<seq_id, PerSeqCache>` and
//! passes the right cache to `Engine::forward_with_cache` on every step.
//!
//! # Why an enum, not a trait object?
//!
//! Each architecture stores KV state in a different type (`candle_llama::Cache`
//! vs `Vec<Option<(Tensor,Tensor)>>`).  The enum lets the compiler devirtualize
//! the dispatch, matching the pattern used in `arch/mod.rs`.
//!
//! # Tensor-parallel note
//!
//! `TpLlama` uses a backend-internal cache shared across all sequences.
//! Per-sequence TP caching requires PagedAttention-style block tables and is
//! tracked separately.  The `LlamaTp` variant here is a zero-size marker so
//! the worker can manage the lifecycle uniformly.

use candle_core::Tensor;
use candle_transformers::models::llama as candle_llama;
use candle_transformers::models::quantized_llama::ModelWeights as GgufModelWeights;

/// KV state for one active sequence.
///
/// Created by `Engine::create_kv_cache`, mutated in-place by
/// `Engine::forward_with_cache`, dropped when the sequence finishes.
pub enum PerSeqCache {
    /// Single-GPU Llama / Mistral — uses `candle_transformers::models::llama::Cache`.
    Llama(candle_llama::Cache),
    /// Mixtral 8×7B — per-layer `(key, value)` tensors; `None` until the
    /// layer is first visited.
    Mixtral(Vec<Option<(Tensor, Tensor)>>),
    /// Tensor-parallel Llama — backend owns the cache internally.
    /// This variant is a marker so the worker can insert/remove it uniformly.
    LlamaTp,
    /// GGUF-quantized Llama — a cheap clone of the template `ModelWeights`
    /// (Arc-backed weights shared, KV cache private to this sequence).
    /// The `ModelWeights` is mutated in-place during the forward pass.
    GgufLlama(GgufModelWeights),
}

impl PerSeqCache {
    /// Attempt a cheap clone of the cache for external-cache architectures.
    ///
    /// Returns `Some` only for `Mixtral` and `GgufLlama`, where the underlying
    /// `Tensor`s are Arc-backed and cloning costs O(num_layers) ref-count bumps.
    /// Returns `None` for `Llama` and `LlamaTp` — callers must use a fallback.
    ///
    /// Used by the speculative decoder to snapshot the draft cache before
    /// generating K candidate tokens so it can be restored on partial accept.
    pub fn try_clone_external(&self) -> Option<Self> {
        match self {
            Self::Mixtral(v) => Some(Self::Mixtral(v.clone())),
            Self::GgufLlama(m) => Some(Self::GgufLlama(m.clone())),
            Self::Llama(_) | Self::LlamaTp => None,
        }
    }
}
