//! Per-architecture model backends.
//!
//! `Backend` is an enum (not a trait object) so the compiler can
//! devirtualize and inline the forward call completely.  When a new
//! architecture is supported, add a variant here and a module below.

pub mod llama;
pub mod llama_tp;
pub mod mixtral;
pub mod phi3;
pub mod qwen2;

use anyhow::Result;
use candle_core::Tensor;
pub use llama::LlamaBackend;
pub use llama_tp::TpLlamaBackend;
pub use mixtral::MixtralBackend;
pub use phi3::Phi3Backend;
pub use qwen2::Qwen2Backend;

use crate::engine::kv_cache::PerSeqCache;

// ── Backend enum ──────────────────────────────────────────────────────────────

// TpLlamaBackend and LlamaBackend are large; stubs are zero-sized.
#[allow(clippy::large_enum_variant)]
pub(crate) enum Backend {
    Llama(LlamaBackend),
    LlamaTp(TpLlamaBackend),
    Mixtral(MixtralBackend), // stub — see arch/mixtral.rs
    Qwen2(Qwen2Backend),     // stub — see arch/qwen2.rs
    Phi3(Phi3Backend),       // stub — see arch/phi3.rs
}

impl Backend {
    #[allow(dead_code)] // used by bench subcommand via Engine::forward
    pub fn forward(&self, token_ids: &[u32], seq_pos: usize) -> Result<Tensor> {
        match self {
            Self::Llama(m) => m.forward(token_ids, seq_pos),
            Self::LlamaTp(m) => m.forward(token_ids, seq_pos),
            Self::Mixtral(m) => m.forward(token_ids, seq_pos),
            Self::Qwen2(m) => m.forward(token_ids, seq_pos),
            Self::Phi3(m) => m.forward(token_ids, seq_pos),
        }
    }

    #[allow(dead_code)]
    pub fn reset_cache(&self) -> Result<()> {
        match self {
            Self::Llama(m) => m.reset_cache(),
            Self::LlamaTp(m) => m.reset_cache(),
            Self::Mixtral(m) => m.reset_cache(),
            Self::Qwen2(m) => m.reset_cache(),
            Self::Phi3(m) => m.reset_cache(),
        }
    }

    // ── Per-sequence cache API ────────────────────────────────────────────────

    /// Allocate a fresh per-sequence KV cache for the given backend.
    pub fn create_kv_cache(&self) -> Result<PerSeqCache> {
        match self {
            Self::Llama(m) => Ok(PerSeqCache::Llama(m.create_kv_cache()?)),
            Self::LlamaTp(_) => Ok(PerSeqCache::LlamaTp),
            Self::Mixtral(m) => Ok(PerSeqCache::Mixtral(m.create_kv_cache())),
            Self::Qwen2(m) => Ok(PerSeqCache::Mixtral(m.create_kv_cache())),
            Self::Phi3(m) => Ok(PerSeqCache::Mixtral(m.create_kv_cache())),
        }
    }

    /// Run one forward step with an externally-owned per-sequence cache.
    pub fn forward_with_cache(
        &self,
        token_ids: &[u32],
        seq_pos: usize,
        cache: &mut PerSeqCache,
    ) -> Result<Tensor> {
        match (self, cache) {
            (Self::Llama(m), PerSeqCache::Llama(c)) => m.forward_with_cache(token_ids, seq_pos, c),
            (Self::LlamaTp(m), PerSeqCache::LlamaTp) => m.forward_with_cache(token_ids, seq_pos),
            (Self::Mixtral(m), PerSeqCache::Mixtral(c)) => {
                m.forward_with_cache(token_ids, seq_pos, c)
            }
            (Self::Qwen2(m), PerSeqCache::Mixtral(c)) => {
                m.forward_with_cache(token_ids, seq_pos, c)
            }
            (Self::Phi3(m), PerSeqCache::Mixtral(c)) => m.forward_with_cache(token_ids, seq_pos, c),
            _ => anyhow::bail!("forward_with_cache: backend/cache type mismatch"),
        }
    }
}

// Ensure PerSeqCache can cross thread boundaries (worker holds it in a HashMap).
// candle_core::Tensor is Send + Sync; candle_llama::Cache contains only Tensors
// and stdlib types — all Send.
fn _assert_send() {
    fn check<T: Send>() {}
    check::<PerSeqCache>();
}
