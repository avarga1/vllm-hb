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
    pub fn forward(&self, token_ids: &[u32], seq_pos: usize) -> Result<Tensor> {
        match self {
            Self::Llama(m) => m.forward(token_ids, seq_pos),
            Self::LlamaTp(m) => m.forward(token_ids, seq_pos),
            Self::Mixtral(m) => m.forward(token_ids, seq_pos),
            Self::Qwen2(m) => m.forward(token_ids, seq_pos),
            Self::Phi3(m) => m.forward(token_ids, seq_pos),
        }
    }

    pub fn reset_cache(&self) -> Result<()> {
        match self {
            Self::Llama(m) => m.reset_cache(),
            Self::LlamaTp(m) => m.reset_cache(),
            Self::Mixtral(m) => m.reset_cache(),
            Self::Qwen2(m) => m.reset_cache(),
            Self::Phi3(m) => m.reset_cache(),
        }
    }
}
