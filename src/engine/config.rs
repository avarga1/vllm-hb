//! Engine configuration types.
//!
//! Kept separate from the model loading logic so main.rs and the worker
//! can construct a ModelConfig without pulling in candle dependencies.

use serde::Deserialize;

// ── Runtime config ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_path:             String,
    pub max_seq_len:            usize,
    pub gpu_memory_utilization: f64,
    /// Force BF16 weights (requires sm_80+ / Ampere GPU).
    /// Falls back to F16 on older hardware when false.
    pub bf16: bool,
}

// ── HuggingFace config.json ───────────────────────────────────────────────────

/// Minimal subset of HuggingFace `config.json` used for architecture
/// detection and metadata logging.  We don't parse the full config here
/// — each arch module does its own full parse.
#[derive(Deserialize)]
pub struct HfMeta {
    pub model_type:          String,
    pub vocab_size:          usize,
    pub hidden_size:         usize,
    pub intermediate_size:   usize,
    pub num_hidden_layers:   usize,
    pub num_attention_heads: usize,
}
