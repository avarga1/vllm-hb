//! Google Gemma 4 architecture backend.
//!
//! # Status: SCAFFOLDING ONLY (Phase 0)
//!
//! Parses [`Gemma4Config`] from `config.json` and logs the detected
//! architecture, then returns an error from [`Gemma4Backend::load`].
//! Forward pass not yet implemented.  See:
//!
//! - Phase 1 (#40) — compressed-tensors pack-quantized weight loader
//! - Phase 2 (#41) — W4A16 int4 GEMM kernel for sm_70
//! - Phase 3 (#42) — decoder block (mixed attention, proportional RoPE)
//! - Phase 4 (#43) — MoE router + 128-expert forward pass
//! - Phase 5 (#44) — full model + KV cache + backend integration
//! - Phase 6 (#45) — parity testing + V100 bench + kernel tuning
//!
//! # Detection
//!
//! Dispatched on `model_type == "gemma4"` in `config.json`.  Gemma 4
//! nests the text-model fields inside a `text_config` block; see
//! [`crate::engine::config::HfMeta`] for how that flattens for the
//! top-level engine dispatch.
//!
//! # The 26B-A4B checkpoint shape (what we're targeting)
//!
//! - 30 decoder layers, hidden 2816
//! - Two attention variants *per model*:
//!   - sliding layers (25 of 30): `head_dim=256`, `num_kv_heads=8`,
//!     default RoPE θ=10k, window=1024
//!   - full layers (5 of 30, every 6th): `global_head_dim=512`,
//!     `num_global_key_value_heads=2`, proportional RoPE θ=1M,
//!     `partial_rotary_factor=0.25`
//! - `attention_k_eq_v=true` — K/V share weights on full layers
//! - MoE per layer: 128 experts, top-8 routing, `moe_intermediate_size=704`
//! - Dense MLP per layer: `intermediate_size=2112`, `gelu_pytorch_tanh`
//! - `final_logit_softcapping=30.0`
//! - Tied word embeddings

#![allow(dead_code, unused_variables, unused_imports)]

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Tensor};
use serde::Deserialize;

// ── Gemma 4 config (text portion only) ────────────────────────────────────────

/// Parsed `text_config` subtree of a Gemma 4 `config.json`.
///
/// Audio and vision towers are skipped — this port is text-only.
#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,

    /// Full-attention layers use this head dim (different from sliding).
    #[serde(default)]
    pub global_head_dim: usize,
    /// Full-attention layers use this KV head count.
    #[serde(default)]
    pub num_global_key_value_heads: usize,

    #[serde(default)]
    pub enable_moe_block: bool,
    #[serde(default)]
    pub num_experts: usize,
    #[serde(default)]
    pub top_k_experts: usize,
    #[serde(default)]
    pub moe_intermediate_size: usize,

    pub sliding_window: usize,
    #[serde(default = "default_true")]
    pub use_sliding_window: bool,

    /// Per-layer type: "sliding_attention" or "full_attention".
    #[serde(default)]
    pub layer_types: Vec<String>,

    #[serde(default)]
    pub final_logit_softcapping: Option<f64>,

    #[serde(default)]
    pub attention_k_eq_v: bool,

    pub rms_norm_eps: f64,

    #[serde(default)]
    pub tie_word_embeddings: bool,

    #[serde(default)]
    pub hidden_activation: Option<String>,
}

fn default_true() -> bool {
    true
}

/// Top-level Gemma 4 config wrapper.  Only `text_config` is used — audio
/// and vision towers are skipped in this port.
#[derive(Debug, Clone, Deserialize)]
pub struct Gemma4Config {
    pub model_type: String,
    pub text_config: Gemma4TextConfig,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

impl Gemma4Config {
    /// Parse a raw `config.json` string into a `Gemma4Config`.
    ///
    /// Returns an error with useful context if the schema doesn't match.
    pub fn from_config_json(s: &str) -> Result<Self> {
        let cfg: Self = serde_json::from_str(s)
            .context("Parsing config.json as Gemma4Config (text_config required)")?;
        if cfg.model_type != "gemma4" {
            bail!(
                "Gemma4Config expected model_type=\"gemma4\", got {:?}",
                cfg.model_type
            );
        }
        Ok(cfg)
    }
}

// ── Backend stub ──────────────────────────────────────────────────────────────

pub struct Gemma4Backend {
    pub config: Gemma4Config,
    pub device: Device,
    #[allow(dead_code)]
    dtype: DType,
}

impl Gemma4Backend {
    pub fn load(
        config_str: &str,
        _shards: &[std::path::PathBuf],
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let cfg = Gemma4Config::from_config_json(config_str)?;
        let t = &cfg.text_config;

        tracing::info!(
            vocab = t.vocab_size,
            hidden = t.hidden_size,
            layers = t.num_hidden_layers,
            heads = t.num_attention_heads,
            kv_heads = t.num_key_value_heads,
            head_dim = t.head_dim,
            global_head_dim = t.global_head_dim,
            moe_enabled = t.enable_moe_block,
            experts = t.num_experts,
            top_k_experts = t.top_k_experts,
            sliding_window = t.sliding_window,
            "Gemma 4 architecture detected (scaffolding only — forward pass not implemented)"
        );

        bail!(
            "Gemma 4 forward pass is not yet implemented. \
             Config parsed successfully; see issues #39-45 for the \
             6-phase port roadmap. This backend can load the config but \
             cannot run inference yet."
        )
    }

    pub fn forward(&self, _token_ids: &[u32], _seq_pos: usize) -> Result<Tensor> {
        unreachable!("Gemma4Backend::load always fails at this phase")
    }

    pub fn reset_cache(&self) -> Result<()> {
        unreachable!("Gemma4Backend::load always fails at this phase")
    }

    pub fn create_kv_cache(&self) -> Vec<Option<(Tensor, Tensor)>> {
        unreachable!("Gemma4Backend::load always fails at this phase")
    }

    pub fn forward_with_cache(
        &self,
        _token_ids: &[u32],
        _seq_pos: usize,
        _cache: &mut [Option<(Tensor, Tensor)>],
    ) -> Result<Tensor> {
        unreachable!("Gemma4Backend::load always fails at this phase")
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal shape matching google/gemma-4-26B-A4B-it config.json.
    /// Not the full file — just enough to exercise the parser.
    const GEMMA4_CONFIG_SNIPPET: &str = r#"{
        "model_type": "gemma4",
        "tie_word_embeddings": true,
        "text_config": {
            "vocab_size": 262144,
            "hidden_size": 2816,
            "intermediate_size": 2112,
            "num_hidden_layers": 30,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "head_dim": 256,
            "max_position_embeddings": 262144,
            "global_head_dim": 512,
            "num_global_key_value_heads": 2,
            "enable_moe_block": true,
            "num_experts": 128,
            "top_k_experts": 8,
            "moe_intermediate_size": 704,
            "sliding_window": 1024,
            "layer_types": [
                "sliding_attention", "sliding_attention", "sliding_attention",
                "sliding_attention", "sliding_attention", "full_attention"
            ],
            "final_logit_softcapping": 30.0,
            "attention_k_eq_v": true,
            "rms_norm_eps": 1e-06,
            "hidden_activation": "gelu_pytorch_tanh"
        }
    }"#;

    #[test]
    fn parses_gemma4_config() {
        let cfg = Gemma4Config::from_config_json(GEMMA4_CONFIG_SNIPPET).unwrap();
        assert_eq!(cfg.model_type, "gemma4");
        assert!(cfg.tie_word_embeddings);
        let t = cfg.text_config;
        assert_eq!(t.num_hidden_layers, 30);
        assert_eq!(t.hidden_size, 2816);
        assert_eq!(t.num_experts, 128);
        assert_eq!(t.top_k_experts, 8);
        assert_eq!(t.sliding_window, 1024);
        assert_eq!(t.final_logit_softcapping, Some(30.0));
        assert!(t.attention_k_eq_v);
        assert_eq!(t.layer_types.len(), 6);
        assert_eq!(t.global_head_dim, 512);
    }

    #[test]
    fn rejects_wrong_model_type() {
        let json = r#"{
            "model_type": "llama",
            "text_config": {
                "vocab_size": 32000,
                "hidden_size": 4096,
                "intermediate_size": 11008,
                "num_hidden_layers": 32,
                "num_attention_heads": 32,
                "num_key_value_heads": 32,
                "head_dim": 128,
                "max_position_embeddings": 4096,
                "sliding_window": 0,
                "rms_norm_eps": 1e-5
            }
        }"#;
        let err = Gemma4Config::from_config_json(json).unwrap_err();
        assert!(err.to_string().contains("expected model_type=\"gemma4\""));
    }
}
