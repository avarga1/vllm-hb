//! Engine configuration types.
//!
//! Kept separate from the model loading logic so main.rs and the worker
//! can construct a ModelConfig without pulling in candle dependencies.

use serde::Deserialize;

// ── Runtime config ────────────────────────────────────────────────────────────

// max_seq_len and gpu_memory_utilization are stored for future use by the
// paged-KV-cache scheduler (scheduler/block_manager.rs).
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_path: String,
    pub max_seq_len: usize,
    pub gpu_memory_utilization: f64,
    /// Force BF16 weights (requires sm_80+ / Ampere GPU).
    /// Falls back to F16 on older hardware when false.
    pub bf16: bool,
    /// Number of GPUs for tensor parallelism.
    ///
    /// `1` = single-GPU (default, no inter-GPU communication).
    /// `N > 1` = column/row-parallel sharding across N CUDA devices.
    pub tensor_parallel_size: usize,
}

// ── HuggingFace config.json ───────────────────────────────────────────────────

/// Minimal subset of HuggingFace `config.json` used for architecture
/// detection and metadata logging.  We don't parse the full config here
/// — each arch module does its own full parse.
///
/// # Nested `text_config` support
///
/// Most HF model configs expose `hidden_size`, `num_hidden_layers`, etc. at
/// the top level.  Multimodal models (Gemma 4, PaliGemma, …) nest them
/// inside a `text_config` block instead.  We parse via an intermediate
/// [`HfMetaRaw`] with optional fields, then [`From`] fills the flat
/// [`HfMeta`] from whichever location populated.  Existing call sites
/// keep accessing `meta.hidden_size` unchanged.
#[derive(Debug, Clone)]
pub struct HfMeta {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
}

#[derive(Deserialize)]
struct HfMetaRaw {
    model_type: String,
    #[serde(default)]
    vocab_size: Option<usize>,
    #[serde(default)]
    hidden_size: Option<usize>,
    #[serde(default)]
    intermediate_size: Option<usize>,
    #[serde(default)]
    num_hidden_layers: Option<usize>,
    #[serde(default)]
    num_attention_heads: Option<usize>,
    #[serde(default)]
    text_config: Option<HfMetaTextConfig>,
}

#[derive(Deserialize)]
struct HfMetaTextConfig {
    #[serde(default)]
    vocab_size: Option<usize>,
    #[serde(default)]
    hidden_size: Option<usize>,
    #[serde(default)]
    intermediate_size: Option<usize>,
    #[serde(default)]
    num_hidden_layers: Option<usize>,
    #[serde(default)]
    num_attention_heads: Option<usize>,
}

impl From<HfMetaRaw> for HfMeta {
    fn from(raw: HfMetaRaw) -> Self {
        let tc = raw.text_config.as_ref();
        let pick = |top: Option<usize>, sub: fn(&HfMetaTextConfig) -> Option<usize>| -> usize {
            top.or_else(|| tc.and_then(sub)).unwrap_or(0)
        };
        Self {
            model_type: raw.model_type,
            vocab_size: pick(raw.vocab_size, |c| c.vocab_size),
            hidden_size: pick(raw.hidden_size, |c| c.hidden_size),
            intermediate_size: pick(raw.intermediate_size, |c| c.intermediate_size),
            num_hidden_layers: pick(raw.num_hidden_layers, |c| c.num_hidden_layers),
            num_attention_heads: pick(raw.num_attention_heads, |c| c.num_attention_heads),
        }
    }
}

impl HfMeta {
    /// Parse from HuggingFace `config.json` string.  Handles both flat
    /// configs (Llama, Qwen, Mixtral) and nested-text-config configs
    /// (Gemma 4, PaliGemma).
    pub fn from_config_json(s: &str) -> Result<Self, serde_json::Error> {
        let raw: HfMetaRaw = serde_json::from_str(s)?;
        Ok(raw.into())
    }
}

// Backwards compat: let existing call sites keep using `serde_json::from_str`.
impl<'de> Deserialize<'de> for HfMeta {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        HfMetaRaw::deserialize(d).map(Self::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_config_llama_style() {
        let json = r#"{
            "model_type": "llama",
            "vocab_size": 32000,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 32
        }"#;
        let m: HfMeta = serde_json::from_str(json).unwrap();
        assert_eq!(m.model_type, "llama");
        assert_eq!(m.hidden_size, 4096);
        assert_eq!(m.num_hidden_layers, 32);
    }

    #[test]
    fn nested_text_config_gemma4_style() {
        // Shape matches google/gemma-4-26B-A4B-it config.json.
        let json = r#"{
            "model_type": "gemma4",
            "text_config": {
                "vocab_size": 262144,
                "hidden_size": 2816,
                "intermediate_size": 2112,
                "num_hidden_layers": 30,
                "num_attention_heads": 16
            }
        }"#;
        let m: HfMeta = serde_json::from_str(json).unwrap();
        assert_eq!(m.model_type, "gemma4");
        assert_eq!(m.hidden_size, 2816);
        assert_eq!(m.num_hidden_layers, 30);
        assert_eq!(m.vocab_size, 262144);
    }

    #[test]
    fn top_level_wins_over_text_config() {
        // When both locations have a value, the top-level one wins
        // (matches HF's own resolution order for multimodal configs).
        let json = r#"{
            "model_type": "hybrid",
            "hidden_size": 9999,
            "text_config": { "hidden_size": 1111 }
        }"#;
        let m: HfMeta = serde_json::from_str(json).unwrap();
        assert_eq!(m.hidden_size, 9999);
    }
}
