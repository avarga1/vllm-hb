//! Model loading and forward pass via Candle.
//!
//! Pure Rust → CUDA through cudarc. No Python. No libtorch. No C++ FFI.

use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama as candle_llama;
use parking_lot::Mutex;

// ── Configuration ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_path:           String,
    pub max_seq_len:          usize,
    pub gpu_memory_utilization: f64,
}

// ── Engine ────────────────────────────────────────────────────────────────────

/// Owns the loaded model weights and KV cache.
///
/// `forward()` is the only public compute method — it accepts a slice of
/// token IDs and a sequence position and returns a logit tensor over the
/// full vocabulary.
pub struct Engine {
    pub config:   ModelConfig,
    model:        candle_llama::Llama,
    pub device:   Device,
    cache:        Mutex<candle_llama::Cache>,
    llama_config: candle_llama::Config,
}

impl Engine {
    /// Load weights from `config.model_path` (a directory of `.safetensors`
    /// files in HuggingFace format).
    pub fn load(config: ModelConfig) -> Result<Self> {
        let model_path = Path::new(&config.model_path);

        // Prefer CUDA; fall back to CPU for testing / CPU-only machines.
        let device = Device::cuda_if_available(0)?;
        tracing::info!(device = ?device, "Compute device");

        // V100 (sm_70) does not support BF16 natively — use FP16 on CUDA.
        let dtype = if device.is_cuda() { DType::F16 } else { DType::F32 };

        // Parse model architecture from HuggingFace `config.json`.
        let config_path = model_path.join("config.json");
        let config_str  = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Reading {}", config_path.display()))?;
        let llama_config: candle_llama::LlamaConfig = serde_json::from_str(&config_str)
            .context("Parsing config.json as LlamaConfig")?;
        let llama_config = llama_config.into_config(false);

        tracing::info!(
            layers  = llama_config.num_hidden_layers,
            hidden  = llama_config.hidden_size,
            heads   = llama_config.num_attention_heads,
            kv_heads = llama_config.num_key_value_heads,
            vocab   = llama_config.vocab_size,
            "Model architecture"
        );

        // Collect and sort safetensors shards.
        let mut shards: Vec<_> = std::fs::read_dir(model_path)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
            .collect();
        shards.sort();
        tracing::info!(shards = shards.len(), dtype = ?dtype, "Loading weights");

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&shards, dtype, &device)?
        };

        let cache = candle_llama::Cache::new(true, dtype, &llama_config, &device)?;
        let model = candle_llama::Llama::load(vb, &llama_config)?;

        tracing::info!("Weights loaded");
        Ok(Self { config, model, device, cache: Mutex::new(cache), llama_config })
    }

    // ── Compute ───────────────────────────────────────────────────────────────

    /// Forward pass: `token_ids` → logit tensor `[vocab_size]` on the device.
    ///
    /// `seq_pos` is the absolute position of the first token in `token_ids`
    /// within the current sequence (0 for prefill, `prompt_len + step` for
    /// each decode step).
    pub fn forward(&self, token_ids: &[u32], seq_pos: usize) -> Result<Tensor> {
        let input   = Tensor::new(token_ids, &self.device)?.unsqueeze(0)?;
        let mut cache = self.cache.lock();
        let logits  = self.model.forward(&input, seq_pos, &mut cache)?;
        Ok(logits.squeeze(0)?)
    }

    /// Reset the KV cache. Call once before each new request.
    pub fn reset_cache(&self) -> Result<()> {
        let dtype = if self.device.is_cuda() { DType::F16 } else { DType::F32 };
        let fresh = candle_llama::Cache::new(true, dtype, &self.llama_config, &self.device)?;
        *self.cache.lock() = fresh;
        Ok(())
    }

    // ── Metadata ──────────────────────────────────────────────────────────────

    /// Approximate parameter count (useful for logging).
    pub fn param_count(&self) -> usize {
        let c    = &self.llama_config;
        let attn = 4 * c.hidden_size * c.hidden_size;
        let ffn  = 3 * c.hidden_size * c.intermediate_size;
        c.vocab_size * c.hidden_size
            + c.num_hidden_layers * (attn + ffn)
            + c.vocab_size * c.hidden_size
    }

    pub fn num_layers(&self) -> usize {
        self.llama_config.num_hidden_layers
    }

    pub fn vocab_size(&self) -> usize {
        self.llama_config.vocab_size
    }
}
