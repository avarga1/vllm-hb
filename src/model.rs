//! Model loading and forward pass via Candle.
//!
//! Architecture is detected automatically from `model_type` in `config.json`.
//! Supported: `llama`, `mistral`.
//!
//! Pure Rust → CUDA through cudarc. No Python. No libtorch. No C++ FFI.

use std::path::Path;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama as candle_llama;
use parking_lot::Mutex;
use serde::Deserialize;

// ── Engine config ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_path:             String,
    pub max_seq_len:            usize,
    pub gpu_memory_utilization: f64,
}

// Minimal HuggingFace config.json fields — used for architecture detection.
#[derive(Deserialize)]
struct HfMeta {
    model_type:          String,
    vocab_size:          usize,
    hidden_size:         usize,
    intermediate_size:   usize,
    num_hidden_layers:   usize,
    num_attention_heads: usize,
}

// ── Engine ────────────────────────────────────────────────────────────────────

/// Owns the loaded model weights and KV cache.
///
/// `forward()` is the only public compute method — it accepts a slice of
/// token IDs and a sequence position and returns a logit tensor over the
/// full vocabulary.
pub struct Engine {
    pub config: ModelConfig,
    model:      candle_llama::Llama,
    pub device: Device,
    cache:      Mutex<candle_llama::Cache>,
    llama_cfg:  candle_llama::Config,
    // Shared metadata for external introspection.
    vocab_size:        usize,
    num_layers:        usize,
    hidden_size:       usize,
    intermediate_size: usize,
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

        // Parse config.json.
        let config_path = model_path.join("config.json");
        let config_str  = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Reading {}", config_path.display()))?;

        let meta: HfMeta = serde_json::from_str(&config_str)
            .context("Parsing config.json for architecture metadata")?;

        match meta.model_type.as_str() {
            "llama" | "mistral" => {}
            other => bail!(
                "Unsupported model_type: {other:?}. \
                 Supported: llama, mistral. \
                 (Mixtral support planned for next release.)"
            ),
        }

        tracing::info!(
            model_type = %meta.model_type,
            layers     = meta.num_hidden_layers,
            hidden     = meta.hidden_size,
            heads      = meta.num_attention_heads,
            vocab      = meta.vocab_size,
            "Model architecture"
        );

        // Collect and sort safetensors shards.
        let mut shards: Vec<_> = std::fs::read_dir(model_path)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
            .collect();
        shards.sort();

        if shards.is_empty() {
            bail!("No .safetensors files found in {}", model_path.display());
        }
        tracing::info!(shards = shards.len(), dtype = ?dtype, "Loading weights");

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&shards, dtype, &device)?
        };

        let llama_cfg: candle_llama::LlamaConfig = serde_json::from_str(&config_str)
            .context("Parsing config.json as LlamaConfig")?;
        let llama_cfg = llama_cfg.into_config(false);

        let cache = candle_llama::Cache::new(true, dtype, &llama_cfg, &device)?;
        let model = candle_llama::Llama::load(vb, &llama_cfg)?;

        tracing::info!("Weights loaded");
        Ok(Self {
            config,
            model,
            device,
            cache:             Mutex::new(cache),
            vocab_size:        meta.vocab_size,
            num_layers:        meta.num_hidden_layers,
            hidden_size:       meta.hidden_size,
            intermediate_size: meta.intermediate_size,
            llama_cfg,
        })
    }

    // ── Compute ───────────────────────────────────────────────────────────────

    /// Forward pass: `token_ids` → logit tensor `[vocab_size]` on the device.
    ///
    /// `seq_pos` is the absolute position of the first token in `token_ids`
    /// within the current sequence (0 for prefill, `prompt_len + step` for
    /// each decode step).
    pub fn forward(&self, token_ids: &[u32], seq_pos: usize) -> Result<Tensor> {
        let input  = Tensor::new(token_ids, &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, seq_pos, &mut self.cache.lock())?;
        Ok(logits.squeeze(0)?)
    }

    /// Reset the KV cache. Call once before each new request.
    pub fn reset_cache(&self) -> Result<()> {
        let dtype = if self.device.is_cuda() { DType::F16 } else { DType::F32 };
        let fresh = candle_llama::Cache::new(true, dtype, &self.llama_cfg, &self.device)?;
        *self.cache.lock() = fresh;
        Ok(())
    }

    // ── Metadata ──────────────────────────────────────────────────────────────

    /// Approximate parameter count (useful for logging).
    pub fn param_count(&self) -> usize {
        let attn = 4 * self.hidden_size * self.hidden_size;
        let ffn  = 3 * self.hidden_size * self.intermediate_size;
        self.vocab_size * self.hidden_size
            + self.num_layers * (attn + ffn)
            + self.vocab_size * self.hidden_size
    }

    pub fn num_layers(&self) -> usize { self.num_layers }
    pub fn vocab_size(&self) -> usize { self.vocab_size }
}
