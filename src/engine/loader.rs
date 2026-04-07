//! Engine — loads model weights and dispatches forward/reset to the
//! correct architecture backend.

use std::path::Path;

use anyhow::{Context, Result, bail};
use candle_core::{Device, Tensor};

use super::arch::{Backend, LlamaBackend, MixtralBackend, Phi3Backend, Qwen2Backend};
use super::config::{HfMeta, ModelConfig};
use super::dtype;

// ── Engine ────────────────────────────────────────────────────────────────────

/// Owns the loaded model weights and KV cache.
///
/// The public API (`forward`, `reset_cache`, metadata accessors) is
/// identical regardless of architecture — callers never see candle types.
pub struct Engine {
    pub config: ModelConfig,
    backend: Backend,
    pub device: Device,
    // Shared metadata for external introspection.
    vocab_size: usize,
    num_layers: usize,
    hidden_size: usize,
    intermediate_size: usize,
}

impl Engine {
    /// Load weights from `config.model_path` (a directory of
    /// `.safetensors` files in HuggingFace format).
    pub fn load(config: ModelConfig) -> Result<Self> {
        let model_path = Path::new(&config.model_path);
        let device = Device::cuda_if_available(0)?;
        tracing::info!(device = ?device, "Compute device");

        let dtype = dtype::resolve(&device, config.bf16);

        let config_path = model_path.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .with_context(|| format!("Reading {}", config_path.display()))?;

        let meta: HfMeta = serde_json::from_str(&config_str)
            .context("Parsing config.json for architecture metadata")?;

        tracing::info!(
            model_type = %meta.model_type,
            layers     = meta.num_hidden_layers,
            hidden     = meta.hidden_size,
            heads      = meta.num_attention_heads,
            vocab      = meta.vocab_size,
            "Model architecture"
        );

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

        let backend = match meta.model_type.as_str() {
            "llama" | "mistral" => {
                Backend::Llama(LlamaBackend::load(&config_str, &shards, dtype, &device)?)
            }
            "mixtral" => {
                Backend::Mixtral(MixtralBackend::load(&config_str, &shards, dtype, &device)?)
            }
            "qwen2" => Backend::Qwen2(Qwen2Backend::load(&config_str, &shards, dtype, &device)?),
            "phi3" => Backend::Phi3(Phi3Backend::load(&config_str, &shards, dtype, &device)?),
            other => bail!(
                "Unsupported model_type: {other:?}. \
                 Supported: llama, mistral. \
                 Planned: mixtral, qwen2, phi3."
            ),
        };

        tracing::info!("Weights loaded");
        Ok(Self {
            config,
            backend,
            device,
            vocab_size: meta.vocab_size,
            num_layers: meta.num_hidden_layers,
            hidden_size: meta.hidden_size,
            intermediate_size: meta.intermediate_size,
        })
    }

    // ── Compute ───────────────────────────────────────────────────────────────

    pub fn forward(&self, token_ids: &[u32], seq_pos: usize) -> Result<Tensor> {
        self.backend.forward(token_ids, seq_pos)
    }

    pub fn reset_cache(&self) -> Result<()> {
        self.backend.reset_cache()
    }

    // ── Metadata ──────────────────────────────────────────────────────────────

    pub fn param_count(&self) -> usize {
        let attn = 4 * self.hidden_size * self.hidden_size;
        let ffn = 3 * self.hidden_size * self.intermediate_size;
        self.vocab_size * self.hidden_size
            + self.num_layers * (attn + ffn)
            + self.vocab_size * self.hidden_size
    }

    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}
