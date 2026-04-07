//! Llama / Mistral architecture backend.
//!
//! Handles any model whose `config.json` has `model_type` = "llama" or
//! "mistral".  Both use the same `candle_transformers::models::llama` module.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama as candle_llama;
use parking_lot::Mutex;

pub struct LlamaBackend {
    model: candle_llama::Llama,
    /// Shared cache used by the legacy `forward()`/`reset_cache()` path
    /// (bench subcommand).  The continuous-batching worker uses
    /// `forward_with_cache` with per-sequence caches instead.
    #[allow(dead_code)]
    cache: Mutex<candle_llama::Cache>,
    config: candle_llama::Config,
    device: Device,
}

impl LlamaBackend {
    pub fn load(
        config_str: &str,
        shards: &[std::path::PathBuf],
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let llama_cfg: candle_llama::LlamaConfig =
            serde_json::from_str(config_str).context("Parsing config.json as LlamaConfig")?;
        let llama_cfg = llama_cfg.into_config(false);

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(shards, dtype, device)? };

        let cache = candle_llama::Cache::new(true, dtype, &llama_cfg, device)?;
        let model = candle_llama::Llama::load(vb, &llama_cfg)?;

        Ok(Self {
            model,
            cache: Mutex::new(cache),
            config: llama_cfg,
            device: device.clone(),
        })
    }

    #[allow(dead_code)]
    pub fn forward(&self, token_ids: &[u32], seq_pos: usize) -> Result<Tensor> {
        let input = Tensor::new(token_ids, &self.device)?.unsqueeze(0)?;
        let logits = self
            .model
            .forward(&input, seq_pos, &mut self.cache.lock())?;
        Ok(logits.squeeze(0)?)
    }

    #[allow(dead_code)]
    pub fn reset_cache(&self) -> Result<()> {
        let dtype = if self.device.is_cuda() {
            DType::F16
        } else {
            DType::F32
        };
        let fresh = candle_llama::Cache::new(true, dtype, &self.config, &self.device)?;
        *self.cache.lock() = fresh;
        Ok(())
    }

    // ── Per-sequence cache API ────────────────────────────────────────────────

    /// Allocate a fresh KV cache for one sequence.
    pub fn create_kv_cache(&self) -> Result<candle_llama::Cache> {
        let dtype = if self.device.is_cuda() {
            DType::F16
        } else {
            DType::F32
        };
        Ok(candle_llama::Cache::new(
            true,
            dtype,
            &self.config,
            &self.device,
        )?)
    }

    /// Run one forward step using an externally-owned per-sequence cache.
    ///
    /// The caller is responsible for passing the same cache object across
    /// successive steps for the same sequence.
    pub fn forward_with_cache(
        &self,
        token_ids: &[u32],
        seq_pos: usize,
        cache: &mut candle_llama::Cache,
    ) -> Result<Tensor> {
        let input = Tensor::new(token_ids, &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, seq_pos, cache)?;
        Ok(logits.squeeze(0)?)
    }
}
