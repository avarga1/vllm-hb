//! Qwen 2 / 2.5 architecture backend.
//!
//! Uses `candle_transformers::models::qwen2` — detected by `model_type == "qwen2"`.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen2;
use parking_lot::Mutex;

pub struct Qwen2Backend {
    model: Mutex<qwen2::ModelForCausalLM>,
    device: Device,
}

impl Qwen2Backend {
    pub fn load(
        config_str: &str,
        shards: &[std::path::PathBuf],
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let cfg: qwen2::Config =
            serde_json::from_str(config_str).context("Parsing config.json as Qwen2Config")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(shards, dtype, device)? };
        let model = qwen2::ModelForCausalLM::new(&cfg, vb)?;
        Ok(Self {
            model: Mutex::new(model),
            device: device.clone(),
        })
    }

    pub fn forward(&self, token_ids: &[u32], seq_pos: usize) -> Result<Tensor> {
        let input = Tensor::new(token_ids, &self.device)?.unsqueeze(0)?;
        let logits = self.model.lock().forward(&input, seq_pos)?;
        Ok(logits.squeeze(0)?)
    }

    pub fn reset_cache(&self) -> Result<()> {
        self.model.lock().clear_kv_cache();
        Ok(())
    }

    pub fn create_kv_cache(&self) -> Vec<Option<(Tensor, Tensor)>> {
        // Clear the model's internal KV cache so each new sequence starts
        // fresh.  Without this, entries from the previous request bleed into
        // the next prefill, causing an attention-mask shape mismatch.
        self.model.lock().clear_kv_cache();
        vec![]
    }

    pub fn forward_with_cache(
        &self,
        token_ids: &[u32],
        seq_pos: usize,
        _cache: &mut [Option<(Tensor, Tensor)>],
    ) -> Result<Tensor> {
        self.forward(token_ids, seq_pos)
    }
}
