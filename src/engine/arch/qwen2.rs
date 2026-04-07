//! Qwen 2 / 2.5 architecture backend.
//!
//! # Status: STUB
//!
//! candle-transformers has a `qwen2::Model`.  Implementation follows
//! the same pattern as `arch/llama.rs` — verify the `Cache::new` and
//! `Model::forward` signatures match, then wire them in.
//!
//! Detected by: `model_type == "qwen2"` in config.json.
//!
//! Tested models: Qwen2.5-7B-Instruct, Qwen2.5-32B-Instruct.

#![allow(dead_code, unused_variables, unused_imports)]

use anyhow::{Result, bail};
use candle_core::{DType, Device, Tensor};

pub struct Qwen2Backend;

impl Qwen2Backend {
    pub fn load(
        _config_str: &str,
        _shards: &[std::path::PathBuf],
        _dtype: DType,
        _device: &Device,
    ) -> Result<Self> {
        bail!(
            "Qwen2 is not yet supported. \
             See arch/qwen2.rs for the implementation checklist."
        )
    }

    pub fn forward(&self, _token_ids: &[u32], _seq_pos: usize) -> Result<Tensor> {
        unreachable!("Qwen2Backend::load always fails")
    }

    pub fn reset_cache(&self) -> Result<()> {
        unreachable!("Qwen2Backend::load always fails")
    }

    pub fn create_kv_cache(&self) -> Vec<Option<(candle_core::Tensor, candle_core::Tensor)>> {
        unreachable!("Qwen2Backend::load always fails")
    }

    pub fn forward_with_cache(
        &self,
        _token_ids: &[u32],
        _seq_pos: usize,
        _cache: &mut Vec<Option<(candle_core::Tensor, candle_core::Tensor)>>,
    ) -> Result<Tensor> {
        unreachable!("Qwen2Backend::load always fails")
    }
}
