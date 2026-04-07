//! Mixtral MoE architecture backend.
//!
//! # Status: STUB
//!
//! candle-transformers 0.9 ships a `mixtral::Model` whose KV cache is
//! stored as private fields inside each attention layer with no public
//! reset method.  Until candle exposes `Model::reset_kv_cache()` (or we
//! vendor + patch the module), we cannot efficiently reset state between
//! requests without reloading all weights from disk.
//!
//! ## Implementation plan
//!
//! Option A (preferred): submit a PR to candle-transformers adding
//! `pub fn reset_kv_cache(&mut self)` to `mixtral::Model`, then update
//! this file to call it in `reset_cache()`.
//!
//! Option B: vendor `candle-transformers/src/models/mixtral.rs` as a
//! local module, add the reset method, and point Cargo.toml `[patch]`
//! at the local fork.
//!
//! Detected by: `model_type == "mixtral"` in config.json.

#![allow(dead_code, unused_variables, unused_imports)]

use anyhow::{Result, bail};
use candle_core::{DType, Device, Tensor};

pub struct MixtralBackend;

impl MixtralBackend {
    pub fn load(
        _config_str: &str,
        _shards: &[std::path::PathBuf],
        _dtype: DType,
        _device: &Device,
    ) -> Result<Self> {
        bail!(
            "Mixtral is not yet supported. \
             Waiting on candle-transformers to expose a public KV cache \
             reset method. Track progress: \
             https://github.com/avarga1/vllm-hb/issues/1"
        )
    }

    pub fn forward(&self, _token_ids: &[u32], _seq_pos: usize) -> Result<Tensor> {
        unreachable!("MixtralBackend::load always fails")
    }

    pub fn reset_cache(&self) -> Result<()> {
        unreachable!("MixtralBackend::load always fails")
    }
}
