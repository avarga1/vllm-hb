//! Microsoft Phi-3 architecture backend.
//!
//! # Status: STUB
//!
//! candle-transformers has `phi3::Model`.  Implementation follows the
//! same pattern as `arch/llama.rs`.
//!
//! Detected by: `model_type == "phi3"` in config.json.

#![allow(dead_code, unused_variables, unused_imports)]

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};

pub struct Phi3Backend;

impl Phi3Backend {
    pub fn load(
        _config_str: &str,
        _shards:     &[std::path::PathBuf],
        _dtype:      DType,
        _device:     &Device,
    ) -> Result<Self> {
        bail!(
            "Phi-3 is not yet supported. \
             See arch/phi3.rs for the implementation checklist."
        )
    }

    pub fn forward(&self, _token_ids: &[u32], _seq_pos: usize) -> Result<Tensor> {
        unreachable!("Phi3Backend::load always fails")
    }

    pub fn reset_cache(&self) -> Result<()> {
        unreachable!("Phi3Backend::load always fails")
    }
}
