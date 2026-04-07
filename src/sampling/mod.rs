//! Token sampling — temperature scaling + top-p nucleus filtering.
//!
//! # Modules
//! - `nucleus` — the math: apply_temperature, softmax, nucleus_filter, renormalize

pub mod nucleus;

use anyhow::Result;
use candle_core::{DType, Tensor};
use rand::distributions::WeightedIndex;
use rand::prelude::*;

use nucleus::{apply_temperature, nucleus_filter, renormalize, softmax_inplace};

/// Sample one token from `logits` using temperature and top-p.
///
/// - `temperature = 0`  → greedy argmax (deterministic, zero allocation)
/// - `top_p = 1.0`      → no nucleus filtering (pure temperature sampling)
/// - `top_p < 1.0`      → nucleus filtering before sampling
pub fn sample(logits: &Tensor, temperature: f32, top_p: f32) -> Result<u32> {
    if temperature < 1e-6 {
        return greedy(logits);
    }

    let logits = logits.to_dtype(DType::F32)?;
    let mut probs: Vec<f32> = logits.to_vec1()?;

    apply_temperature(&mut probs, temperature);
    softmax_inplace(&mut probs);

    if top_p < 1.0 - 1e-6 {
        nucleus_filter(&mut probs, top_p);
        renormalize(&mut probs);
    }

    weighted_sample(&probs)
}

// ── Private ───────────────────────────────────────────────────────────────────

fn greedy(logits: &Tensor) -> Result<u32> {
    Ok(logits.argmax(candle_core::D::Minus1)?.to_scalar::<u32>()?)
}

fn weighted_sample(probs: &[f32]) -> Result<u32> {
    let dist = WeightedIndex::new(probs)
        .map_err(|e| anyhow::anyhow!("Invalid probability distribution: {e}"))?;
    Ok(dist.sample(&mut rand::thread_rng()) as u32)
}
