//! Token sampling — temperature scaling + top-p nucleus filtering.
//!
//! # Modules
//! - `nucleus`  — the math: apply_temperature, softmax, nucleus_filter, renormalize
//! - `stop`     — stop-sequence matching (issue #21)
//! - `logprobs` — per-token log-probability collection (issue #23)
//! - `penalty`  — presence/frequency penalty application (issue #24)

pub mod logprobs;
pub mod nucleus;
pub mod penalty;
pub mod stop;

use anyhow::Result;
use candle_core::{DType, Tensor};
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use rand::rngs::SmallRng;

use nucleus::{apply_temperature, nucleus_filter, renormalize, softmax_inplace};

/// Sample one token from `logits` using temperature and top-p.
///
/// - `temperature = 0`  → greedy argmax (deterministic, zero allocation)
/// - `top_p = 1.0`      → no nucleus filtering (pure temperature sampling)
/// - `top_p < 1.0`      → nucleus filtering before sampling
///
/// Uses the global thread-local RNG.  For reproducible sampling use
/// [`sample_seeded`] with an explicit seed.
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

    weighted_sample_rng(&probs, &mut rand::thread_rng())
}

/// Sample with a seeded RNG for reproducible output.
///
/// Same seed + same logits + same parameters = same token every time
/// (barring floating-point variance across hardware).
///
/// Uses `SmallRng` (xoshiro256++) which is fast and non-cryptographic —
/// appropriate for inference sampling.
pub fn sample_seeded(logits: &Tensor, temperature: f32, top_p: f32, seed: u64) -> Result<u32> {
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

    let mut rng = SmallRng::seed_from_u64(seed);
    weighted_sample_rng(&probs, &mut rng)
}

/// Compute the softmax-normalised probability distribution from `logits`.
///
/// Used by the logprob collector which needs the full distribution, not just
/// the sampled token.
pub fn logits_to_probs(logits: &Tensor, temperature: f32, top_p: f32) -> Result<Vec<f32>> {
    if temperature < 1e-6 {
        // Greedy: delta distribution over the argmax token.
        let id = logits.argmax(candle_core::D::Minus1)?.to_scalar::<u32>()? as usize;
        let n = logits.elem_count();
        let mut probs = vec![0.0_f32; n];
        if id < n {
            probs[id] = 1.0;
        }
        return Ok(probs);
    }

    let logits = logits.to_dtype(DType::F32)?;
    let mut probs: Vec<f32> = logits.to_vec1()?;
    apply_temperature(&mut probs, temperature);
    softmax_inplace(&mut probs);
    if top_p < 1.0 - 1e-6 {
        nucleus_filter(&mut probs, top_p);
        renormalize(&mut probs);
    }
    Ok(probs)
}

// ── Private ───────────────────────────────────────────────────────────────────

fn greedy(logits: &Tensor) -> Result<u32> {
    Ok(logits.argmax(candle_core::D::Minus1)?.to_scalar::<u32>()?)
}

fn weighted_sample_rng(probs: &[f32], rng: &mut impl Rng) -> Result<u32> {
    let dist = WeightedIndex::new(probs)
        .map_err(|e| anyhow::anyhow!("Invalid probability distribution: {e}"))?;
    Ok(dist.sample(rng) as u32)
}
