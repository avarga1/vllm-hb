//! Token sampling — temperature scaling + top-p nucleus filtering.
//!
//! All operations are pure functions over a logit slice; no state.

use anyhow::Result;
use candle_core::{DType, Tensor};
use rand::distributions::WeightedIndex;
use rand::prelude::*;

/// Sample one token from `logits` using temperature and top-p.
///
/// - `temperature = 0`  → greedy argmax (deterministic)
/// - `top_p = 1.0`      → no nucleus filtering (pure temperature)
/// - `top_p < 1.0`      → keep only the smallest set of tokens whose
///                         cumulative probability ≥ top_p, then sample
pub fn sample(logits: &Tensor, temperature: f32, top_p: f32) -> Result<u32> {
    // Greedy path — avoids allocation entirely.
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

// ── Private helpers ───────────────────────────────────────────────────────────

fn greedy(logits: &Tensor) -> Result<u32> {
    Ok(logits.argmax(candle_core::D::Minus1)?.to_scalar::<u32>()?)
}

fn apply_temperature(probs: &mut [f32], temperature: f32) {
    let inv_temp = 1.0 / temperature;
    for p in probs.iter_mut() {
        *p *= inv_temp;
    }
}

fn softmax_inplace(probs: &mut [f32]) {
    let max = probs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for p in probs.iter_mut() {
        *p = (*p - max).exp();
        sum += *p;
    }
    let inv_sum = 1.0 / sum;
    for p in probs.iter_mut() {
        *p *= inv_sum;
    }
}

/// Zero out tokens outside the top-p nucleus.
fn nucleus_filter(probs: &mut [f32], top_p: f32) {
    // Sort indices by descending probability.
    let mut indices: Vec<usize> = (0..probs.len()).collect();
    indices.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

    let mut cumulative = 0.0_f32;
    let mut cutoff = probs.len(); // index after which we zero

    for (rank, &idx) in indices.iter().enumerate() {
        cumulative += probs[idx];
        if cumulative >= top_p {
            cutoff = rank + 1;
            break;
        }
    }

    // Zero out everything outside the nucleus.
    for &idx in &indices[cutoff..] {
        probs[idx] = 0.0;
    }
}

fn renormalize(probs: &mut [f32]) {
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for p in probs.iter_mut() {
            *p *= inv;
        }
    }
}

fn weighted_sample(probs: &[f32]) -> Result<u32> {
    let dist = WeightedIndex::new(probs)
        .map_err(|e| anyhow::anyhow!("Invalid probability distribution: {e}"))?;
    Ok(dist.sample(&mut rand::thread_rng()) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one() {
        let mut v = vec![1.0_f32, 2.0, 3.0, 4.0];
        softmax_inplace(&mut v);
        let sum: f32 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum = {sum}");
    }

    #[test]
    fn nucleus_filter_keeps_top_mass() {
        let mut v = vec![0.6_f32, 0.3, 0.05, 0.05]; // already a distribution
        nucleus_filter(&mut v, 0.9);
        // top-2 cover 0.9; bottom-2 should be zeroed
        assert_eq!(v[2], 0.0);
        assert_eq!(v[3], 0.0);
        assert!(v[0] > 0.0);
        assert!(v[1] > 0.0);
    }
}
