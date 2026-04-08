//! Speculative decoding — draft model + rejection-sampling verification.
//!
//! # Algorithm (Leviathan et al. 2023)
//!
//! Each decode step:
//!
//! 1. **Draft phase** — run the small draft model K times autoregressively
//!    to produce K candidate tokens with their probability distributions.
//!
//! 2. **Verify phase** — run the large target model K+1 times sequentially
//!    (one call per candidate plus one bonus call).  *Sequential* verification
//!    is used rather than a single batched pass so the target's KV cache
//!    always ends at the correct position — no post-step cache fixup needed.
//!
//! 3. **Rejection sampling** — for each candidate i, accept with probability
//!    `min(1, q_i / p_i)` where `q_i` = target probability and `p_i` = draft
//!    probability of the candidate token.  On rejection, replace with a token
//!    drawn from the *correction* distribution `(q − p)⁺` (renormalised).
//!
//! 4. **Bonus token** — if all K candidates are accepted, sample one extra
//!    token from the target distribution at position `seq_pos + K`.
//!
//! Expected outcome: `1 + K·α` tokens accepted per step on average (α =
//! acceptance rate), versus 1 token per step for greedy decoding.  Output
//! distribution is identical to the target model's.
//!
//! # Cache management
//!
//! Per-sequence draft KV caches are maintained across steps so the draft
//! model keeps its full context:
//!
//! - **Full accept** (all K+1 tokens accepted): draft cache is K tokens ahead
//!   of the new sequence position; one extra forward pass syncs it.
//! - **Partial accept** with *external-cache* backends (Mixtral): a pre-draft
//!   snapshot (cheap Arc clone) is restored and re-fed with only the `j`
//!   accepted tokens — O(j ≤ K) draft passes.
//! - **Partial accept** with *internal-cache* backends (Llama, Qwen2/3): the
//!   draft cache is rebuilt from the last `DRAFT_FALLBACK_WINDOW` context
//!   tokens — correct but O(WINDOW) draft passes.
//!
//! The target cache never needs reconciliation: sequential verification
//! stops the loop at the exact rejection boundary.

use std::collections::HashMap;

use anyhow::Result;
use candle_core::{DType, Tensor};
use rand::distributions::WeightedIndex;
use rand::prelude::*;

use crate::engine::{Engine, PerSeqCache};
use crate::sampling::nucleus::{apply_temperature, nucleus_filter, renormalize, softmax_inplace};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Context window used when rebuilding the draft cache from scratch
/// (fallback for Llama / Qwen2/3 internal-cache backends).
///
/// Larger values → better draft prediction quality, higher per-step overhead.
const DRAFT_FALLBACK_WINDOW: usize = 128;

// ── SpeculativeDecoder ────────────────────────────────────────────────────────

/// Pairs a small draft model with the main target model for speculative decoding.
///
/// One `SpeculativeDecoder` is created at server start-up (if `--draft-model`
/// is supplied) and lives for the lifetime of the [`crate::worker::Worker`].
pub struct SpeculativeDecoder {
    pub draft_engine: Engine,
    /// Number of candidate tokens generated per speculative step (K).
    pub speculative_steps: usize,
    /// Per-sequence draft KV caches (seq_id → cache).
    draft_caches: HashMap<u64, PerSeqCache>,
}

impl SpeculativeDecoder {
    pub fn new(draft_engine: Engine, speculative_steps: usize) -> Self {
        Self {
            draft_engine,
            speculative_steps,
            draft_caches: HashMap::new(),
        }
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    /// Warm-start a draft cache for a newly admitted sequence.
    ///
    /// Runs the full prompt through the draft model so the first speculative
    /// step starts with a meaningful KV state (mirrors `step_prefill` on the
    /// target side).
    pub fn init_seq(&mut self, seq_id: u64, prompt_ids: &[u32]) -> Result<()> {
        let mut cache = self.draft_engine.create_kv_cache()?;
        if !prompt_ids.is_empty() {
            self.draft_engine
                .forward_with_cache(prompt_ids, 0, &mut cache)?;
        }
        self.draft_caches.insert(seq_id, cache);
        Ok(())
    }

    /// Drop the draft cache when a sequence finishes.
    pub fn remove_seq(&mut self, seq_id: u64) {
        self.draft_caches.remove(&seq_id);
    }

    // ── Speculative step ──────────────────────────────────────────────────────

    /// Run one speculative decode step.
    ///
    /// Drafts `K = speculative_steps` tokens, verifies them token-by-token
    /// against `target_engine` using rejection sampling, and returns the
    /// accepted tokens (between 1 and K+1 inclusive).
    ///
    /// # Arguments
    ///
    /// * `seq_id`        — Sequence identifier for draft cache lookup.
    /// * `context`       — Full token prefix: `prompt_ids ++ output_ids`.
    /// * `seq_pos`       — Absolute position of `context.last()` in the sequence.
    /// * `temperature`   — Sampling temperature (shared by draft and target).
    /// * `top_p`         — Nucleus filter threshold (shared by draft and target).
    /// * `target_engine` — The large target model.
    /// * `target_cache`  — The target's per-sequence KV cache (mutated in place).
    ///
    /// # Returns
    ///
    /// Non-empty `Vec<u32>` of accepted token ids to append to `output_ids`.
    #[allow(clippy::too_many_arguments)]
    pub fn step(
        &mut self,
        seq_id: u64,
        context: &[u32],
        seq_pos: usize,
        temperature: f32,
        top_p: f32,
        target_engine: &Engine,
        target_cache: &mut PerSeqCache,
    ) -> Result<Vec<u32>> {
        let k = self.speculative_steps;
        let last_token = *context
            .last()
            .ok_or_else(|| anyhow::anyhow!("speculative step: empty context"))?;

        // Remove draft cache from the map so we can borrow &self.draft_engine
        // simultaneously (different struct fields — allowed by Rust's field
        // borrow rules).
        let mut draft_cache = self
            .draft_caches
            .remove(&seq_id)
            .ok_or_else(|| anyhow::anyhow!("no draft cache for seq {seq_id}"))?;

        // Snapshot pre-draft state for cache reconciliation on partial accept.
        let snapshot = draft_cache.try_clone_external();

        // ── Draft phase ───────────────────────────────────────────────────────

        let mut draft_tokens: Vec<u32> = Vec::with_capacity(k);
        // Full probability distributions for each draft position (needed for
        // the correction term in rejection sampling).
        let mut draft_probs: Vec<Vec<f32>> = Vec::with_capacity(k);

        let mut prev = last_token;
        for i in 0..k {
            let logits =
                self.draft_engine
                    .forward_with_cache(&[prev], seq_pos + i, &mut draft_cache)?;
            let probs = logits_to_probs(&logits, temperature, top_p)?;
            let token = sample_probs(&probs)?;
            draft_tokens.push(token);
            draft_probs.push(probs);
            prev = token;
        }

        // ── Verify phase ──────────────────────────────────────────────────────
        //
        // Feed tokens to the target one at a time: input at position seq_pos+i
        // yields logits predicting draft_tokens[i].  We stop early on rejection
        // so the target cache ends at the correct accepted position.

        let mut accepted: Vec<u32> = Vec::with_capacity(k + 1);
        let mut all_accepted = true;
        let mut verify_in = last_token;

        for i in 0..k {
            let logits =
                target_engine.forward_with_cache(&[verify_in], seq_pos + i, target_cache)?;
            let tgt_probs = logits_to_probs(&logits, temperature, top_p)?;

            let p_i = draft_probs[i][draft_tokens[i] as usize].max(1e-10);
            let q_i = tgt_probs[draft_tokens[i] as usize];
            let accept_p = (q_i / p_i).min(1.0_f32);

            if rand::random::<f32>() < accept_p {
                accepted.push(draft_tokens[i]);
                verify_in = draft_tokens[i];
            } else {
                // Rejection: replace with sample from correction distribution.
                let corrected = sample_corrected(&tgt_probs, &draft_probs[i])?;
                accepted.push(corrected);
                all_accepted = false;
                break;
            }
        }

        // All K drafts accepted → sample one bonus token from target at seq_pos+K.
        if all_accepted {
            let bonus_logits =
                target_engine.forward_with_cache(&[verify_in], seq_pos + k, target_cache)?;
            let bonus_probs = logits_to_probs(&bonus_logits, temperature, top_p)?;
            let bonus = sample_probs(&bonus_probs)?;
            accepted.push(bonus);
        }

        // ── Draft cache reconciliation ────────────────────────────────────────

        let j = accepted.len(); // tokens accepted this step (1..=K+1)
        draft_cache = reconcile_draft_cache(
            &self.draft_engine,
            draft_cache,
            snapshot,
            context,
            last_token,
            seq_pos,
            &accepted,
            j,
            k,
        )?;

        self.draft_caches.insert(seq_id, draft_cache);
        Ok(accepted)
    }
}

// ── Cache reconciliation ──────────────────────────────────────────────────────

/// Reconcile the draft cache after a speculative step.
///
/// After generating K draft tokens the cache sits K positions ahead of the
/// last accepted token.  This function advances it to exactly the right place.
///
/// - **Full accept** (`j == K+1`): feed the bonus token (which only the
///   target generated) through the draft — O(1) draft pass.
/// - **Partial accept, snapshot available** (Mixtral external cache): restore
///   the pre-draft snapshot and re-feed `[last_token, accepted[0..j-1]]` —
///   O(j ≤ K) draft passes.
/// - **Partial accept, no snapshot** (Llama / Qwen2/3): rebuild from the last
///   `DRAFT_FALLBACK_WINDOW` tokens of the updated context — O(WINDOW) passes,
///   but correctness is guaranteed.
#[allow(clippy::too_many_arguments)]
fn reconcile_draft_cache(
    draft_engine: &Engine,
    mut draft_cache: PerSeqCache,
    snapshot: Option<PerSeqCache>,
    context: &[u32],
    last_token: u32,
    seq_pos: usize,
    accepted: &[u32],
    j: usize,
    k: usize,
) -> Result<PerSeqCache> {
    if j == k + 1 {
        // Full accept + bonus.  Draft processed K tokens (draft_0..draft_{K-1})
        // but not the bonus.  Feed bonus at seq_pos+K to sync.
        let bonus = accepted[k];
        draft_engine.forward_with_cache(&[bonus], seq_pos + k, &mut draft_cache)?;
        return Ok(draft_cache);
    }

    // Partial accept (j ≤ K): draft cache has K stale entries beyond the
    // accepted prefix.
    match snapshot {
        Some(mut snap) => {
            // Restore pre-draft state (snapshot was at seq_pos entries).
            // Re-feed [last_token, accepted_0, …, accepted_{j-2}] so the
            // cache advances to seq_pos + j entries.
            let replay: Vec<u32> = std::iter::once(last_token)
                .chain(accepted[..j.saturating_sub(1)].iter().copied())
                .collect();
            if !replay.is_empty() {
                draft_engine.forward_with_cache(&replay, seq_pos, &mut snap)?;
            }
            Ok(snap)
        }
        None => {
            // Fallback: rebuild from the last DRAFT_FALLBACK_WINDOW tokens of
            // the fully-updated context (original prefix + accepted tokens).
            let new_ctx: Vec<u32> = context.iter().chain(accepted.iter()).copied().collect();
            let start = new_ctx.len().saturating_sub(DRAFT_FALLBACK_WINDOW);
            let mut fresh = draft_engine.create_kv_cache()?;
            if start < new_ctx.len() {
                draft_engine.forward_with_cache(&new_ctx[start..], start, &mut fresh)?;
            }
            Ok(fresh)
        }
    }
}

// ── Probability helpers ───────────────────────────────────────────────────────

/// Convert raw logits to a probability distribution.
///
/// Handles both `[vocab]` (1-D) and `[1, vocab]` (2-D with a leading batch
/// or sequence dimension) tensors, applying temperature scaling, softmax, and
/// optional nucleus filtering.
///
/// When `temperature < 1e-6` (greedy), returns a delta distribution (prob 1.0
/// at the argmax) so that rejection sampling behaves deterministically.
fn logits_to_probs(logits: &Tensor, temperature: f32, top_p: f32) -> Result<Vec<f32>> {
    // Flatten any leading singleton dimensions → [vocab].
    let flat = if logits.rank() > 1 {
        let vocab = logits.elem_count();
        logits.reshape((vocab,))?
    } else {
        logits.clone()
    };

    let flat = flat.to_dtype(DType::F32)?;
    let mut probs: Vec<f32> = flat.to_vec1()?;

    if temperature < 1e-6 {
        // Greedy: delta distribution at argmax.
        let best = probs
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        probs.iter_mut().enumerate().for_each(|(i, p)| {
            *p = if i == best { 1.0 } else { 0.0 };
        });
        return Ok(probs);
    }

    apply_temperature(&mut probs, temperature);
    softmax_inplace(&mut probs);

    if top_p < 1.0 - 1e-6 {
        nucleus_filter(&mut probs, top_p);
        renormalize(&mut probs);
    }

    Ok(probs)
}

/// Sample one token from a probability vector.
fn sample_probs(probs: &[f32]) -> Result<u32> {
    let dist = WeightedIndex::new(probs).map_err(|e| anyhow::anyhow!("WeightedIndex: {e}"))?;
    Ok(dist.sample(&mut rand::thread_rng()) as u32)
}

/// Sample from the correction distribution `max(0, q − p)` (renormalised).
///
/// Used when a draft token is rejected so that the combined output
/// distribution matches the target's exactly.
fn sample_corrected(q: &[f32], p: &[f32]) -> Result<u32> {
    let mut correction: Vec<f32> = q
        .iter()
        .zip(p.iter())
        .map(|(qi, pi)| (qi - pi).max(0.0))
        .collect();
    let sum: f32 = correction.iter().sum();
    if sum < 1e-10 {
        // Degenerate (q ≤ p everywhere): fall back to the pure target distribution.
        return sample_probs(q);
    }
    correction.iter_mut().for_each(|x| *x /= sum);
    sample_probs(&correction)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    fn make_logits(vals: &[f32]) -> Tensor {
        Tensor::from_vec(vals.to_vec(), vals.len(), &Device::Cpu).unwrap()
    }

    #[test]
    fn logits_to_probs_sums_to_one() {
        let logits = make_logits(&[1.0, 2.0, 3.0, 0.5]);
        let probs = logits_to_probs(&logits, 1.0, 1.0).unwrap();
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "probs sum = {sum}");
    }

    #[test]
    fn logits_to_probs_greedy_is_delta() {
        let logits = make_logits(&[0.1, 5.0, 0.2, 0.3]);
        let probs = logits_to_probs(&logits, 0.0, 1.0).unwrap();
        // Argmax is index 1.
        assert_eq!(probs[1], 1.0);
        assert_eq!(probs.iter().filter(|&&p| p == 0.0).count(), 3);
    }

    #[test]
    fn logits_to_probs_2d_batch_dim() {
        // Simulate [1, vocab] tensor from candle_transformers.
        let vals = vec![1.0f32, 2.0, 3.0];
        let logits = Tensor::from_vec(vals.clone(), (1, vals.len()), &Device::Cpu).unwrap();
        let probs = logits_to_probs(&logits, 1.0, 1.0).unwrap();
        assert_eq!(probs.len(), 3);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn sample_corrected_returns_valid_token() {
        // q is slightly higher on token 2; p is higher on token 0.
        let q = vec![0.1, 0.2, 0.7];
        let p = vec![0.6, 0.3, 0.1];
        let token = sample_corrected(&q, &p).unwrap();
        // Correction = max(0, q-p) = [0, 0, 0.6] → only token 2 survives.
        assert_eq!(token, 2);
    }

    #[test]
    fn sample_corrected_degenerate_falls_back_to_q() {
        // q ≤ p everywhere → correction sums to zero → fall back to q.
        let q = vec![0.1, 0.3, 0.6];
        let p = vec![0.5, 0.4, 0.7]; // p ≥ q at all positions
        // Should not panic; result is a valid index.
        let token = sample_corrected(&q, &p).unwrap();
        assert!(token < 3);
    }

    #[test]
    fn try_clone_external_mixtral() {
        let cache = PerSeqCache::Mixtral(vec![None; 4]);
        let snap = cache.try_clone_external();
        assert!(snap.is_some());
    }

    #[test]
    fn try_clone_external_llama_tp_is_none() {
        let cache = PerSeqCache::LlamaTp;
        assert!(cache.try_clone_external().is_none());
    }
}
