//! Presence and frequency penalties.
//!
//! Applied to logits **before** temperature scaling and nucleus filtering so
//! the penalties interact naturally with the rest of the sampling pipeline.
//!
//! | Penalty            | Effect                                                 |
//! |--------------------|--------------------------------------------------------|
//! | `presence_penalty` | Subtract a fixed value from logits of tokens that have appeared at least once. Encourages the model to introduce new topics. |
//! | `frequency_penalty`| Subtract a value proportional to how many times each token has appeared. Stronger suppression of repeated tokens. |
//!
//! Both are in the range `[-2, 2]` (matching the OpenAI spec).  Negative
//! values encourage repetition; positive values discourage it.  `0.0` is a
//! no-op for both.
//!
//! # Formula (matches OpenAI)
//!
//! ```text
//! logit[t] -= presence_penalty  * (count[t] > 0 ? 1 : 0)
//!           + frequency_penalty * count[t] / total_tokens
//! ```

/// Apply presence and frequency penalties to a mutable logit slice.
///
/// `token_counts` maps token id → number of times it has appeared in the
/// output so far.  Tokens absent from the map are treated as having count 0.
///
/// No-op (fast path) when both penalties are `0.0`.
pub fn apply_penalties(
    logits: &mut [f32],
    token_counts: &[u32],
    presence_penalty: f32,
    frequency_penalty: f32,
) {
    if presence_penalty.abs() < 1e-6 && frequency_penalty.abs() < 1e-6 {
        return;
    }

    let total: u32 = token_counts.iter().sum();
    let total_f = total.max(1) as f32;

    for (id, logit) in logits.iter_mut().enumerate() {
        let count = token_counts.get(id).copied().unwrap_or(0);
        if count == 0 {
            continue;
        }
        *logit -= presence_penalty;
        *logit -= frequency_penalty * (count as f32 / total_f);
    }
}

/// Build a token-count vector from an output token id slice.
///
/// Returns a `Vec<u32>` indexed by token id where each entry is the number of
/// times that token has appeared.  The vector length is `vocab_size`; tokens
/// not present in `output_ids` have count 0.
pub fn count_tokens(output_ids: &[u32], vocab_size: usize) -> Vec<u32> {
    let mut counts = vec![0u32; vocab_size];
    for &id in output_ids {
        if (id as usize) < vocab_size {
            counts[id as usize] += 1;
        }
    }
    counts
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_penalties_are_noop() {
        let mut logits = vec![1.0_f32, 2.0, 3.0];
        let counts = vec![1u32, 0, 2];
        apply_penalties(&mut logits, &counts, 0.0, 0.0);
        assert_eq!(logits, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn presence_penalty_reduces_seen_tokens() {
        let mut logits = vec![1.0_f32, 2.0, 3.0];
        let counts = vec![1u32, 0, 1]; // tokens 0 and 2 seen
        apply_penalties(&mut logits, &counts, 0.5, 0.0);
        assert!((logits[0] - 0.5).abs() < 1e-5, "token 0 reduced");
        assert!((logits[1] - 2.0).abs() < 1e-5, "token 1 unchanged");
        assert!((logits[2] - 2.5).abs() < 1e-5, "token 2 reduced");
    }

    #[test]
    fn frequency_penalty_scales_with_count() {
        let mut logits = vec![5.0_f32, 5.0];
        // token 0 seen once, token 1 seen three times; total = 4
        let counts = vec![1u32, 3];
        apply_penalties(&mut logits, &counts, 0.0, 1.0);
        // logit[0] -= 1.0 * 1/4 = 0.25  →  4.75
        // logit[1] -= 1.0 * 3/4 = 0.75  →  4.25
        assert!((logits[0] - 4.75).abs() < 1e-5);
        assert!((logits[1] - 4.25).abs() < 1e-5);
    }

    #[test]
    fn both_penalties_combined() {
        let mut logits = vec![10.0_f32, 10.0];
        let counts = vec![2u32, 0];
        apply_penalties(&mut logits, &counts, 0.5, 0.5);
        // logit[0] -= 0.5 (presence) + 0.5 * 2/2 (freq) = 1.0  →  9.0
        assert!((logits[0] - 9.0).abs() < 1e-5);
        assert!((logits[1] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn unseen_tokens_not_penalised() {
        let mut logits = vec![3.0_f32, 3.0, 3.0];
        let counts = vec![0u32, 0, 0];
        apply_penalties(&mut logits, &counts, 1.0, 1.0);
        assert_eq!(logits, [3.0, 3.0, 3.0]);
    }

    #[test]
    fn count_tokens_counts_correctly() {
        let ids = vec![0u32, 1, 1, 2, 1];
        let counts = count_tokens(&ids, 4);
        assert_eq!(counts[0], 1);
        assert_eq!(counts[1], 3);
        assert_eq!(counts[2], 1);
        assert_eq!(counts[3], 0);
    }

    #[test]
    fn count_tokens_ignores_out_of_range() {
        let ids = vec![99u32];
        let counts = count_tokens(&ids, 4);
        assert_eq!(counts.len(), 4);
        assert!(counts.iter().all(|&c| c == 0));
    }
}
