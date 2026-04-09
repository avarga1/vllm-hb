//! Per-token log-probability collection.
//!
//! When the client sets `logprobs: true` (and optionally `top_logprobs: N`),
//! the sampler records the log-probability of the chosen token plus the top-N
//! alternatives at each step.  These are attached to the response choice.
//!
//! # Design
//!
//! `LogprobCollector` is created per-sequence when `logprobs` is requested.
//! After each sample the worker calls `record(token_id, probs, decode_fn)`.
//! At the end `finish()` returns the accumulated `Vec<LogprobContent>` which
//! the handler attaches to the response `Choice`.

use serde::Serialize;

/// Log-probability entry for a single token alternative.
#[derive(Debug, Clone, Serialize)]
pub struct TokenLogprob {
    /// The token text as decoded by the tokenizer.
    pub token: String,
    /// Natural log of the sampling probability at this position.
    pub logprob: f32,
    /// UTF-8 byte values of the token (matches OpenAI spec).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
}

/// Per-position log-probability data returned in the response.
#[derive(Debug, Clone, Serialize)]
pub struct LogprobContent {
    /// The token that was actually sampled.
    pub token: String,
    /// Log-probability of the sampled token.
    pub logprob: f32,
    /// UTF-8 byte values.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
    /// Top-N alternative tokens at this position.  Empty when
    /// `top_logprobs` was `0` or not set.
    pub top_logprobs: Vec<TokenLogprob>,
}

/// Accumulates per-token logprob data over the course of one generation.
///
/// Create with [`LogprobCollector::new`] when `logprobs: true`.
/// Call [`LogprobCollector::record`] after each sample.
/// Call [`LogprobCollector::finish`] to retrieve the full list.
pub struct LogprobCollector {
    /// How many top alternatives to record per position (0 = chosen only).
    top_n: usize,
    /// Accumulated per-token entries.
    entries: Vec<LogprobContent>,
}

impl LogprobCollector {
    /// Create a new collector.
    ///
    /// `top_n` is clamped to `[0, 20]` (OpenAI maximum).
    pub fn new(top_n: usize) -> Self {
        Self {
            top_n: top_n.min(20),
            entries: Vec::new(),
        }
    }

    /// Record the sampled token and the full probability distribution.
    ///
    /// - `token_id`  — the token that was sampled
    /// - `probs`     — vocab-length slice, already softmax-normalised
    /// - `decode`    — closure mapping a token id to its display text
    pub fn record(&mut self, token_id: u32, probs: &[f32], decode: impl Fn(u32) -> String) {
        let token_text = decode(token_id);
        let token_bytes = text_bytes(&token_text);

        // log-probability of the chosen token (floor at a tiny value to avoid -inf).
        let chosen_prob = probs.get(token_id as usize).copied().unwrap_or(0.0);
        let chosen_logprob = chosen_prob.max(f32::MIN_POSITIVE).ln();

        // Top-N alternatives (excluding the chosen token to match OAI spec).
        let top_logprobs = if self.top_n > 0 {
            // Collect (id, logprob) pairs, sort descending.
            let mut ranked: Vec<(u32, f32)> = probs
                .iter()
                .enumerate()
                .map(|(id, &p)| (id as u32, p.max(f32::MIN_POSITIVE).ln()))
                .collect();
            ranked.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            ranked
                .into_iter()
                .filter(|&(id, _)| id != token_id)
                .take(self.top_n)
                .map(|(id, lp)| {
                    let t = decode(id);
                    let b = text_bytes(&t);
                    TokenLogprob {
                        token: t,
                        logprob: lp,
                        bytes: b,
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

        self.entries.push(LogprobContent {
            token: token_text,
            logprob: chosen_logprob,
            bytes: token_bytes,
            top_logprobs,
        });
    }

    /// Consume the collector and return the accumulated entries.
    pub fn finish(self) -> Vec<LogprobContent> {
        self.entries
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Return the UTF-8 byte values of a string, or `None` if it is valid ASCII
/// that fits in `[u8]` trivially.  OpenAI always emits this field; we follow.
fn text_bytes(s: &str) -> Option<Vec<u8>> {
    Some(s.as_bytes().to_vec())
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn decode(id: u32) -> String {
        format!("tok{id}")
    }

    #[test]
    fn records_chosen_token() {
        let mut c = LogprobCollector::new(0);
        let probs = vec![0.1_f32, 0.7, 0.2];
        c.record(1, &probs, decode);
        let entries = c.finish();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].token, "tok1");
        assert!((entries[0].logprob - 0.7_f32.ln()).abs() < 1e-5);
        assert!(entries[0].top_logprobs.is_empty());
    }

    #[test]
    fn top_n_excludes_chosen() {
        let mut c = LogprobCollector::new(2);
        let probs = vec![0.6_f32, 0.3, 0.1];
        c.record(0, &probs, decode);
        let entries = c.finish();
        // top_logprobs should NOT contain token 0 (the chosen one)
        assert!(entries[0].top_logprobs.iter().all(|t| t.token != "tok0"));
        assert_eq!(entries[0].top_logprobs.len(), 2);
    }

    #[test]
    fn top_n_sorted_descending() {
        let mut c = LogprobCollector::new(3);
        let probs = vec![0.1_f32, 0.05, 0.8, 0.05];
        c.record(2, &probs, decode);
        let alts = &c.finish()[0].top_logprobs;
        // must be sorted highest logprob first
        for w in alts.windows(2) {
            assert!(w[0].logprob >= w[1].logprob);
        }
    }

    #[test]
    fn top_n_clamped_to_20() {
        let c = LogprobCollector::new(999);
        assert_eq!(c.top_n, 20);
    }

    #[test]
    fn multiple_records_accumulated() {
        let mut c = LogprobCollector::new(0);
        let probs = vec![0.5_f32, 0.5];
        c.record(0, &probs, decode);
        c.record(1, &probs, decode);
        assert_eq!(c.finish().len(), 2);
    }

    #[test]
    fn bytes_field_populated() {
        let mut c = LogprobCollector::new(0);
        c.record(0, &[1.0], |_| "hi".to_string());
        let entries = c.finish();
        assert_eq!(entries[0].bytes, Some(vec![b'h', b'i']));
    }
}
