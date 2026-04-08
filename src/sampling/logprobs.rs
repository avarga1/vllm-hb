//! Per-token log-probability collection.
//!
//! When the client sets `logprobs: true` (and optionally `top_logprobs: N`),
//! the sampler records the log-probability of the chosen token plus the top-N
//! alternatives at each step.  These are attached to the response choice.
//!
//! # Stub status
//! Types and interface are final; collection logic is `todo!()`.
//! See issue #23.

use serde::Serialize;

/// Log-probability entry for a single token.
#[derive(Debug, Clone, Serialize)]
pub struct TokenLogprob {
    /// The token text as decoded by the tokenizer.
    pub token: String,
    /// Natural log of the sampling probability at this position.
    pub logprob: f32,
    /// UTF-8 byte values of the token (matches OpenAI spec).
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
    pub bytes: Option<Vec<u8>>,
    /// Top-N alternative tokens at this position.  Empty when
    /// `top_logprobs` was `0` or not set.
    pub top_logprobs: Vec<TokenLogprob>,
}

/// Accumulates per-token logprob data over the course of one generation.
///
/// Attach to a sequence when `logprobs: true`; call [`LogprobCollector::record`]
/// after each sample; retrieve the full list with [`LogprobCollector::finish`].
pub struct LogprobCollector {
    /// How many top alternatives to record per position (0 = chosen token only).
    top_n: usize,
    /// Accumulated per-token entries.
    _entries: Vec<LogprobContent>,
}

impl LogprobCollector {
    /// Create a new collector.
    ///
    /// `top_n` must be in `0..=20` (OpenAI limit).
    pub fn new(top_n: usize) -> Self {
        Self {
            top_n: top_n.min(20),
            _entries: Vec::new(),
        }
    }

    /// Record the sampled token and the full probability distribution at this
    /// position.
    ///
    /// `probs` must be vocab-length and already softmax-normalised.
    /// `token_id` is the index that was sampled.
    /// `decode` is a closure that converts a token id to its text.
    pub fn record(&mut self, _token_id: u32, _probs: &[f32], _decode: impl Fn(u32) -> String) {
        let _ = self.top_n; // used once real impl lands
        todo!("issue #23 — logprob collection")
    }

    /// Consume the collector and return the accumulated entries.
    pub fn finish(self) -> Vec<LogprobContent> {
        todo!("issue #23 — logprob collection")
    }
}
