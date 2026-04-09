//! Internal inference pipeline types.
//!
//! No HTTP concerns here — these are the types that flow between the
//! worker, engine, and sampling layers.

use serde::Serialize;
use tokio::sync::mpsc;

// ── Sampling parameters ───────────────────────────────────────────────────────

/// Controls how tokens are selected during generation.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    /// Stop strings — generation halts when any is matched in the output.
    pub stop: Vec<String>,
    /// Optional RNG seed for reproducible sampling.  `None` uses the global
    /// thread-local RNG.
    pub seed: Option<u64>,
    /// When `true`, per-token log-probabilities are collected and returned.
    pub logprobs: bool,
    /// Number of top-alternative log-probabilities per position.  Only
    /// meaningful when `logprobs` is `true`.  Clamped to `[0, 20]`.
    pub top_logprobs: u8,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            temperature: 0.7,
            top_p: 1.0,
            stop: Vec::new(),
            seed: None,
            logprobs: false,
            top_logprobs: 0,
        }
    }
}

// ── Generation lifecycle ──────────────────────────────────────────────────────

/// A single token emitted during streaming generation.
#[derive(Debug, Clone)]
pub struct TokenEvent {
    #[allow(dead_code)] // id is emitted on the channel; currently only text is read by handlers
    pub id: u32,
    pub text: String,
}

/// Final statistics emitted when generation completes.
///
/// `ttft_ms`, `total_ms`, and `tokens_per_sec` are recorded per-request
/// and will be surfaced via the Prometheus metrics endpoint (see
/// `server/metrics.rs`).  Suppressed until that feature lands.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub ttft_ms: u64,
    pub total_ms: u64,
    pub tokens_per_sec: f64,
}

/// The result stream a handler receives for one request.
pub enum GenerationEvent {
    Token(TokenEvent),
    Finished {
        finish_reason: FinishReason,
        stats: GenerationStats,
        /// Per-token log-probability data.  `None` when `logprobs` was not
        /// requested.
        logprobs: Option<Vec<crate::sampling::logprobs::LogprobContent>>,
    },
    Error(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
}

impl FinishReason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
        }
    }
}

// ── Worker channel ────────────────────────────────────────────────────────────

/// A unit of work submitted to the inference worker.
pub struct WorkItem {
    pub id: String,
    pub token_ids: Vec<u32>,
    pub params: SamplingParams,
    pub result_tx: mpsc::UnboundedSender<GenerationEvent>,
}
