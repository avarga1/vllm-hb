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
    pub max_tokens:  usize,
    pub temperature: f32,
    pub top_p:       f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self { max_tokens: 512, temperature: 0.7, top_p: 1.0 }
    }
}

// ── Generation lifecycle ──────────────────────────────────────────────────────

/// A single token emitted during streaming generation.
#[derive(Debug, Clone)]
pub struct TokenEvent {
    pub id:   u32,
    pub text: String,
}

/// Final statistics emitted when generation completes.
#[derive(Debug, Clone, Default)]
pub struct GenerationStats {
    pub prompt_tokens:     usize,
    pub completion_tokens: usize,
    pub ttft_ms:           u64,
    pub total_ms:          u64,
    pub tokens_per_sec:    f64,
}

/// The result stream a handler receives for one request.
pub enum GenerationEvent {
    Token(TokenEvent),
    Finished {
        finish_reason: FinishReason,
        stats:         GenerationStats,
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
            Self::Stop   => "stop",
            Self::Length => "length",
        }
    }
}

// ── Worker channel ────────────────────────────────────────────────────────────

/// A unit of work submitted to the inference worker.
pub struct WorkItem {
    pub id:        String,
    pub token_ids: Vec<u32>,
    pub params:    SamplingParams,
    pub result_tx: mpsc::UnboundedSender<GenerationEvent>,
}
