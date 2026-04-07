//! Shared types for the inference pipeline.

use serde::{Deserialize, Serialize};
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
#[derive(Debug, Clone)]
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

// ── OpenAI wire types ─────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model:      String,
    pub messages:   Vec<ChatMessage>,
    #[serde(default = "defaults::max_tokens")]
    pub max_tokens: usize,
    #[serde(default = "defaults::temperature")]
    pub temperature: f32,
    #[serde(default = "defaults::top_p")]
    pub top_p:      f32,
    #[serde(default)]
    pub stream:     bool,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    pub role:    String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id:      String,
    pub object:  &'static str,
    pub created: u64,
    pub model:   String,
    pub choices: Vec<Choice>,
    pub usage:   Usage,
}

#[derive(Debug, Serialize)]
pub struct Choice {
    pub index:         usize,
    pub message:       ChatMessage,
    pub finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
pub struct Usage {
    pub prompt_tokens:     usize,
    pub completion_tokens: usize,
    pub total_tokens:      usize,
}

/// SSE streaming chunk.
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    pub id:      String,
    pub object:  &'static str,
    pub created: u64,
    pub model:   String,
    pub choices: Vec<ChunkChoice>,
}

#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    pub index:         usize,
    pub delta:         Delta,
    pub finish_reason: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role:    Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

mod defaults {
    pub fn max_tokens()  -> usize { 512 }
    pub fn temperature() -> f32   { 0.7 }
    pub fn top_p()       -> f32   { 1.0 }
}
