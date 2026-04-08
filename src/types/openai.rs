//! OpenAI-compatible wire types.
//!
//! These are serialized/deserialized directly from HTTP request/response
//! bodies.  No candle, no tokio — pure serde.
//!
//! # Compatibility
//!
//! Field names and structure follow the OpenAI Chat Completions API v1 so
//! that existing clients (LangChain, OpenAI Python SDK, etc.) work without
//! modification.

use serde::{Deserialize, Serialize};

// ── Request ───────────────────────────────────────────────────────────────────

/// `POST /v1/chat/completions` request body.
///
/// All fields match the OpenAI Chat Completions API.  Optional fields have
/// sensible server-side defaults (see [`defaults`]).
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    /// Model name sent by the client.  The server ignores this and always
    /// uses the loaded model; the value is echoed back in the response.
    pub model: String,

    /// Ordered list of conversation turns.  Must contain at least one message.
    pub messages: Vec<ChatMessage>,

    /// Maximum number of tokens to generate.  Defaults to 512.
    #[serde(default = "defaults::max_tokens")]
    pub max_tokens: usize,

    /// Sampling temperature in `[0, 2]`.  `0` → greedy decoding.
    /// Defaults to 0.7.
    #[serde(default = "defaults::temperature")]
    pub temperature: f32,

    /// Nucleus (top-p) sampling threshold in `(0, 1]`.  `1.0` disables
    /// nucleus filtering.  Defaults to 1.0.
    #[serde(default = "defaults::top_p")]
    pub top_p: f32,

    /// When `true`, tokens are returned as SSE events as they are generated.
    /// When `false` (default), the full response is returned after completion.
    #[serde(default)]
    pub stream: bool,
}

/// A single turn in a conversation.
///
/// Used in both request (`messages`) and non-streaming response (`choice.message`).
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    /// Speaker role: `"system"`, `"user"`, or `"assistant"`.
    pub role: String,
    /// Text content of the message.
    pub content: String,
}

// ── Non-streaming response ────────────────────────────────────────────────────

/// `POST /v1/chat/completions` non-streaming response body.
///
/// Returned when `stream: false` (the default).  Matches the OpenAI
/// `ChatCompletion` object shape.
#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    /// Unique request identifier, e.g. `"chatcmpl-abc123"`.
    pub id: String,
    /// Always `"chat.completion"`.
    pub object: &'static str,
    /// Unix timestamp (seconds) when the response was created.
    pub created: u64,
    /// Echoes back the `model` field from the request.
    pub model: String,
    /// Completed message choices.  Currently always a single-element vec.
    pub choices: Vec<Choice>,
    /// Token usage statistics for billing / observability.
    pub usage: Usage,
}

/// One completion alternative inside a [`ChatCompletionResponse`].
#[derive(Debug, Serialize)]
pub struct Choice {
    /// Zero-based index of this choice.
    pub index: usize,
    /// The generated assistant message.
    pub message: ChatMessage,
    /// Why generation stopped: `"stop"` (EOS token) or `"length"` (max tokens).
    pub finish_reason: &'static str,
}

/// Token counts for a single request.
#[derive(Debug, Serialize)]
pub struct Usage {
    /// Number of tokens in the input prompt.
    pub prompt_tokens: usize,
    /// Number of tokens generated.
    pub completion_tokens: usize,
    /// `prompt_tokens + completion_tokens`.
    pub total_tokens: usize,
}

// ── Streaming response (SSE) ──────────────────────────────────────────────────

/// One SSE event in a streaming `POST /v1/chat/completions` response.
///
/// Matches the OpenAI `ChatCompletionChunk` object.  Each event carries a
/// partial delta; the stream ends with the sentinel string `[DONE]`.
#[derive(Debug, Serialize)]
pub struct ChatCompletionChunk {
    /// Unique request identifier, shared across all chunks for one request.
    pub id: String,
    /// Always `"chat.completion.chunk"`.
    pub object: &'static str,
    /// Unix timestamp (seconds) when generation started.
    pub created: u64,
    /// Echoes back the `model` field from the request.
    pub model: String,
    /// Partial choices.  Currently always a single-element vec.
    pub choices: Vec<ChunkChoice>,
}

/// One partial choice inside a [`ChatCompletionChunk`].
#[derive(Debug, Serialize)]
pub struct ChunkChoice {
    /// Zero-based index.
    pub index: usize,
    /// The incremental content delta for this chunk.
    pub delta: Delta,
    /// `None` until the last chunk, then `"stop"` or `"length"`.
    pub finish_reason: Option<&'static str>,
}

/// Incremental content update inside a streaming chunk.
///
/// The first chunk carries `role: Some("assistant")` with no content.
/// Subsequent chunks carry `content: Some(token_text)` with no role.
#[derive(Debug, Serialize)]
pub struct Delta {
    /// Present only on the first chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    /// Present on every chunk that carries a generated token.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

// ── Defaults ──────────────────────────────────────────────────────────────────

mod defaults {
    pub fn max_tokens() -> usize {
        512
    }
    pub fn temperature() -> f32 {
        0.7
    }
    pub fn top_p() -> f32 {
        1.0
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn deserialize(json: &str) -> ChatCompletionRequest {
        serde_json::from_str(json).expect("valid JSON")
    }

    #[test]
    fn request_minimal_fields() {
        let req = deserialize(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#);
        assert_eq!(req.model, "m");
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, "user");
        assert_eq!(req.messages[0].content, "hi");
    }

    #[test]
    fn request_defaults_applied() {
        let req = deserialize(r#"{"model":"x","messages":[]}"#);
        assert_eq!(req.max_tokens, 512);
        assert!((req.temperature - 0.7).abs() < 1e-5);
        assert!((req.top_p - 1.0).abs() < 1e-5);
        assert!(!req.stream);
    }

    #[test]
    fn request_stream_true() {
        let req = deserialize(r#"{"model":"x","messages":[],"stream":true}"#);
        assert!(req.stream);
    }

    #[test]
    fn request_explicit_params() {
        let req = deserialize(
            r#"{"model":"m","messages":[],"max_tokens":100,"temperature":0.3,"top_p":0.9}"#,
        );
        assert_eq!(req.max_tokens, 100);
        assert!((req.temperature - 0.3).abs() < 1e-5);
        assert!((req.top_p - 0.9).abs() < 1e-5);
    }

    #[test]
    fn response_serializes_total_tokens() {
        let resp = ChatCompletionResponse {
            id: "id".into(),
            object: "chat.completion",
            created: 0,
            model: "m".into(),
            choices: vec![Choice {
                index: 0,
                message: ChatMessage { role: "assistant".into(), content: "hi".into() },
                finish_reason: "stop",
            }],
            usage: Usage { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
        };
        let v: serde_json::Value = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["usage"]["total_tokens"], 15);
        assert_eq!(v["usage"]["prompt_tokens"], 10);
        assert_eq!(v["usage"]["completion_tokens"], 5);
        assert_eq!(v["object"], "chat.completion");
    }

    #[test]
    fn delta_skips_none_fields() {
        let delta = Delta { role: None, content: Some("hello".into()) };
        let v: serde_json::Value = serde_json::to_value(&delta).unwrap();
        assert!(v.get("role").is_none(), "role should be absent when None");
        assert_eq!(v["content"], "hello");
    }

    #[test]
    fn delta_role_only_on_first_chunk() {
        let delta = Delta { role: Some("assistant"), content: None };
        let v: serde_json::Value = serde_json::to_value(&delta).unwrap();
        assert_eq!(v["role"], "assistant");
        assert!(v.get("content").is_none());
    }

    #[test]
    fn request_rejects_invalid_json() {
        let result: Result<ChatCompletionRequest, _> = serde_json::from_str("{bad json}");
        assert!(result.is_err());
    }
}
