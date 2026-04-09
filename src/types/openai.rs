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

use crate::sampling::logprobs::LogprobContent;

// ── Tool call types (issue #22) ───────────────────────────────────────────────

/// A function definition the model may choose to call.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct FunctionDef {
    /// Name of the function.
    pub name: String,
    /// Human-readable description of what the function does.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// JSON Schema object describing the function parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

/// A tool available to the model — currently only `"function"` type.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Tool {
    /// Always `"function"` for now.
    #[serde(rename = "type")]
    pub tool_type: String,
    /// The function specification.
    pub function: FunctionDef,
}

/// Controls which tool (if any) the model must call.
///
/// - `"none"` — model must not call any tool
/// - `"auto"` — model decides (default when tools are provided)
/// - `{"type":"function","function":{"name":"…"}}` — force a specific function
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(untagged)]
pub enum ToolChoice {
    /// `"none"` or `"auto"`.
    Mode(String),
    /// Force a specific function: `{"type":"function","function":{"name":"…"}}`.
    Specific {
        #[serde(rename = "type")]
        tool_type: String,
        function: ToolChoiceFunction,
    },
}

/// Name selector inside a forced [`ToolChoice::Specific`].
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ToolChoiceFunction {
    /// Name of the function to force.
    pub name: String,
}

/// A tool call emitted by the model in a response or streaming delta.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ToolCall {
    /// Unique identifier for this call, e.g. `"call_abc123"`.
    pub id: String,
    /// Always `"function"` for now.
    #[serde(rename = "type")]
    pub tool_type: String,
    /// The function invocation.
    pub function: FunctionCall,
}

/// The function name + JSON-encoded arguments produced by the model.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct FunctionCall {
    /// Name of the function called.
    pub name: String,
    /// JSON string of the arguments object.
    pub arguments: String,
}

// ── Request ───────────────────────────────────────────────────────────────────

/// `POST /v1/chat/completions` request body.
///
/// All fields match the OpenAI Chat Completions API.  Optional fields have
/// sensible server-side defaults (see [`defaults`]).
// Stub fields for in-progress features (#21–#24) are not yet consumed by
// handlers; suppress the lint until each feature is implemented.
#[allow(dead_code)]
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

    // ── Sampling extras (issue #21, #23, #24) ────────────────────────────────
    /// One or more strings that halt generation when produced in the output.
    /// Up to 4 stop strings (OpenAI limit).  Ignored when empty.
    #[serde(default)]
    pub stop: Vec<String>,

    /// If set, deterministic sampling is seeded with this value.
    /// Same seed + same input = same output (barring floating-point variance
    /// across hardware).  See issue #23.
    #[serde(default)]
    pub seed: Option<u64>,

    /// When `true`, the response includes per-token log-probabilities.
    /// See issue #23.
    #[serde(default)]
    pub logprobs: bool,

    /// Number of top-alternative log-probabilities to return per token.
    /// Only meaningful when `logprobs` is `true`.  Clamped to `[0, 20]`.
    #[serde(default)]
    pub top_logprobs: Option<u8>,

    /// Penalise tokens that have already appeared in the output — reduces
    /// repetition.  Range `[-2, 2]`.  Defaults to `0` (no penalty).
    /// See issue #24.
    #[serde(default)]
    pub presence_penalty: f32,

    /// Penalise tokens proportionally to how often they have appeared —
    /// stronger repetition suppression than `presence_penalty`.
    /// Range `[-2, 2]`.  Defaults to `0`.  See issue #24.
    #[serde(default)]
    pub frequency_penalty: f32,

    // ── Tool calls (issue #22) ────────────────────────────────────────────────
    /// List of tools (functions) the model may call.  When provided the model
    /// can respond with a [`ToolCall`] instead of plain text.
    #[serde(default)]
    pub tools: Vec<Tool>,

    /// Controls whether and which tool the model must call.  Defaults to
    /// `"auto"` when `tools` is non-empty, `"none"` otherwise.
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,
}

/// A single turn in a conversation.
///
/// Used in both request (`messages`) and non-streaming response (`choice.message`).
/// Roles: `"system"`, `"user"`, `"assistant"`, `"tool"`.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    /// Speaker role: `"system"`, `"user"`, `"assistant"`, or `"tool"`.
    pub role: String,
    /// Text content of the message.  For `"tool"` messages this is the
    /// function return value serialised as a string.
    pub content: String,
    /// Tool calls emitted by the assistant in this turn.  Present only on
    /// `"assistant"` messages that triggered a function call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// Tool call id this message is responding to.  Present only on
    /// `"tool"` role messages.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
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
    /// Why generation stopped: `"stop"`, `"length"`, or `"tool_calls"`.
    pub finish_reason: &'static str,
    /// Per-token log-probability data.  `None` when `logprobs` was not
    /// requested.  See issue #23.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChoiceLogprobs>,
}

/// Log-probability payload attached to a non-streaming [`Choice`].
#[derive(Debug, Serialize)]
pub struct ChoiceLogprobs {
    /// One entry per generated token.
    pub content: Vec<LogprobContent>,
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
/// Tool-call chunks carry `tool_calls` with partial JSON arguments.
#[derive(Debug, Serialize)]
pub struct Delta {
    /// Present only on the first chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    /// Present on every chunk that carries a generated token.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Streaming tool call deltas.  Each chunk appends partial `arguments`
    /// JSON; the full call is assembled client-side.  See issue #22.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

/// Partial tool call update inside a streaming [`Delta`].
#[derive(Debug, Serialize)]
pub struct ToolCallDelta {
    /// Index of the tool call being streamed (models can emit multiple calls).
    pub index: usize,
    /// Present on the first delta for this call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Always `"function"` when present.
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub tool_type: Option<&'static str>,
    /// Partial function name + arguments fragment.
    pub function: FunctionCallDelta,
}

/// Partial function call content inside a [`ToolCallDelta`].
#[derive(Debug, Serialize)]
pub struct FunctionCallDelta {
    /// Present only on the first delta for this call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Partial JSON arguments string — append successive deltas to reconstruct.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
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
                message: ChatMessage {
                    role: "assistant".into(),
                    content: "hi".into(),
                    tool_calls: None,
                    tool_call_id: None,
                },
                finish_reason: "stop",
                logprobs: None,
            }],
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            },
        };
        let v: serde_json::Value = serde_json::to_value(&resp).unwrap();
        assert_eq!(v["usage"]["total_tokens"], 15);
        assert_eq!(v["usage"]["prompt_tokens"], 10);
        assert_eq!(v["usage"]["completion_tokens"], 5);
        assert_eq!(v["object"], "chat.completion");
    }

    #[test]
    fn delta_skips_none_fields() {
        let delta = Delta {
            role: None,
            content: Some("hello".into()),
            tool_calls: None,
        };
        let v: serde_json::Value = serde_json::to_value(&delta).unwrap();
        assert!(v.get("role").is_none(), "role should be absent when None");
        assert_eq!(v["content"], "hello");
    }

    #[test]
    fn delta_role_only_on_first_chunk() {
        let delta = Delta {
            role: Some("assistant"),
            content: None,
            tool_calls: None,
        };
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
