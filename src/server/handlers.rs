//! Axum request handlers.
//!
//! Three endpoints:
//!   `POST /v1/chat/completions`  — streaming + non-streaming generation
//!   `GET  /v1/models`            — model listing
//!   `GET  /health`               — liveness probe

use std::sync::Arc;

use axum::{
    Json,
    extract::State,
    http::StatusCode,
    response::sse::KeepAlive,
    response::{IntoResponse, Response, Sse},
};
use serde_json::json;
use tokio::sync::mpsc;
use uuid::Uuid;

use super::sse as sse_mod;
use super::{AppState, unix_now};
use crate::tokenize;
use crate::tools::format::{detect_format, inject_tools};
use crate::tools::parser::ToolCallParser;
use crate::types::openai::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, Choice, ChoiceLogprobs,
    CompletionChoice, CompletionRequest, CompletionResponse, EmbeddingObject, EmbeddingRequest,
    EmbeddingResponse, EmbeddingUsage, Usage,
};
use crate::types::pipeline::{
    FinishReason, GenerationEvent, GenerationStats, SamplingParams, WorkItem,
};

// ── Liveness ──────────────────────────────────────────────────────────────────

pub async fn health() -> impl IntoResponse {
    Json(json!({ "status": "ok" }))
}

// ── Model listing ─────────────────────────────────────────────────────────────

pub async fn list_models(State(s): State<Arc<AppState>>) -> impl IntoResponse {
    Json(json!({
        "object": "list",
        "data": [{
            "id":       s.model_name,
            "object":   "model",
            "created":  unix_now(),
            "owned_by": "vllm-hb",
        }]
    }))
}

// ── Chat completions ──────────────────────────────────────────────────────────

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    // Inject tool definitions into the system prompt when tools are provided.
    let messages = if req.tools.is_empty() {
        req.messages.clone()
    } else {
        let template = tokenize::load_chat_template(&state.model_path).unwrap_or_default();
        let fmt = detect_format(&template);
        let tool_system = inject_tools(&req.tools, fmt);
        prepend_system(req.messages.clone(), tool_system)
    };

    let prompt = match tokenize::apply_chat_template(&state.model_path, &messages) {
        Ok(p) => p,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, &e.to_string()),
    };

    let token_ids = match tokenize::encode(&state.tokenizer, &prompt) {
        Ok(ids) => ids,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, &e.to_string()),
    };

    let (event_tx, event_rx) = mpsc::unbounded_channel::<GenerationEvent>();
    let work = WorkItem {
        id: Uuid::new_v4().to_string(),
        token_ids,
        params: SamplingParams {
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            top_p: req.top_p,
            stop: req.stop.clone(),
            seed: req.seed,
            logprobs: req.logprobs,
            top_logprobs: req.top_logprobs.unwrap_or(0),
            has_tools: !req.tools.is_empty(),
            presence_penalty: req.presence_penalty,
            frequency_penalty: req.frequency_penalty,
        },
        result_tx: event_tx,
    };

    if let Err(e) = state.worker.submit(work) {
        return error_response(StatusCode::SERVICE_UNAVAILABLE, &e.to_string());
    }

    if req.stream {
        Sse::new(sse_mod::build_stream(event_rx, req.model))
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        collect_response(event_rx, req.model).await.into_response()
    }
}

// ── Legacy text completions ───────────────────────────────────────────────────

pub async fn completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CompletionRequest>,
) -> Response {
    let token_ids = match tokenize::encode(&state.tokenizer, &req.prompt) {
        Ok(ids) => ids,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, &e.to_string()),
    };

    let (event_tx, event_rx) = mpsc::unbounded_channel::<GenerationEvent>();
    let work = WorkItem {
        id: Uuid::new_v4().to_string(),
        token_ids,
        params: SamplingParams {
            max_tokens: req.max_tokens,
            temperature: req.temperature,
            top_p: req.top_p,
            stop: req.stop.clone(),
            seed: req.seed,
            presence_penalty: req.presence_penalty,
            frequency_penalty: req.frequency_penalty,
            ..SamplingParams::default()
        },
        result_tx: event_tx,
    };

    if let Err(e) = state.worker.submit(work) {
        return error_response(StatusCode::SERVICE_UNAVAILABLE, &e.to_string());
    }

    if req.stream {
        Sse::new(sse_mod::build_stream(event_rx, req.model))
            .keep_alive(KeepAlive::default())
            .into_response()
    } else {
        collect_completion(event_rx, req.model)
            .await
            .into_response()
    }
}

/// Collect a non-streaming `/v1/completions` response.
async fn collect_completion(
    mut rx: mpsc::UnboundedReceiver<GenerationEvent>,
    model: String,
) -> impl IntoResponse {
    let mut token_texts = Vec::<String>::new();
    let mut finish = FinishReason::Length;
    let mut stats = GenerationStats::default();

    while let Some(evt) = rx.recv().await {
        match evt {
            GenerationEvent::Token(t) => token_texts.push(t.text),
            GenerationEvent::Finished {
                finish_reason,
                stats: s,
                ..
            } => {
                finish = finish_reason;
                stats = s;
                break;
            }
            GenerationEvent::Error(e) => {
                return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e);
            }
        }
    }

    Json(CompletionResponse {
        id: format!("cmpl-{}", Uuid::new_v4()),
        object: "text_completion",
        created: unix_now(),
        model,
        choices: vec![CompletionChoice {
            index: 0,
            text: token_texts.join(""),
            finish_reason: finish.as_str(),
        }],
        usage: Usage {
            prompt_tokens: stats.prompt_tokens,
            completion_tokens: stats.completion_tokens,
            total_tokens: stats.prompt_tokens + stats.completion_tokens,
        },
    })
    .into_response()
}

// ── Embeddings ────────────────────────────────────────────────────────────────

pub async fn embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbeddingRequest>,
) -> Response {
    // Only float output is supported.
    if req.encoding_format != "float" {
        return error_response(
            StatusCode::BAD_REQUEST,
            &format!(
                "encoding_format {:?} is not supported — use \"float\"",
                req.encoding_format
            ),
        );
    }

    let embed_table = match &state.embed_tokens {
        Some(t) => t,
        None => {
            return error_response(
                StatusCode::NOT_IMPLEMENTED,
                "This model does not expose a token embedding matrix — \
                 load a dedicated embedding model for /v1/embeddings",
            );
        }
    };

    let inputs = req.input.into_strings();
    let mut data = Vec::with_capacity(inputs.len());
    let mut total_prompt_tokens = 0usize;

    for (index, text) in inputs.iter().enumerate() {
        let token_ids = match tokenize::encode(&state.tokenizer, text) {
            Ok(ids) => ids,
            Err(e) => return error_response(StatusCode::BAD_REQUEST, &e.to_string()),
        };

        if token_ids.is_empty() {
            data.push(EmbeddingObject {
                object: "embedding",
                index,
                embedding: vec![0.0_f32; state.hidden_size],
            });
            continue;
        }

        total_prompt_tokens += token_ids.len();

        // Index embed_table [vocab, hidden] with token_ids → [seq_len, hidden].
        let embedding = match embed_mean_pool(embed_table, &token_ids) {
            Ok(v) => v,
            Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()),
        };

        data.push(EmbeddingObject {
            object: "embedding",
            index,
            embedding,
        });
    }

    Json(EmbeddingResponse {
        object: "list",
        data,
        model: req.model,
        usage: EmbeddingUsage {
            prompt_tokens: total_prompt_tokens,
            total_tokens: total_prompt_tokens,
        },
    })
    .into_response()
}

/// Index into `embed_table` with `token_ids`, mean-pool, L2-normalise.
///
/// Performed on CPU in F32; the embed_table tensor was loaded as F32 in
/// `Engine::load`.  For GPU-backed tables this adds a cheap device→host
/// transfer for the indexed rows only.
fn embed_mean_pool(
    embed_table: &candle_core::Tensor,
    token_ids: &[u32],
) -> anyhow::Result<Vec<f32>> {
    use candle_core::Device;
    let device = Device::Cpu;
    let ids = candle_core::Tensor::new(token_ids, &device)?;
    let rows = embed_table.to_device(&device)?.index_select(&ids, 0)?;
    let mean = rows.mean(0)?;
    let v: Vec<f32> = mean.to_vec1()?;
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9_f32);
    Ok(v.iter().map(|x| x / norm).collect())
}

// ── Non-streaming collection ──────────────────────────────────────────────────

async fn collect_response(
    mut rx: mpsc::UnboundedReceiver<GenerationEvent>,
    model: String,
) -> impl IntoResponse {
    let mut token_texts = Vec::<String>::new();
    let mut finish = FinishReason::Length;
    let mut stats = GenerationStats::default();
    let mut lp_data = None;
    let mut raw_tool_calls = Vec::new();

    while let Some(evt) = rx.recv().await {
        match evt {
            GenerationEvent::Token(t) => token_texts.push(t.text),
            GenerationEvent::Finished {
                finish_reason,
                stats: s,
                logprobs,
                tool_calls,
            } => {
                finish = finish_reason;
                stats = s;
                lp_data = logprobs;
                raw_tool_calls = tool_calls;
                break;
            }
            GenerationEvent::Error(e) => {
                return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e);
            }
        }
    }

    // When tool calls were detected the visible text may already be stripped
    // by the worker, but token_texts contains the raw output — re-parse here
    // to get the visible portion.
    let (content, wire_tool_calls) = if !raw_tool_calls.is_empty() {
        let full_raw = token_texts.join("");
        let parsed = ToolCallParser::parse(&full_raw);
        let tcs: Vec<_> = raw_tool_calls
            .into_iter()
            .map(|c| c.into_tool_call())
            .collect();
        (parsed.visible_text, Some(tcs))
    } else {
        (token_texts.join(""), None)
    };

    Json(ChatCompletionResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4()),
        object: "chat.completion",
        created: unix_now(),
        model,
        choices: vec![Choice {
            index: 0,
            message: ChatMessage {
                role: "assistant".into(),
                content,
                tool_calls: wire_tool_calls,
                tool_call_id: None,
            },
            finish_reason: finish.as_str(),
            logprobs: lp_data.map(|content| ChoiceLogprobs { content }),
        }],
        usage: Usage {
            prompt_tokens: stats.prompt_tokens,
            completion_tokens: stats.completion_tokens,
            total_tokens: stats.prompt_tokens + stats.completion_tokens,
        },
    })
    .into_response()
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Prepend a system message containing `content` to `messages`.
///
/// If a system message already exists at position 0, its content is extended
/// with a double newline + the tool definitions.  Otherwise a new system
/// message is inserted at the front.
fn prepend_system(mut messages: Vec<ChatMessage>, content: String) -> Vec<ChatMessage> {
    if content.is_empty() {
        return messages;
    }
    if messages
        .first()
        .map(|m| m.role == "system")
        .unwrap_or(false)
    {
        messages[0].content = format!("{}\n\n{content}", messages[0].content);
    } else {
        messages.insert(
            0,
            ChatMessage {
                role: "system".into(),
                content,
                tool_calls: None,
                tool_call_id: None,
            },
        );
    }
    messages
}

// ── Error helper ──────────────────────────────────────────────────────────────

pub fn error_response(status: StatusCode, message: &str) -> Response {
    (
        status,
        Json(json!({
            "error": { "message": message, "type": "server_error" }
        })),
    )
        .into_response()
}
