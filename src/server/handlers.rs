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
    response::{IntoResponse, Response, Sse},
    response::sse::KeepAlive,
};
use serde_json::json;
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::types::openai::{
    ChatCompletionResponse, ChatMessage, Choice, Usage, ChatCompletionRequest,
};
use crate::types::pipeline::{
    FinishReason, GenerationEvent, GenerationStats, SamplingParams, WorkItem,
};
use crate::tokenize;
use super::{AppState, unix_now};
use super::sse as sse_mod;

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
    Json(req):    Json<ChatCompletionRequest>,
) -> Response {
    let prompt = match tokenize::apply_chat_template(&state.model_path, &req.messages) {
        Ok(p)  => p,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, &e.to_string()),
    };

    let token_ids = match tokenize::encode(&state.tokenizer, &prompt) {
        Ok(ids) => ids,
        Err(e)  => return error_response(StatusCode::BAD_REQUEST, &e.to_string()),
    };

    let (event_tx, event_rx) = mpsc::unbounded_channel::<GenerationEvent>();
    let work = WorkItem {
        id:        Uuid::new_v4().to_string(),
        token_ids,
        params: SamplingParams {
            max_tokens:  req.max_tokens,
            temperature: req.temperature,
            top_p:       req.top_p,
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

// ── Non-streaming collection ──────────────────────────────────────────────────

async fn collect_response(
    mut rx: mpsc::UnboundedReceiver<GenerationEvent>,
    model:  String,
) -> impl IntoResponse {
    let mut token_texts = Vec::<String>::new();
    let mut finish      = FinishReason::Length;
    let mut stats       = GenerationStats::default();

    while let Some(evt) = rx.recv().await {
        match evt {
            GenerationEvent::Token(t) => token_texts.push(t.text),
            GenerationEvent::Finished { finish_reason, stats: s } => {
                finish = finish_reason;
                stats  = s;
                break;
            }
            GenerationEvent::Error(e) => {
                return error_response(StatusCode::INTERNAL_SERVER_ERROR, &e);
            }
        }
    }

    Json(ChatCompletionResponse {
        id:      format!("chatcmpl-{}", Uuid::new_v4()),
        object:  "chat.completion",
        created: unix_now(),
        model,
        choices: vec![Choice {
            index:         0,
            message:       ChatMessage { role: "assistant".into(), content: token_texts.join("") },
            finish_reason: finish.as_str(),
        }],
        usage: Usage {
            prompt_tokens:     stats.prompt_tokens,
            completion_tokens: stats.completion_tokens,
            total_tokens:      stats.prompt_tokens + stats.completion_tokens,
        },
    }).into_response()
}

// ── Error helper ──────────────────────────────────────────────────────────────

pub fn error_response(status: StatusCode, message: &str) -> Response {
    (status, Json(json!({
        "error": { "message": message, "type": "server_error" }
    }))).into_response()
}
