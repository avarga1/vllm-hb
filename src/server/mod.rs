//! OpenAI-compatible HTTP server.
//!
//! # Modules
//! - `handlers` — axum handler functions (chat, models, health, batch)
//! - `sse`      — SSE streaming response builder
//! - `metrics`  — Prometheus `/metrics` endpoint (stub)
//!
//! # Endpoints
//! | Method | Path                         | Handler                              |
//! |--------|------------------------------|--------------------------------------|
//! | POST   | `/v1/chat/completions`       | `handlers::chat_completions`         |
//! | POST   | `/v1/completions`            | `handlers::completions`              |
//! | POST   | `/v1/embeddings`             | `handlers::embeddings`               |
//! | GET    | `/v1/models`                 | `handlers::list_models`              |
//! | GET    | `/health`                    | `handlers::health`                   |
//! | GET    | `/metrics`                   | `metrics::handler` (stub)            |
//! | POST   | `/v1/files`                  | `handlers::upload_file`              |
//! | POST   | `/v1/batches`                | `handlers::create_batch`             |
//! | GET    | `/v1/batches/{id}`           | `handlers::get_batch`                |
//! | GET    | `/v1/files/{id}/content`     | `handlers::get_file_content`         |
//! | POST   | `/v1/batches/{id}/cancel`    | `handlers::cancel_batch`             |

pub mod handlers;
pub mod metrics;
pub mod sse;

use std::{
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::Result;
use axum::{
    Router,
    routing::{get, post},
};
use candle_core::Tensor;
use tokenizers::Tokenizer;
use tower_http::cors::CorsLayer;

use crate::batch::BatchStore;
#[allow(unused_imports)] // WorkerHandle is used as a field type in AppState
use crate::worker::WorkerHandle;

// ── App state ─────────────────────────────────────────────────────────────────

pub struct AppState {
    pub worker: WorkerHandle,
    pub tokenizer: Tokenizer,
    pub model_name: String,
    pub model_path: String,
    /// Token embedding matrix `[vocab_size, hidden_size]` for `/v1/embeddings`.
    /// `None` when the model's embedding weight was not found at load time.
    pub embed_tokens: Option<Tensor>,
    /// Model hidden size — dimension of each embedding vector.
    pub hidden_size: usize,
    /// Shared in-memory store for uploaded files and batch jobs.
    pub batch_store: Arc<Mutex<BatchStore>>,
}

// ── Router ────────────────────────────────────────────────────────────────────

/// Build the application router without binding to a port.
///
/// Extracted so integration tests can call `router(state).oneshot(request)`
/// directly via [`tower::ServiceExt`] without spinning up a real TCP listener.
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .route("/v1/completions", post(handlers::completions))
        .route("/v1/embeddings", post(handlers::embeddings))
        .route("/v1/models", get(handlers::list_models))
        .route("/health", get(handlers::health))
        .route("/metrics", get(metrics::handler))
        // Batch API
        .route("/v1/files", post(handlers::upload_file))
        .route("/v1/batches", post(handlers::create_batch))
        .route("/v1/batches/{id}", get(handlers::get_batch))
        .route("/v1/files/{id}/content", get(handlers::get_file_content))
        .route("/v1/batches/{id}/cancel", post(handlers::cancel_batch))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// Bind to `addr` and serve forever.
pub async fn serve(state: Arc<AppState>, addr: &str) -> Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(addr, "Server listening");
    axum::serve(listener, router(state)).await?;
    Ok(())
}

// ── Shared util ───────────────────────────────────────────────────────────────

pub(crate) fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}
