//! OpenAI-compatible HTTP server.
//!
//! # Modules
//! - `handlers` — axum handler functions (chat, models, health)
//! - `sse`      — SSE streaming response builder
//! - `metrics`  — Prometheus `/metrics` endpoint (stub)
//!
//! # Endpoints
//! | Method | Path                      | Handler                         |
//! |--------|---------------------------|---------------------------------|
//! | POST   | `/v1/chat/completions`    | `handlers::chat_completions`    |
//! | POST   | `/v1/completions`         | `handlers::completions`         |
//! | GET    | `/v1/models`              | `handlers::list_models`         |
//! | GET    | `/health`                 | `handlers::health`              |
//! | GET    | `/metrics`                | `metrics::handler` (stub)       |

pub mod handlers;
pub mod metrics;
pub mod sse;

use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::Result;
use axum::{
    Router,
    routing::{get, post},
};
use tokenizers::Tokenizer;
use tower_http::cors::CorsLayer;

#[allow(unused_imports)] // WorkerHandle is used as a field type in AppState
use crate::worker::WorkerHandle;

// ── App state ─────────────────────────────────────────────────────────────────

pub struct AppState {
    pub worker: WorkerHandle,
    pub tokenizer: Tokenizer,
    pub model_name: String,
    pub model_path: String,
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
        .route("/v1/models", get(handlers::list_models))
        .route("/health", get(handlers::health))
        .route("/metrics", get(metrics::handler))
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
