//! Prometheus metrics endpoint.
//!
//! # Status: STUB
//!
//! Exposes `/metrics` in the Prometheus text exposition format.
//!
//! ## Planned counters / gauges
//!
//! | Metric                       | Type      | Description                        |
//! |------------------------------|-----------|------------------------------------|
//! | `vllm_hb_requests_total`     | Counter   | HTTP requests received             |
//! | `vllm_hb_tokens_total`       | Counter   | Tokens generated (completion side) |
//! | `vllm_hb_ttft_seconds`       | Histogram | Time-to-first-token distribution   |
//! | `vllm_hb_e2e_seconds`        | Histogram | End-to-end request latency         |
//! | `vllm_hb_queue_depth`        | Gauge     | Requests waiting in worker queue   |
//! | `vllm_hb_active_sequences`   | Gauge     | Sequences currently being decoded  |
//!
//! ## Implementation plan
//!
//! 1. Add `prometheus = "0.13"` to `[dependencies]` under
//!    `[features] metrics = ["dep:prometheus"]`
//! 2. Initialise a `prometheus::Registry` in `AppState`
//! 3. Register counters/histograms at startup
//! 4. Increment them in `handlers.rs` (requests) and `worker/mod.rs` (tokens)
//! 5. Wire this handler to `GET /metrics` in `server/mod.rs`

#![allow(dead_code)]

use axum::{http::StatusCode, response::IntoResponse};

/// Placeholder handler — returns 501 until the metrics feature is implemented.
pub async fn handler() -> impl IntoResponse {
    (
        StatusCode::NOT_IMPLEMENTED,
        "Prometheus metrics not yet implemented.\n\
         Track progress: https://github.com/avarga1/vllm-hb/issues/2",
    )
}
