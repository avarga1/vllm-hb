//! Async batch inference — OpenAI Batch API compatible.
//!
//! # API surface
//!
//! | Method | Path                       | Description                   |
//! |--------|----------------------------|-------------------------------|
//! | POST   | `/v1/files`                | Upload a JSONL file           |
//! | POST   | `/v1/batches`              | Submit a batch job            |
//! | GET    | `/v1/batches/{id}`         | Poll batch status             |
//! | GET    | `/v1/files/{id}/content`   | Retrieve output JSONL         |
//! | POST   | `/v1/batches/{id}/cancel`  | Cancel a pending batch        |
//!
//! # Input format (JSONL)
//!
//! Each line must be a JSON object with the following fields:
//!
//! ```json
//! {"custom_id": "req-1", "method": "POST", "url": "/v1/chat/completions",
//!  "body": {"model": "my-model", "messages": [{"role": "user", "content": "Hi"}]}}
//! ```
//!
//! # Output format (JSONL)
//!
//! ```json
//! {"id": "batch_req_...", "custom_id": "req-1",
//!  "response": {"status_code": 200, "body": { ... }}, "error": null}
//! ```
//!
//! # Lifecycle
//!
//! ```text
//!  POST /v1/files  ─────────────────────────────────► FileObject (file_id)
//!  POST /v1/batches  ──► BatchObject (validating)
//!                         │
//!                         ▼  (background tokio task)
//!                     in_progress
//!                         │  (one WorkItem per line, sequential)
//!                         ▼
//!                     completed  ──► output file (file_id)
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::tokenize;
use crate::types::openai::ChatCompletionRequest;
use crate::types::pipeline::{FinishReason, GenerationEvent, SamplingParams, WorkItem};
use crate::worker::WorkerHandle;

// ── Unix timestamp helper ─────────────────────────────────────────────────────

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ── File store ────────────────────────────────────────────────────────────────

/// Immutable content of an uploaded or generated file.
#[derive(Clone)]
pub struct StoredFile {
    pub id: String,
    pub filename: String,
    pub created_at: u64,
    pub content: String, // raw JSONL bytes
}

// ── Batch job ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BatchStatus {
    Validating,
    InProgress,
    Completed,
    Failed,
    Cancelled,
}

/// Public batch object returned by GET /v1/batches/{id}.
#[derive(Debug, Clone, Serialize)]
pub struct BatchObject {
    pub id: String,
    pub object: &'static str,
    pub endpoint: String,
    pub status: BatchStatus,
    pub input_file_id: String,
    /// Populated once the batch finishes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_file_id: Option<String>,
    pub created_at: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failed_at: Option<u64>,
    pub request_counts: RequestCounts,
}

#[derive(Debug, Clone, Serialize, Default)]
pub struct RequestCounts {
    pub total: usize,
    pub completed: usize,
    pub failed: usize,
}

// ── Store ─────────────────────────────────────────────────────────────────────

/// Central in-memory store shared by all batch handlers.
///
/// Wrapped in `Arc<Mutex<_>>` and placed in `AppState` so it can be accessed
/// from both HTTP handlers and background processing tasks.
#[derive(Default)]
pub struct BatchStore {
    pub files: HashMap<String, StoredFile>,
    pub batches: HashMap<String, BatchObject>,
}

impl BatchStore {
    pub fn new() -> Self {
        Self::default()
    }
}

// ── Request / response JSONL types ───────────────────────────────────────────

/// One line of the input JSONL file.
#[derive(Debug, Deserialize)]
pub struct BatchRequestLine {
    pub custom_id: String,
    #[allow(dead_code)]
    pub method: String,
    pub url: String,
    pub body: serde_json::Value,
}

/// One line of the output JSONL file.
#[derive(Debug, Serialize)]
pub struct BatchResponseLine {
    pub id: String,
    pub custom_id: String,
    pub response: Option<BatchHttpResponse>,
    pub error: Option<BatchErrorResponse>,
}

#[derive(Debug, Serialize)]
pub struct BatchHttpResponse {
    pub status_code: u16,
    pub request_id: String,
    pub body: serde_json::Value,
}

#[derive(Debug, Serialize)]
pub struct BatchErrorResponse {
    pub code: String,
    pub message: String,
}

// ── HTTP request types ────────────────────────────────────────────────────────

/// Body for `POST /v1/batches`.
#[derive(Debug, Deserialize)]
pub struct CreateBatchRequest {
    pub input_file_id: String,
    pub endpoint: String,
    /// Completion window hint (e.g. `"24h"`).  Stored but not enforced.
    #[serde(default = "default_window")]
    #[allow(dead_code)]
    pub completion_window: String,
}

fn default_window() -> String {
    "24h".into()
}

/// `POST /v1/files` response body.
#[derive(Debug, Serialize)]
pub struct FileObject {
    pub id: String,
    pub object: &'static str,
    pub bytes: usize,
    pub created_at: u64,
    pub filename: String,
    pub purpose: &'static str,
}

// ── Background processor ──────────────────────────────────────────────────────

/// Spawn a task that processes all requests in `input_file_id` and updates
/// the batch record in `store` as it goes.
///
/// Each line is processed sequentially; an error in one line is recorded in
/// the output JSONL and does not abort the rest of the batch.
pub fn spawn_processor(
    batch_id: String,
    input_file_id: String,
    store: Arc<Mutex<BatchStore>>,
    worker: WorkerHandle,
    tokenizer_path: String,
) {
    tokio::spawn(async move {
        // Mark in_progress.
        {
            let mut s = store.lock().unwrap();
            if let Some(b) = s.batches.get_mut(&batch_id) {
                b.status = BatchStatus::InProgress;
            }
        }

        // Read input lines.
        let (lines, total) = {
            let s = store.lock().unwrap();
            let content = s
                .files
                .get(&input_file_id)
                .map(|f| f.content.clone())
                .unwrap_or_default();
            let lines: Vec<String> = content
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(str::to_owned)
                .collect();
            let total = lines.len();
            (lines, total)
        };

        // Update total count.
        {
            let mut s = store.lock().unwrap();
            if let Some(b) = s.batches.get_mut(&batch_id) {
                b.request_counts.total = total;
            }
        }

        let mut output_lines: Vec<String> = Vec::with_capacity(total);
        let mut completed = 0usize;
        let mut failed = 0usize;

        for raw_line in &lines {
            let line_id = format!("batch_req_{}", Uuid::new_v4().simple());

            let out_line: BatchResponseLine =
                match process_line(raw_line, line_id.clone(), &worker, &tokenizer_path).await {
                    Ok(resp) => {
                        completed += 1;
                        resp
                    }
                    Err(e) => {
                        failed += 1;
                        // Parse custom_id best-effort for the error line.
                        let custom_id = serde_json::from_str::<serde_json::Value>(raw_line)
                            .ok()
                            .and_then(|v| v["custom_id"].as_str().map(str::to_owned))
                            .unwrap_or_else(|| "unknown".into());
                        BatchResponseLine {
                            id: line_id,
                            custom_id,
                            response: None,
                            error: Some(BatchErrorResponse {
                                code: "processing_error".into(),
                                message: e.to_string(),
                            }),
                        }
                    }
                };

            if let Ok(serialized) = serde_json::to_string(&out_line) {
                output_lines.push(serialized);
            }

            // Update progress.
            let mut s = store.lock().unwrap();
            if let Some(b) = s.batches.get_mut(&batch_id) {
                b.request_counts.completed = completed;
                b.request_counts.failed = failed;
            }
        }

        // Store output file and mark completed.
        let output_content = output_lines.join("\n");
        let output_file_id = format!("file-{}", Uuid::new_v4().simple());
        let bytes = output_content.len();
        let now = unix_now();

        let mut s = store.lock().unwrap();
        s.files.insert(
            output_file_id.clone(),
            StoredFile {
                id: output_file_id.clone(),
                filename: format!("batch_{batch_id}_output.jsonl"),
                created_at: now,
                content: output_content,
            },
        );
        if let Some(b) = s.batches.get_mut(&batch_id) {
            b.status = BatchStatus::Completed;
            b.output_file_id = Some(output_file_id);
            b.completed_at = Some(now);
            b.request_counts.completed = completed;
            b.request_counts.failed = failed;
        }
        drop(s);

        tracing::info!(
            batch_id,
            total,
            completed,
            failed,
            output_bytes = bytes,
            "Batch processing complete"
        );
    });
}

/// Process one JSONL line: parse → submit to worker → collect response.
async fn process_line(
    raw: &str,
    line_id: String,
    worker: &WorkerHandle,
    tokenizer_path: &str,
) -> anyhow::Result<BatchResponseLine> {
    let req_line: BatchRequestLine = serde_json::from_str(raw)?;
    let custom_id = req_line.custom_id.clone();

    // Only /v1/chat/completions supported for now.
    if req_line.url.trim_end_matches('/') != "/v1/chat/completions" {
        anyhow::bail!("unsupported endpoint {:?}", req_line.url);
    }

    let chat_req: ChatCompletionRequest = serde_json::from_value(req_line.body)?;

    // Apply chat template to build the prompt string.
    let prompt = tokenize::apply_chat_template(tokenizer_path, &chat_req.messages)?;

    // Tokenize.
    let tokenizer = tokenize::load(tokenizer_path)?;
    let token_ids = tokenize::encode(&tokenizer, &prompt)?;

    // Submit to worker.
    let (event_tx, mut event_rx) = mpsc::unbounded_channel::<GenerationEvent>();
    let work = WorkItem {
        id: line_id.clone(),
        token_ids,
        params: SamplingParams {
            max_tokens: chat_req.max_tokens,
            temperature: chat_req.temperature,
            top_p: chat_req.top_p,
            stop: chat_req.stop.clone(),
            seed: chat_req.seed,
            ..SamplingParams::default()
        },
        result_tx: event_tx,
    };
    worker.submit(work)?;

    // Collect non-streaming response.
    let mut tokens = Vec::<String>::new();
    let mut finish = FinishReason::Length;

    while let Some(evt) = event_rx.recv().await {
        match evt {
            GenerationEvent::Token(t) => tokens.push(t.text),
            GenerationEvent::Finished {
                finish_reason,
                stats: _,
                ..
            } => {
                finish = finish_reason;
                break;
            }
            GenerationEvent::Error(e) => anyhow::bail!("{e}"),
        }
    }

    let response_body = serde_json::json!({
        "id": format!("chatcmpl-{}", Uuid::new_v4()),
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": tokens.join(""),
            },
            "finish_reason": finish.as_str(),
        }]
    });

    Ok(BatchResponseLine {
        id: line_id,
        custom_id,
        response: Some(BatchHttpResponse {
            status_code: 200,
            request_id: format!("req_{}", Uuid::new_v4().simple()),
            body: response_body,
        }),
        error: None,
    })
}
