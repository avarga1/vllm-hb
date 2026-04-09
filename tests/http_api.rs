//! HTTP API integration tests.
//!
//! These tests build the Axum router with a mock worker (no GPU, no model
//! weights) and call each endpoint via [`tower::ServiceExt::oneshot`].  They
//! verify the OpenAI-compatible wire format without requiring a real model.
//!
//! # Mock worker
//!
//! [`mock_worker`] spawns a Tokio task that drains the worker inbox and
//! replies to every request with a canned "hello world" token stream.  The
//! real worker loop (scheduler, KV cache, GPU forward pass) is bypassed
//! entirely.

use std::sync::Arc;

use axum::{
    body::Body,
    http::{Request, StatusCode, header},
};
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tower::ServiceExt; // oneshot

use vllm_hb::{
    server::AppState,
    types::pipeline::{FinishReason, GenerationEvent, GenerationStats, TokenEvent, WorkItem},
    worker::WorkerHandle,
};

// ── Test helpers ──────────────────────────────────────────────────────────────

/// Fixture directory that contains a ChatML `tokenizer_config.json` so that
/// `tokenize::apply_chat_template` succeeds without a real model.
const FIXTURE_MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/chatml");

/// Minimal `tokenizer.json` fixture used to build a real
/// [`tokenizers::Tokenizer`] without downloading anything from HuggingFace.
const FIXTURE_TOKENIZER_JSON: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/tokenizer.json");

/// Load the fixture tokenizer from disk.
fn fixture_tokenizer() -> tokenizers::Tokenizer {
    tokenizers::Tokenizer::from_file(FIXTURE_TOKENIZER_JSON)
        .expect("test fixture tokenizer must be loadable")
}

/// Spawn a mock inference worker that answers every request with two tokens
/// ("hello" and " world") followed by a `Finished` event.
///
/// Returns a [`WorkerHandle`] backed by the mock task and a join handle.
fn mock_worker() -> (WorkerHandle, tokio::task::JoinHandle<()>) {
    let (tx, mut rx) = mpsc::unbounded_channel::<WorkItem>();
    let handle = WorkerHandle::for_test(tx);
    let task = tokio::spawn(async move {
        while let Some(item) = rx.recv().await {
            let _ = item.result_tx.send(GenerationEvent::Token(TokenEvent {
                id: 1,
                text: "hello".into(),
            }));
            let _ = item.result_tx.send(GenerationEvent::Token(TokenEvent {
                id: 2,
                text: " world".into(),
            }));
            let _ = item.result_tx.send(GenerationEvent::Finished {
                finish_reason: FinishReason::Stop,
                stats: GenerationStats {
                    prompt_tokens: 5,
                    completion_tokens: 2,
                    ttft_ms: 10,
                    total_ms: 20,
                    tokens_per_sec: 100.0,
                },
                logprobs: None,
                tool_calls: Vec::new(),
            });
        }
    });
    (handle, task)
}

/// Dimension of the fake embedding table used in tests.
const TEST_HIDDEN: usize = 8;

/// Build a small random embedding table `[vocab_size, hidden]` for tests.
///
/// Uses a deterministic pattern (token_id * 0.01 per column) so the output
/// is stable across runs.
fn fixture_embed_tokens() -> candle_core::Tensor {
    let tokenizer = fixture_tokenizer();
    let vocab = tokenizer.get_vocab_size(true);
    let data: Vec<f32> = (0..vocab)
        .flat_map(|id| (0..TEST_HIDDEN).map(move |col| id as f32 * 0.01 + col as f32 * 0.001))
        .collect();
    candle_core::Tensor::from_vec(data, (vocab, TEST_HIDDEN), &candle_core::Device::Cpu).unwrap()
}

/// Build a test `AppState` with the fixture tokenizer and mock worker.
fn test_state() -> Arc<AppState> {
    let (worker, _task) = mock_worker();
    Arc::new(AppState {
        worker,
        tokenizer: fixture_tokenizer(),
        model_name: "test-model".into(),
        model_path: FIXTURE_MODEL_PATH.into(),
        embed_tokens: Some(fixture_embed_tokens()),
        hidden_size: TEST_HIDDEN,
    })
}

/// Build the router for a test state.
fn test_router() -> axum::Router {
    vllm_hb::server::router(test_state())
}

/// Collect response body bytes.
async fn body_bytes(resp: axum::response::Response) -> bytes::Bytes {
    axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap()
}

/// Deserialize response body as JSON.
async fn body_json(resp: axum::response::Response) -> Value {
    let bytes = body_bytes(resp).await;
    serde_json::from_slice(&bytes).expect("response must be valid JSON")
}

/// POST `/v1/chat/completions` request builder.
fn chat_request(body: Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

// ── Health endpoint ───────────────────────────────────────────────────────────

#[tokio::test]
async fn health_returns_200() {
    let req = Request::builder()
        .uri("/health")
        .body(Body::empty())
        .unwrap();
    let resp = test_router().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn health_body_is_ok() {
    let req = Request::builder()
        .uri("/health")
        .body(Body::empty())
        .unwrap();
    let resp = test_router().oneshot(req).await.unwrap();
    let body = body_json(resp).await;
    assert_eq!(body["status"], "ok");
}

// ── Model listing endpoint ────────────────────────────────────────────────────

#[tokio::test]
async fn models_returns_200() {
    let req = Request::builder()
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    let resp = test_router().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn models_returns_list_object() {
    let req = Request::builder()
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    let resp = test_router().oneshot(req).await.unwrap();
    let body = body_json(resp).await;
    assert_eq!(body["object"], "list");
}

#[tokio::test]
async fn models_contains_loaded_model() {
    let req = Request::builder()
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();
    let resp = test_router().oneshot(req).await.unwrap();
    let body = body_json(resp).await;
    let id = body["data"][0]["id"].as_str().unwrap();
    assert_eq!(id, "test-model");
}

// ── Non-streaming chat completions ────────────────────────────────────────────

#[tokio::test]
async fn chat_non_streaming_returns_200() {
    let req = chat_request(json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}]
    }));
    let resp = test_router().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn chat_non_streaming_response_shape() {
    let req = chat_request(json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}]
    }));
    let resp = test_router().oneshot(req).await.unwrap();
    let body = body_json(resp).await;

    assert_eq!(body["object"], "chat.completion");
    assert!(body["id"].as_str().unwrap().starts_with("chatcmpl-"));
    assert!(body["created"].is_number());
    assert_eq!(body["model"], "test-model");
}

#[tokio::test]
async fn chat_non_streaming_content_joined() {
    let req = chat_request(json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}]
    }));
    let resp = test_router().oneshot(req).await.unwrap();
    let body = body_json(resp).await;

    let content = body["choices"][0]["message"]["content"].as_str().unwrap();
    assert_eq!(content, "hello world");
}

#[tokio::test]
async fn chat_non_streaming_finish_reason_stop() {
    let req = chat_request(json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}]
    }));
    let resp = test_router().oneshot(req).await.unwrap();
    let body = body_json(resp).await;

    let reason = body["choices"][0]["finish_reason"].as_str().unwrap();
    assert_eq!(reason, "stop");
}

#[tokio::test]
async fn chat_non_streaming_role_is_assistant() {
    let req = chat_request(json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}]
    }));
    let resp = test_router().oneshot(req).await.unwrap();
    let body = body_json(resp).await;

    let role = body["choices"][0]["message"]["role"].as_str().unwrap();
    assert_eq!(role, "assistant");
}

#[tokio::test]
async fn chat_non_streaming_usage_present() {
    let req = chat_request(json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}]
    }));
    let resp = test_router().oneshot(req).await.unwrap();
    let body = body_json(resp).await;

    assert!(body["usage"]["prompt_tokens"].is_number());
    assert!(body["usage"]["completion_tokens"].is_number());
    assert!(body["usage"]["total_tokens"].is_number());
    let total = body["usage"]["total_tokens"].as_u64().unwrap();
    let prompt = body["usage"]["prompt_tokens"].as_u64().unwrap();
    let comp = body["usage"]["completion_tokens"].as_u64().unwrap();
    assert_eq!(total, prompt + comp);
}

// ── Streaming chat completions ────────────────────────────────────────────────

#[tokio::test]
async fn chat_streaming_returns_200() {
    let req = chat_request(json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": true
    }));
    let resp = test_router().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn chat_streaming_content_type_is_event_stream() {
    let req = chat_request(json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": true
    }));
    let resp = test_router().oneshot(req).await.unwrap();
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .unwrap()
        .to_str()
        .unwrap();
    assert!(ct.contains("text/event-stream"), "Content-Type was: {ct}");
}

#[tokio::test]
async fn chat_streaming_body_contains_done_sentinel() {
    let req = chat_request(json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": true
    }));
    let resp = test_router().oneshot(req).await.unwrap();
    let raw = body_bytes(resp).await;
    let text = std::str::from_utf8(&raw).unwrap();
    assert!(text.contains("[DONE]"), "SSE stream must end with [DONE]");
}

#[tokio::test]
async fn chat_streaming_body_contains_role_chunk() {
    let req = chat_request(json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": true
    }));
    let resp = test_router().oneshot(req).await.unwrap();
    let raw = body_bytes(resp).await;
    let text = std::str::from_utf8(&raw).unwrap();
    // First SSE data line should be a JSON chunk with role="assistant".
    let first_data = text
        .lines()
        .find(|l| l.starts_with("data: ") && !l.contains("[DONE]"))
        .expect("at least one data line");
    let json_str = first_data.trim_start_matches("data: ");
    let chunk: Value = serde_json::from_str(json_str).unwrap();
    let role = chunk["choices"][0]["delta"]["role"].as_str().unwrap();
    assert_eq!(role, "assistant");
}

#[tokio::test]
async fn chat_streaming_token_chunks_have_content() {
    let req = chat_request(json!({
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": true
    }));
    let resp = test_router().oneshot(req).await.unwrap();
    let raw = body_bytes(resp).await;
    let text = std::str::from_utf8(&raw).unwrap();

    // Skip the first (role) chunk and [DONE]; remaining data lines are tokens.
    let token_texts: Vec<&str> = text
        .lines()
        .filter(|l| l.starts_with("data: ") && !l.contains("[DONE]"))
        .skip(1) // skip role chunk
        .filter_map(|l| {
            let json_str = l.trim_start_matches("data: ");
            let chunk: Value = serde_json::from_str(json_str).ok()?;
            chunk["choices"][0]["delta"]["content"]
                .as_str()
                .map(|s| s.to_owned())
                .map(|_| l)
        })
        .collect();
    // Mock sends two token events ("hello" and " world").
    assert!(
        !token_texts.is_empty(),
        "Expected at least one content chunk"
    );
}

// ── Legacy text completions ───────────────────────────────────────────────────

fn completion_request(body: Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

#[tokio::test]
async fn completion_returns_200() {
    let req = completion_request(json!({
        "model": "test-model",
        "prompt": "Hello, world!"
    }));
    let resp = test_router().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn completion_response_shape() {
    let req = completion_request(json!({
        "model": "test-model",
        "prompt": "Hello"
    }));
    let resp = test_router().oneshot(req).await.unwrap();
    let body = body_json(resp).await;

    assert_eq!(body["object"], "text_completion");
    assert!(body["id"].as_str().unwrap().starts_with("cmpl-"));
    assert!(body["created"].is_number());
    assert_eq!(body["model"], "test-model");
}

#[tokio::test]
async fn completion_text_joined() {
    let req = completion_request(json!({
        "model": "test-model",
        "prompt": "Say something"
    }));
    let resp = test_router().oneshot(req).await.unwrap();
    let body = body_json(resp).await;

    let text = body["choices"][0]["text"].as_str().unwrap();
    assert_eq!(text, "hello world");
}

#[tokio::test]
async fn completion_finish_reason_stop() {
    let req = completion_request(json!({
        "model": "test-model",
        "prompt": "Say something"
    }));
    let resp = test_router().oneshot(req).await.unwrap();
    let body = body_json(resp).await;

    assert_eq!(body["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn completion_usage_present() {
    let req = completion_request(json!({
        "model": "test-model",
        "prompt": "Say something"
    }));
    let resp = test_router().oneshot(req).await.unwrap();
    let body = body_json(resp).await;

    assert!(body["usage"]["prompt_tokens"].is_number());
    assert!(body["usage"]["completion_tokens"].is_number());
    let total = body["usage"]["total_tokens"].as_u64().unwrap();
    let prompt = body["usage"]["prompt_tokens"].as_u64().unwrap();
    let comp = body["usage"]["completion_tokens"].as_u64().unwrap();
    assert_eq!(total, prompt + comp);
}

#[tokio::test]
async fn completion_streaming_returns_200() {
    let req = completion_request(json!({
        "model": "test-model",
        "prompt": "Hello",
        "stream": true
    }));
    let resp = test_router().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn completion_streaming_event_stream() {
    let req = completion_request(json!({
        "model": "test-model",
        "prompt": "Hello",
        "stream": true
    }));
    let resp = test_router().oneshot(req).await.unwrap();
    let ct = resp
        .headers()
        .get(header::CONTENT_TYPE)
        .unwrap()
        .to_str()
        .unwrap();
    assert!(ct.contains("text/event-stream"), "Content-Type was: {ct}");
}

// ── Embeddings ────────────────────────────────────────────────────────────────

fn embedding_request(body: serde_json::Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/embeddings")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

#[tokio::test]
async fn embedding_returns_200() {
    let req = embedding_request(json!({
        "model": "test-model",
        "input": "Hello world"
    }));
    let resp = test_router().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn embedding_response_shape() {
    let req = embedding_request(json!({
        "model": "test-model",
        "input": "Hello"
    }));
    let resp = test_router().oneshot(req).await.unwrap();
    let body = body_json(resp).await;

    assert_eq!(body["object"], "list");
    assert_eq!(body["model"], "test-model");
    assert!(body["data"].is_array());
    assert_eq!(body["data"][0]["object"], "embedding");
    assert_eq!(body["data"][0]["index"], 0);
}

#[tokio::test]
async fn embedding_vector_dimension() {
    let req = embedding_request(json!({
        "model": "test-model",
        "input": "Hello"
    }));
    let resp = test_router().oneshot(req).await.unwrap();
    let body = body_json(resp).await;

    let emb = body["data"][0]["embedding"].as_array().unwrap();
    assert_eq!(
        emb.len(),
        TEST_HIDDEN,
        "embedding dimension should match hidden size"
    );
}

#[tokio::test]
async fn embedding_is_l2_normalized() {
    let req = embedding_request(json!({
        "model": "test-model",
        "input": "normalise me"
    }));
    let resp = test_router().oneshot(req).await.unwrap();
    let body = body_json(resp).await;

    let emb: Vec<f64> = body["data"][0]["embedding"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    let norm: f64 = emb.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "embedding L2 norm should be ~1, got {norm}"
    );
}

#[tokio::test]
async fn embedding_batch_input() {
    let req = embedding_request(json!({
        "model": "test-model",
        "input": ["first sentence", "second sentence"]
    }));
    let resp = test_router().oneshot(req).await.unwrap();
    let body = body_json(resp).await;

    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 2);
    assert_eq!(data[0]["index"], 0);
    assert_eq!(data[1]["index"], 1);
}

#[tokio::test]
async fn embedding_usage_present() {
    let req = embedding_request(json!({
        "model": "test-model",
        "input": "count my tokens"
    }));
    let resp = test_router().oneshot(req).await.unwrap();
    let body = body_json(resp).await;

    assert!(body["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
    assert_eq!(
        body["usage"]["prompt_tokens"],
        body["usage"]["total_tokens"]
    );
}

#[tokio::test]
async fn embedding_invalid_encoding_format_returns_400() {
    let req = embedding_request(json!({
        "model": "test-model",
        "input": "hello",
        "encoding_format": "base64"
    }));
    let resp = test_router().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn embedding_no_embed_table_returns_501() {
    // Build state with embed_tokens = None.
    let (worker, _task) = mock_worker();
    let state = Arc::new(AppState {
        worker,
        tokenizer: fixture_tokenizer(),
        model_name: "test-model".into(),
        model_path: FIXTURE_MODEL_PATH.into(),
        embed_tokens: None,
        hidden_size: TEST_HIDDEN,
    });
    let req = embedding_request(json!({
        "model": "test-model",
        "input": "hi"
    }));
    let resp = vllm_hb::server::router(state).oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
}

// ── Error handling ────────────────────────────────────────────────────────────

#[tokio::test]
async fn chat_invalid_json_returns_422() {
    let req = Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from("this is not json"))
        .unwrap();
    let resp = test_router().oneshot(req).await.unwrap();
    // Axum returns 400 Bad Request for malformed JSON bodies.
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn unknown_route_returns_404() {
    let req = Request::builder()
        .uri("/nonexistent")
        .body(Body::empty())
        .unwrap();
    let resp = test_router().oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}
