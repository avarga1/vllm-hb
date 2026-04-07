//! Server-sent event (SSE) streaming response.
//!
//! Converts the internal `GenerationEvent` channel into an OpenAI-compatible
//! SSE stream: one `data: {...}` line per token, then `data: [DONE]`.

use axum::response::sse::Event;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::types::openai::{ChatCompletionChunk, ChunkChoice, Delta};
use crate::types::pipeline::GenerationEvent;

/// Build an SSE event stream from the per-request generation channel.
///
/// Emits:
/// 1. An opening delta with `role: "assistant"`
/// 2. One delta per token
/// 3. A final `[DONE]` sentinel
pub fn build_stream(
    rx:    mpsc::UnboundedReceiver<GenerationEvent>,
    model: String,
) -> impl futures_core::stream::Stream<Item = Result<Event, std::convert::Infallible>> {
    use tokio_stream::StreamExt as _;

    let now    = super::unix_now();
    let gen_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

    // Opening chunk: role assignment.
    let role_chunk = make_chunk(&gen_id, &model, now, Some("assistant"), None, None);
    let prefix = tokio_stream::once(Ok(
        Event::default().data(serde_json::to_string(&role_chunk).unwrap())
    ));

    let gen_id_clone = gen_id.clone();
    let body = UnboundedReceiverStream::new(rx).filter_map(move |evt| {
        match evt {
            GenerationEvent::Token(t) => {
                let chunk = make_chunk(&gen_id_clone, &model, now, None, Some(t.text), None);
                Some(Ok(Event::default().data(serde_json::to_string(&chunk).unwrap())))
            }
            GenerationEvent::Finished { .. } | GenerationEvent::Error(_) => {
                Some(Ok(Event::default().data("[DONE]")))
            }
        }
    });

    prefix.chain(body)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn make_chunk(
    id:            &str,
    model:         &str,
    created:       u64,
    role:          Option<&'static str>,
    content:       Option<String>,
    finish_reason: Option<&'static str>,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id:      id.to_string(),
        object:  "chat.completion.chunk",
        created,
        model:   model.to_string(),
        choices: vec![ChunkChoice {
            index:         0,
            delta:         Delta { role, content },
            finish_reason,
        }],
    }
}
