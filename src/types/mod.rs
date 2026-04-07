//! Type definitions for the inference pipeline and OpenAI wire protocol.
//!
//! - `pipeline` — internal types (WorkItem, GenerationEvent, SamplingParams, …)
//! - `openai`   — HTTP request/response types (ChatCompletionRequest, …)

pub mod openai;
pub mod pipeline;

// Re-export everything so `use crate::types::*` still works at call sites
// that don't care about the internal/external distinction.
#[allow(unused_imports)]
pub use openai::*;
#[allow(unused_imports)]
pub use pipeline::*;
