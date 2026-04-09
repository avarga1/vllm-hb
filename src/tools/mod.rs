//! Tool-call support for function-calling models.
//!
//! # Overview
//!
//! When a client sends `tools` in the request the pipeline:
//!
//! 1. **Formats** the tool definitions into the system prompt so the model
//!    knows what functions are available (`format.rs`).
//! 2. **Generates** text normally via the existing worker/sampling pipeline.
//! 3. **Parses** the raw generated text for tool call syntax (`parser.rs`).
//!    Two formats are detected:
//!    - **JSON-block** — used by Llama-3, Qwen, Mistral function-calling models
//!    - **XML** — used by Claude-family and Hermes-style models
//! 4. **Strips** the tool-call markup from the visible assistant text and
//!    returns the parsed call(s) as `tool_calls` in the response.
//!
//! # Non-tool requests
//!
//! When `tools` is empty every function in this module is a no-op; the hot
//! path (no tools) has zero overhead.

pub mod format;
pub mod parser;

pub use format::inject_tools;
pub use parser::{ParsedToolCall, ToolCallParser};
