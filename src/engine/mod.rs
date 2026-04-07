//! Inference engine — model loading, architecture dispatch, forward pass.
//!
//! # Modules
//! - `config`  — ModelConfig + HfMeta (no candle dependency)
//! - `dtype`   — DType resolution (F16 / BF16 based on GPU + flag)
//! - `loader`  — Engine struct: load(), forward(), reset_cache()
//! - `arch/`   — per-architecture backends (Llama ✅, Mixtral 🔜, Qwen2 🔜, Phi3 🔜)

pub mod arch;
pub mod config;
pub mod dtype;
pub mod loader;

// Primary public API — callers import `engine::Engine` and `engine::ModelConfig`.
pub use config::ModelConfig;
pub use loader::Engine;
