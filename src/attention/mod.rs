//! Attention kernel selection.
//!
//! # Modules
//! - `sdpa`  — standard scaled dot-product attention hook (stub)
//! - `flash` — flash attention, sm_80+ only (stub; enable with `--features flash-attn`)
//!
//! When flash-attn feature is enabled, `engine/arch/llama.rs` passes
//! `use_flash_attn = true` to candle-transformers' `into_config()`.

pub mod flash;
pub mod sdpa;
