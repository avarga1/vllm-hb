//! Flash Attention (sm_80+ only).
//!
//! # Status: STUB
//!
//! candle-transformers exposes flash attention via a `flash-attn` Cargo
//! feature that links against the flash-attention-2 CUDA kernels.
//!
//! ## To enable
//!
//! 1. Add to `Cargo.toml`:
//!    ```toml
//!    [features]
//!    flash-attn = ["candle-transformers/flash-attn"]
//!
//!    [dependencies]
//!    candle-transformers = { version = "0.9", features = ["cuda"] }
//!    ```
//!
//! 2. In `engine/arch/llama.rs`, change:
//!    ```rust
//!    let llama_cfg = llama_cfg.into_config(false);   // false = no flash attn
//!    // becomes:
//!    let use_flash = cfg!(feature = "flash-attn");
//!    let llama_cfg = llama_cfg.into_config(use_flash);
//!    ```
//!
//! ## Hardware requirement
//!
//! CUDA compute capability ≥ 8.0 (Ampere).  V100 is sm_70 — the kernel
//! will compile but produce incorrect results or crash.  Guard with a
//! runtime capability check before enabling.
//!
//! ## Expected speedup
//!
//! ~2× reduction in attention memory bandwidth → ~20-40% wall-clock
//! improvement on long-context requests (2k+ tokens).

#![allow(dead_code)]

#[cfg(feature = "flash-attn")]
pub fn is_available() -> bool { true }

#[cfg(not(feature = "flash-attn"))]
pub fn is_available() -> bool { false }
