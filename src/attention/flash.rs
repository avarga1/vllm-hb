//! Flash Attention 2 availability probe.
//!
//! The actual FA2 kernels live in `candle-flash-attn` (linked when the
//! `flash-attn` feature is enabled).  Architecture backends call
//! `candle_flash_attn::flash_attn` directly behind `#[cfg(feature = "flash-attn")]`
//! guards; this module just exposes a runtime flag for logging and health checks.
//!
//! # Hardware requirement
//!
//! CUDA compute capability ≥ 8.0 (Ampere / Ada Lovelace / Hopper).
//! RTX 4090 = sm_89 ✅ · A100 = sm_80 ✅ · V100 = sm_70 ❌
//!
//! # How to enable
//!
//! ```sh
//! cargo build --release --features flash-attn
//! ```
//!
//! # Backends that use FA2
//!
//! | Backend        | FA2 path                              |
//! |----------------|---------------------------------------|
//! | `LlamaBackend` | `candle-transformers` built-in        |
//! | `TpLlamaBackend` | `candle_flash_attn::flash_attn`     |
//! | `MixtralBackend` | `candle_flash_attn::flash_attn`     |

/// Returns `true` when the binary was compiled with `--features flash-attn`.
pub fn is_available() -> bool {
    cfg!(feature = "flash-attn")
}
