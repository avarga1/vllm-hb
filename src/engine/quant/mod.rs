//! Quantization support for weight-only quantized checkpoints.
//!
//! # Supported formats
//!
//! - **compressed-tensors pack-quantized** ([`compressed_tensors`]) — the
//!   format used by Red Hat / Neural Magic quantizers and by many HF
//!   checkpoints including `cyankiwi/gemma-4-*-AWQ-4bit`.  W4A16 symmetric
//!   with per-group scales, int32-packed weights (8 int4 values per int32).
//!
//! # Phase status
//!
//! - ✅ Phase 1 (#40) — dequant-on-load fallback: packed weights →
//!   fp16/bf16 tensors at load time, reuse existing matmul path.  Correct
//!   but uses 4× more VRAM than keeping packed (defeats the point on
//!   memory-tight hardware).
//! - 🔜 Phase 2 (#41) — W4A16 fused GEMM kernel for sm_70.  Keeps weights
//!   packed in VRAM, dequants per-tile in-register during matmul.

pub mod compressed_tensors;

pub use compressed_tensors::{
    CompressedTensorsConfig, CompressedTensorsQuantArgs, dequantize_pack_quantized,
};
