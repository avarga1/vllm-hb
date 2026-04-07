//! Standard scaled dot-product attention (SDPA).
//!
//! Currently the attention computation lives inside candle-transformers'
//! arch implementations.  This file becomes the hook point for:
//!
//! 1. Custom paged-attention CUDA kernels (continuous batching)
//! 2. Any attention variant that doesn't come for free via candle
//!
//! For now it's a stub — nothing imports this yet.

#![allow(dead_code)]
