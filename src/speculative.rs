//! Speculative decoding.
//!
//! # Status: STUB
//!
//! ## Concept
//!
//! A small "draft" model generates K candidate tokens cheaply.  The large
//! "target" model verifies all K tokens in a single forward pass.  If the
//! target accepts token i, it moves on; if it rejects it, generation
//! backtracks to that position.  Expected speedup: 2-3× at no quality loss.
//!
//! ## Key components to implement
//!
//! 1. **Draft model** — a small model (e.g. 160M params) loaded alongside
//!    the main model.  Uses the same `Engine` abstraction.
//!
//! 2. **Speculative sampler** — generates K draft tokens, runs target
//!    verification, applies the rejection sampling correction.
//!
//! 3. **Worker integration** — `worker/mod.rs` gains a `speculative_steps: usize`
//!    field; when > 0, each "decode step" runs the draft loop instead of
//!    a single target forward.
//!
//! ## References
//!
//! - Leviathan et al. (2023): "Fast Inference from Transformers via Speculative Decoding"
//! - Chen et al. (2023): "Accelerating Large Language Model Decoding with Speculative Sampling"

#![allow(dead_code)]

pub struct SpeculativeDecoder;
