//! Weight sharding strategies for tensor parallelism.
//!
//! # Status: STUB
//!
//! ## Column-parallel linear (QKV projection)
//!
//! Split weight matrix columns across `world_size` GPUs.
//! Each GPU computes a partial output; no all-reduce needed until
//! after the row-parallel layer.
//!
//! ## Row-parallel linear (output projection, FFN down)
//!
//! Split weight matrix rows.  Each GPU holds a shard; an all-reduce
//! across GPUs produces the final output.
//!
//! ## Embedding sharding
//!
//! Vocabulary is partitioned across GPUs.  Each GPU holds
//! `vocab_size / world_size` rows; a gather collects the full logits.
//!
//! ## Integration point
//!
//! `engine/arch/llama.rs` passes `world_size` to candle-transformers'
//! `LlamaConfig` when building the VarBuilder shards.

#![allow(dead_code)]

pub fn shard_count() -> usize {
    1 // TODO: return world_size when tensor parallelism is active
}
