//! Tensor parallelism (multi-GPU).
//!
//! # Status: STUB
//!
//! # Modules
//! - `comm`  — all-reduce, broadcast, all-gather primitives
//! - `shard` — column/row-parallel weight sharding strategies
//!
//! # Integration plan
//!
//! 1. Add `--tensor-parallel-size N` CLI flag to `ServeArgs`
//! 2. Start N worker tasks (one per GPU via `Device::cuda_if_available(rank)`)
//! 3. Each worker loads its weight shard via `shard::` helpers
//! 4. After each linear layer, call `comm::all_reduce()` to synchronise
//!
//! When `tensor_parallel_size == 1` (the default), all comm ops are no-ops
//! and the existing single-GPU path is unaffected.

pub mod comm;
pub mod shard;

/// World size — number of GPUs used for tensor parallelism.
/// Always 1 until multi-GPU is implemented.
#[allow(dead_code)]
pub fn world_size() -> usize {
    1
}
