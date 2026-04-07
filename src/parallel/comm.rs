//! Inter-GPU communication primitives.
//!
//! # Status: STUB
//!
//! ## Planned operations
//!
//! - `all_reduce(tensor)` — sum across all ranks (used after row-parallel linear)
//! - `broadcast(tensor, src_rank)` — send from one rank to all
//! - `all_gather(tensor)` — collect shards from all ranks
//!
//! ## Implementation options
//!
//! 1. **NCCL via cudarc** — cudarc has NCCL bindings. Lowest latency,
//!    requires NCCL installed alongside CUDA.
//!
//! 2. **Ring all-reduce in Rust** — implement the ring algorithm over
//!    cudarc device-to-device copies.  No extra dependency, but higher
//!    latency than NCCL for large tensors.
//!
//! When `world_size == 1`, all operations are no-ops — single-GPU
//! deployment is unaffected by this module.

#![allow(dead_code)]

/// No-op all-reduce for single-GPU deployments.
pub fn all_reduce() {
    // TODO: NCCL or ring all-reduce for world_size > 1
}
