//! Tensor parallelism — multi-GPU inference support.
//!
//! # Modules
//!
//! | Module  | Contents                                                    |
//! |---------|-------------------------------------------------------------|
//! | `world` | [`TpWorld`] — GPU device registry, rank assignment          |
//! | `comm`  | [`all_reduce`], [`all_gather`] — inter-GPU collective ops   |
//! | `shard` | [`column_shard`], [`row_shard`] — weight partitioning       |
//!
//! # Typical data-flow (4-GPU, 1 transformer layer)
//!
//! ```text
//! input [seq, hidden]
//!    │
//!    ├─ column_shard(Q/K/V weight, rank, 4)  → partial QKV [seq, hidden/4]
//!    │   …attention on each rank…
//!    ├─ row_shard(out_proj weight, rank, 4)  → partial output [seq, hidden]
//!    │
//!    └─ all_reduce([partial_outputs], device) → full output [seq, hidden]
//! ```
//!
//! # Single-GPU fast path
//!
//! `TpWorld::is_single()` returns `true` for `world_size == 1`.  In that
//! case `all_reduce` is a no-op move and `column_shard`/`row_shard` with
//! `world_size=1` return the original tensor unchanged.

pub mod comm;
pub mod shard;
pub mod world;

#[allow(unused_imports)]
pub use comm::{all_gather, all_reduce};
#[allow(unused_imports)]
pub use shard::{bias_shard, column_chunk_size, column_shard, row_chunk_size, row_shard};
#[allow(unused_imports)]
pub use world::TpWorld;
