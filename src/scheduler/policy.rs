//! Scheduling and preemption policies.
//!
//! # Status: STUB
//!
//! Keeps policy logic out of `block_manager.rs` so alternative policies
//! can be swapped without touching the allocator.
//!
//! ## Policies to implement
//!
//! - **FCFS** (first-come-first-served) — the default vLLM policy.
//!   Preempts the sequence group that arrived most recently when GPU
//!   memory is exhausted.
//!
//! - **Priority** — assigns numeric priority to each request.
//!   Useful for API tiering (paid vs free users).
//!
//! ## Interface sketch
//!
//! ```rust
//! pub trait Policy {
//!     fn sort_by_priority<'a>(
//!         &self,
//!         now:      Instant,
//!         seq_groups: &'a [SequenceGroup],
//!     ) -> Vec<&'a SequenceGroup>;
//! }
//!
//! pub struct FcfsPolicy;
//! impl Policy for FcfsPolicy { ... }
//! ```

#![allow(dead_code)]

pub struct FcfsPolicy;
