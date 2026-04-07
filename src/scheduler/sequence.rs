//! Per-sequence state machine for continuous batching.
//!
//! # Status: STUB
//!
//! ## Sequence lifecycle
//!
//! ```text
//!  Waiting ──► Running ──► Finished
//!                │
//!                └──► Swapped ──► Running  (preempted, blocks moved to CPU)
//! ```
//!
//! ## Key types to implement
//!
//! ```rust
//! pub enum SequenceStatus { Waiting, Running, Swapped, Finished(FinishReason) }
//!
//! pub struct Sequence {
//!     id:          u64,
//!     prompt_ids:  Vec<u32>,          // original prompt tokens
//!     output_ids:  Vec<u32>,          // tokens generated so far
//!     block_table: Vec<usize>,        // logical → physical block mapping
//!     status:      SequenceStatus,
//!     params:      SamplingParams,
//!     result_tx:   UnboundedSender<GenerationEvent>,
//! }
//!
//! pub struct SequenceGroup {
//!     request_id: String,
//!     seqs:       Vec<Sequence>,      // >1 for beam search / spec decoding
//!     arrival_time: Instant,
//! }
//! ```

#![allow(dead_code)]

pub struct SequenceGroup;
pub struct Sequence;
