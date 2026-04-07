//! Continuous batching scheduler (paged KV cache).
//!
//! # Status: STUB — the largest single roadmap item
//!
//! This is what turns vllm-hb from a single-sequence server into a true
//! high-throughput inference engine.  Nothing in the current serving path
//! imports this module yet.
//!
//! # Modules
//! - `block_manager` — paged KV-cache physical block allocator
//! - `sequence`      — per-sequence state machine (Waiting/Running/Swapped/Finished)
//! - `policy`        — preemption and priority policies (FCFS, priority)
//!
//! # Integration plan
//!
//! When this is real, `worker/mod.rs` is updated to:
//! 1. Hold a `Scheduler` instead of raw channels
//! 2. Call `scheduler.add_sequence_group(work_item)` on new requests
//! 3. Each decode step: call `scheduler.schedule()` to get a `SchedulerOutputs`
//!    containing the sequences to prefill and those to decode
//! 4. Run a batched forward pass over all ready sequences
//! 5. Call `scheduler.update(outputs)` to advance state

pub mod block_manager;
pub mod policy;
pub mod sequence;
