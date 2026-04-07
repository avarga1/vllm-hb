//! Continuous batching scheduler (paged KV cache).
//!
//! The scheduler drives the vLLM continuous batching loop:
//!
//! ```text
//!  loop {
//!      outputs = scheduler.schedule();
//!      // prefill outputs.to_prefill, decode outputs.to_decode
//!      scheduler.update(outputs);
//!  }
//! ```
//!
//! # Queues
//! - `waiting`  — new requests, not yet allocated blocks
//! - `running`  — active sequences with GPU blocks
//! - `swapped`  — preempted sequences with blocks on CPU
//!
//! # Preemption
//! When GPU memory is exhausted the lowest-priority running group is preempted:
//! swap to CPU if CPU space exists, otherwise free and requeue for recompute.
//! Swapped groups are swapped back in when GPU memory becomes available.

pub mod block_manager;
pub mod policy;
pub mod sequence;

use std::collections::{HashSet, VecDeque};
use std::time::Instant;

use crate::scheduler::{
    block_manager::BlockManager,
    policy::{FcfsPolicy, Policy},
    sequence::{SequenceGroup, SequenceStatus},
};

// ── Scheduler outputs ─────────────────────────────────────────────────────────

/// The decision produced by one `schedule()` call.
pub struct SchedulerOutputs {
    /// Groups whose prompt tokens need a prefill forward pass.
    pub to_prefill: Vec<SequenceGroup>,
    /// Groups whose next token needs a decode forward pass.
    pub to_decode: Vec<SequenceGroup>,
    /// `(cpu_block, gpu_block)` pairs: blocks moved CPU → GPU this step.
    #[allow(dead_code)]
    pub blocks_to_swap_in: Vec<(usize, usize)>,
    /// `(gpu_block, cpu_block)` pairs: blocks moved GPU → CPU this step.
    #[allow(dead_code)]
    pub blocks_to_swap_out: Vec<(usize, usize)>,
    /// `(src_block, dst_block)` copy-on-write pairs the engine must handle.
    #[allow(dead_code)]
    pub blocks_to_copy: Vec<(usize, usize)>,
}

impl SchedulerOutputs {
    fn empty() -> Self {
        Self {
            to_prefill: Vec::new(),
            to_decode: Vec::new(),
            blocks_to_swap_in: Vec::new(),
            blocks_to_swap_out: Vec::new(),
            blocks_to_copy: Vec::new(),
        }
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.to_prefill.is_empty() && self.to_decode.is_empty()
    }
}

// ── Scheduler ─────────────────────────────────────────────────────────────────

pub struct Scheduler {
    pub block_manager: BlockManager,
    policy: Box<dyn Policy + Send + Sync>,
    waiting: VecDeque<SequenceGroup>,
    running: Vec<SequenceGroup>,
    swapped: Vec<SequenceGroup>,
}

impl Scheduler {
    pub fn new(num_gpu_blocks: usize, num_cpu_blocks: usize) -> Self {
        Self {
            block_manager: BlockManager::new(num_gpu_blocks, num_cpu_blocks),
            policy: Box::new(FcfsPolicy),
            waiting: VecDeque::new(),
            running: Vec::new(),
            swapped: Vec::new(),
        }
    }

    /// Enqueue a new request.
    pub fn add_sequence_group(&mut self, group: SequenceGroup) {
        self.waiting.push_back(group);
    }

    pub fn num_waiting(&self) -> usize {
        self.waiting.len()
    }

    pub fn num_running(&self) -> usize {
        self.running.len()
    }

    pub fn num_swapped(&self) -> usize {
        self.swapped.len()
    }

    // ── schedule ──────────────────────────────────────────────────────────────

    /// Decide what to run this step.
    ///
    /// 1. Swap swapped groups back to GPU (highest priority first, stop when OOM).
    /// 2. Check running groups can append; preempt lowest-priority ones if OOM.
    /// 3. Admit waiting groups into GPU (FCFS order, stop when OOM).
    pub fn schedule(&mut self) -> SchedulerOutputs {
        let mut out = SchedulerOutputs::empty();
        let block_size = self.block_manager.block_size();

        // ── Phase 1: swap-in swapped groups ───────────────────────────────────
        let swap_in_indices = self.priority_indices(&self.swapped);
        let mut keep_swapped: Vec<bool> = vec![true; self.swapped.len()];

        for idx in swap_in_indices {
            let needed = self.swapped[idx].num_logical_blocks(block_size);
            if self.block_manager.num_free_gpu_blocks() < needed {
                break;
            }
            keep_swapped[idx] = false;
        }

        // Drain swapped into a temp vec, process, put non-swapped-in back.
        let old_swapped: Vec<SequenceGroup> = self.swapped.drain(..).collect();
        for (mut group, keep) in old_swapped.into_iter().zip(keep_swapped) {
            if keep {
                self.swapped.push(group);
            } else {
                match self.block_manager.swap_in(&group) {
                    Ok(mapping) => {
                        out.blocks_to_swap_in.extend(mapping);
                        for seq in &mut group.seqs {
                            if seq.status == SequenceStatus::Swapped {
                                seq.status = SequenceStatus::Running;
                            }
                        }
                        self.running.push(group);
                    }
                    Err(_) => self.swapped.push(group),
                }
            }
        }

        // ── Phase 2: running groups — append slot or preempt ─────────────────
        // Sort running by priority (highest first); preempt from the tail.
        let run_indices = self.priority_indices(&self.running);
        let mut to_preempt: HashSet<usize> = HashSet::new();

        for &idx in run_indices.iter().rev() {
            if !self.block_manager.can_append_slot(&self.running[idx]) {
                to_preempt.insert(idx);
            }
        }

        let old_running: Vec<SequenceGroup> = self.running.drain(..).collect();
        for (mut group, idx) in old_running.into_iter().zip(0usize..) {
            if to_preempt.contains(&idx) {
                let needed = group.num_logical_blocks(block_size);
                if self.block_manager.num_free_cpu_blocks() >= needed {
                    // Swap to CPU.
                    if let Ok(mapping) = self.block_manager.swap_out(&group) {
                        out.blocks_to_swap_out.extend(mapping);
                        for seq in &mut group.seqs {
                            if seq.status == SequenceStatus::Running {
                                seq.status = SequenceStatus::Swapped;
                            }
                        }
                        self.swapped.push(group);
                    }
                } else {
                    // Free and requeue — engine will recompute the prompt.
                    self.block_manager.free(&group);
                    for seq in &mut group.seqs {
                        seq.status = SequenceStatus::Waiting;
                        seq.output_ids.clear();
                    }
                    self.waiting.push_front(group);
                }
            } else {
                self.running.push(group);
            }
        }

        // Append token slots for all remaining running groups.
        for group in &self.running {
            if let Ok(cow) = self.block_manager.append_slot(group) {
                out.blocks_to_copy.extend(cow);
            }
        }

        // ── Phase 3: admit waiting groups ─────────────────────────────────────
        // Sort by priority, greedily admit while GPU blocks remain.
        // Track free blocks during the scan so we don't over-admit.
        let waiting_vec: Vec<SequenceGroup> = self.waiting.drain(..).collect();
        let admit_indices = self.priority_indices(&waiting_vec);

        let mut free_remaining = self.block_manager.num_free_gpu_blocks();
        let mut will_admit: HashSet<usize> = HashSet::new();
        for idx in &admit_indices {
            let needed = waiting_vec[*idx].num_logical_blocks(block_size);
            if needed > free_remaining {
                break;
            }
            free_remaining -= needed;
            will_admit.insert(*idx);
        }

        // Consume vec into Options so we can take by index in priority order.
        let mut waiting_opts: Vec<Option<SequenceGroup>> =
            waiting_vec.into_iter().map(Some).collect();

        // Admit in priority order so to_prefill is ordered highest → lowest.
        for idx in &admit_indices {
            if will_admit.contains(idx) {
                let mut group = waiting_opts[*idx].take().unwrap();
                if self.block_manager.allocate(&group).is_ok() {
                    for seq in &mut group.seqs {
                        seq.status = SequenceStatus::Running;
                    }
                    out.to_prefill.push(group);
                }
            }
        }

        // Return non-admitted groups to the waiting queue.
        for opt in waiting_opts.into_iter().flatten() {
            self.waiting.push_back(opt);
        }

        // Move current running groups to to_decode.
        out.to_decode.extend(self.running.drain(..));

        out
    }

    // ── update ────────────────────────────────────────────────────────────────

    /// Feed engine outputs back to the scheduler.
    ///
    /// Finished groups are freed; all others return to `running`.
    pub fn update(&mut self, mut outputs: SchedulerOutputs) {
        for group in outputs
            .to_prefill
            .drain(..)
            .chain(outputs.to_decode.drain(..))
        {
            if group.is_finished() {
                self.block_manager.free(&group);
            } else {
                self.running.push(group);
            }
        }
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    /// Return indices into `groups` sorted by policy priority (highest first).
    fn priority_indices(&self, groups: &[SequenceGroup]) -> Vec<usize> {
        let sorted = self.policy.sort_by_priority(Instant::now(), groups);
        sorted
            .iter()
            .map(|g| groups.iter().position(|s| std::ptr::eq(s, *g)).unwrap())
            .collect()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::sequence::Sequence;
    use crate::types::pipeline::SamplingParams;
    use tokio::sync::mpsc::unbounded_channel;

    fn make_group(id: &str, prompt_len: usize) -> SequenceGroup {
        let (tx, _) = unbounded_channel();
        let seq = Sequence::new(0, vec![0u32; prompt_len], SamplingParams::default(), tx);
        SequenceGroup::new(id.into(), vec![seq])
    }

    #[test]
    fn schedule_admits_waiting_group() {
        let mut sched = Scheduler::new(8, 4);
        sched.add_sequence_group(make_group("r0", 16)); // needs 1 block
        assert_eq!(sched.num_waiting(), 1);

        let out = sched.schedule();
        assert_eq!(out.to_prefill.len(), 1);
        assert_eq!(out.to_prefill[0].request_id, "r0");
        assert_eq!(sched.num_waiting(), 0);
    }

    #[test]
    fn schedule_respects_gpu_limit() {
        // 1 GPU block → only first group fits (both need 1 block each).
        let mut sched = Scheduler::new(1, 0);
        sched.add_sequence_group(make_group("r0", 16));
        sched.add_sequence_group(make_group("r1", 16));

        let out = sched.schedule();
        assert_eq!(out.to_prefill.len(), 1);
        assert_eq!(sched.num_waiting(), 1);
    }

    #[test]
    fn update_moves_prefill_to_running() {
        let mut sched = Scheduler::new(8, 4);
        sched.add_sequence_group(make_group("r0", 16));

        let out = sched.schedule();
        assert_eq!(sched.num_running(), 0); // still in to_prefill, not yet running

        sched.update(out);
        assert_eq!(sched.num_running(), 1);
    }

    #[test]
    fn finished_group_freed_on_update() {
        let mut sched = Scheduler::new(8, 4);
        sched.add_sequence_group(make_group("r0", 16));

        let mut out = sched.schedule();
        for g in &mut out.to_prefill {
            for s in &mut g.seqs {
                s.status = SequenceStatus::Finished(crate::types::pipeline::FinishReason::Stop);
            }
        }
        sched.update(out);
        assert_eq!(sched.num_running(), 0);
        assert_eq!(sched.block_manager.num_free_gpu_blocks(), 8);
    }

    #[test]
    fn fcfs_order_respected_across_waiting_queue() {
        use std::time::Duration;
        let mut sched = Scheduler::new(8, 0);

        let mut g_old = make_group("old", 16);
        let mut g_new = make_group("new", 16);
        g_old.arrival_time = Instant::now() - Duration::from_millis(100);
        g_new.arrival_time = Instant::now() - Duration::from_millis(10);

        // Enqueue new before old — FCFS should still put old first.
        sched.add_sequence_group(g_new);
        sched.add_sequence_group(g_old);

        let out = sched.schedule();
        assert_eq!(out.to_prefill.len(), 2);
        assert_eq!(out.to_prefill[0].request_id, "old");
        assert_eq!(out.to_prefill[1].request_id, "new");
    }

    #[test]
    fn preempted_group_goes_to_swapped_when_cpu_available() {
        // 1 GPU block: admit r0, then on next step it can't append (no free blocks).
        let mut sched = Scheduler::new(1, 4);
        sched.add_sequence_group(make_group("r0", 16));

        // Step 1: admit r0 into prefill.
        let out = sched.schedule();
        assert_eq!(out.to_prefill.len(), 1);
        sched.update(out); // r0 now running

        // Step 2: r0 needs to append but GPU is full → preempt → swap to CPU.
        let out2 = sched.schedule();
        assert!(!out2.blocks_to_swap_out.is_empty());
        assert_eq!(sched.num_swapped(), 1);
        assert_eq!(sched.block_manager.num_free_gpu_blocks(), 1); // freed
    }
}
