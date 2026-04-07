//! Per-sequence state machine for continuous batching.
//!
//! ```text
//! Waiting ──► Running ──► Finished
//!               │
//!               └──► Swapped ──► Running   (preempted; blocks on CPU)
//! ```

use std::time::Instant;

use tokio::sync::mpsc::UnboundedSender;

use crate::types::pipeline::{FinishReason, GenerationEvent, SamplingParams};

// ── Status ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SequenceStatus {
    Waiting,
    Running,
    Swapped,
    Finished(FinishReason),
}

impl SequenceStatus {
    pub fn is_finished(&self) -> bool {
        matches!(self, Self::Finished(_))
    }
}

// ── Sequence ──────────────────────────────────────────────────────────────────

pub struct Sequence {
    pub id: u64,
    pub prompt_ids: Vec<u32>,
    pub output_ids: Vec<u32>,
    pub status: SequenceStatus,
    #[allow(dead_code)]
    pub params: SamplingParams,
    #[allow(dead_code)]
    pub result_tx: UnboundedSender<GenerationEvent>,
}

impl Sequence {
    pub fn new(
        id: u64,
        prompt_ids: Vec<u32>,
        params: SamplingParams,
        result_tx: UnboundedSender<GenerationEvent>,
    ) -> Self {
        Self {
            id,
            prompt_ids,
            output_ids: Vec::new(),
            status: SequenceStatus::Waiting,
            params,
            result_tx,
        }
    }

    /// Total tokens (prompt + generated).
    pub fn len(&self) -> usize {
        self.prompt_ids.len() + self.output_ids.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[allow(dead_code)]
    pub fn all_token_ids(&self) -> Vec<u32> {
        self.prompt_ids
            .iter()
            .chain(self.output_ids.iter())
            .copied()
            .collect()
    }

    pub fn num_logical_blocks(&self, block_size: usize) -> usize {
        self.len().div_ceil(block_size)
    }

    /// Slots used in the last (possibly partial) block.
    pub fn last_block_num_filled(&self, block_size: usize) -> usize {
        let rem = self.len() % block_size;
        if rem == 0 && !self.is_empty() {
            block_size
        } else {
            rem
        }
    }
}

// ── SequenceGroup ─────────────────────────────────────────────────────────────

pub struct SequenceGroup {
    pub request_id: String,
    pub seqs: Vec<Sequence>,
    pub arrival_time: Instant,
}

impl SequenceGroup {
    pub fn new(request_id: String, seqs: Vec<Sequence>) -> Self {
        Self {
            request_id,
            seqs,
            arrival_time: Instant::now(),
        }
    }

    #[allow(dead_code)]
    pub fn max_len(&self) -> usize {
        self.seqs.iter().map(|s| s.len()).max().unwrap_or(0)
    }

    pub fn num_logical_blocks(&self, block_size: usize) -> usize {
        self.seqs
            .iter()
            .map(|s| s.num_logical_blocks(block_size))
            .max()
            .unwrap_or(0)
    }

    pub fn is_finished(&self) -> bool {
        self.seqs.iter().all(|s| s.status.is_finished())
    }

    pub fn running_seqs(&self) -> impl Iterator<Item = &Sequence> {
        self.seqs
            .iter()
            .filter(|s| s.status == SequenceStatus::Running)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::mpsc::unbounded_channel;

    fn make_seq(id: u64, prompt_len: usize) -> Sequence {
        let (tx, _) = unbounded_channel();
        Sequence::new(id, vec![0u32; prompt_len], SamplingParams::default(), tx)
    }

    #[test]
    fn seq_len_is_prompt_plus_output() {
        let mut seq = make_seq(0, 10);
        assert_eq!(seq.len(), 10);
        seq.output_ids.push(42);
        assert_eq!(seq.len(), 11);
    }

    #[test]
    fn num_logical_blocks_rounds_up() {
        assert_eq!(make_seq(0, 17).num_logical_blocks(16), 2);
        assert_eq!(make_seq(0, 16).num_logical_blocks(16), 1);
        assert_eq!(make_seq(0, 1).num_logical_blocks(16), 1);
    }

    #[test]
    fn last_block_filled_partial() {
        assert_eq!(make_seq(0, 17).last_block_num_filled(16), 1);
    }

    #[test]
    fn last_block_filled_full() {
        assert_eq!(make_seq(0, 32).last_block_num_filled(16), 16);
    }

    #[test]
    fn seq_group_max_len() {
        let grp = SequenceGroup::new("r0".into(), vec![make_seq(0, 10), make_seq(1, 20)]);
        assert_eq!(grp.max_len(), 20);
    }

    #[test]
    fn seq_group_finished_when_all_done() {
        let mut s1 = make_seq(0, 5);
        let mut s2 = make_seq(1, 5);
        s1.status = SequenceStatus::Finished(FinishReason::Stop);
        s2.status = SequenceStatus::Finished(FinishReason::Length);
        assert!(SequenceGroup::new("r0".into(), vec![s1, s2]).is_finished());
    }

    #[test]
    fn seq_group_not_finished_while_any_running() {
        let mut s1 = make_seq(0, 5);
        s1.status = SequenceStatus::Finished(FinishReason::Stop);
        let s2 = make_seq(1, 5); // still Waiting
        assert!(!SequenceGroup::new("r0".into(), vec![s1, s2]).is_finished());
    }
}
