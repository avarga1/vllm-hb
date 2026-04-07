//! Scheduling and preemption policies.
//!
//! `Policy` decides which sequence groups to run when GPU memory is scarce.
//! Keeping it as a trait means you can swap FCFS for priority-based scheduling
//! without touching the allocator.

use std::time::Instant;

use super::sequence::SequenceGroup;

// ── Trait ─────────────────────────────────────────────────────────────────────

pub trait Policy {
    /// Sort `seq_groups` by priority, highest priority first.
    ///
    /// `now` is the current instant — policies can use it to compute
    /// waiting time or age-based priority.
    fn sort_by_priority<'a>(
        &self,
        now: Instant,
        seq_groups: &'a [SequenceGroup],
    ) -> Vec<&'a SequenceGroup>;
}

// ── FCFS ──────────────────────────────────────────────────────────────────────

/// First-come-first-served — the default vLLM policy.
///
/// Groups that arrived earlier have higher priority.  When GPU memory is
/// exhausted, the most recently arrived group is preempted first.
pub struct FcfsPolicy;

impl Policy for FcfsPolicy {
    fn sort_by_priority<'a>(
        &self,
        _now: Instant,
        seq_groups: &'a [SequenceGroup],
    ) -> Vec<&'a SequenceGroup> {
        let mut sorted: Vec<&SequenceGroup> = seq_groups.iter().collect();
        sorted.sort_by_key(|g| g.arrival_time);
        sorted
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::sequence::Sequence;
    use crate::types::pipeline::SamplingParams;
    use std::time::Duration;
    use tokio::sync::mpsc::unbounded_channel;

    fn make_group(id: &str, delay_ms: u64) -> SequenceGroup {
        let (tx, _) = unbounded_channel();
        let seq = Sequence::new(0, vec![1u32], SamplingParams::default(), tx);
        let mut g = SequenceGroup::new(id.into(), vec![seq]);
        // Backdate arrival time to simulate earlier arrival
        g.arrival_time = Instant::now() - Duration::from_millis(delay_ms);
        g
    }

    #[test]
    fn fcfs_sorts_oldest_first() {
        let groups = vec![
            make_group("new", 10),   // arrived 10ms ago
            make_group("old", 100),  // arrived 100ms ago
            make_group("mid", 50),   // arrived 50ms ago
        ];
        let policy = FcfsPolicy;
        let sorted = policy.sort_by_priority(Instant::now(), &groups);
        assert_eq!(sorted[0].request_id, "old");
        assert_eq!(sorted[1].request_id, "mid");
        assert_eq!(sorted[2].request_id, "new");
    }

    #[test]
    fn fcfs_single_group_unchanged() {
        let groups = vec![make_group("only", 0)];
        let sorted = FcfsPolicy.sort_by_priority(Instant::now(), &groups);
        assert_eq!(sorted.len(), 1);
        assert_eq!(sorted[0].request_id, "only");
    }
}
