//! Paged KV-cache block allocator.
//!
//! The KV cache is split into fixed-size physical blocks (default 16 tokens).
//! Each sequence gets a block table mapping logical block indices to physical
//! block IDs.  When GPU memory is exhausted the scheduler can swap blocks to
//! CPU RAM or preempt and recompute.
//!
//! ```text
//!  Sequence A:  logical [0, 1, 2]  →  physical [4, 7, 12]
//!  Sequence B:  logical [0, 1]     →  physical [2, 9]
//!
//!  GPU block pool:  [0, 1, 3, 5, 6, 8, 10, 11, ...]  (free)
//! ```

use std::collections::{HashMap, VecDeque};

use anyhow::{Result, bail};

use super::sequence::SequenceGroup;

// ── Block size ────────────────────────────────────────────────────────────────

/// Tokens per KV-cache block.  Must divide evenly into max_seq_len.
pub const BLOCK_SIZE: usize = 16;

// ── Physical block ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct PhysicalBlock {
    block_id: usize,
    /// Reference count — >1 when shared via copy-on-write (beam search).
    ref_count: usize,
}

// ── Block allocator (one per device) ─────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockDevice {
    Gpu,
    Cpu,
}

#[allow(dead_code)]
struct BlockAllocator {
    device: BlockDevice,
    num_blocks: usize,
    free_blocks: VecDeque<usize>,
    allocated: HashMap<usize, PhysicalBlock>,
}

impl BlockAllocator {
    fn new(device: BlockDevice, num_blocks: usize) -> Self {
        Self {
            device,
            num_blocks,
            free_blocks: (0..num_blocks).collect(),
            allocated: HashMap::new(),
        }
    }

    fn allocate(&mut self) -> Option<usize> {
        let id = self.free_blocks.pop_front()?;
        self.allocated.insert(id, PhysicalBlock { block_id: id, ref_count: 1 });
        Some(id)
    }

    fn free(&mut self, block_id: usize) {
        if let Some(block) = self.allocated.get_mut(&block_id) {
            block.ref_count -= 1;
            if block.ref_count == 0 {
                self.allocated.remove(&block_id);
                self.free_blocks.push_back(block_id);
            }
        }
    }

    fn num_free(&self) -> usize {
        self.free_blocks.len()
    }

    fn ref_count(&self, block_id: usize) -> usize {
        self.allocated.get(&block_id).map(|b| b.ref_count).unwrap_or(0)
    }

    #[allow(dead_code)]
    fn increment_ref(&mut self, block_id: usize) {
        if let Some(b) = self.allocated.get_mut(&block_id) {
            b.ref_count += 1;
        }
    }
}

// ── Block table (per sequence) ────────────────────────────────────────────────

/// Maps logical block indices → physical block IDs.
#[derive(Debug, Default, Clone)]
pub struct BlockTable {
    pub blocks: Vec<usize>,
}

// ── BlockManager ──────────────────────────────────────────────────────────────

pub struct BlockManager {
    block_size: usize,
    gpu: BlockAllocator,
    cpu: BlockAllocator,
    /// seq_id → block table (GPU blocks).
    block_tables: HashMap<u64, BlockTable>,
}

impl BlockManager {
    pub fn new(num_gpu_blocks: usize, num_cpu_blocks: usize) -> Self {
        Self {
            block_size: BLOCK_SIZE,
            gpu: BlockAllocator::new(BlockDevice::Gpu, num_gpu_blocks),
            cpu: BlockAllocator::new(BlockDevice::Cpu, num_cpu_blocks),
            block_tables: HashMap::new(),
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn num_free_gpu_blocks(&self) -> usize {
        self.gpu.num_free()
    }

    pub fn num_free_cpu_blocks(&self) -> usize {
        self.cpu.num_free()
    }

    // ── Allocation ────────────────────────────────────────────────────────────

    /// `true` if enough free GPU blocks exist to run this sequence group.
    pub fn can_allocate(&self, group: &SequenceGroup) -> bool {
        let needed = group.num_logical_blocks(self.block_size);
        needed <= self.gpu.num_free()
    }

    /// Allocate GPU blocks for every sequence in the group.
    ///
    /// Call after `can_allocate` returns `true`.
    pub fn allocate(&mut self, group: &SequenceGroup) -> Result<()> {
        let needed = group.num_logical_blocks(self.block_size);
        if needed > self.gpu.num_free() {
            bail!("not enough GPU blocks: need {needed}, have {}", self.gpu.num_free());
        }
        for seq in &group.seqs {
            let mut table = BlockTable::default();
            for _ in 0..needed {
                let id = self.gpu.allocate().unwrap();
                table.blocks.push(id);
            }
            self.block_tables.insert(seq.id, table);
        }
        Ok(())
    }

    /// Free all GPU blocks held by a sequence group.
    pub fn free(&mut self, group: &SequenceGroup) {
        for seq in &group.seqs {
            if let Some(table) = self.block_tables.remove(&seq.id) {
                for block_id in table.blocks {
                    self.gpu.free(block_id);
                }
            }
        }
    }

    // ── Slot append ───────────────────────────────────────────────────────────

    /// `true` if all running sequences can append one more token slot.
    ///
    /// A slot is available when the last block is not full, OR there is a
    /// free GPU block to allocate.
    pub fn can_append_slot(&self, group: &SequenceGroup) -> bool {
        for seq in group.running_seqs() {
            let table = match self.block_tables.get(&seq.id) {
                Some(t) => t,
                None => return false,
            };
            let last_filled = seq.last_block_num_filled(self.block_size);
            let needs_new_block = last_filled == self.block_size;
            if needs_new_block && self.gpu.num_free() == 0 {
                return false;
            }
            // Copy-on-write: if last block is shared, need a free block to copy into.
            if let Some(&last_id) = table.blocks.last() {
                if self.gpu.ref_count(last_id) > 1 && self.gpu.num_free() == 0 {
                    return false;
                }
            }
        }
        true
    }

    /// Append a slot for the next token in each running sequence.
    ///
    /// Returns a list of `(src_block, dst_block)` copy-on-write pairs that
    /// the engine must handle before writing to the new slot.
    pub fn append_slot(&mut self, group: &SequenceGroup) -> Result<Vec<(usize, usize)>> {
        let mut cow_pairs = Vec::new();

        // Collect seq IDs to avoid borrow issues
        let seq_ids: Vec<u64> = group.running_seqs().map(|s| s.id).collect();
        let last_filled: HashMap<u64, usize> = group
            .running_seqs()
            .map(|s| (s.id, s.last_block_num_filled(self.block_size)))
            .collect();

        for seq_id in seq_ids {
            let filled = last_filled[&seq_id];
            let table = self
                .block_tables
                .get_mut(&seq_id)
                .ok_or_else(|| anyhow::anyhow!("no block table for seq {seq_id}"))?;

            if filled == self.block_size || table.blocks.is_empty() {
                // Last block full — allocate a new one.
                let new_id = self
                    .gpu
                    .allocate()
                    .ok_or_else(|| anyhow::anyhow!("no free GPU blocks for append"))?;
                table.blocks.push(new_id);
            } else {
                // Last block has space — but may need copy-on-write.
                let last_id = *table.blocks.last().unwrap();
                if self.gpu.ref_count(last_id) > 1 {
                    let new_id = self
                        .gpu
                        .allocate()
                        .ok_or_else(|| anyhow::anyhow!("no free GPU blocks for CoW"))?;
                    self.gpu.free(last_id); // decrement ref
                    *table.blocks.last_mut().unwrap() = new_id;
                    cow_pairs.push((last_id, new_id));
                }
            }
        }

        Ok(cow_pairs)
    }

    // ── Swap ──────────────────────────────────────────────────────────────────

    /// Swap a sequence group's blocks from GPU → CPU (preemption).
    ///
    /// Returns a map of `gpu_block_id → cpu_block_id` for the engine to
    /// copy KV tensors.
    pub fn swap_out(&mut self, group: &SequenceGroup) -> Result<HashMap<usize, usize>> {
        let mut mapping = HashMap::new();
        let seq_ids: Vec<u64> = group.seqs.iter().map(|s| s.id).collect();

        for seq_id in seq_ids {
            let table = self
                .block_tables
                .get_mut(&seq_id)
                .ok_or_else(|| anyhow::anyhow!("no block table for seq {seq_id}"))?;

            for gpu_id in table.blocks.iter_mut() {
                let cpu_id = self
                    .cpu
                    .allocate()
                    .ok_or_else(|| anyhow::anyhow!("no free CPU blocks for swap_out"))?;
                mapping.insert(*gpu_id, cpu_id);
                self.gpu.free(*gpu_id);
                *gpu_id = cpu_id; // table now tracks CPU blocks
            }
        }
        Ok(mapping)
    }

    /// Swap a sequence group's blocks from CPU → GPU (resume after preemption).
    ///
    /// Returns a map of `cpu_block_id → gpu_block_id`.
    pub fn swap_in(&mut self, group: &SequenceGroup) -> Result<HashMap<usize, usize>> {
        let mut mapping = HashMap::new();
        let seq_ids: Vec<u64> = group.seqs.iter().map(|s| s.id).collect();

        for seq_id in seq_ids {
            let table = self
                .block_tables
                .get_mut(&seq_id)
                .ok_or_else(|| anyhow::anyhow!("no block table for seq {seq_id}"))?;

            for cpu_id in table.blocks.iter_mut() {
                let gpu_id = self
                    .gpu
                    .allocate()
                    .ok_or_else(|| anyhow::anyhow!("no free GPU blocks for swap_in"))?;
                mapping.insert(*cpu_id, gpu_id);
                self.cpu.free(*cpu_id);
                *cpu_id = gpu_id;
            }
        }
        Ok(mapping)
    }

    // ── Introspection ─────────────────────────────────────────────────────────

    pub fn get_block_table(&self, seq_id: u64) -> Option<&BlockTable> {
        self.block_tables.get(&seq_id)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::sequence::Sequence;
    use crate::types::pipeline::SamplingParams;
    use tokio::sync::mpsc::unbounded_channel;

    fn make_group(request_id: &str, prompt_len: usize) -> SequenceGroup {
        let (tx, _) = unbounded_channel();
        let seq = Sequence::new(0, vec![0u32; prompt_len], SamplingParams::default(), tx);
        SequenceGroup::new(request_id.into(), vec![seq])
    }

    #[test]
    fn new_manager_has_correct_free_counts() {
        let bm = BlockManager::new(8, 4);
        assert_eq!(bm.num_free_gpu_blocks(), 8);
        assert_eq!(bm.num_free_cpu_blocks(), 4);
    }

    #[test]
    fn can_allocate_respects_free_blocks() {
        let bm = BlockManager::new(2, 0);
        let grp = make_group("r0", 16); // needs 1 block
        assert!(bm.can_allocate(&grp));

        let grp_big = make_group("r1", 48); // needs 3 blocks
        assert!(!bm.can_allocate(&grp_big));
    }

    #[test]
    fn allocate_reduces_free_count() {
        let mut bm = BlockManager::new(8, 0);
        let grp = make_group("r0", 17); // needs 2 blocks (ceil(17/16))
        bm.allocate(&grp).unwrap();
        assert_eq!(bm.num_free_gpu_blocks(), 6);
    }

    #[test]
    fn free_returns_blocks_to_pool() {
        let mut bm = BlockManager::new(8, 0);
        let grp = make_group("r0", 16);
        bm.allocate(&grp).unwrap();
        assert_eq!(bm.num_free_gpu_blocks(), 7);
        bm.free(&grp);
        assert_eq!(bm.num_free_gpu_blocks(), 8);
    }

    #[test]
    fn allocate_too_large_errors() {
        let mut bm = BlockManager::new(1, 0);
        let grp = make_group("r0", 48); // needs 3 blocks
        assert!(bm.allocate(&grp).is_err());
    }

    #[test]
    fn get_block_table_returns_correct_length() {
        let mut bm = BlockManager::new(8, 0);
        let grp = make_group("r0", 33); // needs 3 blocks (ceil(33/16))
        bm.allocate(&grp).unwrap();
        let table = bm.get_block_table(0).unwrap();
        assert_eq!(table.blocks.len(), 3);
    }

    #[test]
    fn swap_out_moves_to_cpu() {
        let mut bm = BlockManager::new(4, 4);
        let grp = make_group("r0", 16);
        bm.allocate(&grp).unwrap();
        assert_eq!(bm.num_free_gpu_blocks(), 3);

        let mapping = bm.swap_out(&grp).unwrap();
        assert_eq!(mapping.len(), 1);
        assert_eq!(bm.num_free_gpu_blocks(), 4); // GPU block returned
        assert_eq!(bm.num_free_cpu_blocks(), 3); // CPU block used
    }

    #[test]
    fn swap_in_restores_to_gpu() {
        let mut bm = BlockManager::new(4, 4);
        let grp = make_group("r0", 16);
        bm.allocate(&grp).unwrap();
        bm.swap_out(&grp).unwrap();
        bm.swap_in(&grp).unwrap();

        assert_eq!(bm.num_free_gpu_blocks(), 3); // back on GPU
        assert_eq!(bm.num_free_cpu_blocks(), 4); // CPU block returned
    }
}
