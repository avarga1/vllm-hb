//! Paged KV-cache block allocator.
//!
//! # Status: STUB
//!
//! ## Concept
//!
//! The KV cache is divided into fixed-size physical blocks (e.g. 16 tokens
//! per block).  Each sequence gets a "block table" mapping logical block
//! indices to physical blocks.  When the physical pool is exhausted, the
//! scheduler preempts low-priority sequences (swapping their blocks to CPU
//! RAM or simply recomputing them) to free space for high-priority ones.
//!
//! ## Key types to implement
//!
//! ```rust
//! pub struct PhysicalBlock { id: usize, ref_count: usize }
//! pub struct BlockTable    { blocks: Vec<Option<usize>> }
//! pub struct BlockManager  {
//!     block_size:      usize,       // tokens per block
//!     num_gpu_blocks:  usize,
//!     num_cpu_blocks:  usize,
//!     free_gpu_blocks: Vec<usize>,
//!     free_cpu_blocks: Vec<usize>,
//!     gpu_allocator:   BlockAllocator,
//!     cpu_allocator:   BlockAllocator,
//! }
//! ```
//!
//! ## Key methods to implement
//!
//! - `allocate(seq_group) -> Result<()>`
//! - `free(seq_group)`
//! - `can_allocate(seq_group) -> bool`
//! - `can_append_slot(seq_group) -> bool`
//! - `append_slot(seq_group) -> Option<(src, dst)>` — copy-on-write slot
//! - `swap_in(seq_group)` — CPU → GPU
//! - `swap_out(seq_group)` — GPU → CPU
//! - `get_block_table(seq_id) -> &[usize]`

#![allow(dead_code)]

/// Number of tokens per KV-cache block.
pub const BLOCK_SIZE: usize = 16;

pub struct BlockManager;

impl BlockManager {
    pub fn new(_num_gpu_blocks: usize, _num_cpu_blocks: usize) -> Self {
        todo!("BlockManager: paged KV cache not yet implemented")
    }
}
