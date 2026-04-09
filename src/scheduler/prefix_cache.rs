//! Automatic prefix caching — block-level KV hash deduplication.
//!
//! When multiple requests share a common prompt prefix (e.g. the same system
//! prompt or few-shot examples) the KV computation for those tokens only needs
//! to happen once.  Subsequent requests can reuse the cached physical blocks
//! and skip the corresponding portion of the prefill forward pass.
//!
//! # How it works
//!
//! Each completed (full) block of 16 tokens is identified by a 64-bit FNV-1a
//! hash of its token IDs.  When a sequence finishes and its blocks are freed,
//! the block manager registers them here instead of returning them immediately
//! to the free pool.  Future sequences that produce the same hash for a given
//! logical block position get the cached physical block back (ref-count bumped)
//! rather than a fresh allocation.
//!
//! # Capacity and eviction
//!
//! The cache is bounded by `capacity` entries (default: 128 blocks).  When the
//! cache is full the oldest inserted entry is evicted (FIFO approximation of
//! LRU — good enough for shared system-prompt workloads where the hot prefix
//! stays warm continuously).
//!
//! # Limitations
//!
//! Block sharing only saves KV *computation* when the engine can populate a
//! new sequence's cache from the cached tensors.  Tensor-level KV restore is
//! architecture-specific (see `PerSeqCache::try_restore_prefix`).  Even
//! without tensor reuse, shared blocks reduce GPU memory pressure and prefill
//! allocation latency.

use std::collections::{HashMap, VecDeque};

// ── Hash ─────────────────────────────────────────────────────────────────────

/// FNV-1a 64-bit hash of a token-id slice.
///
/// Non-cryptographic, extremely fast, and collision-resistant enough for
/// block deduplication over a 32 k vocab.
pub fn hash_block(tokens: &[u32]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET;
    for &tok in tokens {
        hash ^= u64::from(tok);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

// ── PrefixCache ───────────────────────────────────────────────────────────────

/// LRU-approximated cache mapping token-block hashes → physical block IDs.
pub struct PrefixCache {
    capacity: usize,
    /// hash → physical block id
    entries: HashMap<u64, usize>,
    /// Insertion-order queue for eviction (oldest first).
    order: VecDeque<u64>,
    /// Reverse map: block_id → hash, for efficient `remove_by_block_id`.
    block_to_hash: HashMap<usize, u64>,
}

impl PrefixCache {
    /// Create a new cache with the given maximum number of entries.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            entries: HashMap::new(),
            order: VecDeque::new(),
            block_to_hash: HashMap::new(),
        }
    }

    /// Look up a cached physical block for the given token-block hash.
    ///
    /// Returns `Some(block_id)` on a hit, `None` on a miss.
    pub fn lookup(&self, hash: u64) -> Option<usize> {
        self.entries.get(&hash).copied()
    }

    /// Register a physical block as a cached prefix candidate.
    ///
    /// If the cache is at capacity the oldest entry is evicted first.  Callers
    /// must NOT insert a block that is still actively used by a sequence.
    pub fn insert(&mut self, hash: u64, block_id: usize) {
        // If already present (e.g. re-freed after CoW), just update block_id.
        if let Some(&old_id) = self.entries.get(&hash) {
            if old_id != block_id {
                self.block_to_hash.remove(&old_id);
            }
            self.entries.insert(hash, block_id);
            self.block_to_hash.insert(block_id, hash);
            return;
        }

        // Evict the oldest entry if at capacity.
        if self.entries.len() >= self.capacity
            && let Some(evicted_hash) = self.order.pop_front()
            && let Some(evicted_id) = self.entries.remove(&evicted_hash)
        {
            self.block_to_hash.remove(&evicted_id);
        }

        self.entries.insert(hash, block_id);
        self.order.push_back(hash);
        self.block_to_hash.insert(block_id, hash);
    }

    /// Remove a block from the cache by its physical block ID.
    ///
    /// Called when a block is permanently reclaimed (evicted from the free
    /// pool by the allocator, not just freed by a sequence).
    pub fn remove_by_block_id(&mut self, block_id: usize) {
        if let Some(hash) = self.block_to_hash.remove(&block_id) {
            self.entries.remove(&hash);
            // Leave the stale entry in `order` — it will be skipped on eviction
            // because the hash is no longer in `entries`.
        }
    }

    /// Total number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` when the cache holds no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Configured capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_block_deterministic() {
        let tokens = vec![1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        assert_eq!(hash_block(&tokens), hash_block(&tokens));
    }

    #[test]
    fn hash_block_different_tokens_differ() {
        let a = vec![1u32; 16];
        let b = vec![2u32; 16];
        assert_ne!(hash_block(&a), hash_block(&b));
    }

    #[test]
    fn hash_block_order_sensitive() {
        let a = vec![1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let mut b = a.clone();
        b.swap(0, 15);
        assert_ne!(hash_block(&a), hash_block(&b));
    }

    #[test]
    fn lookup_miss_on_empty() {
        let cache = PrefixCache::new(8);
        assert!(cache.lookup(42).is_none());
    }

    #[test]
    fn insert_then_lookup() {
        let mut cache = PrefixCache::new(8);
        cache.insert(0xdeadbeef, 7);
        assert_eq!(cache.lookup(0xdeadbeef), Some(7));
    }

    #[test]
    fn evicts_oldest_when_full() {
        let mut cache = PrefixCache::new(2);
        cache.insert(1, 10);
        cache.insert(2, 20);
        cache.insert(3, 30); // evicts hash=1
        assert!(cache.lookup(1).is_none(), "oldest should be evicted");
        assert_eq!(cache.lookup(2), Some(20));
        assert_eq!(cache.lookup(3), Some(30));
    }

    #[test]
    fn remove_by_block_id_clears_entry() {
        let mut cache = PrefixCache::new(8);
        cache.insert(0xabc, 5);
        cache.remove_by_block_id(5);
        assert!(cache.lookup(0xabc).is_none());
    }

    #[test]
    fn len_tracks_entries() {
        let mut cache = PrefixCache::new(8);
        assert_eq!(cache.len(), 0);
        cache.insert(1, 10);
        cache.insert(2, 20);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn insert_same_hash_updates_block_id() {
        let mut cache = PrefixCache::new(8);
        cache.insert(0xff, 3);
        cache.insert(0xff, 7); // updated
        assert_eq!(cache.lookup(0xff), Some(7));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn capacity_accessible() {
        let cache = PrefixCache::new(64);
        assert_eq!(cache.capacity(), 64);
    }
}
