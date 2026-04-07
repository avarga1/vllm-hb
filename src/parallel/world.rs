//! Tensor-parallel world — tracks GPU devices and rank assignments.
//!
//! A `TpWorld` is created once at startup from `--tensor-parallel-size N`.
//! It detects available CUDA devices, assigns one device per rank, and
//! provides the context needed by `comm::` and `shard::`.
//!
//! # Single-GPU fast path
//!
//! When `world_size == 1` every operation in this module is a no-op or
//! trivial pass-through.  Single-GPU deployments pay zero overhead.
//!
//! # Multi-GPU path
//!
//! Each rank owns one `candle_core::Device`.  The world is constructed on
//! the main thread and then `Arc`-shared into per-rank worker tasks.
//!
//! ```text
//!  rank 0 → Device::Cuda(0)
//!  rank 1 → Device::Cuda(1)
//!  rank 2 → Device::Cuda(2)
//!  rank 3 → Device::Cuda(3)
//! ```

use std::sync::Arc;

use anyhow::{Result, bail};
use candle_core::Device;

// ── TpWorld ───────────────────────────────────────────────────────────────────

/// Shared tensor-parallel context.
///
/// Clone cheaply via the inner `Arc` — all clones share the same device list.
#[derive(Clone, Debug)]
pub struct TpWorld(Arc<Inner>);

#[derive(Debug)]
struct Inner {
    /// One device per rank. `devices[rank]` is the device for that rank.
    devices: Vec<Device>,
}

impl TpWorld {
    /// Build a world with `world_size` ranks.
    ///
    /// - `world_size = 1` → single CPU or CUDA:0 device, no communication.
    /// - `world_size > 1` → requires at least `world_size` CUDA devices.
    ///
    /// # Errors
    ///
    /// Returns an error if `world_size > 1` and fewer than `world_size`
    /// CUDA devices are available.
    pub fn new(world_size: usize) -> Result<Self> {
        assert!(world_size >= 1, "world_size must be >= 1");

        if world_size == 1 {
            // Single device — CUDA if available, CPU otherwise.
            let device = Device::cuda_if_available(0)?;
            return Ok(Self(Arc::new(Inner {
                devices: vec![device],
            })));
        }

        // Multi-GPU: verify we have enough devices.
        let available = Self::cuda_device_count();
        if available < world_size {
            bail!(
                "tensor_parallel_size={world_size} requested but only \
                 {available} CUDA device(s) detected. \
                 Use --tensor-parallel-size 1 for single-GPU."
            );
        }

        let devices = (0..world_size)
            .map(|rank| Device::new_cuda(rank))
            .collect::<candle_core::Result<Vec<_>>>()?;

        Ok(Self(Arc::new(Inner { devices })))
    }

    /// Number of ranks (GPUs) in this world.
    #[inline]
    pub fn world_size(&self) -> usize {
        self.0.devices.len()
    }

    /// The device assigned to `rank`.
    ///
    /// # Panics
    ///
    /// Panics if `rank >= world_size()`.
    #[inline]
    pub fn device(&self, rank: usize) -> &Device {
        &self.0.devices[rank]
    }

    /// Iterator over `(rank, device)` pairs.
    pub fn ranks(&self) -> impl Iterator<Item = (usize, &Device)> {
        self.0.devices.iter().enumerate()
    }

    /// Returns `true` when running on a single GPU (no communication needed).
    #[inline]
    pub fn is_single(&self) -> bool {
        self.0.devices.len() == 1
    }

    // ── Private ───────────────────────────────────────────────────────────────

    /// Count available CUDA devices via candle.
    ///
    /// Probes devices 0..31; stops at the first failure.
    fn cuda_device_count() -> usize {
        (0..32).take_while(|&i| Device::new_cuda(i).is_ok()).count()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_size_1_always_works() {
        // world_size=1 must succeed on any machine (CPU fallback).
        let world = TpWorld::new(1).expect("world_size=1 should always succeed");
        assert_eq!(world.world_size(), 1);
        assert!(world.is_single());
    }

    #[test]
    fn device_rank_0_accessible() {
        let world = TpWorld::new(1).unwrap();
        // Just assert it doesn't panic — we don't care if it's CPU or CUDA.
        let _dev = world.device(0);
    }

    #[test]
    fn ranks_iterator_yields_all() {
        let world = TpWorld::new(1).unwrap();
        let pairs: Vec<_> = world.ranks().collect();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, 0); // rank 0
    }

    #[test]
    fn world_size_too_large_errors() {
        // Request more GPUs than any reasonable CI machine has.
        // This should fail gracefully, not panic.
        let result = TpWorld::new(64);
        // Either it succeeds (64 GPUs somehow exist) or returns a clean error.
        if result.is_err() {
            let msg = result.unwrap_err().to_string();
            assert!(
                msg.contains("tensor_parallel_size") || msg.contains("CUDA"),
                "error message should mention tensor_parallel_size or CUDA, got: {msg}"
            );
        }
    }

    #[test]
    fn clone_shares_inner() {
        let world = TpWorld::new(1).unwrap();
        let clone = world.clone();
        // Both should report the same world_size — they share the Arc.
        assert_eq!(world.world_size(), clone.world_size());
    }
}
