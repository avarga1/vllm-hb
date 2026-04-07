//! Tensor-parallel world — tracks GPU devices, rank assignments, and NCCL comms.
//!
//! A `TpWorld` is created once at startup from `--tensor-parallel-size N`.
//! It detects available CUDA devices, assigns one device per rank, and
//! provides the context needed by `comm::` and `shard::`.
//!
//! # Single-GPU fast path
//!
//! When `world_size == 1` every operation is a no-op or trivial pass-through.
//! Single-GPU deployments pay zero overhead.
//!
//! # Multi-GPU path
//!
//! ```text
//!  rank 0 → Device::Cuda(0)
//!  rank 1 → Device::Cuda(1)
//!  rank 2 → Device::Cuda(2)
//!  rank 3 → Device::Cuda(3)
//! ```
//!
//! # NCCL (`--features nccl`)
//!
//! When built with the `nccl` feature, `TpWorld` initialises one
//! `ncclComm_t` per rank using `Comm::from_devices`.  `all_reduce` then
//! replaces the host-mediated CPU accumulation with a true device-to-device
//! ring-reduce via `ncclAllReduce`, eliminating the GPU→CPU→GPU round-trip
//! that dominates TP communication cost on large tensors.
//!
//! Single-process multi-GPU NCCL uses `group_start` / `group_end` to batch
//! sequential per-rank calls so NCCL treats them as a single collective.

use std::sync::Arc;

use anyhow::{Result, bail};
use candle_core::Device;

// ── TpWorld ───────────────────────────────────────────────────────────────────

/// Shared tensor-parallel context.
///
/// Clone cheaply via the inner `Arc` — all clones share the same device list
/// and NCCL communicators.
#[derive(Clone, Debug)]
pub struct TpWorld(Arc<Inner>);

#[derive(Debug)]
struct Inner {
    /// One device per rank.
    devices: Vec<Device>,
    /// NCCL communicators, one per rank.  Present only when `nccl` feature
    /// is enabled and `world_size > 1`.
    #[cfg(feature = "nccl")]
    comms: Vec<NcclComm>,
}

// ── NCCL wrapper ──────────────────────────────────────────────────────────────

/// Newtype around `cudarc::nccl::safe::Comm` that opts into `Sync`.
///
/// # Safety
///
/// `ncclComm_t` is not thread-safe for concurrent calls, but it IS safe to
/// call sequentially from any thread.  We never call two NCCL operations on
/// the same `Comm` concurrently — the group_start/group_end pattern serialises
/// them — so `Sync` is safe here.
#[cfg(feature = "nccl")]
pub struct NcclComm(pub cudarc::nccl::safe::Comm);

#[cfg(feature = "nccl")]
// SAFETY: see doc comment above.
unsafe impl Sync for NcclComm {}

#[cfg(feature = "nccl")]
impl std::fmt::Debug for NcclComm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NcclComm(rank={})", self.0.rank())
    }
}

// ── Construction ──────────────────────────────────────────────────────────────

impl TpWorld {
    /// Build a world with `world_size` ranks.
    ///
    /// - `world_size = 1` → single CPU or CUDA:0 device, no communication.
    /// - `world_size > 1` → requires at least `world_size` CUDA devices.
    ///
    /// When built with the `nccl` feature, also initialises NCCL communicators
    /// for multi-GPU worlds (world_size > 1).
    ///
    /// # Errors
    ///
    /// Returns an error if `world_size > 1` and fewer than `world_size`
    /// CUDA devices are available, or if NCCL initialisation fails.
    pub fn new(world_size: usize) -> Result<Self> {
        assert!(world_size >= 1, "world_size must be >= 1");

        if world_size == 1 {
            let device = Device::cuda_if_available(0)?;
            return Ok(Self(Arc::new(Inner {
                devices: vec![device],
                #[cfg(feature = "nccl")]
                comms: Vec::new(), // no comms needed for single GPU
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
            .map(Device::new_cuda)
            .collect::<candle_core::Result<Vec<_>>>()?;

        #[cfg(feature = "nccl")]
        let comms = init_nccl_comms(&devices)?;

        Ok(Self(Arc::new(Inner {
            devices,
            #[cfg(feature = "nccl")]
            comms,
        })))
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

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
    #[allow(dead_code)]
    pub fn ranks(&self) -> impl Iterator<Item = (usize, &Device)> {
        self.0.devices.iter().enumerate()
    }

    /// Returns `true` when running on a single GPU (no communication needed).
    #[inline]
    pub fn is_single(&self) -> bool {
        self.0.devices.len() == 1
    }

    // ── All-reduce ────────────────────────────────────────────────────────────

    /// Element-wise sum of `shards` from all ranks, returned on rank 0's device.
    ///
    /// - **NCCL feature enabled**: performs a true device-to-device ring-reduce
    ///   via `ncclAllReduce`.  No GPU→CPU transfer; BW limited only by NVLink
    ///   or PCIe interconnect.
    /// - **NCCL feature disabled**: host-mediated fallback — copies all shards
    ///   to CPU, accumulates, copies result back.  Functionally correct but
    ///   penalised by PCIe transfers on every TP step.
    pub fn all_reduce(&self, shards: Vec<candle_core::Tensor>) -> Result<candle_core::Tensor> {
        #[cfg(feature = "nccl")]
        {
            if shards.len() > 1 {
                return crate::parallel::comm::nccl_all_reduce(&self.0.comms, shards);
            }
        }
        // Fallback: CPU-mediated reduction (or single-shard no-op).
        crate::parallel::comm::all_reduce(&shards, self.device(0))
    }

    // ── Private ───────────────────────────────────────────────────────────────

    fn cuda_device_count() -> usize {
        (0..32).take_while(|&i| Device::new_cuda(i).is_ok()).count()
    }
}

// ── NCCL initialisation ───────────────────────────────────────────────────────

/// Build one `NcclComm` per rank using `Comm::from_devices`.
///
/// `from_devices` handles the barrier internally — all comms are ready when
/// the function returns.
#[cfg(feature = "nccl")]
fn init_nccl_comms(devices: &[Device]) -> Result<Vec<NcclComm>> {
    use candle_core::cuda::CudaDevice;

    let streams: Vec<std::sync::Arc<cudarc::driver::CudaStream>> = devices
        .iter()
        .map(|d| match d {
            Device::Cuda(cd) => Ok(cd.cuda_stream()),
            _ => bail!("NCCL requires CUDA devices; got a non-CUDA device"),
        })
        .collect::<Result<_>>()?;

    let comms = cudarc::nccl::safe::Comm::from_devices(streams)
        .map_err(|e| anyhow::anyhow!("NCCL communicator init failed: {e:?}"))?;

    tracing::info!(world_size = comms.len(), "NCCL communicators initialised");

    Ok(comms.into_iter().map(NcclComm).collect())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_size_1_always_works() {
        let world = TpWorld::new(1).expect("world_size=1 should always succeed");
        assert_eq!(world.world_size(), 1);
        assert!(world.is_single());
    }

    #[test]
    fn device_rank_0_accessible() {
        let world = TpWorld::new(1).unwrap();
        let _dev = world.device(0);
    }

    #[test]
    fn ranks_iterator_yields_all() {
        let world = TpWorld::new(1).unwrap();
        let pairs: Vec<_> = world.ranks().collect();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, 0);
    }

    #[test]
    fn world_size_too_large_errors() {
        let result = TpWorld::new(64);
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
        assert_eq!(world.world_size(), clone.world_size());
    }
}
