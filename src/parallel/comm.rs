//! Inter-GPU communication primitives for tensor parallelism.
//!
//! # Design
//!
//! All operations are expressed over `candle_core::Tensor` slices so they
//! compose naturally with the rest of the engine.
//!
//! # Single-GPU fast path
//!
//! `all_reduce` with a single shard returns the shard (moved to `device`)
//! without any copy or arithmetic — zero overhead for `tp=1`.
//!
//! # Multi-GPU implementation
//!
//! Reduction is host-mediated: each shard is moved to CPU, accumulated with
//! `Tensor::add`, then moved to the target device.  This avoids the NCCL
//! dependency for a first working implementation.  When NCCL bindings in
//! `cudarc` are stable the accumulation loop can be replaced by a single
//! `ncclAllReduce` call without changing the call-sites.
//!
//! # All-gather
//!
//! `all_gather` concatenates shards along a specified dimension — used to
//! reassemble column-parallel outputs before feeding the next layer.

use anyhow::{Result, bail};
use candle_core::{Device, Tensor};

// ── all_reduce ────────────────────────────────────────────────────────────────

/// Sum `shards` element-wise and place the result on `device`.
///
/// - `shards.len() == 1` — returns the shard moved to `device`; no arithmetic.
/// - `shards.len() > 1`  — host-mediated reduce: CPU sum → target device.
///
/// All shards must have identical shapes.
///
/// # Errors
///
/// Returns an error if `shards` is empty, shapes differ, or any tensor
/// operation fails.
pub fn all_reduce(shards: &[Tensor], device: &Device) -> Result<Tensor> {
    match shards.len() {
        0 => bail!("all_reduce: shard list is empty"),
        1 => Ok(shards[0].to_device(device)?),
        _ => {
            // Accumulate on CPU to avoid NCCL dependency.
            let cpu = &Device::Cpu;
            let mut acc = shards[0].to_device(cpu)?;
            for shard in &shards[1..] {
                let s = shard.to_device(cpu)?;
                acc = acc.add(&s)?;
            }
            Ok(acc.to_device(device)?)
        }
    }
}

// ── all_gather ────────────────────────────────────────────────────────────────

/// Concatenate `shards` along `dim` and place the result on `device`.
///
/// Used after a column-parallel linear to reassemble the full output
/// before the next row-parallel layer.
///
/// # Errors
///
/// Returns an error if `shards` is empty or any tensor operation fails.
#[allow(dead_code)]
pub fn all_gather(shards: &[Tensor], dim: usize, device: &Device) -> Result<Tensor> {
    match shards.len() {
        0 => bail!("all_gather: shard list is empty"),
        1 => Ok(shards[0].to_device(device)?),
        _ => {
            let cpu = &Device::Cpu;
            let cpu_shards = shards
                .iter()
                .map(|s| s.to_device(cpu))
                .collect::<candle_core::Result<Vec<_>>>()?;
            let gathered = Tensor::cat(&cpu_shards, dim)?;
            Ok(gathered.to_device(device)?)
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    fn cpu_tensor(data: &[f32], shape: &[usize]) -> Tensor {
        Tensor::from_vec(data.to_vec(), shape, &Device::Cpu).unwrap()
    }

    // ── all_reduce ────────────────────────────────────────────────────────────

    #[test]
    fn all_reduce_single_shard_is_passthrough() {
        let t = cpu_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let result = all_reduce(&[t], &Device::Cpu).unwrap();
        let data: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn all_reduce_two_shards_sums_elementwise() {
        let a = cpu_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = cpu_tensor(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
        let result = all_reduce(&[a, b], &Device::Cpu).unwrap();
        let data: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(data, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn all_reduce_four_shards() {
        // 4-way reduce: each rank contributes 1.0 → sum = 4.0
        let shards: Vec<Tensor> = (0..4).map(|_| cpu_tensor(&[1.0, 1.0], &[1, 2])).collect();
        let result = all_reduce(&shards, &Device::Cpu).unwrap();
        let data: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(data, vec![4.0, 4.0]);
    }

    #[test]
    fn all_reduce_empty_errors() {
        let result = all_reduce(&[], &Device::Cpu);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }

    #[test]
    fn all_reduce_preserves_dtype() {
        let t = Tensor::zeros((2, 3), DType::F32, &Device::Cpu).unwrap();
        let result = all_reduce(&[t], &Device::Cpu).unwrap();
        assert_eq!(result.dtype(), DType::F32);
    }

    // ── all_gather ────────────────────────────────────────────────────────────

    #[test]
    fn all_gather_single_shard_is_passthrough() {
        let t = cpu_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let result = all_gather(&[t], 0, &Device::Cpu).unwrap();
        let data: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn all_gather_two_shards_along_dim0() {
        // Rank 0: rows [1, 2], rank 1: rows [3, 4] → gathered: [1, 2, 3, 4]
        let a = cpu_tensor(&[1.0, 2.0], &[1, 2]);
        let b = cpu_tensor(&[3.0, 4.0], &[1, 2]);
        let result = all_gather(&[a, b], 0, &Device::Cpu).unwrap();
        assert_eq!(result.dims(), &[2, 2]);
        let data: Vec<f32> = result.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn all_gather_two_shards_along_dim1() {
        // Rank 0: cols [1, 3], rank 1: cols [2, 4] → gathered shape [2, 4]
        let a = cpu_tensor(&[1.0, 3.0], &[2, 1]);
        let b = cpu_tensor(&[2.0, 4.0], &[2, 1]);
        let result = all_gather(&[a, b], 1, &Device::Cpu).unwrap();
        assert_eq!(result.dims(), &[2, 2]);
    }

    #[test]
    fn all_gather_empty_errors() {
        let result = all_gather(&[], 0, &Device::Cpu);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty"));
    }
}
