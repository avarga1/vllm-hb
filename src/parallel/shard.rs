//! Weight-sharding strategies for tensor parallelism.
//!
//! Two fundamental patterns cover the entire transformer:
//!
//! ```text
//! Column-parallel (QKV, gate/up projections)
//! ─────────────────────────────────────────
//!  weight  [out, in]
//!  rank 0  [out/N, in]   ← narrow(dim=0, start=0,       len=out/N)
//!  rank 1  [out/N, in]   ← narrow(dim=0, start=out/N,   len=out/N)
//!  …
//!  Each rank computes a partial output independently; no all-reduce here.
//!
//! Row-parallel (output projection, FFN down)
//! ──────────────────────────────────────────
//!  weight  [out, in]
//!  rank 0  [out, in/N]   ← narrow(dim=1, start=0,      len=in/N)
//!  rank 1  [out, in/N]   ← narrow(dim=1, start=in/N,   len=in/N)
//!  …
//!  Each rank multiplies its input shard by its weight shard, then an
//!  all-reduce sums the partial outputs across ranks.
//! ```
//!
//! Both functions are pure `candle_core::Tensor` operations — no device
//! I/O, no CUDA required.  A CPU tensor of any shape can be sharded.
//!
//! # Bias sharding
//!
//! `bias_shard` covers column-parallel biases (split same way as weight
//! rows).  Row-parallel biases are added by rank 0 only; that caller-side
//! convention is outside this module.

use anyhow::{Result, bail};
use candle_core::Tensor;

// ── Column-parallel ───────────────────────────────────────────────────────────

/// Return the column-parallel shard for `rank` out of `world_size`.
///
/// Splits the *output* dimension (dim 0) of a 2-D weight matrix
/// `[out_features, in_features]`.
///
/// # Errors
///
/// Returns an error if `out_features` is not evenly divisible by `world_size`.
pub fn column_shard(weight: &Tensor, rank: usize, world_size: usize) -> Result<Tensor> {
    debug_assert!(world_size >= 1);
    let rows = weight.dim(0)?;
    if rows % world_size != 0 {
        bail!(
            "column_shard: out_features ({rows}) is not divisible by world_size ({world_size})"
        );
    }
    let chunk = rows / world_size;
    Ok(weight.narrow(0, rank * chunk, chunk)?)
}

// ── Row-parallel ──────────────────────────────────────────────────────────────

/// Return the row-parallel shard for `rank` out of `world_size`.
///
/// Splits the *input* dimension (dim 1) of a 2-D weight matrix
/// `[out_features, in_features]`.
///
/// # Errors
///
/// Returns an error if `in_features` is not evenly divisible by `world_size`.
pub fn row_shard(weight: &Tensor, rank: usize, world_size: usize) -> Result<Tensor> {
    debug_assert!(world_size >= 1);
    let cols = weight.dim(1)?;
    if cols % world_size != 0 {
        bail!(
            "row_shard: in_features ({cols}) is not divisible by world_size ({world_size})"
        );
    }
    let chunk = cols / world_size;
    Ok(weight.narrow(1, rank * chunk, chunk)?)
}

// ── Bias ──────────────────────────────────────────────────────────────────────

/// Shard a 1-D bias vector for a column-parallel layer.
///
/// Bias has shape `[out_features]`; we split the same way as `column_shard`.
///
/// # Errors
///
/// Returns an error if `out_features` is not evenly divisible by `world_size`.
pub fn bias_shard(bias: &Tensor, rank: usize, world_size: usize) -> Result<Tensor> {
    debug_assert!(world_size >= 1);
    let len = bias.dim(0)?;
    if len % world_size != 0 {
        bail!(
            "bias_shard: out_features ({len}) is not divisible by world_size ({world_size})"
        );
    }
    let chunk = len / world_size;
    Ok(bias.narrow(0, rank * chunk, chunk)?)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Number of output features owned by `rank` in a column-parallel split.
#[inline]
pub fn column_chunk_size(out_features: usize, world_size: usize) -> usize {
    out_features / world_size
}

/// Number of input features owned by `rank` in a row-parallel split.
#[inline]
pub fn row_chunk_size(in_features: usize, world_size: usize) -> usize {
    in_features / world_size
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn cpu_tensor(data: &[f32], shape: &[usize]) -> Tensor {
        Tensor::from_vec(data.to_vec(), shape, &Device::Cpu).unwrap()
    }

    // ── column_shard ─────────────────────────────────────────────────────────

    #[test]
    fn column_shard_2way_shape() {
        // weight [4, 3] → each rank gets [2, 3]
        let w = cpu_tensor(&[0.0f32; 12], &[4, 3]);
        let s0 = column_shard(&w, 0, 2).unwrap();
        let s1 = column_shard(&w, 1, 2).unwrap();
        assert_eq!(s0.dims(), &[2, 3]);
        assert_eq!(s1.dims(), &[2, 3]);
    }

    #[test]
    fn column_shard_2way_values() {
        // rows: [0,1,2,3]; rank 0 → rows [0,1], rank 1 → rows [2,3]
        let w = cpu_tensor(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[4, 2]);
        let s0 = column_shard(&w, 0, 2).unwrap();
        let s1 = column_shard(&w, 1, 2).unwrap();

        let d0: Vec<f32> = s0.flatten_all().unwrap().to_vec1().unwrap();
        let d1: Vec<f32> = s1.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(d0, vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(d1, vec![4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn column_shard_4way() {
        let w = cpu_tensor(&[0.0f32; 8], &[8, 1]);
        for rank in 0..4 {
            let s = column_shard(&w, rank, 4).unwrap();
            assert_eq!(s.dims(), &[2, 1], "rank {rank} wrong shape");
        }
    }

    #[test]
    fn column_shard_1way_is_full_weight() {
        let w = cpu_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let s = column_shard(&w, 0, 1).unwrap();
        assert_eq!(s.dims(), w.dims());
        let d: Vec<f32> = s.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(d, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn column_shard_not_divisible_errors() {
        let w = cpu_tensor(&[0.0f32; 9], &[3, 3]); // 3 rows, world_size=2
        let result = column_shard(&w, 0, 2);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("divisible"), "unexpected error: {msg}");
    }

    // ── row_shard ─────────────────────────────────────────────────────────────

    #[test]
    fn row_shard_2way_shape() {
        // weight [3, 4] → each rank gets [3, 2]
        let w = cpu_tensor(&[0.0f32; 12], &[3, 4]);
        let s0 = row_shard(&w, 0, 2).unwrap();
        let s1 = row_shard(&w, 1, 2).unwrap();
        assert_eq!(s0.dims(), &[3, 2]);
        assert_eq!(s1.dims(), &[3, 2]);
    }

    #[test]
    fn row_shard_2way_values() {
        // weight [2, 4]; rank 0 → cols [0,1], rank 1 → cols [2,3]
        let w = cpu_tensor(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[2, 4]);
        let s0 = row_shard(&w, 0, 2).unwrap();
        let s1 = row_shard(&w, 1, 2).unwrap();

        let d0: Vec<f32> = s0.flatten_all().unwrap().to_vec1().unwrap();
        let d1: Vec<f32> = s1.flatten_all().unwrap().to_vec1().unwrap();
        // Row 0: [0,1,2,3], Row 1: [4,5,6,7]
        // Rank 0 gets cols 0-1: [0,1,4,5], rank 1 gets cols 2-3: [2,3,6,7]
        assert_eq!(d0, vec![0.0, 1.0, 4.0, 5.0]);
        assert_eq!(d1, vec![2.0, 3.0, 6.0, 7.0]);
    }

    #[test]
    fn row_shard_4way() {
        let w = cpu_tensor(&[0.0f32; 8], &[1, 8]);
        for rank in 0..4 {
            let s = row_shard(&w, rank, 4).unwrap();
            assert_eq!(s.dims(), &[1, 2], "rank {rank} wrong shape");
        }
    }

    #[test]
    fn row_shard_not_divisible_errors() {
        let w = cpu_tensor(&[0.0f32; 9], &[3, 3]); // 3 cols, world_size=2
        let result = row_shard(&w, 0, 2);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("divisible"), "unexpected error: {msg}");
    }

    // ── bias_shard ────────────────────────────────────────────────────────────

    #[test]
    fn bias_shard_2way() {
        let b = cpu_tensor(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let s0 = bias_shard(&b, 0, 2).unwrap();
        let s1 = bias_shard(&b, 1, 2).unwrap();

        let d0: Vec<f32> = s0.to_vec1().unwrap();
        let d1: Vec<f32> = s1.to_vec1().unwrap();
        assert_eq!(d0, vec![1.0, 2.0]);
        assert_eq!(d1, vec![3.0, 4.0]);
    }

    // ── chunk size helpers ────────────────────────────────────────────────────

    #[test]
    fn chunk_size_helpers() {
        assert_eq!(column_chunk_size(4096, 4), 1024);
        assert_eq!(row_chunk_size(4096, 8), 512);
        assert_eq!(column_chunk_size(1024, 1), 1024);
    }

    // ── round-trip: shard then gather matches original ────────────────────────

    #[test]
    fn column_shard_round_trips_via_cat() {
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let w = cpu_tensor(&data, &[4, 3]);

        let shards: Vec<Tensor> = (0..2)
            .map(|rank| column_shard(&w, rank, 2).unwrap())
            .collect();
        let reassembled = Tensor::cat(&shards, 0).unwrap();

        let orig: Vec<f32> = w.flatten_all().unwrap().to_vec1().unwrap();
        let got: Vec<f32> = reassembled.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(orig, got);
    }

    #[test]
    fn row_shard_round_trips_via_cat() {
        let data: Vec<f32> = (0..12).map(|x| x as f32).collect();
        let w = cpu_tensor(&data, &[3, 4]);

        let shards: Vec<Tensor> = (0..2)
            .map(|rank| row_shard(&w, rank, 2).unwrap())
            .collect();
        let reassembled = Tensor::cat(&shards, 1).unwrap();

        let orig: Vec<f32> = w.flatten_all().unwrap().to_vec1().unwrap();
        let got: Vec<f32> = reassembled.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(orig, got);
    }
}
