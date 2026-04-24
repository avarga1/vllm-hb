//! KV slot-assign kernel.
//!
//! Writes one token's K or V vector into a pre-allocated `[1, nkv, max_seq, head_dim]`
//! buffer at a specific sequence position.  The output tensor always has the same shape
//! as the input buffer — this constant shape is what makes CUDA-graph capture possible.
//!
//! # Cost model
//! Each call does one DtoD clone of the full buffer (constant = `max_seq * nkv * head_dim`
//! elements) plus one small kernel write (1 token worth of data).  For a pre-alloc buffer
//! sized to the benchmark window (≤ 2048 tokens) this is negligible compared to the matmuls.
//!
//! # Why not `apply_op1_no_bwd` / true in-place?
//! candle's `Tensor` is immutable — there is no public `slice_assign`.  `CustomOp2::cuda_fwd`
//! must produce a new `CudaStorage`.  We DtoD-clone the full buffer into `dst`, then the
//! kernel patches one slot.  The net allocation per decode step is one fixed-size buffer
//! instead of an ever-growing cat-allocated tensor.

use anyhow::Result;
use candle_core::{Device, Tensor};

#[cfg(feature = "cuda")]
static PTX: &str = include_str!(concat!(env!("KERNEL_OUT_DIR"), "/kv_assign.ptx"));

#[cfg(feature = "cuda")]
const MODULE: &str = "vllm_hb_kv_assign";

/// Write one token's K or V into a pre-allocated buffer at sequence position `offset`.
///
/// - `buf`    : `[1, nkv, max_seq, head_dim]`  (fixed shape)
/// - `src`    : `[1, nkv,       1, head_dim]`  (one token after RoPE)
/// - `offset` : sequence slot index (= current `seqlen_offset`)
///
/// Returns a new tensor of the same shape as `buf` with `src` written at `offset`.
pub fn assign_slot(buf: &Tensor, src: &Tensor, offset: usize) -> Result<Tensor> {
    #[cfg(feature = "cuda")]
    if matches!(buf.device(), Device::Cuda(_)) {
        return Ok(buf
            .apply_op2_no_bwd(src, &KvSlotAssign { offset })
            .map_err(anyhow::Error::from)?);
    }
    anyhow::bail!("kv_assign::assign_slot requires the `cuda` feature and a CUDA device")
}

// ── CustomOp2 ─────────────────────────────────────────────────────────────────

#[cfg(feature = "cuda")]
struct KvSlotAssign {
    offset: usize,
}

#[cfg(feature = "cuda")]
impl candle_core::CustomOp2 for KvSlotAssign {
    fn name(&self) -> &'static str {
        "kv-slot-assign"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle_core::CpuStorage,
        _l1: &candle_core::Layout,
        _s2: &candle_core::CpuStorage,
        _l2: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        // assign_slot() checks the device before calling apply_op2_no_bwd.
        candle_core::bail!("kv-slot-assign: unreachable on CPU")
    }

    /// `s1` = buf  `[1, nkv, max_seq, head_dim]`
    /// `s2` = src  `[1, nkv,       1, head_dim]`
    ///
    /// Strategy: DtoD-clone the full buf into `dst`, then launch the kernel to
    /// patch slot `offset` with the values from `src`.  We use a scoped block so
    /// the `CudaView` borrows from `dst` are dropped before we move `dst` into
    /// the return value — this resolves the "CudaView has no try_clone" error from
    /// the previous attempt where we sliced first and then tried to own the view.
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage, // buf
        l1: &candle_core::Layout,
        s2: &candle_core::CudaStorage, // src
        l2: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::{CudaStorageSlice, WrapErr};

        let dev = s1.device();
        let shape = l1.shape(); // [1, nkv, max_seq, head_dim]
        let dims = shape.dims();

        let nkv = dims[1] as i32;
        let max_seq = dims[2] as i32;
        let head_dim = dims[3] as i32;
        let offset_i = self.offset as i32;

        // One thread per (kv_head × head_dim) element.
        let total_threads = (nkv * head_dim) as u32;
        let block = 128u32;
        let grid = total_threads.div_ceil(block);

        let cfg = LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };

        let slice = match (&s1.slice, &s2.slice) {
            // ── F16 ──────────────────────────────────────────────────────────
            (CudaStorageSlice::F16(buf_sl), CudaStorageSlice::F16(src_sl)) => {
                let func = dev.get_or_load_custom_func("kv_slot_assign_f16", MODULE, PTX)?;
                // buf_sl : &CudaSlice<f16>  (owned by s1.slice, we have a ref)
                // try_clone() does a DtoD memcpy → new owned CudaSlice<f16>
                let dst = buf_sl.try_clone().map_err(candle_core::Error::wrap)?;
                {
                    // CudaView borrows from dst; drop before we move dst.
                    let dst_view = dst.slice(l1.start_offset()..);
                    let src_view = src_sl.slice(l2.start_offset()..);
                    let mut b = func.builder();
                    b.arg(&dst_view)
                        .arg(&src_view)
                        .arg(&nkv)
                        .arg(&max_seq)
                        .arg(&head_dim)
                        .arg(&offset_i);
                    unsafe { b.launch(cfg) }.w()?;
                } // dst_view, src_view dropped here
                CudaStorageSlice::F16(dst)
            }
            // ── F32 ──────────────────────────────────────────────────────────
            (CudaStorageSlice::F32(buf_sl), CudaStorageSlice::F32(src_sl)) => {
                let func = dev.get_or_load_custom_func("kv_slot_assign_f32", MODULE, PTX)?;
                let dst = buf_sl.try_clone().map_err(candle_core::Error::wrap)?;
                {
                    let dst_view = dst.slice(l1.start_offset()..);
                    let src_view = src_sl.slice(l2.start_offset()..);
                    let mut b = func.builder();
                    b.arg(&dst_view)
                        .arg(&src_view)
                        .arg(&nkv)
                        .arg(&max_seq)
                        .arg(&head_dim)
                        .arg(&offset_i);
                    unsafe { b.launch(cfg) }.w()?;
                }
                CudaStorageSlice::F32(dst)
            }
            _ => candle_core::bail!("kv-slot-assign: dtype mismatch between buf and src"),
        };

        let out = candle_core::CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((out, shape.clone()))
    }
}
