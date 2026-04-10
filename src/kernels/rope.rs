//! Fused RoPE (Rotary Position Embedding) CUDA kernel.
//!
//! Applies pre-computed cos/sin tables to Q and K in a single pass per tensor,
//! replacing candle's multi-op sequence with one kernel launch per tensor.
//!
//! Layout expectations (Qwen2/Llama convention):
//!   q, k:    [batch, num_heads, seq_len, head_dim]  (contiguous, F32 or F16)
//!   cos/sin: [seq_len, head_dim/2]                  (pre-sliced for current pos)
//!
//! Falls back to the candle eager path on CPU or unsupported dtype.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

#[cfg(feature = "cuda")]
static PTX: &str = include_str!(concat!(env!("KERNEL_OUT_DIR"), "/rope.ptx"));

#[cfg(feature = "cuda")]
const MODULE: &str = "vllm_hb_rope";

/// Fused RoPE: applies cos/sin to both `q` and `k`, returns `(out_q, out_k)`.
///
/// - `q`, `k`    : `[batch, num_heads, seq_len, head_dim]`
/// - `cos`, `sin`: `[seq_len, head_dim/2]`
pub fn apply(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let dtype = q.dtype();
    if !matches!(dtype, DType::F32 | DType::F16) {
        return candle_fallback(q, k, cos, sin);
    }

    #[cfg(feature = "cuda")]
    if matches!(q.device(), Device::Cuda(_)) {
        let q_dims = q.dims();
        if q_dims.len() == 4 {
            let seq_len  = q_dims[2] as i32;
            let head_dim = q_dims[3] as i32;
            let op = RopeSingle { sin: sin.clone(), seq_len, head_dim };
            let out_q = q.apply_op2_no_bwd(cos, &op)?;
            let out_k = k.apply_op2_no_bwd(cos, &op)?;
            return Ok((out_q, out_k));
        }
    }

    candle_fallback(q, k, cos, sin)
}

// ── CustomOp2 implementation (CUDA) ──────────────────────────────────────────

/// Applies `rope_single_{f32,f16}` to one tensor using the captured `sin` table.
/// Grid: (batch * num_heads * seq_len,)  Block: (head_dim / 2,)
#[cfg(feature = "cuda")]
struct RopeSingle {
    sin:      Tensor,
    seq_len:  i32,
    head_dim: i32,
}

#[cfg(feature = "cuda")]
impl candle_core::CustomOp2 for RopeSingle {
    fn name(&self) -> &'static str {
        "fused-rope"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle_core::CpuStorage,
        _l1: &candle_core::Layout,
        _s2: &candle_core::CpuStorage,
        _l2: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        // apply() routes CPU through candle_fallback before reaching this path.
        candle_core::bail!("fused-rope: unreachable cpu_fwd")
    }

    fn cuda_fwd(
        &self,
        s_inp: &candle_core::CudaStorage,
        l_inp: &candle_core::Layout,
        s_cos: &candle_core::CudaStorage,
        l_cos: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::{CudaStorageSlice, WrapErr};

        let dev = s_inp.device();

        let (sin_st, sin_layout) = self.sin.storage_and_layout();
        let s_sin = match &*sin_st {
            candle_core::Storage::Cuda(cs) => cs,
            _ => candle_core::bail!("rope: sin must be on CUDA"),
        };

        let shape    = l_inp.shape();
        let n        = shape.elem_count();
        let tokens   = n / self.head_dim as usize;  // batch * heads * seq_len
        let half_dim = self.head_dim / 2;

        // Grid: one block per (batch, head, token) triple.
        // Block: one thread per half-dim element.
        let cfg = LaunchConfig {
            grid_dim:         (tokens as u32, 1, 1),
            block_dim:        (half_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let inp_off = l_inp.start_offset();
        let cos_off = l_cos.start_offset();
        let sin_off = sin_layout.start_offset();
        let seq_len  = self.seq_len;
        let head_dim = self.head_dim;

        let slice = match (&s_inp.slice, &s_cos.slice, &s_sin.slice) {
            (
                CudaStorageSlice::F32(inp_sl),
                CudaStorageSlice::F32(cos_sl),
                CudaStorageSlice::F32(sin_sl),
            ) => {
                let inp_sl = inp_sl.slice(inp_off..);
                let cos_sl = cos_sl.slice(cos_off..);
                let sin_sl = sin_sl.slice(sin_off..);
                let dst = unsafe { dev.alloc::<f32>(n) }?;
                let func = dev.get_or_load_custom_func("rope_single_f32", MODULE, PTX)?;
                let mut b = func.builder();
                b.arg(&dst).arg(&inp_sl).arg(&cos_sl).arg(&sin_sl)
                 .arg(&seq_len).arg(&head_dim);
                unsafe { b.launch(cfg) }.w()?;
                CudaStorageSlice::F32(dst)
            }
            (
                CudaStorageSlice::F16(inp_sl),
                CudaStorageSlice::F16(cos_sl),
                CudaStorageSlice::F16(sin_sl),
            ) => {
                let inp_sl = inp_sl.slice(inp_off..);
                let cos_sl = cos_sl.slice(cos_off..);
                let sin_sl = sin_sl.slice(sin_off..);
                let dst = unsafe { dev.alloc::<half::f16>(n) }?;
                let func = dev.get_or_load_custom_func("rope_single_f16", MODULE, PTX)?;
                let mut b = func.builder();
                b.arg(&dst).arg(&inp_sl).arg(&cos_sl).arg(&sin_sl)
                 .arg(&seq_len).arg(&head_dim);
                unsafe { b.launch(cfg) }.w()?;
                CudaStorageSlice::F16(dst)
            }
            _ => candle_core::bail!("rope: dtype mismatch between inp, cos, sin"),
        };

        let out = candle_core::CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((out, shape.clone()))
    }
}

// ── Candle fallback (CPU or unsupported dtype) ────────────────────────────────

fn candle_fallback(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    use candle_core::D;
    let half = q.dim(D::Minus1)? / 2;

    let q0 = q.narrow(D::Minus1, 0, half)?;
    let q1 = q.narrow(D::Minus1, half, half)?;
    let k0 = k.narrow(D::Minus1, 0, half)?;
    let k1 = k.narrow(D::Minus1, half, half)?;

    let out_q = Tensor::cat(
        &[
            &(q0.broadcast_mul(cos)? - q1.broadcast_mul(sin)?)?,
            &(q0.broadcast_mul(sin)? + q1.broadcast_mul(cos)?)?,
        ],
        D::Minus1,
    )?;
    let out_k = Tensor::cat(
        &[
            &(k0.broadcast_mul(cos)? - k1.broadcast_mul(sin)?)?,
            &(k0.broadcast_mul(sin)? + k1.broadcast_mul(cos)?)?,
        ],
        D::Minus1,
    )?;

    Ok((out_q, out_k))
}
