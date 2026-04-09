//! Fused RMSNorm CUDA kernel.
//!
//! Replaces candle's multi-op sequence (sqr → mean → add_eps → sqrt → div → mul)
//! with a single kernel that uses warp-level reduction.  On a 7B model with
//! 28 layers and 2 RMSNorm ops per layer that's 56 × 5 → 56 × 1 kernel launches.
//!
//! Falls back to the candle eager path on CPU or unsupported dtype.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

#[cfg(feature = "cuda")]
static PTX: &str = include_str!(concat!(env!("KERNEL_OUT_DIR"), "/rms_norm.ptx"));

#[cfg(feature = "cuda")]
const MODULE: &str = "vllm_hb_rms_norm";

/// `out = x / rms(x) * weight`
///
/// - `x`      : `[..., hidden]` (F32 or F16, any leading dims)
/// - `weight` : `[hidden]` (same dtype)
/// - `eps`    : numerical stability constant (typically 1e-5 or 1e-6)
pub fn apply(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = x.dtype();
    if !matches!(dtype, DType::F32 | DType::F16) {
        return candle_fallback(x, weight, eps);
    }

    #[cfg(feature = "cuda")]
    if matches!(x.device(), Device::Cuda(_)) {
        return Ok(x.apply_op2_no_bwd(weight, &FusedRmsNorm { eps })?);
    }

    candle_fallback(x, weight, eps)
}

// ── CustomOp2 implementation (CUDA) ──────────────────────────────────────────

#[cfg(feature = "cuda")]
struct FusedRmsNorm {
    eps: f64,
}

#[cfg(feature = "cuda")]
impl candle_core::CustomOp2 for FusedRmsNorm {
    fn name(&self) -> &'static str {
        "fused-rms-norm"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle_core::CpuStorage,
        _l1: &candle_core::Layout,
        _s2: &candle_core::CpuStorage,
        _l2: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        // apply() calls candle_fallback before reaching this path on CPU.
        candle_core::bail!("fused-rms-norm: unreachable cpu_fwd")
    }

    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &candle_core::Layout,
        s2: &candle_core::CudaStorage,
        _l2: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use candle_core::backend::BackendStorage;
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::{CudaStorageSlice, WrapErr};

        let dev = s1.device();

        let shape  = l1.shape();
        let hidden = *shape.dims().last().unwrap();
        let rows   = shape.elem_count() / hidden;

        let block: u32 = if hidden <= 256 { 256 } else { 512 };
        let fn_name = match (s1.dtype(), block) {
            (candle_core::DType::F32, 256) => "rms_norm_f32_256",
            (candle_core::DType::F32, _)   => "rms_norm_f32_512",
            (candle_core::DType::F16, 256) => "rms_norm_f16_256",
            (candle_core::DType::F16, _)   => "rms_norm_f16_512",
            (dt, _) => candle_core::bail!("rms_norm: unsupported dtype {:?}", dt),
        };

        let func = dev.get_or_load_custom_func(fn_name, MODULE, PTX)?;

        let cfg = LaunchConfig {
            grid_dim:         (rows as u32, 1, 1),
            block_dim:        (block, 1, 1),
            shared_mem_bytes: 32 * 4,
        };

        let hidden_i = hidden as i32;
        let eps_f    = self.eps as f32;

        let slice = match (&s1.slice, &s2.slice) {
            (CudaStorageSlice::F32(x_sl), CudaStorageSlice::F32(w_sl)) => {
                let x_sl = x_sl.slice(l1.start_offset()..);
                let dst = unsafe { dev.alloc::<f32>(shape.elem_count()) }.w()?;
                let mut b = func.builder();
                b.arg(&dst).arg(&x_sl).arg(w_sl).arg(&hidden_i).arg(&eps_f);
                unsafe { b.launch(cfg) }.w()?;
                CudaStorageSlice::F32(dst)
            }
            (CudaStorageSlice::F16(x_sl), CudaStorageSlice::F16(w_sl)) => {
                let x_sl = x_sl.slice(l1.start_offset()..);
                let dst = unsafe { dev.alloc::<half::f16>(shape.elem_count()) }.w()?;
                let mut b = func.builder();
                b.arg(&dst).arg(&x_sl).arg(w_sl).arg(&hidden_i).arg(&eps_f);
                unsafe { b.launch(cfg) }.w()?;
                CudaStorageSlice::F16(dst)
            }
            _ => candle_core::bail!("rms_norm: x and weight dtype mismatch"),
        };

        let out = candle_core::CudaStorage {
            slice,
            device: dev.clone(),
        };
        Ok((out, shape.clone()))
    }
}

// ── Candle fallback (CPU or unsupported dtype) ────────────────────────────────

fn candle_fallback(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    use candle_core::D;
    let rms = (x.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?;
    Ok(x.broadcast_div(&rms)?.broadcast_mul(weight)?)
}
