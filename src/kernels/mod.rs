//! Custom CUDA kernels for hot-path LLM operations.
//!
//! Each submodule wraps one `.cu` file compiled by the build script into PTX,
//! then JIT-loaded via cudarc at runtime.
//!
//! # Feature gate
//!
//! All kernels are gated on the `cuda` Cargo feature.  Without it the module
//! stubs return `Err` so callers fall back to the candle eager path.

#[cfg(feature = "cuda")]
pub mod kv_assign;
#[cfg(feature = "cuda")]
pub mod rms_norm;
#[cfg(feature = "cuda")]
pub mod rope;

// CPU-only fallbacks — used when the `cuda` feature is disabled.
// These replicate the candle eager computation so non-GPU builds still work.
#[cfg(not(feature = "cuda"))]
pub mod kv_assign {
    use anyhow::Result;
    use candle_core::Tensor;
    pub fn assign_slot(_buf: &Tensor, _src: &Tensor, _offset: usize) -> Result<Tensor> {
        anyhow::bail!("kv_assign::assign_slot: `cuda` feature not enabled")
    }
}

#[cfg(not(feature = "cuda"))]
pub mod rms_norm {
    use anyhow::Result;
    use candle_core::{D, Tensor};
    pub fn apply(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
        let rms = (x.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?;
        Ok(x.broadcast_div(&rms)?.broadcast_mul(weight)?)
    }
}
#[cfg(not(feature = "cuda"))]
pub mod rope {
    use anyhow::Result;
    use candle_core::{D, Tensor};
    pub fn apply(q: &Tensor, k: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<(Tensor, Tensor)> {
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
}
