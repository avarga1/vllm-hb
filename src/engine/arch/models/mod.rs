//! Vendored model implementations with fused kernel patches.
//!
//! These are local copies of candle-transformers models with the RMSNorm
//! forward pass swapped out for our single-kernel implementation.

pub mod qwen2;
pub mod qwen3;

// ── Shared fused RMSNorm ──────────────────────────────────────────────────────

/// Thin wrapper that loads the weight via VarBuilder and dispatches to
/// `crate::kernels::rms_norm::apply`.  Replaces `with_tracing::RmsNorm`.
#[derive(Debug, Clone)]
pub(super) struct RmsNorm {
    weight: candle_core::Tensor,
    eps:    f64,
}

impl RmsNorm {
    pub(super) fn new(
        size: usize,
        eps: f64,
        vb: candle_nn::VarBuilder,
    ) -> candle_core::Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }
}

impl candle_core::Module for RmsNorm {
    fn forward(&self, x: &candle_core::Tensor) -> candle_core::Result<candle_core::Tensor> {
        crate::kernels::rms_norm::apply(x, &self.weight, self.eps)
            .map_err(candle_core::Error::wrap)
    }
}
