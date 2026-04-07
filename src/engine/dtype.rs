//! DType resolution for model weights.
//!
//! Centralises the "which floating-point type should I use?" decision so
//! the --dtype / --bf16 CLI flag has a single place to land.
//!
//! # Hardware notes
//! - V100  (sm_70): no native BF16 — emulates in software, ~3× slower.
//!                  Stick to F16.
//! - A100  (sm_80): native BF16 + F16, both fast.
//! - H100  (sm_90): native BF16 + F16 + FP8.
//! - RTX 3090+ (sm_86): native BF16.
//!
//! Rule: bf16=true is ignored (falls back to F16) below sm_80.
//! We currently don't query compute capability at runtime, so the flag is
//! the user's responsibility.  TODO: query via cudarc DeviceAttribute.

use candle_core::{DType, Device};

/// Pick the weight dtype for the given device and user preference.
pub fn resolve(device: &Device, bf16: bool) -> DType {
    match device {
        Device::Cuda(_) => {
            if bf16 {
                DType::BF16
            } else {
                DType::F16
            }
        }
        _ => DType::F32,
    }
}
